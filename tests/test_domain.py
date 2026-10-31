"""Source-confounding regression tests: no background/color/lecture shortcuts."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sokolov.config import load_config
from sokolov.io import read_csv, read_json, write_csv

ROOT = Path(__file__).resolve().parents[1]


class DomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_mask_blocks_background_and_affine_light_changes(self):
        import torch
        from sokolov.preprocessing import preprocess_face, face_mask
        torch.manual_seed(42)
        x = torch.rand(1, 3, 112, 112)*.6+.2
        original = preprocess_face(x, 'identity_v1')
        changed = x.clone()
        changed[:, :, face_mask(x) == 0] = torch.rand_like(changed[:, :, face_mask(x) == 0])
        torch.testing.assert_close(original, preprocess_face(changed, 'identity_v1'), rtol=0, atol=1e-6)
        torch.testing.assert_close(original, preprocess_face(x*.65+.15, 'identity_v1'), rtol=0, atol=2e-6)
        self.assertTrue(torch.equal(original[:, 0], original[:, 1]))
        gradient_input = x.clone().requires_grad_(True)
        output = preprocess_face(gradient_input, 'identity_v1')
        (output*torch.rand_like(output)).sum().backward()
        self.assertTrue(torch.isfinite(gradient_input.grad).all())
        self.assertEqual(gradient_input.grad[:, :, face_mask(x) == 0].abs().max().item(), 0.)
        flat = torch.full_like(x, .5, requires_grad=True)
        preprocess_face(flat, 'identity_v1').sum().backward()
        self.assertTrue(torch.isfinite(flat.grad).all())

    def test_legacy_and_generation_use_the_same_normalization(self):
        import torch
        from torchvision.transforms import Normalize
        from sokolov.preprocessing import normalized_face
        from sokolov.synthesize import normalized_pixels
        for name, side in [('resnet18', 224), ('arcface', 112)]:
            x = torch.rand(3, side, side)
            mean, std = ((.485, .456, .406), (.229, .224, .225)) if name == 'resnet18' else ((.5,)*3, (.5,)*3)
            torch.testing.assert_close(normalized_face(x, name), Normalize(mean, std)(x), rtol=0, atol=0)
            torch.testing.assert_close(normalized_face(x, name, 'identity_v1'),
                normalized_pixels(x.unsqueeze(0), name, 'identity_v1')[0], rtol=0, atol=0)

    def test_old_checkpoints_keep_their_original_rgb_pipeline(self):
        from dataclasses import asdict
        from sokolov.synthesize import checkpoint_config
        old = asdict(load_config(ROOT/'configs/resnet18_face.toml'))
        old['model'].pop('face_preprocessing')
        old['data'].pop('extra_manifest');old['data'].pop('reuse_prepared_manifest')
        old['split'].pop('train_only_sources')
        old['training'].pop('augmentation');old['training'].pop('source_balanced')
        config = checkpoint_config({'config': old})
        self.assertEqual(config.model.face_preprocessing, 'rgb')
        self.assertEqual(config.training.augmentation, 'light')
        self.assertFalse(config.training.source_balanced)

    def test_sampling_equalizes_domains_and_sources_without_changing_class_mass(self):
        from collections import defaultdict
        from sokolov.data.dataset import source_sampling_weights
        rows = ([{'label': 1, 'source_id': 'lecture1'}]*100 +
                [{'label': 1, 'source_id': 'web:a'}]*2 + [{'label': 1, 'source_id': 'web:b'}] +
                [{'label': 0, 'source_id': 'fairface:a'}]*20 + [{'label': 0, 'source_id': 'web:c'}]*4)
        weights = source_sampling_weights(rows)
        mass = defaultdict(float)
        for row, weight in zip(rows, weights):
            mass[(row['label'], row['source_id'].split(':')[0])] += weight
        self.assertAlmostEqual(mass[(1, 'lecture1')], 103/2)
        self.assertAlmostEqual(mass[(1, 'web')], 103/2)
        self.assertAlmostEqual(mass[(0, 'fairface')], 24/2)
        self.assertAlmostEqual(mass[(0, 'web')], 24/2)
        self.assertAlmostEqual(sum(w for r, w in zip(rows, weights) if r['source_id'] == 'web:a'), 103/4)

    def test_group_split_keeps_lectures_and_near_duplicates_together_and_reuses_crops(self):
        import numpy as np
        from PIL import Image
        from test_pipeline import fixtures, FakeDetector
        from sokolov.data.download import download_data
        from sokolov.data.prepare import prepare
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixtures(root)
            pixels = np.asarray(Image.open(root/'positive/p0.png')).copy()
            Image.fromarray(pixels+1).save(root/'positive/p1.png')
            download_data(config)
            grouped = replace(config, split=replace(config.split, mode='group_stratified'))
            prepare(grouped, detector=FakeDetector())
            rows = read_csv(config.data.prepared_manifest)
            by_source = {}
            for row in rows:
                by_source.setdefault(row['source_id'], set()).add(row['split'])
            self.assertTrue(all(len(parts) == 1 for parts in by_source.values()))
            mapping = {r['image_id']: r for r in rows}
            self.assertEqual(mapping['p0']['split'], mapping['p1']['split'])
            self.assertEqual(read_json(root/'prepared/data_audit.json')['near_duplicate_cross_split'], [])
            class NoRedetection(FakeDetector):
                def __call__(self, _):
                    raise AssertionError('Unchanged audited crops must be reused')
            reused = replace(grouped, data=replace(grouped.data,
                reuse_prepared_manifest=config.data.prepared_manifest,
                prepared_manifest=str(root/'reused/manifest.csv')))
            prepare(reused, detector=NoRedetection())
            self.assertEqual(read_json(root/'reused/data_audit.json')['reused_detections'], len(rows))
            self.assertEqual([r['crop_sha256'] for r in rows],
                             [r['crop_sha256'] for r in read_csv(reused.data.prepared_manifest)])

    def test_extra_manifest_has_provenance_and_invalidates_selection_cache(self):
        from PIL import Image
        from test_pipeline import fixtures
        from sokolov.data.download import download_data
        with TemporaryDirectory() as temporary:
            root = Path(temporary);config = fixtures(root)
            photo = root/'external.png';Image.new('RGB', (120, 120), 'red').save(photo)
            extra = root/'extra.csv'
            write_csv(extra, [{'image_id': 'external1', 'path': str(photo), 'label': 1,
                'source_id': 'web:portrait', 'source_url': 'https://example.org/verified'}])
            config = replace(config, data=replace(config.data, extra_manifest=str(extra)))
            download_data(config)
            added = next(r for r in read_csv(config.data.raw_manifest) if r['image_id'] == 'external1')
            self.assertEqual(added['source_url'], 'https://example.org/verified')
            self.assertEqual(added['source_id'], 'web:portrait')
            extra.write_text(extra.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'different'):
                download_data(config)

    def test_source_macro_ap_exposes_failures_hidden_by_lecture_frames(self):
        from sokolov.metrics import validation_ap_by_domain
        rows = [{'label': 1, 'score': .99, 'source_id': 'lecture'}]*100
        rows += [{'label': 0, 'score': .2, 'source_id': 'fairface:a'}]*10
        rows += [{'label': 1, 'score': .1, 'source_id': 'web:held_out'}]
        result = validation_ap_by_domain(rows)
        self.assertEqual(result['lecture'], 1.)
        self.assertLess(result['web'], .1)

    def test_validation_threshold_respects_source_weights(self):
        from sokolov.metrics import choose_threshold
        self.assertEqual(choose_threshold([0, 1, 1], [.8, .9, .7])['threshold'], .7)
        selected = choose_threshold([0, 1, 1], [.8, .9, .7], sample_weight=[10, 1, 1])
        self.assertEqual(selected['threshold'], .9)
        self.assertEqual(selected['selection_partition'], 'val')

    def test_domain_metrics_and_selection_criterion_are_in_the_report(self):
        from PIL import Image
        from sokolov.metrics import binary_metrics
        from sokolov.io import write_json
        from sokolov.report import make_report
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = binary_metrics([0, 1], [.1, .9], .5)
            for name in ('resnet18', 'arcface'):
                folder = root/name
                write_json(folder/'status.json', {'status': 'evaluated', 'technical_test': True})
                write_json(folder/'test_metrics.json', {'image_level': result, 'face_level': result,
                    'by_positive_domain': {'web': result}, 'checkpoint_bytes': 1, 'benchmark': {}, 'coverage': {}})
                write_json(folder/'training_summary.json', {'data_fingerprint': {'split': 'shared'},
                    'training_seconds': 0, 'parameters': 1, 'pretrained_provenance': {},
                    'selection_metric': 'source_macro_ap'})
                write_json(folder/'environment.json', {'device': 'cpu'})
                write_json(folder/'threshold.json', {'threshold': .5, 'selection_partition': 'val'})
                write_csv(folder/'training.csv', [{'epoch': 1, 'train_loss': 1,
                    'validation_ap': .9, 'selection_score': .7}])
                write_csv(folder/'errors.csv', [], fields=['image_id', 'label', 'score', 'error_type', 'path'])
                (folder/'plots').mkdir()
                for plot in ('curves', 'confusion_matrix', 'scores'):
                    Image.new('RGB', (10, 10)).save(folder/'plots'/f'{plot}.png')
            write_json(root/'data/data_audit.json', {'warnings': ['Fixture only'], 'counts': {},
                'exclusions': [], 'near_duplicate_candidates': [], 'detection_seconds': 0,
                'detector': {'name': 'fixture'}})
            report = make_report(root)
            self.assertIn('source_macro_ap', (report/'report.md').read_text())
            self.assertIn('positive_domain_web', (report/'metrics.csv').read_text())
            self.assertIn('positive_domain_web', (report/'report.html').read_text())

    def test_development_photos_and_linked_variants_are_train_only(self):
        from sokolov.contracts import RawSample
        from sokolov.data.split import split_groups
        config = load_config(ROOT/'configs/resnet18_face.toml').split
        rows = []
        for label in (0, 1):
            for group in range(12):
                source = f'web:{label}:{group}'
                rows += [RawSample(f'{label}_{group}_{i}', Path('unused'), label, source, '', None)
                         for i in range(2)]
        rows += [RawSample('user', Path('unused'), 1, 'web:user_provided_batch', '', None),
                 RawSample('near_variant', Path('unused'), 1, 'web:linked_photo', '', None)]
        groups = {r.image_id: r.source_id for r in rows}
        groups['user'] = groups['near_variant'] = 'merged-development-group'
        assignments = split_groups(rows, groups, config)
        self.assertEqual(assignments, split_groups(rows, groups, config))
        mapping = {r.image_id: r.split for r in assignments}
        self.assertEqual(mapping['user'], 'train')
        self.assertEqual(mapping['near_variant'], 'train')
        partitions = {group: {a.split for a in assignments if a.group_id == group}
                      for group in set(groups.values())}
        self.assertTrue(all(len(values) == 1 for values in partitions.values()))
        for part in ('val', 'test'):
            for label in (0, 1):
                source_groups = {groups[r.image_id] for r in rows
                    if r.label == label and r.source_id.startswith('web:') and mapping[r.image_id] == part}
                self.assertGreaterEqual(len(source_groups), 2)
        with self.assertRaisesRegex(ValueError, 'requires group_stratified'):
            split_groups(rows, groups, replace(config, mode='random_stratified'))

    def test_only_train_is_augmented_and_sampling_is_repeatable(self):
        import random
        import numpy as np
        import torch
        from PIL import Image
        from sokolov.data.dataset import transforms
        config = load_config(ROOT/'configs/resnet18_face.toml')
        image = Image.fromarray(np.random.default_rng(42).integers(0, 255, (224, 224, 3), dtype=np.uint8))
        val = transforms(config)
        self.assertTrue(torch.equal(val(image), val(image)))
        augment = transforms(config, train=True)
        random.seed(42);torch.manual_seed(42);first = augment(image)
        random.seed(42);torch.manual_seed(42);repeated = augment(image)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, val(image)))
        self.assertTrue(torch.isfinite(first).all())
        self.assertEqual(tuple(first.shape), (3, 224, 224))

    def test_local_extra_restore_and_private_bundle_exclude_originals(self):
        import importlib.util
        from PIL import Image
        from sokolov.data.download import image_info
        from sokolov.io import sha256
        def module(name):
            spec = importlib.util.spec_from_file_location(name, ROOT/'tools'/f'{name}.py')
            result = importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
            return result
        builder, packer = module('build_domain_dataset'), module('package_kaggle')
        with TemporaryDirectory() as temporary:
            root = Path(temporary); originals = root/'provided';originals.mkdir()
            original = originals/'sample.png'
            Image.new('RGB', (100, 100), 'red').save(original)
            expected = root/'expected.png';Image.new('RGB', (47, 47), 'red').save(expected)
            item = {'image_id': 'user', 'path': 'data/raw/sokolov_domain_v3/user.png', 'label': 1,
                    'source_id': 'web:user_provided_batch', 'source_url': 'user-provided',
                    'user_original': 'sample.png', 'original_sha256': sha256(original),
                    'bbox': [30, 30, 70, 70], 'margin_fraction': .08, 'max_side': 320,
                    'pixel_sha256': image_info(expected)[1]}
            catalogue = {'items': [item]}
            builder.restore(root, catalogue, user_photo_dir=originals)
            builder.restore(root, catalogue, fetch=lambda _: self.fail('cached run used network'))
            for name in ('pyproject.toml', 'requirements-kaggle.txt', 'README.md'):
                (root/name).write_text('fixture')
            secret = root/'data/raw/sokolov_domain_v3/user_originals/private.jpg'
            secret.parent.mkdir();secret.write_bytes(b'must not be bundled')
            (root/'data/manifests/domain_v3/private.local.json').write_text('private paths')
            archive = packer.package(root, root/'bundle.zip')
            import zipfile
            with zipfile.ZipFile(archive) as bundled:
                self.assertIn(item['path'], bundled.namelist())
                self.assertIn('data/manifests/domain_v3/web.csv', bundled.namelist())
                self.assertFalse(any('user_originals' in p or '.local.' in p for p in bundled.namelist()))
            (root/item['path']).write_bytes(b'changed')
            with self.assertRaises((OSError, ValueError)):
                builder.restore(root, catalogue)


if __name__ == '__main__':
    unittest.main()
