"""Group split invariants, near-duplicate guard, manifest v2 merge, primary face, augmentation."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sokolov.config import load_config
from sokolov.contracts import RawSample
from sokolov.io import read_csv, read_json, write_csv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT/'configs/resnet18_face.toml')


def sample(i, label, group, domain, dataset):
    return RawSample(f'{group}-{i}', Path(f'{group}-{i}.jpg'), label, group, '', None, group, domain, dataset,
                     'sokolov' if label else 'unknown')


def corpus(v1=4, web=(('lecture_room', 3), ('talk_stage', 2), ('interview', 1)), persons=10, fairface=300):
    samples = []
    for lecture in range(v1):
        samples += [sample(i, 1, f'v1:lecture-{lecture}', 'lecture_webcam', 'v1') for i in range(50)]
    for domain, count in web:
        for k in range(count):
            group = f'{domain}-src{k}'
            samples += [sample(i, 1, group, domain, 'web_positive_v2') for i in range(20)]
            samples += [sample(100+i, 0, group, domain, 'web_positive_v2') for i in range(3)]  # bystanders
    for k in range(persons):
        domain = ('lecture_room', 'talk_stage')[k % 2]
        samples += [sample(i, 0, f'person:p{k}', domain, 'web_negative_v2') for i in range(10)]
    samples += [sample(0, 0, f'fairface:{k}.jpg', 'fairface', 'fairface') for k in range(fairface)]
    return samples


def split_of(samples, config):
    from sokolov.data.split import split_groups
    result = split_groups(samples, {s.image_id: s.group for s in samples}, config)
    return result, {a.group_id: a.split for a in result}


class GroupSplitTests(unittest.TestCase):
    def test_groups_disjoint_stratified_deterministic(self):
        samples = corpus()
        result, groups = split_of(samples, CONFIG.split)
        per_group = defaultdict(set)
        for a in result:
            per_group[a.group_id].add(a.split)
        self.assertTrue(all(len(v) == 1 for v in per_group.values()))
        self.assertEqual(result, split_of(samples, CONFIG.split)[0])
        self.assertNotEqual(groups, split_of(samples, replace(CONFIG.split, seed=7))[1])
        fairface = Counter(s for g, s in groups.items() if g.startswith('fairface:'))
        self.assertEqual(fairface, {'train': 195, 'val': 45, 'test': 60})  # 300 groups: 15%/20%
        persons = Counter(s for g, s in groups.items() if g.startswith('person:'))
        self.assertEqual(persons, {'train': 6, 'val': 2, 'test': 2})  # two strata of 5: 1 val + 1 test each
        labels = defaultdict(set)
        for s, a in zip(samples, result):
            labels[a.split].add(s.label)
            self.assertEqual((a.label, a.domain, a.dataset), (s.label, s.domain, s.dataset))
        self.assertEqual(labels, {'train': {0, 1}, 'val': {0, 1}, 'test': {0, 1}})

    def test_pinned_groups_keep_user_partition(self):
        from sokolov.data.split import split_summary
        pinned = [replace(sample(i, 1, f'u3:g{k}', 'interview', 'user_v3'), pinned_split=split)
                  for k, split in enumerate(('test', 'test', 'train')) for i in range(3)]
        pinned += [replace(sample(i, 0, f'u3:n{k}', 'interview', 'user_v3'), pinned_split=split)
                   for k, split in enumerate(('test', 'train', 'train')) for i in range(2)]
        samples = corpus() + pinned
        result, groups = split_of(samples, CONFIG.split)
        expected = {'u3:g0': 'test', 'u3:g1': 'test', 'u3:g2': 'train', 'u3:n0': 'test', 'u3:n1': 'train', 'u3:n2': 'train'}
        self.assertEqual({g: groups[g] for g in expected}, expected)
        checks = split_summary(samples, result, CONFIG.split)['checks']
        self.assertEqual((checks['pinned_groups'], checks['pinned_violations']), (6, []))
        self.assertEqual(checks['groups_in_several_splits'], [])
        mixed = samples + [replace(sample(9, 1, 'u3:g0', 'interview', 'user_v3'), pinned_split='train')]
        with self.assertRaisesRegex(ValueError, 'mixes pinned'):
            split_of(mixed, CONFIG.split)
        with self.assertRaisesRegex(ValueError, 'group_stratified'):
            split_of(samples, replace(CONFIG.split, mode='random_stratified'))

    def test_guarantees_and_summary(self):
        from sokolov.data.split import split_summary
        # Singletons/pairs round to train; guarantees must move groups explicitly.
        samples = corpus(v1=3, web=(('lecture_room', 1), ('talk_stage', 1)), persons=2, fairface=40)
        result, groups = split_of(samples, CONFIG.split)
        self.assertEqual(groups['lecture_room-src0'], 'test')
        self.assertEqual(groups['talk_stage-src0'], 'test')
        self.assertEqual(sum(groups[f'v1:lecture-{i}'] == 'test' for i in range(3)), 1)
        self.assertEqual(sum(groups[f'v1:lecture-{i}'] == 'val' for i in range(3)), 1)
        summary = split_summary(samples, result, CONFIG.split)
        checks = summary['checks']
        self.assertEqual(checks['groups_in_several_splits'], [])
        self.assertTrue(checks['both_labels_every_split'])
        self.assertEqual(checks['test_web_positive_domains'], ['lecture_room', 'talk_stage'])
        self.assertEqual(len(checks['test_v1_lectures']), 1)
        self.assertEqual(checks['val_positive_groups'], 1)
        cells = {(c['split'], c['label'], c['domain']): c for c in summary['cells']}
        self.assertEqual(cells[('test', 1, 'talk_stage')], {'split': 'test', 'label': 1, 'domain': 'talk_stage',
                                                            'groups': 1, 'images': 20})
        self.assertEqual(sum(c['images'] for c in summary['cells']), len(samples))
        # Not enough independent positive groups: refuse instead of leaking.
        with self.assertRaisesRegex(ValueError, 'Both classes'):
            split_of(corpus(v1=1, web=(), persons=2, fairface=40), CONFIG.split)

    def test_explicit_modes_preserved(self):
        samples = corpus(fairface=100)
        _, groups = split_of(samples, replace(CONFIG.split, mode='random_stratified'))
        with self.assertRaisesRegex(ValueError, 'External'):
            split_of(samples, replace(CONFIG.split, mode='external'))


class NearDuplicateTests(unittest.TestCase):
    def test_hamming_helpers_match_python(self):
        import random
        from sokolov.data.dedup import close_pairs, hamming, nearest
        rng = random.Random(0)
        a = [rng.getrandbits(64) for _ in range(300)]
        b = [rng.getrandbits(64) for _ in range(270)]
        matrix = hamming(a, b)
        self.assertTrue(all(matrix[i, j] == (a[i] ^ b[j]).bit_count() for i in range(0, 300, 7) for j in range(0, 270, 5)))
        distance, index = nearest(a, b)
        self.assertEqual(distance.tolist(), [min((x ^ y).bit_count() for y in b) for x in a])
        self.assertTrue(all((a[i] ^ b[index[i]]).bit_count() == distance[i] for i in range(300)))
        near = a[:5] + [a[0] ^ 0b111, a[1] ^ (2**63)]
        expected = [(i, j, (near[i] ^ near[j]).bit_count()) for i in range(7) for j in range(i+1, 7)
                    if (near[i] ^ near[j]).bit_count() <= 3]
        self.assertEqual(close_pairs(near, 3), expected)

    def test_val_test_crops_near_train_or_val_are_excluded(self):
        from sokolov.data.prepare import near_duplicate_guard
        base = 0x0123456789abcdef
        def row(image_id, split, value, label=1, status='ok'):
            return {'image_id': image_id, 'split': split, 'status': status, 'label': label, 'domain': 'photo',
                    'crop_phash': f'{value:016x}', 'near_duplicate_of': '', 'near_duplicate_distance': ''}
        far = base ^ (2**40 - 1)  # 40 bits from base
        rows = [row('t1', 'train', base), row('t2', 'train', far, 0),
                row('v_near', 'val', base ^ 0b111),          # 3 bits from train -> excluded
                row('v_far', 'val', base ^ (2**20 - 1)),       # 20 bits -> kept
                row('x_edge', 'test', far ^ 0xff, 0),          # exactly 8 bits from train -> excluded
                row('x_nine', 'test', base ^ (0x1ff << 50)),   # 9 bits from train -> kept unless near val
                row('x_val', 'test', (base ^ (2**20 - 1)) ^ 0b11),  # 2 bits from kept val crop
                row('x_noface', 'test', 0, status='no_face')]
        guard, flagged = near_duplicate_guard(rows, 8)
        status = {r['image_id']: r['status'] for r in rows}
        self.assertEqual(status, {'t1': 'ok', 't2': 'ok', 'v_near': 'near_duplicate', 'v_far': 'ok',
                                  'x_edge': 'near_duplicate', 'x_nine': 'ok', 'x_val': 'near_duplicate',
                                  'x_noface': 'no_face'})
        self.assertEqual({f['image_id']: (f['nearest'], f['distance']) for f in flagged},
                         {'v_near': ('t1', 3), 'x_edge': ('t2', 8), 'x_val': ('v_far', 2)})
        self.assertEqual(guard['excluded'], {'val': {'0': 0, '1': 1}, 'test': {'0': 1, '1': 1}})
        self.assertEqual(guard['min_cross_split_distance_before'], {'val-train': 3, 'test-train': 8, 'test-val': 2})
        after = guard['min_cross_split_distance_after']
        self.assertTrue(all(value > 8 for value in after.values()))


class ManifestMergeTests(unittest.TestCase):
    def build(self, root, negative_manifest=True):
        import numpy as np
        from PIL import Image
        def image(path, seed):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.random.default_rng(seed).integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(path)
        v1 = root/'data/raw/sokolov_positive_v1'
        for i in range(4):
            image(v1/f'positive/p{i}.jpg', i)
        write_csv(v1/'manifest.csv', [{'file': f'positive/p{i}.jpg', 'label': 1, 'class_name': 'sokolov',
                                       'source_video': f'лекция-0{i % 2}.mp4', 'timestamp_seconds': i}
                                      for i in range(4)])
        fields = ['image_id', 'path', 'label', 'source_id', 'group_id', 'domain', 'person', 'source_url',
                  'timestamp_seconds', 'det_score', 'face_px', 'sokolov_similarity', 'review_status', 'notes']
        def harvest(name, prefix, items):
            rows = []
            for n, (label, source, group, domain, person, status) in enumerate(items):
                rel = f'data/raw/{name}/faces/{source}/{prefix}_{source}_{n:04d}.jpg'
                if status != 'missing':
                    image(root/rel, 100 + n + (50 if prefix == 'wn2' else 0))
                rows.append(dict(zip(fields, [f'{prefix}_{source}_{n:04d}', rel, label, source, group, domain,
                    person, 'https://example.org/v', 12.5, .9, 80, .6 if label else .1,
                    'accepted' if status == 'missing' else status, ''])))
            write_csv(root/f'data/raw/{name}/manifest.csv', rows, fields=fields)
        harvest('web_positive_v2', 'wp2', [
            (1, 'talk1', 'talk1', 'talk_stage', 'sokolov', 'accepted'),
            (1, 'talk1', 'talk1', 'talk_stage', 'sokolov', 'ambiguous'),
            (0, 'talk1', 'talk1', 'talk_stage', 'unknown', 'accepted'),
            (1, 'room1', 'room1', 'lecture_room', 'sokolov', 'excluded'),
            (1, 'room1', 'room1', 'lecture_room', 'other', 'accepted'),      # label/person mismatch
            (1, 'room1', 'room1', 'lecture_room', 'sokolov', 'missing')])    # accepted but file absent
        if negative_manifest:
            harvest('web_negative_v2', 'wn2', [
                (0, 'conf1', 'person:ivanov', 'talk_stage', 'ivanov', 'accepted'),
                (0, 'conf2', 'person:ivanov', 'interview', 'ivanov', 'accepted')])
        for i in range(6):
            image(root/f'data/raw/fairface/train/{i}.jpg', 500+i)
        (root/'data/raw/fairface/SOURCE.md').write_text('FairFace, CC BY 4.0')
        return replace(CONFIG, data=replace(CONFIG.data, positive_dir=str(v1), negative_count=5, extra_manifest='',
            fairface_dir=str(root/'data/raw/fairface'), web_positive_dir=str(root/'data/raw/web_positive_v2'),
            web_negative_dir=str(root/'data/raw/web_negative_v2'), user_dir=str(root/'data/raw/user_v3'),
            raw_manifest=str(root/'manifests/raw.csv')))

    def test_merge_accepted_only_with_groups_domains(self):
        from sokolov.data.download import RAW_FIELDS, download_data
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.build(root)
            download_data(config)
            rows = read_csv(config.data.raw_manifest)
            self.assertEqual(list(rows[0]), RAW_FIELDS)
            by_id = {r['image_id']: r for r in rows}
            self.assertEqual(Counter(r['dataset'] for r in rows),
                             {'v1': 4, 'web_positive_v2': 2, 'web_negative_v2': 2, 'fairface': 5})
            self.assertEqual(by_id['p1']['group_id'], 'v1:лекция-01.mp4')
            self.assertEqual((by_id['p1']['domain'], by_id['p1']['person']), ('lecture_webcam', 'sokolov'))
            self.assertEqual({k for k in by_id if k.startswith('wp2_')}, {'wp2_talk1_0000', 'wp2_talk1_0002'})
            self.assertEqual(by_id['wp2_talk1_0002']['label'], '0')
            self.assertEqual(by_id['wp2_talk1_0002']['group_id'], 'talk1')
            self.assertEqual({by_id[k]['group_id'] for k in by_id if k.startswith('wn2_')}, {'person:ivanov'})
            fair = [r for r in rows if r['dataset'] == 'fairface']
            self.assertTrue(all(r['group_id'].startswith('fairface:train/') and r['domain'] == 'fairface'
                                and r['label'] == '0' for r in fair))
            selection = read_json(root/'manifests/selection.local.json')
            reasons = Counter(e['reason'] for e in selection['exclusions'])
            self.assertEqual(reasons, {'review_status=ambiguous': 1, 'review_status=excluded': 1,
                                       'label/person mismatch': 1, 'file missing': 1})
            before = Path(config.data.raw_manifest).read_bytes()
            download_data(config)  # cached
            self.assertEqual(before, Path(config.data.raw_manifest).read_bytes())
            manifest = root/'data/raw/web_negative_v2/manifest.csv'
            manifest.write_text(manifest.read_text().replace('accepted', 'excluded', 1))
            with self.assertRaisesRegex(ValueError, 'different inputs'):
                download_data(config)

    def test_user_dataset_partition_is_pinned(self):
        from sokolov.data.download import download_data
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.build(root)
            user = root/'data/raw/user_v3'
            (user/'faces').mkdir(parents=True)
            import numpy as np
            from PIL import Image
            rows = []
            for n, (label, partition) in enumerate([(1, 'test'), (0, 'train'), (1, '')]):
                Image.fromarray(np.full((40, 40, 3), 30*n, dtype=np.uint8)).save(user/f'faces/u3_{n:03d}_f1.jpg')
                rows.append({'image_id': f'u3_{n:03d}_f1', 'path': f'data/raw/user_v3/faces/u3_{n:03d}_f1.jpg',
                    'label': label, 'source_id': f'u3_{n:03d}', 'group_id': f'u3:g{n}', 'domain': 'interview',
                    'person': 'sokolov' if label else 'unknown', 'source_url': '', 'timestamp_seconds': '',
                    'det_score': .9, 'face_px': 200, 'sokolov_similarity': .5, 'review_status': 'accepted',
                    'notes': '', 'partition': partition})
            write_csv(user/'manifest.csv', rows, fields=list(rows[0]))
            download_data(config)
            by_id = {r['image_id']: r for r in read_csv(config.data.raw_manifest)}
            self.assertEqual((by_id['u3_000_f1']['pinned_split'], by_id['u3_001_f1']['pinned_split']), ('test', 'train'))
            self.assertNotIn('u3_002_f1', by_id)  # missing partition is rejected, not silently split
            self.assertEqual({by_id[k]['pinned_split'] for k in by_id if not k.startswith('u3_')}, {''})
            reasons = Counter(e['reason'] for e in read_json(root/'manifests/selection.local.json')['exclusions'])
            self.assertEqual(reasons['pinned dataset needs partition=train/test'], 1)

    def test_missing_harvest_folder_warns_and_fairface_is_local_only(self):
        from sokolov.data.download import download_data
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.build(root, negative_manifest=False)
            download_data(config)
            selection = read_json(root/'manifests/selection.local.json')
            self.assertTrue(any('web_negative_v2' in w for w in selection['warnings']))
            missing = replace(config, data=replace(config.data, fairface_dir=str(root/'none'),
                                                   raw_manifest=str(root/'other/raw.csv')))
            with self.assertRaisesRegex(FileNotFoundError, 'fairface_download'):
                download_data(missing)


class PrimaryFaceTests(unittest.TestCase):
    def test_nearest_centre_then_larger_box(self):
        from sokolov.data.prepare import primary_face
        big_corner = {'bbox': [0, 0, 60, 60], 'confidence': .99}
        small_centre = {'bbox': [90, 40, 110, 60], 'confidence': .6}
        self.assertEqual(primary_face([big_corner, small_centre], (200, 100)), 1)
        small = {'bbox': [90, 40, 110, 60], 'confidence': .99}
        large = {'bbox': [80, 30, 120, 70], 'confidence': .5}
        self.assertEqual(primary_face([small, large], (200, 100)), 1)
        self.assertEqual(primary_face([large, dict(large)], (200, 100)), 0)


class AugmentationTests(unittest.TestCase):
    def test_seeded_train_augmentation_and_deterministic_eval(self):
        import numpy as np
        import torch
        from PIL import Image
        from sokolov.data.dataset import RandomDownscale, RandomJPEG, transforms
        image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8))
        for name in ('resnet18', 'arcface'):
            config = load_config(ROOT/f'configs/{name}_face.toml')
            size = config.model.image_size
            train = transforms(config, train=True)
            self.assertIn('RandomJPEG', repr(train))
            outputs = []
            for seed in (3, 3, 4):
                torch.manual_seed(seed)
                outputs.append(train(image))
            self.assertEqual(outputs[0].shape, (3, size, size))
            self.assertTrue(torch.equal(outputs[0], outputs[1]))
            self.assertFalse(torch.equal(outputs[0], outputs[2]))
            evaluation = transforms(config, train=False)
            self.assertTrue(torch.equal(evaluation(image), evaluation(image)))
            self.assertNotIn('Random', repr(evaluation))
            legacy = transforms(replace(config, training=replace(config.training, augmentation='light')), train=True)
            self.assertNotIn('RandomJPEG', repr(legacy))
        torch.manual_seed(0)
        low = np.asarray(RandomDownscale(24, 24, 1.)(image), dtype=float)
        self.assertEqual(low.shape, (224, 224, 3))
        self.assertLess(np.abs(np.diff(low, axis=1)).mean(), np.abs(np.diff(np.asarray(image, float), axis=1)).mean()/2)
        self.assertEqual(RandomJPEG(20, 20, 1.)(image).size, image.size)
        self.assertIs(RandomJPEG(20, 90, 0.)(image), image)

    def test_finetune_blocks(self):
        from sokolov.model import build_model
        from sokolov.train import apply_training_stage
        model = build_model(CONFIG.model, pretrained=False)
        apply_training_stage(model, CONFIG, True)
        trainable = {name for name in ('layer1', 'layer2', 'layer3', 'layer4', 'fc')
                     if all(p.requires_grad for p in getattr(model, name).parameters())}
        self.assertEqual(trainable, {'layer3', 'layer4', 'fc'})
        self.assertFalse(model.layer2.training)
        one = replace(CONFIG, training=replace(CONFIG.training, finetune_blocks=1))
        apply_training_stage(model, one, True)
        self.assertFalse(model.layer3[0].conv1.weight.requires_grad)
        self.assertTrue(model.layer4[0].conv1.weight.requires_grad)
        apply_training_stage(model, CONFIG, False)
        self.assertFalse(any(p.requires_grad for n, p in model.named_parameters() if not n.startswith('fc.')))


class EvaluationStatisticsTests(unittest.TestCase):
    def test_group_bootstrap_and_breakdown(self):
        from sokolov.metrics import breakdown, group_bootstrap
        labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        scores = [.9, .8, .7, .2, .1, .3, .6, .2, .1, .05]
        groups = ['a', 'a', 'b', 'b', 'a', 'n1', 'n2', 'n3', 'n4', 'n5']
        detected = [True]*9 + [False]
        first = group_bootstrap(labels, scores, .5, detected, groups, resamples=200, seed=1)
        self.assertEqual(first, group_bootstrap(labels, scores, .5, detected, groups, resamples=200, seed=1))
        self.assertEqual((first['positive_groups'], first['negative_groups']), (2, 5))
        for name, ci in first['intervals'].items():
            self.assertLessEqual(ci['low'], ci['high'])
            self.assertEqual(ci['valid_resamples'], 200)
        perfect = group_bootstrap([0, 1, 0, 1], [.1, .9, .2, .8], .5, None, ['n1', 'p1', 'n2', 'p2'], resamples=50)
        self.assertEqual(perfect['intervals']['roc_auc'], {'low': 1., 'high': 1., 'valid_resamples': 50})
        self.assertIsNone(group_bootstrap([0, 0], [.1, .2], .5, None, ['x', 'y'])['intervals'])
        rows = [{'label': l, 'score': s, 'detected': d, 'domain': dom, 'source_id': src}
                for l, s, d, dom, src in zip(labels, scores, detected,
                    ['photo']*4 + ['photo', 'fairface', 'fairface', 'talk_stage', 'fairface', 'fairface'],
                    ['s1', 's1', 's2', 's2'] + ['x']*6)]
        result = breakdown(rows, .5)
        self.assertEqual(result['positive_sources']['s1']['recall'], 1.)
        self.assertEqual(result['positive_sources']['s2']['recall'], .5)
        self.assertEqual(result['negatives_by_kind']['fairface'],
                         {'negatives': 4, 'false_positives': 1, 'false_positive_rate': .25})
        self.assertEqual(result['negatives_by_kind']['in_domain']['false_positive_rate'], 0.)
        self.assertEqual(result['by_domain']['photo']['roc_auc'], 1.)


if __name__ == '__main__':
    unittest.main()
