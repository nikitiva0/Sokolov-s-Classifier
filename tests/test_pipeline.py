"""Offline tests: fixtures, actual PyTorch forward/backward, no external weights/data."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sokolov.config import load_config
from sokolov.contracts import RawSample
from sokolov.io import read_csv, read_json, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]


class SplitTests(unittest.TestCase):
    def test_random_stratified_reproducible_sizes_and_external_groups(self):
        from sokolov.data.split import split_groups
        # Legacy frame-level split stays available only as an explicit option.
        config = replace(load_config(ROOT/'configs/resnet18_face.toml').split,
                         mode='random_stratified', test_fraction=.15)
        samples = [RawSample(str(i), Path(str(i)), int(i < 1000), f'lecture-{i%4}', '', None)
                   for i in range(6000)]
        groups = {s.image_id: s.source_id for s in samples}
        result = split_groups(samples, groups, config)
        self.assertEqual(result, split_groups(samples, groups, config))
        from collections import Counter
        self.assertEqual(Counter(a.split for a in result), {'train': 4200, 'val': 900, 'test': 900})
        self.assertEqual(Counter(s.label for s, a in zip(samples, result) if a.split == 'test'), {0: 750, 1: 150})
        external = [RawSample(str(i), Path(str(i)), i%2, f'group{i//4}', '',
                             'test' if i >= 80 else 'train') for i in range(100)]
        result = split_groups(external, {s.image_id: s.source_id for s in external}, replace(config, mode='external'))
        self.assertTrue(all(a.split == 'test' for a in result[80:]))
        self.assertFalse({a.group_id for a in result if a.split == 'train'} &
                         {a.group_id for a in result if a.split == 'val'})
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            split_groups(external, {s.image_id: 'shared' for s in external}, config)


class MetricTests(unittest.TestCase):
    def test_perfect_known_metrics_and_no_face_negative_at_zero(self):
        from sokolov.metrics import binary_metrics, choose_threshold
        result = binary_metrics([0, 0, 1, 1], [.1, .2, .8, .9], .5)
        self.assertEqual(result['confusion_matrix'], [[2, 0], [0, 2]])
        self.assertEqual(result['metrics']['roc_auc'], 1.)
        self.assertEqual(result['metrics']['macro_f1'], 1.)
        self.assertEqual(choose_threshold([0, 0, 1, 1], [.1, .2, .8, .9])['threshold'], .8)
        result = binary_metrics([0, 1], [0., .8], 0., [False, True])
        self.assertEqual(result['confusion_matrix'], [[1, 0], [0, 1]])
        self.assertIsNone(binary_metrics([0], [.1], .5)['metrics']['roc_auc'])
        missed = binary_metrics([0, 1], [.1, 0.], .5, [True, False])
        self.assertEqual(missed['confusion_matrix'], [[1, 0], [1, 0]])
        self.assertIsNone(missed['metrics']['mcc'])


class FakeDetector:
    metadata = {'name': 'FIXTURE_ONLY', 'sha256': 'fixture', 'version': 1}
    def __call__(self, image):
        return [{'bbox': [0, 0, 95, 95], 'confidence': .99,
                 'landmarks': [[32, 35], [64, 35], [48, 51], [35, 69], [63, 69]]}]


def fixtures(folder, count=40):
    import numpy as np
    from PIL import Image
    positive, fairface = folder/'positive', folder/'fairface'
    positive.mkdir(); fairface.mkdir()
    rows = []
    for i in range(count):
        # Low-frequency random structure: aligned crops get distinct pHashes (no accidental near-duplicates).
        coarse = np.random.default_rng(1000+i).integers(0, 100, (6, 6, 3), dtype=np.uint8)
        pixels = np.kron(coarse, np.ones((16, 16, 1), dtype=np.uint8))
        pixels = pixels + np.random.default_rng(i).integers(0, 20, (96, 96, 3), dtype=np.uint8)//4
        if i < count//2:
            pixels = pixels+130
            path = positive/f'p{i}.png'
            rows.append({'file': path.name, 'label': 1, 'source_video': f'lecture{i%4}'})
        else:
            path = fairface/f'n{i}.png'
        Image.fromarray(pixels).save(path)
    write_csv(positive/'manifest.csv', rows)
    config = load_config(ROOT/'configs/resnet18_face.toml')
    return replace(config, data=replace(config.data,
        positive_dir=str(positive), fairface_dir=str(fairface), negative_count=count//2,
        web_positive_dir=str(folder/'absent_web_positive'), web_negative_dir=str(folder/'absent_web_negative'),
        user_dir=str(folder/'absent_user'),
        raw_manifest=str(folder/'raw.csv'), prepared_manifest=str(folder/'prepared/manifest.csv')),
        training=replace(config.training, num_workers=0, device='cpu', batch_size=4,
                         head_epochs=2, finetune_epochs=0))


class PreparationTests(unittest.TestCase):
    def test_cached_selection_dedup_conflict_and_missing_faces(self):
        from sokolov.data.download import download_data
        from sokolov.data.prepare import prepare
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = fixtures(folder)
            # Corrupt candidate and byte-identical duplicate must not enter the selected set.
            (folder/'fairface/broken.jpg').write_bytes(b'bad image')
            import shutil
            shutil.copy2(folder/'fairface/n20.png', folder/'fairface/duplicate.png')
            download_data(config)
            before = Path(config.data.raw_manifest).read_bytes()
            download_data(config)
            self.assertEqual(before, Path(config.data.raw_manifest).read_bytes())
            self.assertEqual(len(read_csv(config.data.raw_manifest)), 40)
            detector = FakeDetector()
            class Missing(FakeDetector):
                def __call__(self, image):
                    self.counter = getattr(self, 'counter', 0)+1
                    if self.counter == 1: return []
                    if self.counter == 2: return super().__call__(image)*2
                    return super().__call__(image)
            prepare(config, detector=Missing())
            rows = read_csv(config.data.prepared_manifest)
            self.assertEqual(rows[0]['status'], 'no_face')
            # Several detections: the face nearest the centre is the labelled target.
            self.assertEqual((rows[1]['status'], rows[1]['face_count'], rows[1]['primary_face']), ('ok', '2', '0'))
            raw_rows = read_csv(config.data.raw_manifest)
            self.assertEqual({r['group_id'] for r in raw_rows if r['label'] == '1'},
                             {f'v1:lecture{i}' for i in range(4)})
            self.assertTrue(all(r['domain'] == 'fairface' and r['group_id'] == r['source_id']
                                for r in raw_rows if r['label'] == '0'))
            split_rows = read_csv(Path(config.data.prepared_manifest).parent/'split.csv')
            group_splits = {}
            for r in split_rows:
                group_splits.setdefault(r['group_id'], set()).add(r['split'])
            self.assertTrue(all(len(v) == 1 for v in group_splits.values()))
            summary = read_json(Path(config.data.prepared_manifest).parent/'split_summary.json')
            self.assertEqual(summary['checks']['groups_in_several_splits'], [])
            self.assertEqual(len(summary['checks']['test_v1_lectures']), 1)
            audit = read_json(Path(config.data.prepared_manifest).parent/'data_audit.json')
            self.assertEqual(audit['near_duplicate_guard']['max_distance'], 8)
            self.assertGreater(audit['near_duplicate_guard']['min_cross_split_distance_after']['test-train'], 8)
            self.assertTrue(all(Path(r['crop_path']).is_file() for r in rows if r['status'] == 'ok'))
            split = read_csv(Path(config.data.prepared_manifest).parent/'split.csv')
            self.assertEqual({r['image_id']: r['split'] for r in rows}, {r['image_id']: r['split'] for r in split})
            prepare(config, detector=detector)  # cache, not a redetection
            with self.assertRaisesRegex(ValueError, 'changed'):
                prepare(replace(config, split=replace(config.split, seed=7)), detector=detector)
            self.assertEqual(read_csv(config.data.prepared_manifest)[0]['status'], 'no_face')
            raw = read_csv(config.data.raw_manifest)
            raw[1]['path'] = raw[0]['path']; raw[1]['label'] = '0'; raw[1]['pixel_sha256'] = raw[0]['pixel_sha256']
            write_csv(config.data.raw_manifest, raw)
            changed = replace(config, data=replace(config.data, prepared_manifest=str(folder/'conflict/manifest.csv')))
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                prepare(changed, detector=detector)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_shapes_frozen_parameters_statistics_and_strict_arcface_weights(self):
        import torch
        from sokolov.model import build_model, set_training_stage, frozen_eval, state_fingerprint
        for name, size in [('resnet18', 224), ('arcface', 112)]:
            config = load_config(ROOT/f'configs/{name}_face.toml')
            torch.manual_seed(42)
            model = build_model(config.model, pretrained=False)
            set_training_stage(model, name, False)
            frozen_before = state_fingerprint({k: v for k, v in model.state_dict().items()
                if not k.startswith('fc.') and not k.startswith('head.')})
            head = model.fc if name == 'resnet18' else model.head
            before = head.weight.detach().clone()
            optimizer = torch.optim.Adam(head.parameters(), lr=.01)
            output = model(torch.rand(2, 3, size, size))
            self.assertEqual(output.shape, (2, 1))
            torch.nn.functional.binary_cross_entropy_with_logits(output, torch.ones_like(output)).backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, head.weight))
            frozen_after = state_fingerprint({k: v for k, v in model.state_dict().items()
                if not k.startswith('fc.') and not k.startswith('head.')})
            self.assertEqual(frozen_before, frozen_after)
            if name == 'resnet18':
                set_training_stage(model, name, True)
                self.assertTrue(model.layer4[0].conv1.weight.requires_grad)
                self.assertFalse(model.layer3[0].conv1.weight.requires_grad)
            else:
                with TemporaryDirectory() as tmp:
                    path = Path(tmp)/'official-format.pth'
                    torch.save(model.backbone.state_dict(), path)
                    restored = build_model(replace(config.model, checkpoint=str(path), checkpoint_sha256=''), pretrained=True)
                    self.assertEqual(state_fingerprint(model.backbone.state_dict()),
                                     state_fingerprint(restored.backbone.state_dict()))
                    torch.save({'wrong': torch.ones(1)}, path)
                    with self.assertRaises(RuntimeError):
                        build_model(replace(config.model, checkpoint=str(path), checkpoint_sha256=''), pretrained=True)
                    with self.assertRaises(FileNotFoundError):
                        build_model(replace(config.model, checkpoint=str(path.parent/'absent.pth'), checkpoint_sha256=''), pretrained=True)

    def test_end_to_end_resume_report_publication(self):
        import torch
        from sokolov.data.download import download_data
        from sokolov.data.prepare import prepare
        from sokolov.train import train
        from sokolov.evaluate import evaluate_test
        from sokolov.report import make_report, publish_report
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = fixtures(folder)
            download_data(config); prepare(config, detector=FakeDetector())
            run = folder/'technical-run'
            from sokolov.train import predict
            with patch('sokolov.train.predict', wraps=predict) as predictions:
                train(config, run, pretrained=False, stop_after_epoch=1)
                self.assertTrue(all(call.args[-1] == 'val' for call in predictions.call_args_list))
            with self.assertRaisesRegex(ValueError, 'finish'):
                evaluate_test(config, run/'resnet18/best.pt')
            train(config, run, pretrained=False, resume=True)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                train(replace(config, training=replace(config.training, head_lr=.02)), run,
                      pretrained=False, resume=True)
            resumed = torch.load(run/'resnet18/last.pt', weights_only=False, map_location='cpu')
            self.assertEqual(len(resumed['history']), 2)
            best = torch.load(run/'resnet18/best.pt', weights_only=False, map_location='cpu')
            self.assertEqual(best['epoch'], 0)  # AP tie keeps the earlier checkpoint.
            weights = torch.load(run/'resnet18/best_weights.pt', weights_only=True)
            self.assertTrue(all(torch.equal(v, weights[k]) for k, v in best['model'].items()))
            del best, weights
            # An uninterrupted run must match resume exactly, including training augmentations.
            train(config, folder/'uninterrupted', pretrained=False)
            full = torch.load(folder/'uninterrupted/resnet18/last.pt', weights_only=False, map_location='cpu')
            self.assertEqual(resumed['history'], full['history'])
            self.assertTrue(all(torch.equal(v, full['model'][k]) for k, v in resumed['model'].items()))
            evaluate_test(config, run/'resnet18/best.pt')
            arc = load_config(ROOT/'configs/arcface_face.toml')
            arc = replace(arc, data=config.data, split=config.split,
                          training=replace(config.training, head_epochs=1))
            train(arc, run, pretrained=False)
            evaluate_test(arc, run/'arcface/best.pt')
            self.assertTrue((run/'arcface/best_weights.pt').is_file())
            import shutil
            self.assertTrue((run/'data/data_audit.json').is_file())
            self.assertTrue((run/'data/split_summary.json').is_file())
            metrics = read_json(run/'resnet18/test_metrics.json')
            self.assertEqual(metrics['bootstrap']['image_level']['resamples'], 1000)
            self.assertEqual(set(metrics['breakdown']['negatives_by_kind']), {'fairface'})
            self.assertEqual(len(metrics['breakdown']['positive_sources']), 1)
            report = make_report(run)
            text = (report/'report.md').read_text()
            self.assertIn('Протокол оценки', text)
            self.assertIn('near_duplicate', text)
            self.assertNotIn('Random frame split', text)
            self.assertIn('ci95_low', read_csv(report/'metrics.csv')[0])
            import os
            output = os.environ.get('SOKOLOV_SMOKE_OUTPUT')
            if output:
                output = Path(output)
                shutil.copytree(report, output/'report', dirs_exist_ok=True)
                for name in ('resnet18', 'arcface'):
                    (output/name).mkdir(parents=True, exist_ok=True)
                    for artifact in ('environment.json', 'training.csv', 'training_summary.json', 'test_metrics.json'):
                        shutil.copy2(run/name/artifact, output/name/artifact)
            self.assertIn('data:image/png;base64,', (report/'report.html').read_text())
            self.assertTrue((run/'resnet18/error_examples.jpg').exists())
            with self.assertRaisesRegex(ValueError, 'real pretrained'):
                publish_report(run, transport=lambda *a: self.fail('must reject technical test'))
            # Exercise publication transport/allowlist without network or real measurements.
            complete = read_json(report/'completed.json'); complete['technical_test'] = False
            write_json(report/'completed.json', complete)
            received = []
            publish_report(run, transport=lambda *args: received.append(args))
            self.assertEqual(received[0][1], 'technical-run')
            self.assertFalse(any(x.endswith(('.pt', '.jpg')) for x in received[0][2]['files']))
            with patch('sokolov.report.subprocess.run', side_effect=OSError('network down')):
                with self.assertRaises(OSError): publish_report(run)
            self.assertTrue((report/'report.md').exists())
            (report/'report.md').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'modified'):
                publish_report(run, transport=lambda *a: None)
            # Complete assignment path: persist the validation winner and generate
            # five initial noises without touching training data or source weights.
            from sokolov.synthesize import select_best_model, generate_best, validate_synthesis
            select_best_model(run)
            self.assertEqual(read_json(run/'best_model.json')['architecture'], 'resnet18')
            # Arbitrary photos: every detected face is scored, the photo score is the maximum.
            from sokolov.evaluate import format_prediction, predict_images
            class TwoFaces(FakeDetector):
                def __call__(self, image):
                    face = super().__call__(image)[0]
                    return [face, {**face, 'bbox': [10, 10, 60, 60]}]
            photo = next(Path(config.data.positive_dir).glob('*.png'))
            result = predict_images(run, [photo], detector=TwoFaces(), device='cpu')[0]
            self.assertEqual(len(result['faces']), 2)
            self.assertEqual(result['score'], max(f['score'] for f in result['faces']))
            threshold = read_json(run/'resnet18/threshold.json')['threshold']
            self.assertEqual(result['decision'], 'sokolov' if result['score'] >= threshold else 'not_sokolov')
            self.assertIn('face 2: bbox=[10, 10, 60, 60] px', format_prediction(result))
            empty = predict_images(run, [photo], which='arcface', detector=lambda image: [], device='cpu')[0]
            self.assertEqual((empty['score'], empty['decision']), (0., 'not_sokolov'))
            Path(config.data.raw_manifest).unlink()
            generation = replace(config.generation, steps=3, snapshot_every=2, seeds=(0, 1, 2, 3, 4))
            generate_best(run, generation=generation, device='cpu')
            synthesis = validate_synthesis(run/'synthesis')
            self.assertEqual(synthesis['seeds'], [0, 1, 2, 3, 4])
            self.assertTrue(all(r['weights_unchanged'] for r in synthesis['results']))
            report = make_report(run)
            self.assertIn('Генерация из случайного шума', (report/'report.md').read_text())
            self.assertTrue((report/'synthesis/seed_4/progression.png').exists())
            with self.assertRaisesRegex(ValueError, 'real pretrained'):
                publish_report(run, transport=lambda *a: None)
            if output:
                shutil.copytree(report, output/'report', dirs_exist_ok=True)
                shutil.copy2(run/'best_model.json', output/'best_model.json')



if __name__ == '__main__':
    unittest.main()
