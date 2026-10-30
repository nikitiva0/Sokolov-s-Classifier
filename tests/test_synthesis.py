"""Offline activation-maximization tests on actual PyTorch classifiers."""
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sokolov.config import load_config
from sokolov.io import read_csv, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


class SynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_selection_uses_validation_not_test_and_preserves_checkpoint(self):
        import torch
        from sokolov.synthesize import select_best_model
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, ap in [('resnet18', .7), ('arcface', .9)]:
                folder = root/name; folder.mkdir()
                config = load_config(ROOT/f'configs/{name}_face.toml')
                torch.save({'best_ap': ap, 'epoch': 2, 'technical_test': True,
                    'data_fingerprint': {'split': 'shared'}, 'config': asdict(config)}, folder/'best.pt')
                write_json(folder/'status.json', {'status': 'evaluated'})
                write_json(folder/'test_metrics.json', {'average_precision': 1-ap})
            winner = select_best_model(root)
            self.assertEqual(read_json(root/'best_model.json')['architecture'], 'arcface')
            self.assertEqual(sha256(winner), sha256(root/'arcface/best.pt'))
            # Ties use a stable architecture order, independently of test scores.
            saved = torch.load(root/'resnet18/best.pt', weights_only=False)
            saved['best_ap'] = .9; torch.save(saved, root/'resnet18/best.pt')
            select_best_model(root)
            self.assertEqual(read_json(root/'best_model.json')['architecture'], 'resnet18')

    def test_actual_models_pixel_gradients_png_scores_determinism_and_cache(self):
        import torch
        from PIL import Image
        import numpy as np
        from sokolov.model import build_model
        from sokolov.synthesize import synthesize, normalized_pixels
        for architecture in ('resnet18', 'arcface'):
            with self.subTest(architecture=architecture), TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = load_config(ROOT/f'configs/{architecture}_face.toml')
                config = replace(config, generation=replace(config.generation,
                    steps=3, snapshot_every=2, seeds=(7,)),
                    training=replace(config.training, device='cpu'))
                torch.manual_seed(42)
                model = build_model(config.model, pretrained=False).eval()
                checkpoint = root/'best_model.pt'
                torch.save({'model': model.state_dict(), 'config': asdict(config), 'technical_test': True}, checkpoint)
                digest = sha256(checkpoint)
                first = synthesize(config, checkpoint, 7)
                rows = read_csv(first.trajectory)
                self.assertEqual([int(r['step']) for r in rows], [0, 1, 2, 3])
                self.assertEqual([r['step'] for r in rows if r['snapshot']], ['0', '2', '3'])
                self.assertNotEqual(sha256(first.snapshots_dir/'step_0000.png'), sha256(first.final_image))
                self.assertEqual(digest, sha256(checkpoint))
                meta = read_json(first.trajectory.parent/'metadata.json')
                self.assertGreater(meta['first_input_gradient_norm'], 0)
                self.assertEqual(meta['model_state_sha256_before'], meta['model_state_sha256_after'])
                self.assertGreaterEqual(meta['pixel_min'], 0)
                self.assertLessEqual(meta['pixel_max'], 1)
                for row in rows:
                    if not row['snapshot']: continue
                    with Image.open(first.trajectory.parent/row['snapshot']) as image:
                        self.assertEqual(image.size, (config.model.image_size,)*2)
                        x = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float()[None]/255
                    with torch.no_grad():
                        score = model(normalized_pixels(x, architecture)).sigmoid().item()
                    self.assertAlmostEqual(score, float(row['score']), places=6)
                # Repeated seed and same checkpoint reproduce exactly, including PNG quantization.
                again = synthesize(config, checkpoint, 7, output_dir=root/'replay')
                self.assertEqual(first.final_image.read_bytes(), again.final_image.read_bytes())
                self.assertEqual(rows, read_csv(again.trajectory))
                with patch('sokolov.synthesize.build_model', side_effect=AssertionError('cache missed')):
                    synthesize(config, checkpoint, 7)
                changed = replace(config, generation=replace(config.generation, lr=.01))
                with self.assertRaisesRegex(ValueError, 'changed'):
                    synthesize(changed, checkpoint, 7)
                first.final_image.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'modified'):
                    synthesize(config, checkpoint, 7)

    def test_publication_synthesis_allowlist_with_test_transport(self):
        from sokolov.report import publish_report
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'publication-fixture'
            report = root/'report'; report.mkdir(parents=True)
            files = {'report.md', 'report.html', 'metrics.json', 'metrics.csv',
                     'data_audit_summary.json', 'best_model.json', 'synthesis/summary.json',
                     'synthesis/seed_comparison.png', 'synthesis/scores.png'}
            files |= {f'{name}/{plot}.png' for name in ('resnet18', 'arcface')
                      for plot in ('training', 'curves', 'confusion_matrix', 'scores')}
            files |= {f'synthesis/seed_{seed}/{name}' for seed in (0, 1)
                      for name in ('progression.png', 'trajectory.csv')}
            for name in files:
                path = report/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'publication transport fixture')
            completed = {'status': 'completed', 'technical_test': False, 'models': ['resnet18', 'arcface'],
                'best_model': True, 'synthesis': {'seeds': [0, 1], 'technical_test': False},
                'files': {name: sha256(report/name) for name in files}}
            write_json(report/'completed.json', completed)
            received = []
            publish_report(root, transport=lambda *args: received.append(args))
            self.assertEqual(set(received[0][2]['files']), files)
            completed['files']['best_model.pt'] = 'not-allowed'
            write_json(report/'completed.json', completed)
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                publish_report(root, transport=lambda *a: self.fail('weights must not be sent'))
