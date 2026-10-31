"""Offline activation-maximization tests on actual PyTorch classifiers (random weights)."""
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sokolov.config import load_config
from sokolov.io import read_csv, read_json, sha256, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
BUFFALO = ROOT/'checkpoints/insightface/models/buffalo_l'


def small_config(architecture, *, steps=3, seeds=(7,), **data):
    config = load_config(ROOT/f'configs/{architecture}_face.toml')
    return replace(config, generation=replace(config.generation, steps=steps, snapshot_every=2, seeds=seeds),
                   training=replace(config.training, device='cpu'),
                   data=replace(config.data, **data) if data else config.data)


def save_checkpoint(config, path, seed=42, **extra):
    import torch
    from sokolov.model import build_model
    torch.manual_seed(seed)
    model = build_model(config.model, pretrained=False).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': asdict(config), 'technical_test': True, **extra}, path)
    return model


def png_score(model, path, architecture, policy='rgb'):
    import numpy as np
    import torch
    from PIL import Image
    from sokolov.synthesize import normalized_pixels
    with Image.open(path) as image:
        x = torch.from_numpy(np.array(image.convert('RGB')).copy()).permute(2, 0, 1).float()[None]/255
    with torch.no_grad():
        return model(normalized_pixels(x, architecture, policy)).sigmoid().item(), image.size


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

    def test_generation_config_variants_validation_and_legacy_checkpoints(self):
        from sokolov.config import generation_config
        from sokolov.synthesize import legacy_generation, override_generation
        for name in ('resnet18', 'arcface'):
            generation = load_config(ROOT/f'configs/{name}_face.toml').generation
            self.assertEqual([v.name for v in generation.variants], ['pixel', 'robust', 'fourier'])
            self.assertEqual(generation.seeds, (0, 1, 2, 3, 4))
        raw = asdict(generation)
        self.assertEqual(generation_config(raw), generation)
        bad = [dict(raw, variants=[]), dict(raw, variants=raw['variants']*2),
               dict(raw, variants=[dict(raw['variants'][0], parametrization='gan')]),
               dict(raw, variants=[dict(raw['variants'][2], blur_every=4, blur_sigma_px=.5)]),
               dict(raw, variants=[dict(raw['variants'][1], blur_every=4, blur_sigma_px=0.)]),
               dict(raw, variants=[dict(raw['variants'][1], scale_min=1.2, scale_max=1.1)]),
               dict(raw, variants=[dict(raw['variants'][0], name='../x')]),
               dict(raw, variants=[dict(raw['variants'][0], name='seed_0')]),
               dict(raw, variants=[dict(raw['variants'][0], lr=True)]),
               dict(raw, variants=[dict(raw['variants'][0], unknown=1)])]
        for value in bad:
            with self.assertRaises(ValueError):
                generation_config(value)
        legacy = legacy_generation({'steps': 5, 'lr': .05, 'tv_weight': 1e-4, 'l2_weight': 1e-5,
                                    'seeds': [0, 1], 'snapshot_every': 2})
        self.assertEqual(legacy.variants[0].name, 'pixel')
        self.assertEqual((legacy.variants[0].lr, legacy.variants[0].tv_weight), (.05, 1e-4))
        changed = override_generation(generation, variants=['fourier', 'pixel'], seeds=[3], steps=10)
        self.assertEqual(([v.name for v in changed.variants], changed.seeds, changed.steps, changed.snapshot_every),
                         (['fourier', 'pixel'], (3,), 10, 10))
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            override_generation(generation, variants=['missing'])

    def test_bilinear_warp_matches_grid_sample_forward_and_backward(self):
        import math
        import torch
        import torch.nn.functional as F
        from sokolov.synthesize import _bilinear, sampling_grid, gaussian_blur
        angle, scale = math.radians(4), 1.07
        theta = torch.tensor([[[math.cos(angle)/scale, -math.sin(angle)/scale, .06],
                               [math.sin(angle)/scale, math.cos(angle)/scale, -.1]]], dtype=torch.float64)
        x = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(0))
        a, b = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        index, weight = sampling_grid(theta, 32)
        ours = _bilinear().apply(a, index, weight, index, weight)
        reference = F.grid_sample(b, F.affine_grid(theta.float(), (1, 3, 32, 32), align_corners=False),
                                  mode='bilinear', padding_mode='reflection', align_corners=False)
        self.assertLess((ours-reference).abs().max().item(), 1e-5)
        upstream = torch.randn(ours.shape, generator=torch.Generator().manual_seed(1))
        (ours*upstream).sum().backward(); (reference*upstream).sum().backward()
        self.assertLess((a.grad-b.grad).abs().max().item(), 1e-5)
        blurred = gaussian_blur(x, 1.)
        self.assertEqual(blurred.shape, x.shape)
        self.assertGreaterEqual(blurred.min().item(), 0); self.assertLessEqual(blurred.max().item(), 1)
        self.assertLess(blurred.diff(dim=-1).abs().mean().item(), x.diff(dim=-1).abs().mean().item())

    def test_actual_models_pixel_gradients_png_scores_determinism_and_cache(self):
        import torch
        from sokolov.synthesize import synthesize
        for architecture in ('resnet18', 'arcface'):
            with self.subTest(architecture=architecture), TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = small_config(architecture)
                checkpoint = root/'best_model.pt'
                model = save_checkpoint(config, checkpoint)
                digest = sha256(checkpoint)
                first = synthesize(config, checkpoint, 7)
                self.assertEqual((first.variant, first.trajectory.parent), ('pixel', root/'synthesis/pixel/seed_7'))
                rows = read_csv(first.trajectory)
                self.assertEqual([int(r['step']) for r in rows], [0, 1, 2, 3])
                self.assertEqual([r['step'] for r in rows if r['snapshot']], ['0', '2', '3'])
                self.assertNotEqual(sha256(first.snapshots_dir/'step_0000.png'), sha256(first.final_image))
                self.assertEqual(digest, sha256(checkpoint))
                for row in rows:
                    if row['snapshot']:
                        # Scores use the checkpoint face preprocessing (identity_v1 in the configs).
                        score, size = png_score(model, first.trajectory.parent/row['snapshot'], architecture,
                                                config.model.face_preprocessing)
                        self.assertEqual(size, (config.model.image_size,)*2)
                        self.assertAlmostEqual(score, float(row['score']), places=6)
                # Repeated seed and same checkpoint reproduce exactly, including PNG quantization.
                again = synthesize(config, checkpoint, 7, output_dir=root/'replay')
                self.assertEqual(first.final_image.read_bytes(), again.final_image.read_bytes())
                self.assertEqual(rows, read_csv(again.trajectory))
                with patch('sokolov.synthesize.build_model', side_effect=AssertionError('cache missed')):
                    synthesize(config, checkpoint, 7)
                variants = config.generation.variants
                changed = replace(config, generation=replace(config.generation,
                    variants=(replace(variants[0], lr=.01),)+variants[1:]))
                with self.assertRaisesRegex(ValueError, 'changed'):
                    synthesize(changed, checkpoint, 7)
                first.final_image.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'modified'):
                    synthesize(config, checkpoint, 7)

    def test_every_variant_both_architectures_frozen_weights_range_gradient_determinism(self):
        import torch
        from sokolov.model import state_fingerprint
        from sokolov.synthesize import synthesize
        for architecture in ('resnet18', 'arcface'):
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = small_config(architecture)
                checkpoint = root/'best_model.pt'
                model = save_checkpoint(config, checkpoint)
                saved = state_fingerprint(model.state_dict())
                for variant in config.generation.variants:
                    with self.subTest(architecture=architecture, variant=variant.name):
                        run = synthesize(config, checkpoint, 7, variant=variant.name)
                        meta = read_json(run.trajectory.parent/'metadata.json')
                        self.assertEqual(meta['model_state_sha256_before'], saved)
                        self.assertEqual(meta['model_state_sha256_after'], saved)
                        self.assertGreater(meta['first_input_gradient_norm'], 0)
                        self.assertGreaterEqual(meta['pixel_min'], 0)
                        self.assertLessEqual(meta['pixel_max'], 1)
                        rows = read_csv(run.trajectory)
                        self.assertEqual(rows[0]['optimized_logit'], '')
                        self.assertTrue(all(r['optimized_logit'] != '' for r in rows[1:]))
                        transformed = variant.name != 'pixel'
                        self.assertEqual(all(r['scale'] != '' for r in rows[1:]), transformed)
                        score, _ = png_score(model, run.final_image, architecture,
                                             config.model.face_preprocessing)
                        self.assertAlmostEqual(score, float(rows[-1]['score']), places=6)
                        again = synthesize(config, checkpoint, 7, variant=variant.name, output_dir=root/'replay')
                        self.assertEqual(run.final_image.read_bytes(), again.final_image.read_bytes())
                        self.assertEqual(rows, read_csv(again.trajectory))
                        other = synthesize(config, checkpoint, 8, variant=variant.name, output_dir=root/'other')
                        self.assertNotEqual(run.final_image.read_bytes(), other.final_image.read_bytes())
                self.assertEqual(state_fingerprint(torch.load(checkpoint, weights_only=False)['model']), saved)

    def test_experiment_analysis_degrades_gracefully_and_report_section(self):
        from PIL import Image
        from sokolov.analysis import publication_files, report_section
        from sokolov.synthesize import run_experiment, validate_synthesis
        with TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'
            config = small_config('resnet18', steps=2, seeds=(0, 1), detector_root=str(Path(tmp)/'no-models'),
                                  prepared_manifest=str(Path(tmp)/'missing/manifest.csv'))
            checkpoint = run/'best_model.pt'
            save_checkpoint(config, checkpoint)
            runs = run_experiment(config, checkpoint, output_dir=run/'synthesis', run_dir=run)
            self.assertEqual([(r.variant, r.seed) for r in runs],
                             [(v, s) for v in ('pixel', 'robust', 'fourier') for s in (0, 1)])
            summary = validate_synthesis(run/'synthesis')
            analysis = summary['analysis']
            for key in ('detector', 'reference', 'transfer', 'real'):
                self.assertFalse(analysis[key]['available'])
                self.assertTrue(analysis[key]['reason'])
            self.assertIn('det_10g.onnx', analysis['detector']['reason'])
            self.assertEqual(len(analysis['images']), 12)
            self.assertTrue(all(r['class'] == 'N/A' and r['faces'] is None for r in analysis['images']))
            self.assertEqual(set(analysis['per_variant']), {'pixel', 'robust', 'fourier'})
            self.assertEqual(analysis['per_variant']['robust']['final']['classes']['N/A'], 2)
            self.assertTrue(all(isinstance(line, str) for line in summary['interpretation']))
            self.assertTrue(all(r['weights_unchanged'] for r in summary['results']))
            self.assertEqual(set(summary['runtime']), {'pixel', 'robust', 'fourier'})
            self.assertEqual(len(read_csv(run/'synthesis/trajectories.csv')), 6*3)
            for path in ('grid.png', 'scores.png', 'by_seed/seed_0.png', 'by_seed/seed_1.png',
                         'robust/seed_1/progression.png', 'analysis.json', 'analysis.csv'):
                self.assertTrue((run/'synthesis'/path).is_file(), path)
            with Image.open(run/'synthesis/fourier/seed_0/progression.gif') as gif:
                self.assertEqual(getattr(gif, 'n_frames', 1), 2)
            with patch('sokolov.synthesize.build_model', side_effect=AssertionError('cache missed')):
                run_experiment(config, checkpoint, output_dir=run/'synthesis', run_dir=run)
            lines = report_section(run, run/'report/synthesis', summary)
            text = '\n'.join(lines)
            self.assertIn('## Генерация из случайного шума', text)
            self.assertIn('N/A', text)
            copied = {'synthesis/'+p.relative_to(run/'report/synthesis').as_posix()
                      for p in (run/'report/synthesis').rglob('*') if p.is_file()}
            self.assertEqual(copied, publication_files(summary))
            changed = replace(config, generation=replace(config.generation, seeds=(0,)))
            with self.assertRaisesRegex(ValueError, 'changed'):
                run_experiment(changed, checkpoint, output_dir=run/'synthesis')

    @unittest.skipUnless((BUFFALO/'det_10g.onnx').is_file() and (BUFFALO/'w600k_r50.onnx').is_file(),
                         'buffalo_l SCRFD/ArcFace files are not available locally')
    def test_analysis_with_local_detector_reference_and_transfer(self):
        import numpy as np
        from PIL import Image
        from sokolov.analysis import FaceTools, analyze_experiment
        from sokolov.data.prepare import align_face
        from sokolov.synthesize import synthesize
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = small_config('resnet18', steps=2, seeds=(0,), detector_root=str(BUFFALO.parents[1]),
                                  prepared_manifest=str(tmp/'prepared/manifest.csv'))
            tools = FaceTools(config.data)
            frames = sorted((ROOT/'data/raw/sokolov_positive_v1/positive').glob('*.jpg'))[::200][:5]
            rows = []
            for i, frame in enumerate(frames):
                with Image.open(frame) as image:
                    rgb = np.asarray(image.convert('RGB'))
                boxes, landmarks = tools.detector.detect(rgb[:, :, ::-1].copy(), max_num=0)
                crop = tmp/f'crop_{i}.png'
                align_face(rgb, landmarks[int(boxes[:, 4].argmax())]).save(crop)
                rows.append({'image_id': f'p{i}', 'split': 'train' if i < 3 else 'test', 'label': 1,
                             'status': 'ok', 'crop_path': str(crop), 'crop_sha256': sha256(crop)})
            noise = tmp/'noise.png'
            Image.fromarray((np.random.default_rng(0).random((224, 224, 3))*255).astype(np.uint8)).save(noise)
            rows.append({'image_id': 'n0', 'split': 'test', 'label': 0, 'status': 'ok',
                         'crop_path': str(noise), 'crop_sha256': sha256(noise)})
            write_csv(tmp/'prepared/manifest.csv', rows)
            run = tmp/'run'
            save_checkpoint(config, run/'resnet18/best.pt')
            save_checkpoint(small_config('arcface'), run/'arcface/best.pt')
            output = run/'synthesis'
            runs = [synthesize(config, run/'resnet18/best.pt', 0, variant=v.name, output_dir=output)
                    for v in config.generation.variants]
            analysis = analyze_experiment(runs, output, config=config, checkpoint=run/'resnet18/best.pt', run_dir=run)
            self.assertTrue(analysis['detector']['available'])
            self.assertEqual((analysis['reference']['available'], analysis['reference']['count']), (True, 3))
            self.assertEqual(analysis['transfer']['architecture'], 'arcface')
            positive = analysis['real']['positive']
            self.assertEqual(positive['n'], 2)
            self.assertEqual(positive['classes']['face-like'], 2)
            self.assertTrue(-1 <= positive['arcface_cosine']['mean'] <= 1)
            self.assertEqual(analysis['real']['negative']['classes']['pattern'], 1)
            for record in analysis['images']:
                self.assertIn(record['class'], ('face-like', 'many faces', 'pattern'))
                self.assertTrue(0 <= record['transfer_score'] <= 1)

    def test_publication_synthesis_allowlist_with_test_transport(self):
        from sokolov.analysis import publication_files
        from sokolov.report import publish_report
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'publication-fixture'
            report = root/'report'; report.mkdir(parents=True)
            synthesis = {'seeds': [0, 1], 'technical_test': False}
            files = {'report.md', 'report.html', 'metrics.json', 'metrics.csv',
                     'data_audit_summary.json', 'best_model.json'} | publication_files(synthesis)
            files |= {f'{name}/{plot}.png' for name in ('resnet18', 'arcface')
                      for plot in ('training', 'curves', 'confusion_matrix', 'scores')}
            self.assertIn('synthesis/seed_1/progression.png', files)
            self.assertFalse(any(name.endswith(('.gif', '.pt', '.jpg')) for name in files))
            for name in files:
                path = report/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'publication transport fixture')
            completed = {'status': 'completed', 'technical_test': False, 'models': ['resnet18', 'arcface'],
                'best_model': True, 'synthesis': synthesis,
                'files': {name: sha256(report/name) for name in files}}
            write_json(report/'completed.json', completed)
            received = []
            publish_report(root, transport=lambda *args: received.append(args))
            self.assertEqual(set(received[0][2]['files']), files)
            completed['files']['best_model.pt'] = 'not-allowed'
            write_json(report/'completed.json', completed)
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                publish_report(root, transport=lambda *a: self.fail('weights must not be sent'))


if __name__ == '__main__':
    unittest.main()
