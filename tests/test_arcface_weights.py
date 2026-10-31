"""Converted InsightFace w600k_r50 weights: exact PyTorch/onnxruntime parity; skipped without the ONNX."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sokolov.config import ModelConfig
from sokolov.io import sha256

ROOT = Path(__file__).resolve().parents[1]
ONNX = Path('checkpoints/insightface/models/buffalo_l/w600k_r50.onnx')


def converter():
    spec = importlib.util.spec_from_file_location('convert_w600k_onnx', ROOT/'tools/convert_w600k_onnx.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArcFaceWeightsTests(unittest.TestCase):
    # cwd-relative pack, like SCRFD: an existing folder guarantees insightface does not download.
    @unittest.skipUnless((Path.cwd()/ONNX).is_file(), 'InsightFace buffalo_l w600k_r50.onnx absent')
    def test_missing_w600k_checkpoint_is_converted_and_matches_onnxruntime(self):
        import numpy as np
        import onnxruntime
        import torch
        from sokolov.model import build_model, state_fingerprint
        torch.set_num_threads(2)
        tool = converter()
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'iresnet50_w600k.pt'
            # Model fields only: independent of unrelated [data]/[generation] tables.
            model = build_model(ModelConfig('arcface', 'iresnet50', tool.WEIGHTS, 112, 512, str(path)))
            self.assertIn('WebFace600K', model.pretrained_provenance['source'])
            self.assertTrue(path.is_file())
            self.assertEqual(model.pretrained_provenance['sha256'], sha256(path))
            local = ROOT/tool.OUTPUT
            if local.is_file():
                self.assertEqual(sha256(local), sha256(path))
            backbone = model.backbone.eval()
            self.assertEqual(state_fingerprint(backbone.state_dict()), tool.STATE_SHA256)
            pixels = np.random.default_rng(0).uniform(-1, 1, (4, 3, 112, 112)).astype(np.float32)
            with torch.no_grad():
                ours = backbone(torch.from_numpy(pixels)).numpy()
            session = onnxruntime.InferenceSession(str(Path.cwd()/ONNX), providers=['CPUExecutionProvider'])
            name = session.get_inputs()[0].name
            theirs = np.concatenate([session.run(None, {name: x[None]})[0] for x in pixels])
            self.assertLessEqual(float(np.abs(ours - theirs).max()), 1e-4)

    def test_unknown_onnx_is_rejected(self):
        tool = converter()
        with TemporaryDirectory() as tmp:
            fake = Path(tmp)/'models/buffalo_l/w600k_r50.onnx'
            fake.parent.mkdir(parents=True)
            fake.write_bytes(b'not the official model')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                tool.convert_checkpoint(Path(tmp)/'out/iresnet50_w600k.pt', root=tmp)
            self.assertFalse((Path(tmp)/'out/iresnet50_w600k.pt').exists())


if __name__ == '__main__':
    unittest.main()
