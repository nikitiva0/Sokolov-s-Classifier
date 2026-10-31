"""Regression tests for tight face crops, source coordinates and coverage gates."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sokolov.config import load_config
from sokolov.data.prepare import SCRFDDetector, detector_input, validate_detector_coverage
from sokolov.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


class DetectorTests(unittest.TestCase):
    def test_tight_face_crop_is_not_magnified_or_stretched(self):
        import numpy as np
        from PIL import Image
        # Real failure: a tight 224px FairFace face was expanded to 640px.
        rgb = np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        padded, scale = detector_input(Image.fromarray(rgb))
        self.assertEqual(scale, 1.)
        self.assertEqual(padded.shape, (640, 640, 3))
        np.testing.assert_array_equal(padded[:224, :224], rgb[:, :, ::-1])
        self.assertFalse(padded[224:].any())
        self.assertFalse(padded[:, 224:].any())
        rectangular, scale = detector_input(Image.new('RGB', (240, 116), 'white'))
        self.assertEqual(scale, 1.)
        self.assertTrue((rectangular[:116, :240] == 255).all())
        self.assertFalse(rectangular[116:].any())

    def test_coordinates_return_to_original_image_after_downscaling(self):
        import numpy as np
        from PIL import Image
        class Network:
            def detect(self, image, max_num):
                assert image.shape == (640, 640, 3)
                return (np.array([[10., 20., 30., 40., .9]], dtype=np.float32),
                        np.array([[[10., 20.]]*5], dtype=np.float32))
        detector = SCRFDDetector.__new__(SCRFDDetector)
        detector.model = Network()
        result = detector(Image.new('RGB', (1920, 1080)))[0]
        np.testing.assert_allclose(result['bbox'], [30., 60., 90., 120.])
        np.testing.assert_allclose(result['landmarks'], [[30., 60.]]*5)
        self.assertAlmostEqual(result['confidence'], .9)

    def test_coverage_checks_each_class_and_partition(self):
        counts = {'labels_original': {'0': 3500, '1': 700},
                  'labels_usable': {'0': 33, '1': 697}}
        with self.assertRaisesRegex(ValueError, 'train/not_sokolov: 33/3500'):
            validate_detector_coverage({'counts': {'train': counts}}, .8)
        counts['labels_usable']['0'] = 3400
        validate_detector_coverage({'counts': {'train': counts}}, .8)
        with self.assertRaisesRegex(ValueError, 'val/sokolov'):
            validate_detector_coverage({'counts': {'train': counts, 'val': {
                'labels_original': {'0': 750, '1': 150},
                'labels_usable': {'0': 740, '1': 5}}}}, .8)

    def test_low_coverage_does_not_complete_preparation_or_bypass_cached_gate(self):
        from test_pipeline import fixtures, FakeDetector
        from sokolov.data.download import download_data
        from sokolov.data.prepare import prepare
        from sokolov.train import validate_prepared
        with TemporaryDirectory() as temporary:
            config = fixtures(Path(temporary))
            download_data(config)
            prepare(config, detector=FakeDetector())
            audit_path = Path(config.data.prepared_manifest).parent/'data_audit.json'
            audit = read_json(audit_path)
            audit['counts']['train']['labels_usable']['0'] = 0
            write_json(audit_path, audit)
            with self.assertRaisesRegex(ValueError, 'coverage'):
                prepare(config, detector=FakeDetector())
            with self.assertRaisesRegex(ValueError, 'coverage'):
                validate_prepared(config)
            # Old pre-fix crops cannot silently be reused under the new code.
            metadata = audit_path.parent/'prepared.json'
            saved = read_json(metadata)
            saved['signature']['schema_version'] = 1
            write_json(metadata, saved)
            with self.assertRaisesRegex(ValueError, 'changed'):
                prepare(config, detector=FakeDetector())
            with self.assertRaisesRegex(ValueError, 'differs'):
                validate_prepared(config)


if __name__ == '__main__':
    unittest.main()
