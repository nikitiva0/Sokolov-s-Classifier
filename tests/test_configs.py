"""Проверки согласованности моделей, вариантов данных и внешних partition."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sokolov.config import load_config
from sokolov.contracts import PreparedSample, RawSample


ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigTests(unittest.TestCase):
    def load_changed(self, filename: str, old: str, new: str):
        source = (ROOT / "configs" / filename).read_text()
        self.assertIn(old, source)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "experiment.toml"
            path.write_text(source.replace(old, new))
            return load_config(path)

    def test_two_face_experiments_share_source_and_split(self):
        configs = [
            load_config(ROOT / "configs" / filename)
            for filename in ("resnet18_face.toml", "arcface_face.toml")
        ]
        self.assertEqual(
            [(c.model.architecture, c.data.variant) for c in configs],
            [("resnet18", "face_crop"), ("arcface", "face_crop")],
        )
        self.assertEqual({c.data.raw_manifest for c in configs}, {"data/manifests/raw.csv"})
        self.assertTrue(all(c.split == configs[0].split for c in configs))
        self.assertTrue(all(c.generation == configs[0].generation for c in configs))
        self.assertEqual(configs[0].split.validation_fraction, 0.15)
        self.assertEqual(configs[1].training.finetune_epochs, 0)

    def test_both_models_reject_full_image(self):
        for filename in ("resnet18_face.toml", "arcface_face.toml"):
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "face_crop"):
                self.load_changed(filename, 'variant = "face_crop"', 'variant = "full_image"')

    def test_detector_coverage_must_be_a_positive_fraction(self):
        for invalid in ('0', '1.1', '-0.1', 'true', 'nan'):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.load_changed('resnet18_face.toml', 'detector_threshold = 0.5',
                    'detector_threshold = 0.5\nmin_detector_coverage = '+invalid)

    def test_models_reject_wrong_input_size(self):
        for filename, old, new in (
            ("resnet18_face.toml", "image_size = 224", "image_size = 112"),
            ("arcface_face.toml", "image_size = 112", "image_size = 224"),
        ):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.load_changed(filename, old, new)

    def test_arcface_rejects_onnx_without_input_gradients(self):
        with self.assertRaisesRegex(ValueError, "ONNX"):
            self.load_changed("arcface_face.toml", "iresnet50_ms1mv3.pt", "iresnet50_ms1mv3.onnx")

    def test_unknown_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.load_changed("resnet18_face.toml", 'architecture = "resnet18"', 'architecture = "efficientnet_b0"')

    def test_unknown_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError, "variant"):
            self.load_changed("resnet18_face.toml", 'variant = "face_crop"', 'variant = "background_removed"')

    def test_validation_fraction_is_a_proper_fraction(self):
        for invalid in ("0", "1", "-0.2", "1.1", "true", "nan"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.load_changed("resnet18_face.toml", "validation_fraction = 0.15", f"validation_fraction = {invalid}")

    def test_imported_test_cannot_be_configured_as_a_new_fraction(self):
        with self.assertRaisesRegex(ValueError, "split"):
            self.load_changed("resnet18_face.toml", "validation_fraction = 0.15", "train = 0.7\nval = 0.15\ntest = 0.15")

    def test_other_person_crop_can_have_negative_label_on_positive_parent(self):
        parent = RawSample("image-1", Path("data/raw/photo.jpg"), 1, "session-1", "", "test")
        crop = PreparedSample(
            sample=parent,
            sample_id="image-1-face-2",
            variant="face_crop",
            path=Path("data/processed/face_crop/test/image-1-face-2.jpg"),
            label=0,
            sha256="crop-hash",
            phash="crop-phash",
            group_id="session-1",
            split="test",
        )
        self.assertEqual(crop.parent_image_id, parent.image_id)
        self.assertEqual(crop.label, 0)
        self.assertEqual(crop.sample.partition, "test")


if __name__ == "__main__":
    unittest.main()
