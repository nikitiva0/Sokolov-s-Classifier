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
        self.assertEqual({c.data.raw_manifest for c in configs}, {"data/manifests/domain_v3/raw.csv"})
        self.assertTrue(all(c.split.mode == 'group_stratified' for c in configs))
        self.assertTrue(all(c.training.augmentation == 'robust' for c in configs))
        self.assertTrue(all(c.model.face_preprocessing == 'identity_v1' for c in configs))
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
            self.load_changed("arcface_face.toml", "iresnet50_w600k.pt", "iresnet50_w600k.onnx")

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

    def test_group_split_protocol_defaults_and_bounds(self):
        for filename in ("resnet18_face.toml", "arcface_face.toml"):
            config = load_config(ROOT / "configs" / filename)
            self.assertEqual((config.split.mode, config.split.test_fraction, config.split.dedup_max_distance),
                             ("group_stratified", 0.20, 8))
            self.assertEqual((config.data.fairface_dir, config.data.negative_count, config.data.fairface_download),
                             ("data/raw/fairface", 3000, False))
            self.assertEqual(config.training.augmentation, 'robust')
            self.assertEqual((config.training.head_epochs, config.training.device), (5, "auto"))
        self.assertEqual(load_config(ROOT / "configs/resnet18_face.toml").training.finetune_blocks, 2)
        self.assertEqual(load_config(ROOT / "configs/resnet18_face.toml").training.finetune_epochs, 10)
        for old, new in (('mode = "group_stratified"', 'mode = "by_frame"'),
                         ("dedup_max_distance = 8", "dedup_max_distance = 65"),
                         ('augmentation = "robust"', 'augmentation = "heavy"'),
                         ("aug_translate = 0.0", "aug_translate = 1.5"),
                         ("finetune_blocks = 2", "finetune_blocks = 5"),
                         ("aug_downscale_max = 112", "aug_downscale_max = 16"),
                         ("aug_jpeg_max = 90", "aug_jpeg_max = 101"),
                         ("aug_blur_p = 0.2", "aug_blur_p = 1.5"),
                         ("fairface_download = false", 'fairface_download = "no"')):
            with self.subTest(change=new), self.assertRaises(ValueError):
                self.load_changed("resnet18_face.toml", old, new)
        legacy = self.load_changed("resnet18_face.toml", 'mode = "group_stratified"', 'mode = "random_stratified"')
        self.assertEqual(legacy.split.mode, "random_stratified")

    def test_cli_check_config_and_predict_arguments(self):
        import contextlib
        import io
        from sokolov.__main__ import main
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["check-config", "--config", str(ROOT / "configs/resnet18_face.toml")]), 0)
        self.assertIn("group_stratified", output.getvalue())
        with TemporaryDirectory() as temp, contextlib.redirect_stderr(errors):
            self.assertEqual(main(["predict", "--run-dir", temp, "--model", "resnet18",
                                   str(Path(temp) / "absent.jpg")]), 2)
        self.assertIn("absent.jpg", errors.getvalue())

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
