"""Photo inference and the localhost upload contract, without external services."""
from dataclasses import asdict, replace
from http.client import HTTPConnection
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import re
import threading
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from sokolov.config import load_config
from sokolov.inference import PhotoClassifier, decode_photo, MAX_UPLOAD_BYTES
from sokolov.io import sha256, write_json
from sokolov.web import create_server

ROOT = Path(__file__).resolve().parents[1]


def image_bytes(color='white', *, size=(100, 80), exif=None):
    stream = BytesIO()
    image = Image.new('RGB', size, color)
    image.save(stream, format='JPEG', **({'exif': exif} if exif else {}))
    return stream.getvalue()


class PhotoInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def fixture(self, root):
        import torch
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
            def forward(self, images):
                return images.mean((1, 2, 3)).unsqueeze(1)*self.weight
        model = TinyModel()
        config = load_config(ROOT/'configs/resnet18_face.toml')
        config = replace(config, data=replace(config.data, detector_root=str(root/'detector')),
                         model=replace(config.model, face_preprocessing='rgb'))
        path = root/'detector/models/buffalo_l/det_10g.onnx'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'FIXTURE_DETECTOR_ONLY')
        torch.save({'config': asdict(config), 'model': model.state_dict(), 'technical_test': False},
                   root/'best_model.pt')
        self.metadata(root)
        write_json(root/'data/data_audit.json', {'detector': {
            'resize_policy': 'shrink_only_top_left_pad_v1', 'sha256': sha256(path), 'threshold': .5}})
        detection = {'bbox': [0., 0., 99., 79.],
                     'landmarks': [[30, 30], [60, 30], [45, 45], [30, 60], [60, 60]]}
        detector = Mock(return_value=[detection])
        with patch('sokolov.inference.build_model', return_value=model), \
             patch('sokolov.inference.SCRFDDetector', return_value=detector):
            classifier = PhotoClassifier(root, device='cpu')
        return classifier, detector, detection

    def metadata(self, root):
        digest = sha256(root/'best_model.pt')
        write_json(root/'best_model.json', {'architecture': 'resnet18',
            'checkpoint_sha256': digest, 'selection_partition': 'val'})
        write_json(root/'resnet18/threshold.json', {'threshold': .5,
            'checkpoint_sha256': digest, 'selection_partition': 'val'})

    def test_positive_negative_no_face_and_frozen_model(self):
        import torch
        with TemporaryDirectory() as temp:
            classifier, detector, _ = self.fixture(Path(temp))
            before = classifier.model.weight.clone()
            with patch('sokolov.inference.align_face', side_effect=lambda image, _: image):
                positive = classifier.classify(image_bytes())
                negative = classifier.classify(image_bytes('black'))
            self.assertEqual(positive['status'], 'sokolov')
            self.assertGreater(positive['faces'][0]['score'], .5)
            self.assertEqual(negative['status'], 'not_sokolov')
            detector.return_value = []
            result = classifier.classify(image_bytes())
            self.assertEqual(result['status'], 'no_face')
            self.assertEqual(result['faces'], [])
            self.assertTrue(torch.equal(before, classifier.model.weight))
            self.assertIsNone(classifier.model.weight.grad)

    def test_multiple_faces_and_failed_alignment_are_not_silently_negative(self):
        with TemporaryDirectory() as temp:
            classifier, detector, detection = self.fixture(Path(temp))
            detector.return_value = [detection, detection]
            with patch('sokolov.inference.align_face', side_effect=[
                    Image.new('RGB', (224, 224), 'black'), Image.new('RGB', (224, 224), 'white')]):
                result = classifier.classify(image_bytes())
            self.assertEqual(result['status'], 'sokolov')
            self.assertEqual([x['positive'] for x in result['faces']], [False, True])
            with patch('sokolov.inference.align_face', side_effect=[
                    Image.new('RGB', (224, 224), 'black'), ValueError('bad landmarks')]):
                result = classifier.classify(image_bytes())
            self.assertEqual(result['status'], 'uncertain')
            self.assertIsNone(result['faces'][1]['score'])

    def test_rejects_random_weights_wrong_threshold_and_legacy_detector(self):
        import torch
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            write_json(root/'resnet18/threshold.json', {'checkpoint_sha256': 'wrong',
                'selection_partition': 'val', 'threshold': .5})
            with self.assertRaisesRegex(ValueError, 'Порог'):
                PhotoClassifier(root, device='cpu')
            self.metadata(root)
            write_json(root/'data/data_audit.json', {'detector': {'resize_policy': 'legacy'}})
            with self.assertRaisesRegex(ValueError, 'масштаба'):
                PhotoClassifier(root, device='cpu')
            saved = torch.load(root/'best_model.pt', weights_only=False)
            saved['technical_test'] = True
            torch.save(saved, root/'best_model.pt')
            self.metadata(root)
            with self.assertRaisesRegex(ValueError, 'технический'):
                PhotoClassifier(root, device='cpu')

    def test_decode_corruption_limits_and_exif_orientation(self):
        for content in (b'', b'not an image', b'x'*(MAX_UPLOAD_BYTES+1)):
            with self.subTest(length=len(content)), self.assertRaises(ValueError):
                decode_photo(content)
        exif = Image.Exif(); exif[274] = 6
        image = decode_photo(image_bytes(size=(20, 10), exif=exif))
        self.assertEqual(image.size, (10, 20))
        self.assertEqual(image.mode, 'RGB')


class WebUploadTests(unittest.TestCase):
    def setUp(self):
        self.classifier = Mock()
        self.classifier.info.return_value = {'ready': True, 'model': 'fixture', 'device': 'cpu'}
        self.classifier.classify.return_value = {'status': 'no_face', 'faces': []}
        self.server = create_server(self.classifier, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        self.connection.request('GET', '/')
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200)
        html = response.read().decode()
        self.token = re.search("const token='([^']+)'", html).group(1)

    def tearDown(self):
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_upload_and_error_paths(self):
        self.connection.request('POST', '/api/predict', b'image', {'X-Request-Token': self.token})
        response = self.connection.getresponse()
        self.assertEqual(response.status, 200); response.read()
        self.classifier.classify.assert_called_once_with(b'image')
        self.classifier.classify.side_effect = ValueError('Повреждённое изображение')
        self.connection.request('POST', '/api/predict', b'bad', {'X-Request-Token': self.token})
        response = self.connection.getresponse()
        self.assertEqual(response.status, 400)
        self.assertIn('Повреждённое', response.read().decode())

    def test_foreign_request_and_oversized_upload_rejected_before_inference(self):
        self.connection.request('POST', '/api/predict', b'image')
        response = self.connection.getresponse()
        self.assertEqual(response.status, 403); response.read()
        self.connection.request('POST', '/api/predict', headers={
            'X-Request-Token': self.token, 'Content-Length': str(MAX_UPLOAD_BYTES+1)})
        response = self.connection.getresponse()
        self.assertEqual(response.status, 413); response.read()
        self.classifier.classify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
