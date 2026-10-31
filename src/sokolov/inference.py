"""Local photo inference using the saved validation winner and its exact face pipeline."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import threading
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from .data.dataset import transforms
from .data.prepare import SCRFDDetector, align_face, DETECTOR_RESIZE_POLICY
from .io import read_json, sha256
from .model import build_model
from .synthesize import checkpoint_config
from .train import device_for

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 40_000_000


def decode_photo(data):
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('Выберите изображение размером до 15 МБ.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as original:
                if original.format not in ('JPEG', 'PNG', 'WEBP'):
                    raise ValueError('Поддерживаются фотографии JPG, PNG и WebP.')
                if original.width * original.height > MAX_PIXELS:
                    raise ValueError('Изображение слишком большое. Уменьшите его до 40 мегапикселей.')
                image = ImageOps.exif_transpose(original).convert('RGB')
                image.load()
                return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as error:
        raise ValueError('Не удалось прочитать изображение. Выберите другой JPG, PNG или WebP.') from error


def preview_data(image):
    preview = image.copy()
    preview.thumbnail((1600, 1600))
    stream = BytesIO()
    preview.save(stream, format='JPEG', quality=90)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


class PhotoClassifier:
    def __init__(self, run_dir, *, device='auto', detector_root=None):
        import torch
        self.run_dir = Path(run_dir).resolve()
        selected = read_json(self.run_dir/'best_model.json')
        checkpoint = self.run_dir/'best_model.pt'
        digest = sha256(checkpoint)
        if digest != selected['checkpoint_sha256']:
            raise ValueError('Контрольная сумма лучшей модели не совпадает.')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['technical_test']:
            raise ValueError('Для проверки фото нужны обученные веса, а не технический тест.')
        self.config = checkpoint_config(saved)
        self.architecture = self.config.model.architecture
        if selected['architecture'] != self.architecture or selected['selection_partition'] != 'val':
            raise ValueError('Некорректные сведения о выборе лучшей модели.')
        threshold = read_json(self.run_dir/self.architecture/'threshold.json')
        if threshold['checkpoint_sha256'] != digest or threshold['selection_partition'] != 'val':
            raise ValueError('Порог должен быть выбран на validation для этой модели.')
        self.threshold = float(threshold['threshold'])
        if not 0 <= self.threshold <= 1.000001:
            raise ValueError('Некорректный порог классификатора.')
        audit = read_json(self.run_dir/'data/data_audit.json')
        expected = audit['detector']
        if expected.get('resize_policy') != DETECTOR_RESIZE_POLICY:
            raise ValueError('Нужна модель, обученная после исправления масштаба детектора (v2).')
        from dataclasses import replace
        data = self.config.data
        if detector_root is not None:
            data = replace(data, detector_root=str(detector_root))
        path = Path(data.detector_root)/'models'/data.detector_name/'det_10g.onnx'
        if not path.is_file():
            raise FileNotFoundError(f'Локальный детектор не найден: {path}')
        if sha256(path) != expected['sha256'] or data.detector_threshold != expected['threshold']:
            raise ValueError('Детектор отличается от использованного при обучении.')
        self.detector = SCRFDDetector(data)
        self.device = device_for(device)
        self.model = build_model(self.config.model, pretrained=False)
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.transform = transforms(self.config, train=False)
        self.lock = threading.Lock()

    def info(self):
        return {'ready': True, 'model': self.architecture, 'device': str(self.device)}

    def classify(self, data):
        import torch
        image = decode_photo(data)
        faces = []
        with self.lock, torch.inference_mode():
            detections = self.detector(image)
            for detection in detections:
                bbox = [float(v) for v in detection['bbox']]
                face = {'bbox': bbox, 'score': None, 'positive': False, 'error': None}
                try:
                    crop = align_face(image, detection['landmarks'])
                    pixels = self.transform(crop).unsqueeze(0).to(self.device)
                    score = float(self.model(pixels).reshape(-1).sigmoid().item())
                    if not 0 <= score <= 1:
                        raise ValueError('Некорректная оценка модели')
                    face.update(score=score, positive=score >= self.threshold)
                except ValueError:
                    face['error'] = 'Не удалось проверить это лицо'
                faces.append(face)
        if any(face['positive'] for face in faces):
            status, title = 'sokolov', 'На фото найден Соколов'
        elif not faces:
            status, title = 'no_face', 'Лицо не найдено'
        elif any(face['error'] for face in faces):
            status, title = 'uncertain', 'Не удалось проверить все лица'
        else:
            status, title = 'not_sokolov', 'Соколов не найден'
        return {'status': status, 'title': title, 'faces': faces,
                'image': {'width': image.width, 'height': image.height,
                          'preview': preview_data(image)}}
