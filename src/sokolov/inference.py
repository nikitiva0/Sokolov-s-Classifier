"""Photo inference with a saved model and its exact face pipeline.

One path for the localhost UI (classify: uploaded bytes) and the `predict` CLI
(predict_images: files): SCRFD (same resize policy) -> primary-independent scoring of
every detected face -> aligned crop -> deterministic preprocessing -> validation threshold.
"""
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
    """model: 'best' (validated winner best_model.pt) or 'resnet18'/'arcface' (that model's best.pt).

    allow_technical: accept technical-test checkpoints (random weights; CLI prints a warning,
    the UI refuses them). detector: injected detector (tests); otherwise the local SCRFD file
    must match the run's audit.
    """

    def __init__(self, run_dir, *, device='auto', detector_root=None, model='best', allow_technical=False,
                 detector=None):
        import torch
        self.run_dir = Path(run_dir).resolve()
        if model == 'best':
            selected = read_json(self.run_dir/'best_model.json')
            checkpoint = self.run_dir/'best_model.pt'
        elif model in ('resnet18', 'arcface'):
            selected, checkpoint = None, self.run_dir/model/'best.pt'
        else:
            raise ValueError('model: best, resnet18 or arcface')
        digest = sha256(checkpoint)
        if selected is not None and digest != selected['checkpoint_sha256']:
            raise ValueError('Контрольная сумма лучшей модели не совпадает.')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.technical_test = bool(saved['technical_test'])
        if self.technical_test and not allow_technical:
            raise ValueError('Для проверки фото нужны обученные веса, а не технический тест.')
        self.config = checkpoint_config(saved)
        self.architecture = self.config.model.architecture
        if selected is not None and (selected['architecture'] != self.architecture
                                     or selected['selection_partition'] != 'val'):
            raise ValueError('Некорректные сведения о выборе лучшей модели.')
        if selected is None and self.architecture != model:
            raise ValueError('Архитектура checkpoint не совпадает с запрошенной моделью.')
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
        if detector is None:
            from dataclasses import replace
            data = self.config.data
            if detector_root is not None:
                data = replace(data, detector_root=str(detector_root))
            path = Path(data.detector_root)/'models'/data.detector_name/'det_10g.onnx'
            if not path.is_file():
                raise FileNotFoundError(f'Локальный детектор не найден: {path}')
            if sha256(path) != expected['sha256'] or data.detector_threshold != expected['threshold']:
                raise ValueError('Детектор отличается от использованного при обучении.')
            detector = SCRFDDetector(data)
        self.detector = detector
        self.device = device_for(device)
        self.model = build_model(self.config.model, pretrained=False)
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.transform = transforms(self.config, train=False)
        self.lock = threading.Lock()

    def info(self):
        return {'ready': True, 'model': self.architecture, 'device': str(self.device)}

    def classify_image(self, image):
        """Score every detected face of an RGB PIL image; a failed face keeps score None."""
        import torch
        faces = []
        with self.lock, torch.inference_mode():
            detections = self.detector(image)
            for detection in detections:
                bbox = [float(v) for v in detection['bbox']]
                confidence = detection.get('confidence')
                face = {'bbox': bbox, 'confidence': None if confidence is None else float(confidence),
                        'score': None, 'positive': False, 'error': None}
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
        return faces

    @staticmethod
    def verdict(faces):
        """A photo is positive when any face passes the validation threshold."""
        if any(face['positive'] for face in faces):
            return 'sokolov', 'На фото найден Соколов'
        if not faces:
            return 'no_face', 'Лицо не найдено'
        if any(face['error'] for face in faces):
            return 'uncertain', 'Не удалось проверить все лица'
        return 'not_sokolov', 'Соколов не найден'

    def classify(self, data):
        image = decode_photo(data)
        faces = self.classify_image(image)
        status, title = self.verdict(faces)
        return {'status': status, 'title': title, 'faces': faces,
                'image': {'width': image.width, 'height': image.height,
                          'preview': preview_data(image)}}


def predict_images(run_dir, images, *, which='best', device=None, detector=None, detector_root=None):
    """`predict` CLI: the same PhotoClassifier as the UI, for image files.

    Photo score = maximum face score (0 without a scored face); decision 'sokolov' iff
    some face passes the validation threshold. Technical checkpoints are allowed and flagged.
    """
    from .data.download import image_info
    classifier = PhotoClassifier(run_dir, device=device or 'auto', detector_root=detector_root, model=which,
                                 allow_technical=True, detector=detector)
    results = []
    for path in images:
        image, _ = image_info(path)
        faces = classifier.classify_image(image)
        status, _ = classifier.verdict(faces)
        scored = [f['score'] for f in faces if f['score'] is not None]
        results.append({'path': str(path), 'architecture': classifier.architecture,
                        'threshold': classifier.threshold, 'technical_test': classifier.technical_test,
                        'faces': faces, 'status': status, 'score': max(scored, default=0.),
                        'decision': 'sokolov' if status == 'sokolov' else 'not_sokolov'})
    return results


def format_prediction(result):
    lines = [f"{result['path']}: {result['decision']} [{result['status']}] (score={result['score']:.4f}, "
             f"threshold={result['threshold']:.4f}, model={result['architecture']}, faces={len(result['faces'])})"]
    if result['technical_test']:
        lines.append('  WARNING: technical test checkpoint (random weights), not a trained classifier')
    for i, face in enumerate(result['faces'], 1):
        bbox = ', '.join(f'{v:.0f}' for v in face['bbox'])
        detection = 'n/a' if face['confidence'] is None else f"{face['confidence']:.3f}"
        score = f"n/a ({face['error']})" if face['score'] is None else f"{face['score']:.4f}"
        lines.append(f"  face {i}: bbox=[{bbox}] px, det={detection}, score={score}")
    return '\n'.join(lines)
