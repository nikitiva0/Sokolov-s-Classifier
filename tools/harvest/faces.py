"""SCRFD (как в sokolov.data.prepare) + ArcFace w600k_r50 на выровненных 112×112 лицах."""
from __future__ import annotations
import tomllib
from pathlib import Path
from types import SimpleNamespace

from . import REPO

CONFIG = REPO / 'configs/arcface_face.toml'


def phash_int(image):
    import imagehash
    return int(str(imagehash.phash(image)), 16)


def hamming(values, other):
    """Попарные расстояния Хэмминга 64-битных pHash: values[N] против other[M] → [N, M]."""
    import numpy as np
    a = np.asarray(values, dtype=np.uint64).reshape(-1, 1)
    b = np.asarray(other, dtype=np.uint64).reshape(1, -1)
    x = np.bitwise_xor(a, b)
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


class FaceEngine:
    def __init__(self):
        from importlib.metadata import version
        from insightface.model_zoo import get_model
        from insightface.utils.storage import ensure_available
        from sokolov.data.prepare import SCRFDDetector
        from sokolov.io import sha256
        # Только настройки детектора из [data]: не зависим от остальной схемы конфига.
        with CONFIG.open('rb') as stream:
            raw = tomllib.load(stream)['data']
        data = SimpleNamespace(detector_name=raw['detector_name'], detector_threshold=raw['detector_threshold'],
                               detector_root=str(REPO / raw['detector_root']))
        self.detector = SCRFDDetector(data)
        path = Path(ensure_available('models', data.detector_name, root=data.detector_root)) / 'w600k_r50.onnx'
        if not path.is_file():
            raise FileNotFoundError(f'{path} не найден (пакет buffalo_l)')
        self.arcface = get_model(str(path), providers=['CPUExecutionProvider'])
        self.arcface.prepare(ctx_id=-1)
        self.metadata = {'detector': self.detector.metadata,
                         'embedder': {'name': 'ArcFace/buffalo_l/w600k_r50.onnx', 'sha256': sha256(path),
                                      'insightface': version('insightface'), 'input': '112x112 aligned',
                                      'alignment': 'sokolov.data.prepare.align_face 224x224 -> INTER_AREA 112x112',
                                      'provider': 'CPUExecutionProvider'}}

    def detect(self, image, min_det=0.0, min_px=0):
        """Лица ≥ min_det и ≥ min_px (max(w, h) bbox, px), от крупных к мелким."""
        faces = []
        for face in self.detector(image):
            x1, y1, x2, y2 = face['bbox']
            px = max(x2 - x1, y2 - y1)
            if face['confidence'] >= min_det and px >= min_px:
                faces.append({**face, 'face_px': px})
        return sorted(faces, key=lambda f: -f['face_px'])

    def describe(self, image, face):
        """(L2-нормированный эмбеддинг float32[512], pHash выровненного кропа 224×224).
        ValueError, если выравнивание невозможно."""
        import cv2
        import numpy as np
        from sokolov.data.prepare import align_face
        aligned = align_face(image, face['landmarks'])
        small = cv2.resize(np.asarray(aligned), (112, 112), interpolation=cv2.INTER_AREA)
        embedding = self.arcface.get_feat([np.ascontiguousarray(small[:, :, ::-1])])[0].astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('Нулевой эмбеддинг')
        return embedding / norm, phash_int(aligned)
