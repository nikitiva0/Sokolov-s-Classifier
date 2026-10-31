"""Геометрия кропа по контракту и равномерные моменты кадров. Без ML-зависимостей."""
from __future__ import annotations

CROP_SCALE = 2.0
MAX_SIDE = 512


def crop_box(bbox, width, height, scale=CROP_SCALE):
    """Квадрат стороной scale·max(w, h) с центром в bbox; сдвигается внутрь кадра,
    уменьшается только если кадр меньше квадрата. Возвращает (left, top, right, bottom), px."""
    x1, y1, x2, y2 = map(float, bbox)
    if width < 1 or height < 1 or x2 <= x1 or y2 <= y1:
        raise ValueError(f'Некорректный bbox/кадр: {bbox}, {width}×{height}')
    side = max(1, int(round(min(scale * max(x2 - x1, y2 - y1), width, height))))
    left = int(round((x1 + x2) / 2 - side / 2))
    top = int(round((y1 + y2) / 2 - side / 2))
    left = min(max(left, 0), width - side)
    top = min(max(top, 0), height - side)
    return left, top, left + side, top + side


def crop_face(image, bbox, max_side=MAX_SIDE):
    """PIL-кроп в исходном разрешении; только уменьшение, если сторона > max_side."""
    from PIL import Image
    box = crop_box(bbox, *image.size)
    crop = image.crop(box)
    if crop.width > max_side:
        crop = crop.resize((max_side, max_side), Image.LANCZOS)
    return crop, box


def timestamps(duration, start=0.0, end=None, count=None, every=None):
    """Детерминированные моменты (s): count штук по центрам равных интервалов либо шаг every."""
    start = max(0.0, float(start or 0))
    end = float(duration) if end is None else min(float(end), float(duration))
    if end <= start:
        raise ValueError(f'Пустой интервал: start={start:g} s, end={end:g} s')
    if (count is None) == (every is None):
        raise ValueError('Укажите ровно одно из --count / --every')
    if count is not None:
        if count < 1:
            raise ValueError('--count >= 1')
        step = (end - start) / count
        return [round(start + (i + .5) * step, 3) for i in range(count)]
    if every <= 0:
        raise ValueError('--every > 0')
    values = [start + every * (i + .5) for i in range(int((end - start) // every) + 1)]
    return [round(v, 3) for v in values if v < end] or [round((start + end) / 2, 3)]
