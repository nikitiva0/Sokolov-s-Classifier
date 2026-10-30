"""План: выбрать порог на validation, затем один раз оценить test.

Сохраняются ROC-AUC, average precision, precision/recall/F1, confusion matrix,
число изображений и независимых групп каждого класса, предсказания по image_id.
Для face_crop score исходного фото — максимум оценок его лиц; no_face даёт
отрицательное решение с отдельным флагом. Порог и checkpoint выбираются по
image-level метрикам validation с той же агрегацией; crop-метрики отдельно.
Все исходные test image_id сохраняются, включая пропуски детектора.
При отсутствии одного класса показывается ограничение метрики; test автоматически
не перераспределяется. Пригодность набора проверяется до начала обучения.
"""

from pathlib import Path

from .config import Config
from .contracts import EvaluationArtifacts


def select_threshold(config: Config, checkpoint: Path) -> float:
    """Выбрать порог по максимальному F1 только на validation."""
    raise NotImplementedError("Выбор порога ещё не реализован")


def evaluate_test(config: Config, checkpoint: Path, threshold: float) -> EvaluationArtifacts:
    """Оценить test с заранее зафиксированными checkpoint и порогом."""
    raise NotImplementedError("Оценка на test ещё не реализована")
