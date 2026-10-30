"""План: две модели на общем фиксированном split и собственной предобработке.

ResNet18: обучение головы, затем fine-tuning последнего блока backbone.
ArcFace: pretrained iresnet50 encoder заморожен, обучается бинарная голова;
finetune_epochs=0 в базовом конфиге. В обоих случаях BCEWithLogitsLoss.

Validation определяет checkpoint по image-level average precision. Для лиц
агрегация и no_face совпадают с оценкой, crop-метрики показываются отдельно.
Test не участвует.
Число эпох, seed, конфигурация и отпечаток манифеста сохраняются с checkpoint.
"""

from .config import Config
from .contracts import TrainingArtifacts


def train(config: Config) -> TrainingArtifacts:
    """Обучить модель на зафиксированном split и сохранить артефакты."""
    raise NotImplementedError("Обучение ещё не реализовано и не запускалось")
