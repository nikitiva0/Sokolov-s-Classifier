"""План: два адаптера с одним логитом присутствия и embedding_dim=512.

ResNet18: torchvision IMAGENET1K_V1, 224x224, Linear(512,1).
ArcFace: InsightFace arcface_torch iresnet50, pretrained MS1MV3, 112x112;
L2-normalized embedding -> Linear(512,1). ArcFace — loss предобучения, а не
отдельный тип бинарной головы. PyTorch checkpoint загружается локально,
его SHA-256 сохраняется; скачиваний при импорте или config validation нет.

Обучение использует BCEWithLogitsLoss; sigmoid применяется только к оценкам.
Генерация использует ту же нормализацию входа, что обучение и оценка.
Оба адаптера сохраняют autograd по входу, включая замороженный ArcFace encoder.
"""

from typing import Any

from .config import ModelConfig


def build_model(config: ModelConfig, *, pretrained: bool = True) -> Any:
    """Вернуть nn.Module с выходом [batch, 1]; веса пока не загружаются."""
    raise NotImplementedError("Построение pretrained-модели ещё не реализовано")
