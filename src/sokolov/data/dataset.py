"""План: Dataset вырезанных лиц и transforms под выбранную модель.

Train: умеренные аугментации, сохраняющие идентичность и присутствие человека.
ResNet18: выровненное по пяти точкам лицо масштабируется до 224x224, затем
нормализация ImageNet. ArcFace: то же лицо 112x112, RGB [0,1]
нормализуется mean=std=(0.5,0.5,0.5). Детекция и выравнивание выполняются заранее,
их параметры и версия сохраняются. Val/test без случайных аугментаций.
Crop не должен незаметно удалить человека при сохранении положительной метки.
"""

from typing import Any

from ..config import Config
from ..contracts import Split


def build_dataset(config: Config, split: Split) -> Any:
    """Вернуть Dataset с (image_tensor, label, sample_id, parent_image_id).

    В face_crop метки проверяются по каждому лицу; метку исходного изображения нельзя
    автоматически назначать каждому обнаруженному лицу.
    """
    raise NotImplementedError("Dataset и transforms ещё не реализованы")
