"""План: Dataset двух вариантов и transforms под выбранную модель.

Train: умеренные аугментации, сохраняющие идентичность и присутствие человека.
ResNet18: letterbox до 224x224, затем нормализация ImageNet; работают full_image
и face_crop. ArcFace: только выровненное по пяти точкам лицо 112x112, RGB [0,1]
нормализуется mean=std=(0.5,0.5,0.5). Детекция и выравнивание выполняются заранее,
их параметры и версия сохраняются. Val/test без случайных аугментаций.
Crop не должен незаметно удалить человека при сохранении положительной метки.
"""

from typing import Any

from ..config import Config
from ..contracts import Split


def build_dataset(config: Config, split: Split) -> Any:
    """Вернуть Dataset с (image_tensor, label, sample_id, parent_image_id).

    В face_crop метки проверяются по каждому лицу; метку полного кадра нельзя
    автоматически назначать каждому обнаруженному лицу.
    """
    raise NotImplementedError("Dataset и transforms ещё не реализованы")
