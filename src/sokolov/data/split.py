"""План: групповое разбиение до аугментаций и аудит утечек.

Группы source/duplicates не делятся между выборками. Внешние train/test из raw
не перераспределяются. При обнаружении группы одновременно в train и test —
ошибка аудита, а не перенос данных. Val выделяется только из train.
Назначения создаются на полном master dataset до фильтрации/детекции лиц;
full_image и face_crop наследуют один ParentAssignment. Оба класса обязательны.
"""

from ..config import SplitConfig
from ..contracts import ParentAssignment, RawSample


def split_groups(
    samples: list[RawSample], group_ids: dict[str, str], config: SplitConfig
) -> list[ParentAssignment]:
    """Назначить split исходным image_id; group_ids получены аудитом master dataset."""
    raise NotImplementedError("Групповое разбиение ещё не реализовано")
