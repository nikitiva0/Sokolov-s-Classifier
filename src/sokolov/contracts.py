"""Типы исходников, разбиения и артефактов моделей."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Label = Literal[0, 1]
Split = Literal["train", "val", "test"]
Partition = Literal["train", "test"]
DatasetVariant = Literal["face_crop"]


@dataclass(frozen=True)
class RawSample:
    """Исходник raw.csv. group_id — единица независимости при разбиении.

    Пустой group_id означает source_id (старые манифесты без колонки).
    domain: lecture_webcam, lecture_room, talk_stage, interview, photo, fairface.
    """

    image_id: str
    path: Path
    label: Label
    source_id: str
    source_url: str
    partition: Partition | None
    group_id: str = ''
    domain: str = ''
    dataset: str = ''
    person: str = ''
    pinned_split: str = ''  # train/test, закреплённый вручную набором (user_v3)

    @property
    def group(self) -> str:
        return self.group_id or self.source_id


@dataclass(frozen=True)
class ParentAssignment:
    """Split исходного изображения; создаётся до детекции лиц.

    group_stratified (по умолчанию): группа целиком в одной выборке; группы,
    связанные pHash-кандидатами исходников, объединяются до разбиения.
    random_stratified — явный устаревший режим, группы могут пересекаться.
    external сохраняет train/test и целостность групп; val выделяется из train.
    """

    image_id: str
    group_id: str
    partition: Partition | None
    split: Split
    label: int = -1
    domain: str = ''
    dataset: str = ''


@dataclass(frozen=True)
class PreparedSample:
    """Производный образец наследует split/group_id из общего ParentAssignment.

    Метка crop проверяется отдельно: лицо другого человека на положительном
    исходном кадре имеет label=0. sample хранит исходное изображение.
    """

    sample: RawSample
    sample_id: str
    variant: DatasetVariant
    path: Path
    label: Label
    sha256: str
    phash: str
    group_id: str
    split: Split | None = None

    @property
    def parent_image_id(self) -> str:
        return self.sample.image_id


@dataclass(frozen=True)
class TrainingArtifacts:
    checkpoint: Path
    resolved_config: Path
    history: Path
    split_manifest_sha256: str
    pretrained_checkpoint_sha256: str


@dataclass(frozen=True)
class EvaluationArtifacts:
    metrics: Path
    predictions: Path
    confusion_matrix: Path
    threshold: float


@dataclass(frozen=True)
class SynthesisArtifacts:
    seed: int
    final_image: Path
    snapshots_dir: Path
    trajectory: Path
