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
    image_id: str
    path: Path
    label: Label
    source_id: str
    source_url: str
    partition: Partition | None


@dataclass(frozen=True)
class ParentAssignment:
    """Split исходного изображения; создаётся до детекции лиц.

    В group_stratified лекции и близкие дубликаты не пересекают split.
    В external сохраняются train/test и целостность групп; val выделяется из train.
    random_stratified оставлен как явно выбранный устаревший протокол.
    """

    image_id: str
    group_id: str
    partition: Partition | None
    split: Split


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
