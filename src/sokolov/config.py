"""Строгая загрузка TOML средствами стандартной библиотеки Python 3.11+."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class DataConfig:
    variant: str
    raw_manifest: str
    prepared_manifest: str
    artifacts_dir: str
    phash_max_distance: int
    positive_dir: str = "data/raw/sokolov_positive_v1"
    fairface_handle: str = "mehmoodsheikh/fairface-dataset"
    fairface_dir: str = ""
    negative_count: int = 5000
    detector_root: str = "checkpoints/insightface"
    detector_name: str = "buffalo_l"
    detector_threshold: float = 0.5


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    backbone: str
    weights: str
    image_size: int
    embedding_dim: int
    checkpoint: str
    checkpoint_url: str = ""
    checkpoint_sha256: str = ""


@dataclass(frozen=True)
class SplitConfig:
    validation_fraction: float
    seed: int
    test_fraction: float = 0.15
    mode: str = "random_stratified"


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    head_epochs: int
    finetune_epochs: int
    head_lr: float
    finetune_lr: float
    weight_decay: float
    selection_metric: str
    device: str = "auto"
    num_workers: int = 2


@dataclass(frozen=True)
class GenerationConfig:
    steps: int
    lr: float
    tv_weight: float
    l2_weight: float
    seeds: tuple[int, ...]
    snapshot_every: int


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig
    split: SplitConfig
    training: TrainingConfig
    generation: GenerationConfig


T = TypeVar("T")


def _table(raw: dict[str, Any], name: str, cls: type[T]) -> T:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Отсутствует таблица [{name}]")
    try:
        return cls(**value)
    except TypeError as error:
        raise ValueError(f"Неверные поля [{name}]: {error}") from error


def _integer(name: str, value: Any, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: ожидается целое число >= {minimum}")


def _number(name: str, value: Any, *, positive: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name}: ожидается конечное число")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name}: ожидается {'положительное' if positive else 'неотрицательное'} число")


def load_config(path: str | Path) -> Config:
    """Проверить параметры; не проверяет наличие датасета или ускорителя.

    Пути внутри TOML интерпретируются будущими стадиями относительно корня
    проекта (рабочего каталога команды), а не каталога configs/.
    """
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    expected = {"data", "model", "split", "training", "generation"}
    if set(raw) != expected:
        raise ValueError(f"Ожидаются таблицы {sorted(expected)}, получены {sorted(raw)}")
    data = _table(raw, "data", DataConfig)
    model = _table(raw, "model", ModelConfig)
    split = _table(raw, "split", SplitConfig)
    training = _table(raw, "training", TrainingConfig)
    generation_raw = dict(raw["generation"])
    seeds = generation_raw.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("generation.seeds: ожидается непустой массив целых чисел")
    for seed in seeds:
        _integer("generation.seeds", seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("generation.seeds: начальные шумы должны иметь разные seed")
    generation_raw["seeds"] = tuple(seeds)
    generation = _table({"generation": generation_raw}, "generation", GenerationConfig)

    for field in ("raw_manifest", "prepared_manifest", "artifacts_dir"):
        value = getattr(data, field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.{field}: ожидается непустой путь")
    if data.variant != "face_crop":
        raise ValueError("data.variant: поддерживается только face_crop (вырезанное лицо)")
    _integer("data.phash_max_distance", data.phash_max_distance)
    if data.phash_max_distance > 64:
        raise ValueError("data.phash_max_distance: максимум 64 для 64-битного pHash")
    if model.architecture == "resnet18":
        expected_model = ("resnet18", "IMAGENET1K_V1", 224, 512)
        if model.checkpoint != "":
            raise ValueError("model.checkpoint: для ResNet18 используем torchvision weights, пустой путь")
    elif model.architecture == "arcface":
        expected_model = ("iresnet50", "MS1MV3", 112, 512)
        if not isinstance(model.checkpoint, str) or not model.checkpoint.strip():
            raise ValueError("model.checkpoint: для ArcFace нужен путь к локальному PyTorch checkpoint")
        if Path(model.checkpoint).suffix not in (".pt", ".pth"):
            raise ValueError("model.checkpoint: для ArcFace нужен .pt/.pth, а не ONNX")
    else:
        raise ValueError("model.architecture: предусмотрены только resnet18 и arcface")
    _integer("model.image_size", model.image_size, 1)
    _integer("model.embedding_dim", model.embedding_dim, 1)
    if (model.backbone, model.weights, model.image_size, model.embedding_dim) != expected_model:
        raise ValueError(f"Несовместимые параметры модели; ожидаются {expected_model}")
    _number("split.validation_fraction", split.validation_fraction, positive=True)
    if split.validation_fraction >= 1:
        raise ValueError("split.validation_fraction должна быть строго между 0 и 1")
    if split.mode not in ("random_stratified", "external"):
        raise ValueError("split.mode: random_stratified or external")
    _number("split.test_fraction", split.test_fraction, positive=True)
    if split.validation_fraction + split.test_fraction >= 1:
        raise ValueError("split: train fraction must be positive")
    _integer("data.negative_count", data.negative_count, 1)
    _number("data.detector_threshold", data.detector_threshold, positive=True)
    if data.detector_threshold > 1:
        raise ValueError("data.detector_threshold must be <= 1")
    if data.detector_name != "buffalo_l":
        raise ValueError("data.detector_name: supported SCRFD pack is buffalo_l")
    if model.checkpoint_sha256 and (len(model.checkpoint_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in model.checkpoint_sha256)):
        raise ValueError("model.checkpoint_sha256: expected lowercase SHA-256")
    if model.checkpoint_url and not model.checkpoint_sha256:
        raise ValueError("model.checkpoint_url requires checkpoint_sha256")
    if model.checkpoint_url and not model.checkpoint_url.startswith("https://"):
        raise ValueError("model.checkpoint_url requires HTTPS")
    if training.device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("training.device: auto/cpu/cuda/mps")
    _integer("training.num_workers", training.num_workers)
    _integer("split.seed", split.seed)
    for field in ("batch_size", "head_epochs"):
        _integer(f"training.{field}", getattr(training, field), 1)
    _integer("training.finetune_epochs", training.finetune_epochs)
    for field in ("head_lr", "finetune_lr"):
        _number(f"training.{field}", getattr(training, field), positive=True)
    _number("training.weight_decay", training.weight_decay)
    if training.selection_metric != "average_precision":
        raise ValueError("training.selection_metric: предусмотрен average_precision")
    _integer("generation.steps", generation.steps, 1)
    _integer("generation.snapshot_every", generation.snapshot_every, 1)
    if generation.snapshot_every > generation.steps:
        raise ValueError("generation.snapshot_every не должен превышать steps")
    _number("generation.lr", generation.lr, positive=True)
    _number("generation.tv_weight", generation.tv_weight)
    _number("generation.l2_weight", generation.l2_weight)
    return Config(data, model, split, training, generation)
