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
    fairface_dir: str = "data/raw/fairface"
    negative_count: int = 3000
    detector_root: str = "checkpoints/insightface"
    detector_name: str = "buffalo_l"
    detector_threshold: float = 0.5
    web_positive_dir: str = "data/raw/web_positive_v2"
    web_negative_dir: str = "data/raw/web_negative_v2"
    # Ручной набор: HARVEST CONTRACT + колонка partition (train/test), группы закреплены.
    user_dir: str = "data/raw/user_v3"
    # kagglehub — только явный запасной вариант при отсутствии fairface_dir.
    fairface_download: bool = False


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
    test_fraction: float = 0.20
    mode: str = "group_stratified"
    # Hamming distance (bits) of 64-bit pHash between aligned crops of different splits.
    dedup_max_distance: int = 8


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
    # ResNet18 finetune stage: number of last residual stages unfrozen (2 = layer3+layer4).
    finetune_blocks: int = 2
    # Domain-robust train-only augmentation (both classes); sizes in px.
    augment: bool = True
    aug_downscale_min: int = 24
    aug_downscale_max: int = 112
    aug_downscale_p: float = 0.5
    aug_jpeg_min: int = 20
    aug_jpeg_max: int = 90
    aug_jpeg_p: float = 0.5
    aug_blur_p: float = 0.2
    aug_crop_scale_min: float = 0.85
    aug_rotation_deg: float = 10.0
    aug_color_jitter: float = 0.2
    aug_grayscale_p: float = 0.05


@dataclass(frozen=True)
class GenerationVariant:
    """Один вариант ограничений генерации ([[generation.variants]]).

    pixel — оптимизируются RGB-пиксели; fourier — спектр с масштабом 1/f^decay_power
    и декоррелированным цветом, затем sigmoid. Сдвиг и sigma размытия заданы в px
    для входа 112 px и масштабируются пропорционально размеру входа модели.
    """
    name: str
    parametrization: str = "pixel"
    lr: float = 0.05
    tv_weight: float = 0.0
    l2_weight: float = 0.0
    translate_px: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    rotate_deg: float = 0.0
    blur_every: int = 0
    blur_sigma_px: float = 0.0
    decay_power: float = 1.0
    init_std: float = 0.01


@dataclass(frozen=True)
class GenerationConfig:
    steps: int
    seeds: tuple[int, ...]
    snapshot_every: int
    variants: tuple[GenerationVariant, ...]


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


def generation_config(raw: Any) -> GenerationConfig:
    """Разобрать и проверить [generation] (TOML или словарь из checkpoint)."""
    if not isinstance(raw, dict):
        raise ValueError("Отсутствует таблица [generation]")
    raw = dict(raw)
    seeds = raw.get("seeds")
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("generation.seeds: ожидается непустой массив целых чисел")
    for seed in seeds:
        _integer("generation.seeds", seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("generation.seeds: начальные шумы должны иметь разные seed")
    raw["seeds"] = tuple(seeds)
    variants = raw.get("variants")
    if not isinstance(variants, (list, tuple)) or not variants:
        raise ValueError("generation.variants: нужен хотя бы один [[generation.variants]]")
    if not all(isinstance(v, dict) for v in variants):
        raise ValueError("generation.variants: ожидаются таблицы [[generation.variants]]")
    raw["variants"] = tuple(_table({"generation.variants": v}, "generation.variants", GenerationVariant)
                            for v in variants)
    generation = _table({"generation": raw}, "generation", GenerationConfig)
    _integer("generation.steps", generation.steps, 1)
    _integer("generation.snapshot_every", generation.snapshot_every, 1)
    if generation.snapshot_every > generation.steps:
        raise ValueError("generation.snapshot_every не должен превышать steps")
    names = [v.name for v in generation.variants]
    if len(set(names)) != len(names):
        raise ValueError("generation.variants: имена вариантов должны быть уникальными")
    for v in generation.variants:
        # Имя становится именем папки: только [a-z0-9_], без префикса seed_.
        if (not isinstance(v.name, str) or not 1 <= len(v.name) <= 32 or not v.name[0].isascii()
                or not v.name[0].islower() or v.name.startswith("seed_")
                or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in v.name)):
            raise ValueError(f"generation.variants.name: ожидается [a-z][a-z0-9_]* до 32 символов, получено {v.name!r}")
        prefix = f"generation.variants.{v.name}"
        if v.parametrization not in ("pixel", "fourier"):
            raise ValueError(f"{prefix}.parametrization: pixel или fourier")
        for field in ("lr", "init_std", "scale_min", "scale_max"):
            _number(f"{prefix}.{field}", getattr(v, field), positive=True)
        for field in ("tv_weight", "l2_weight", "translate_px", "rotate_deg", "blur_sigma_px", "decay_power"):
            _number(f"{prefix}.{field}", getattr(v, field))
        _integer(f"{prefix}.blur_every", v.blur_every)
        if v.scale_min > v.scale_max or v.scale_max > 2:
            raise ValueError(f"{prefix}: ожидается 0 < scale_min <= scale_max <= 2")
        if v.translate_px >= 56:
            raise ValueError(f"{prefix}.translate_px: меньше половины входа 112 px")
        if v.rotate_deg > 180:
            raise ValueError(f"{prefix}.rotate_deg: не больше 180°")
        if (v.blur_every > 0) != (v.blur_sigma_px > 0):
            raise ValueError(f"{prefix}: blur_every и blur_sigma_px задаются только вместе")
        if v.blur_sigma_px > 8:
            raise ValueError(f"{prefix}.blur_sigma_px: не больше 8 px")
        if v.parametrization == "fourier" and v.blur_every:
            raise ValueError(f"{prefix}: периодическое размытие предусмотрено только для pixel")
    return generation


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
    generation = generation_config(raw["generation"])

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
        if model.weights not in ("MS1MV3", "WebFace600K (InsightFace buffalo_l w600k_r50, converted)"):
            raise ValueError("model.weights: MS1MV3 или конвертированный WebFace600K (buffalo_l)")
        expected_model = ("iresnet50", model.weights, 112, 512)
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
    if split.mode not in ("group_stratified", "random_stratified", "external"):
        raise ValueError("split.mode: group_stratified, random_stratified or external")
    _number("split.test_fraction", split.test_fraction, positive=True)
    if split.validation_fraction + split.test_fraction >= 1:
        raise ValueError("split: train fraction must be positive")
    _integer("split.dedup_max_distance", split.dedup_max_distance)
    if split.dedup_max_distance > 64:
        raise ValueError("split.dedup_max_distance: максимум 64 для 64-битного pHash")
    _integer("data.negative_count", data.negative_count, 1)
    for field in ("positive_dir", "web_positive_dir", "web_negative_dir", "user_dir", "fairface_dir"):
        if not isinstance(getattr(data, field), str):
            raise ValueError(f"data.{field}: ожидается строка-путь")
    if type(data.fairface_download) is not bool:
        raise ValueError("data.fairface_download: ожидается true/false")
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
    _integer("training.finetune_blocks", training.finetune_blocks, 1)
    if training.finetune_blocks > 4:
        raise ValueError("training.finetune_blocks: у ResNet18 четыре блока layer1..layer4")
    if type(training.augment) is not bool:
        raise ValueError("training.augment: ожидается true/false")
    _integer("training.aug_downscale_min", training.aug_downscale_min, 8)
    _integer("training.aug_downscale_max", training.aug_downscale_max, training.aug_downscale_min)
    _integer("training.aug_jpeg_min", training.aug_jpeg_min, 1)
    _integer("training.aug_jpeg_max", training.aug_jpeg_max, training.aug_jpeg_min)
    if training.aug_jpeg_max > 100:
        raise ValueError("training.aug_jpeg_max: качество JPEG не больше 100")
    for field in ("aug_downscale_p", "aug_jpeg_p", "aug_blur_p", "aug_grayscale_p", "aug_color_jitter"):
        _number(f"training.{field}", getattr(training, field))
        if getattr(training, field) > 1:
            raise ValueError(f"training.{field}: ожидается значение в [0, 1]")
    _number("training.aug_crop_scale_min", training.aug_crop_scale_min, positive=True)
    if training.aug_crop_scale_min > 1:
        raise ValueError("training.aug_crop_scale_min: ожидается (0, 1]")
    _number("training.aug_rotation_deg", training.aug_rotation_deg)
    if training.aug_rotation_deg > 180:
        raise ValueError("training.aug_rotation_deg: не больше 180°")
    return Config(data, model, split, training, generation)
