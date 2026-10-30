"""CLI каркаса: доступна только проверка конфигурации."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sokolov")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-config", help="Проверить TOML без запуска ML")
    check.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError, TypeError) as error:
        print(f"Ошибка конфигурации: {error}", file=sys.stderr)
        return 2
    print(f"Конфигурация корректна: {args.config}")
    print(
        f"Модель: {config.model.architecture} / {config.model.backbone}, {config.model.weights}, "
        f"{config.model.image_size}×{config.model.image_size}"
    )
    print(
        f"Датасет: {config.data.variant}; внешние train/test сохраняются; "
        f"validation из train: {config.split.validation_fraction:g}; seed={config.split.seed}"
    )
    print(f"Синтез: {config.generation.steps} шагов, seeds={list(config.generation.seeds)}")
    print("Проверена структура параметров. Данные, обучение и синтез не запускались.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
