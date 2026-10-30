"""CLI: lazy ML imports keep check-config free of downloads/dependencies."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog='sokolov')
    commands = parser.add_subparsers(dest='command', required=True)
    names = ['check-config', 'download-data', 'prepare', 'train', 'evaluate', 'report',
             'run-all', 'publish-report', 'check-weights']
    for name in names:
        sub = commands.add_parser(name)
        if name not in ('report', 'publish-report'):
            sub.add_argument('--config', type=Path, default=Path('configs/resnet18_face.toml'))
            sub.add_argument('--fairface-dir', type=Path)
            sub.add_argument('--arcface-checkpoint', type=Path)
            sub.add_argument('--arcface-sha256')
        if name in ('run-all', 'check-weights'):
            sub.add_argument('--configs', type=Path, nargs='+', default=[
                Path('configs/resnet18_face.toml'), Path('configs/arcface_face.toml')])
        if name in ('train', 'evaluate', 'report', 'run-all', 'publish-report'):
            sub.add_argument('--run-dir', type=Path, required=True)
        if name in ('train', 'run-all'):
            sub.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'publish-report':
            from .report import publish_report
            print(publish_report(args.run_dir))
            return 0
        if args.command == 'report':
            from .report import make_report
            print(make_report(args.run_dir))
            return 0

        def resolve(path):
            config = load_config(path)
            if args.arcface_checkpoint and config.model.architecture == 'arcface':
                config = replace(config, model=replace(config.model,
                    checkpoint=str(args.arcface_checkpoint.resolve()), checkpoint_url=''))
            if args.arcface_sha256 and config.model.architecture == 'arcface':
                config = replace(config, model=replace(config.model, checkpoint_sha256=args.arcface_sha256))
            if args.fairface_dir:
                config = replace(config, data=replace(config.data, fairface_dir=str(args.fairface_dir.resolve())))
            return config

        config = resolve(args.config)
        if args.command == 'check-config':
            print(f'Конфигурация корректна: {args.config}')
            print(f'Модель: {config.model.architecture}; лица {config.model.image_size}×{config.model.image_size}')
            print(f'Split: {config.split.mode}; val={config.split.validation_fraction:g}, test={config.split.test_fraction:g}')
            print('Проверены только параметры. Загрузки и обучение не запускались.')
            return 0
        if args.command in ('run-all', 'check-weights'):
            configs = [resolve(path) for path in args.configs]
            if len(configs) != 2 or {c.model.architecture for c in configs} != {'resnet18', 'arcface'}:
                raise ValueError('Supply exactly one ResNet18 and one ArcFace configuration')
            if any(c.data != configs[0].data or c.split != configs[0].split for c in configs):
                raise ValueError('Both models must share identical data and split configurations')
            from .model import build_model
            # Check ArcFace first: fail before large downloads/preparation if unavailable.
            for c in sorted(configs, key=lambda c: c.model.architecture):
                model = build_model(c.model, pretrained=True)
                print(f'Pretrained weights verified: {c.model.architecture}')
                del model
            if args.command == 'check-weights':
                return 0
            from .pipeline import run_all
            print(run_all(configs, args.run_dir, resume=args.resume))
            return 0
        if args.command == 'download-data':
            from .data.download import download_data
            print(download_data(config))
        elif args.command == 'prepare':
            from .data.prepare import prepare
            print(prepare(config))
        elif args.command == 'train':
            from .train import train
            print(train(config, args.run_dir, resume=args.resume))
        elif args.command == 'evaluate':
            from .evaluate import evaluate_test
            print(evaluate_test(config, args.run_dir/config.model.architecture/'best.pt'))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f'Ошибка: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
