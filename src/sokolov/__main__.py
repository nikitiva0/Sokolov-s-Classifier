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
             'run-all', 'publish-report', 'check-weights', 'select-best', 'synthesize', 'export-mobile',
             'build-manifest', 'predict']
    for name in names:
        sub = commands.add_parser(name)
        if name not in ('report', 'publish-report', 'select-best', 'synthesize', 'export-mobile', 'predict'):
            sub.add_argument('--config', type=Path, default=Path('configs/resnet18_face.toml'))
            sub.add_argument('--fairface-dir', type=Path)
            sub.add_argument('--arcface-checkpoint', type=Path)
            sub.add_argument('--arcface-sha256')
        if name in ('download-data', 'build-manifest', 'run-all'):
            # Explicit opt-in: kagglehub only when fairface_dir is missing.
            sub.add_argument('--download-fairface', action='store_true')
        if name == 'predict':
            sub.add_argument('--model', choices=['resnet18', 'arcface', 'best'], default='best')
            sub.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'])
            sub.add_argument('images', type=Path, nargs='+')
        if name in ('run-all', 'check-weights'):
            sub.add_argument('--configs', type=Path, nargs='+', default=[
                Path('configs/resnet18_face.toml'), Path('configs/arcface_face.toml')])
        if name in ('train', 'evaluate', 'report', 'run-all', 'publish-report', 'select-best', 'synthesize',
                    'export-mobile', 'predict'):
            sub.add_argument('--run-dir', type=Path, required=True)
        if name == 'export-mobile':
            sub.add_argument('--output', type=Path, required=True)
            sub.add_argument('--detector', type=Path)
            sub.add_argument('--allow-technical', action='store_true')
        if name == 'synthesize':
            sub.add_argument('--generation-config', type=Path)
            sub.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'])
            sub.add_argument('--output-dir', type=Path)
            sub.add_argument('--model', choices=['best', 'resnet18', 'arcface'], default='best',
                             help='best: победитель по validation AP; иначе best.pt этой модели')
            sub.add_argument('--variants', nargs='+', metavar='NAME', help='подмножество [[generation.variants]]')
            sub.add_argument('--seeds', type=int, nargs='+', metavar='SEED')
            sub.add_argument('--steps', type=int)
        if name in ('train', 'run-all'):
            sub.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'predict':
            from .evaluate import format_prediction, predict_images
            missing = [str(path) for path in args.images if not path.is_file()]
            if missing:
                raise FileNotFoundError(f'Images not found: {missing}')
            for result in predict_images(args.run_dir, args.images, which=args.model, device=args.device):
                print(format_prediction(result))
            return 0
        if args.command == 'export-mobile':
            from .mobile import export_mobile
            print(export_mobile(args.run_dir, args.output, detector=args.detector, allow_technical=args.allow_technical))
            return 0
        if args.command == 'select-best':
            from .synthesize import select_best_model
            print(select_best_model(args.run_dir))
            return 0
        if args.command == 'synthesize':
            from .synthesize import generate_best
            generation = load_config(args.generation_config).generation if args.generation_config else None
            runs = generate_best(args.run_dir, generation=generation, device=args.device, output_dir=args.output_dir,
                                 model=args.model, variants=args.variants, seeds=args.seeds, steps=args.steps)
            print(runs[0].trajectory.parents[2]/'summary.json')
            return 0
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
            if getattr(args, 'download_fairface', False):
                config = replace(config, data=replace(config.data, fairface_download=True))
            return config

        config = resolve(args.config)
        if args.command == 'check-config':
            print(f'Конфигурация корректна: {args.config}')
            print(f'Модель: {config.model.architecture}; лица {config.model.image_size}×{config.model.image_size}')
            print(f'Split: {config.split.mode}; val={config.split.validation_fraction:g}, test={config.split.test_fraction:g}; '
                  f'pHash-дубликаты между выборками: <= {config.split.dedup_max_distance} бит')
            print(f'Аугментации train: {"включены" if config.training.augment else "только flip/jitter"}')
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
        if args.command in ('download-data', 'build-manifest'):
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
