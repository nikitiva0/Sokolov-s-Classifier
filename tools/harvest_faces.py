"""Сбор лиц из публичных видео/фото по HARVEST CONTRACT. Документация: docs/HARVEST.md.

python tools/harvest_faces.py {reference,calibrate,video,images,sheet,dedup-v1,set-status,stats} ...
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'tools'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from harvest import DATASETS, DOMAINS, MODES, REPO, REVIEW  # noqa: E402


def parser():
    main = argparse.ArgumentParser(prog='harvest_faces.py', description=__doc__.splitlines()[0])
    main.add_argument('--root', default=str(REPO), help='корень данных (по умолчанию репозиторий; для тестов — временный)')
    sub = main.add_subparsers(dest='command', required=True)
    sub.add_parser('reference', help='эмбеддинги 1000 позитивов v1 → data/reference/')
    p = sub.add_parser('calibrate', help='пороги из reference + лиц других людей → tools/harvest/thresholds.json')
    p.add_argument('--negatives', nargs='+', required=True, help='manifest.csv с лицами заведомо не Соколова')
    p.add_argument('--margin', type=float, default=0.05)
    p.add_argument('--note', default='', help='кто эти люди и откуда (для basis)')

    def harvest_options(p):
        p.add_argument('--dataset', required=True, choices=sorted(DATASETS))
        p.add_argument('--source-id', required=True)
        p.add_argument('--mode', required=True, choices=MODES)
        p.add_argument('--person', help='slug известного другого человека (только --mode negative; '
                                        'присваивается самому крупному лицу кадра)')
        p.add_argument('--max-faces', type=int, default=0, help='не больше N крупнейших лиц на кадр (0 — все)')
        p.add_argument('--min-det', type=float, default=0.6)
        p.add_argument('--min-face', type=float, default=28, help='минимум max(w, h) bbox, px')
        p.add_argument('--verification', help='текст identity_verification для data/sources')
        p.add_argument('--jobs', type=int, default=3, help='параллельных загрузок')
        p.add_argument('--max-failures', type=int, default=8, help='останов после N неудач подряд')
        p.add_argument('--stall', type=float, default=180, help='останов, если ни один кадр не готов N s')
        p.add_argument('--max-seconds', type=float, default=0, help='общий бюджет времени, s (0 — нет)')

    p = sub.add_parser('video', help='кадры видео (yt-dlp + ffmpeg/HLS) → лица')
    p.add_argument('url')
    harvest_options(p)
    p.add_argument('--domain', required=True, choices=[d for d in DOMAINS if d not in ('photo', 'fairface')])
    when = p.add_mutually_exclusive_group(required=True)
    when.add_argument('--count', type=int)
    when.add_argument('--every', type=float, help='шаг между кадрами, s')
    p.add_argument('--start', type=float, default=0.0)
    p.add_argument('--end', type=float)
    p.add_argument('--max-height', type=int, default=720)
    p.add_argument('--max-kbps', type=float, default=2500, help='не брать форматы с битрейтом выше (kbit/s), если есть другие')
    p.add_argument('--fetch', choices=['auto', 'hls', 'seek', 'download'], default='auto',
                   help='auto: download, если файл ≤ --download-max-mib; иначе hls/seek')
    p.add_argument('--download-max-mib', type=int, default=48)
    p.add_argument('--exact', action='store_true', help='точный кадр t вместо ключевого ≤ t (дороже)')
    p.add_argument('--ytdlp-args', default='', help='доп. аргументы yt-dlp, напр. "--cookies FILE"')
    p.add_argument('--max-resolves', type=int, default=3, help='повторных resolve при сбоях узла CDN/истёкшей ссылке')

    p = sub.add_parser('images', help='фотографии (URL или каталог) → лица')
    p.add_argument('urls', nargs='*')
    p.add_argument('--dir', help='каталог с уже скачанными фото (нужен --source-url)')
    harvest_options(p)
    p.add_argument('--domain', default='photo', choices=DOMAINS)
    p.add_argument('--source-url', help='страница-источник')
    p.add_argument('--title')
    p.add_argument('--date', help='YYYY-MM-DD')

    p = sub.add_parser('sheet', help='контактные листы для аудита')
    p.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    p.add_argument('--source')
    p.add_argument('--sort', choices=['time', 'sim'], default='time')
    p.add_argument('--status', nargs='+', choices=REVIEW)

    p = sub.add_parser('dedup-v1', help='accepted-лица, почти совпадающие с v1 → ambiguous')
    p.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    p.add_argument('--dry-run', action='store_true')

    p = sub.add_parser('set-status', help='решение аудита')
    p.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--status', required=True, choices=REVIEW)
    p.add_argument('--note', required=True)
    p.add_argument('--label', type=int, choices=[0, 1])
    p.add_argument('--person', help='slug человека для label 0')

    p = sub.add_parser('stats', help='счётчики по источникам')
    p.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    return main


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'images' and bool(args.urls) == bool(args.dir):
        print('Ошибка: укажите URL изображений либо --dir', file=sys.stderr)
        return 2
    from harvest import commands
    handler = {'reference': commands.reference, 'calibrate': commands.calibrate, 'video': commands.video,
               'images': commands.images, 'sheet': commands.sheet, 'dedup-v1': commands.dedup_v1,
               'set-status': commands.set_status, 'stats': commands.stats}[args.command]
    try:
        handler(args)
    except KeyboardInterrupt:
        print('Прервано: состояние сохранено, повторный запуск продолжит с места остановки', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'Ошибка: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
