"""Restore the extra photo manifest from documented sources; never start training."""
from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sokolov.data.download import image_info
from sokolov.io import read_json, sha256, write_csv


def restore(root, catalogue, *, user_photo_dir=None, skip_user_photos=False, fetch=None):
    """Use checked local copies first. Changed remote image content fails closed."""
    from PIL import Image, ImageOps
    rows = []
    for item in catalogue['items']:
        is_user = 'user_original' in item
        if is_user and skip_user_photos:
            continue
        destination = root/item['path']
        if not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError('Photo destination escapes project')
        if destination.exists():
            if image_info(destination)[1] != item['pixel_sha256']:
                raise ValueError(f"Cached image changed: {item['image_id']}")
        else:
            if is_user:
                if user_photo_dir is None:
                    raise FileNotFoundError('Provide --user-photo-dir with the five original photos, '
                                            'or use --skip-user-photos explicitly')
                original = Path(user_photo_dir)/item['user_original']
                if sha256(original) != item['original_sha256']:
                    raise ValueError('User photo differs from the annotated original')
                image, _ = image_info(original)
                x1, y1, x2, y2 = item['bbox']
                mx, my = (x2-x1)*item['margin_fraction'], (y2-y1)*item['margin_fraction']
                box = (max(0, int(x1-mx)), max(0, int(y1-my)),
                       min(image.width, int(x2+mx)), min(image.height, int(y2+my)))
                image = image.crop(box)
                image.thumbnail((item['max_side'],)*2, Image.Resampling.LANCZOS)
            else:
                if fetch is None:
                    request = Request(item['download_url'], headers={'User-Agent': 'Sokolov-research/1.0'})
                    with urlopen(request, timeout=30) as response:
                        content = response.read(25*1024*1024+1)
                else:
                    content = fetch(item['download_url'])
                if len(content) > 25*1024*1024:
                    raise ValueError('Remote image exceeds 25 MB')
                with Image.open(BytesIO(content)) as source:
                    image = ImageOps.exif_transpose(source).convert('RGB')
                    image.load()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.tmp.png')
            image.save(temporary)
            if image_info(temporary)[1] != item['pixel_sha256']:
                temporary.unlink()
                raise ValueError(f"Source pixels changed: {item['image_id']}; review provenance before relabelling")
            temporary.replace(destination)
        rows.append({key: item[key] for key in ('image_id', 'path', 'label', 'source_id', 'source_url')})
    write_csv(root/'data/manifests/domain_v3/web.csv', rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user-photo-dir', type=Path)
    parser.add_argument('--skip-user-photos', action='store_true')
    args = parser.parse_args()
    rows = restore(ROOT, read_json(ROOT/'data/sources/domain_v3_sources.json'),
                   user_photo_dir=args.user_photo_dir, skip_user_photos=args.skip_user_photos)
    print(f'Extra photos ready: {len(rows)}; labels: {dict(Counter(r["label"] for r in rows))}')
    print('Manifest: data/manifests/domain_v3/web.csv. Training was not started.')


if __name__ == '__main__':
    main()
