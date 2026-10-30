"""Deterministic, cached acquisition of positive frames and FairFace negatives."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from PIL import Image, ImageOps

from ..io import read_csv, read_json, sha256, write_csv, write_json


def image_info(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        image.load()
    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return image, digest


def download_data(config, fairface_dir=None):
    data = config.data
    raw = Path(data.raw_manifest)
    selection = raw.parent / 'selection.local.json'
    request = {'handle': data.fairface_handle, 'local_dir': str(fairface_dir or data.fairface_dir),
               'positive_dir': str(Path(data.positive_dir).resolve()),
               'negative_count': data.negative_count, 'seed': config.split.seed}
    if raw.exists():
        if config.split.mode == 'external':
            return raw
        if selection.exists() and read_json(selection)['request'] == request:
            if read_json(selection).get('raw_sha256') != sha256(raw):
                raise ValueError('Cached raw manifest was modified')
            return raw
        raise ValueError('Raw manifest already exists with different/missing selection metadata; use a new path')
    local = fairface_dir or data.fairface_dir
    if local:
        folder = Path(local).resolve()
    else:
        import kagglehub
        folder = Path(kagglehub.dataset_download(data.fairface_handle)).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f'FairFace directory not found: {folder}')
    positive = Path(data.positive_dir)
    provenance = read_csv(positive / 'manifest.csv')
    rows, seen, exclusions = [], {}, []

    def add(path, label, image_id, source_id):
        try:
            _, digest = image_info(path)
        except (OSError, ValueError) as error:
            exclusions.append({'path': str(path), 'reason': f'decode: {type(error).__name__}'})
            return False
        if digest in seen:
            if seen[digest]['label'] != label:
                raise ValueError(f'Conflicting labels on identical pixels: {image_id}')
            exclusions.append({'path': str(path), 'reason': 'exact_duplicate',
                               'duplicate_of': seen[digest]['image_id']})
            return False
        row = {'image_id': image_id, 'path': str(path.resolve()), 'label': label,
               'source_id': source_id, 'source_url': '', 'partition': '', 'pixel_sha256': digest}
        seen[digest] = row
        rows.append(row)
        return True

    for item in provenance:
        if item['label'] != '1':
            raise ValueError('Positive manifest must contain only label=1')
        path = (positive / item['file']).resolve()
        if not path.is_relative_to(positive.resolve()):
            raise ValueError('Positive path escapes dataset directory')
        add(path, 1, path.stem, item['source_video'])
    candidates = sorted(p for p in folder.rglob('*')
                        if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'} and p.is_file())
    random.Random(config.split.seed).shuffle(candidates)
    accepted = 0
    for path in candidates:
        rel = path.relative_to(folder).as_posix()
        image_id = 'fairface_' + hashlib.sha256(rel.encode()).hexdigest()[:20]
        if add(path, 0, image_id, 'fairface:' + rel):
            accepted += 1
        if accepted == data.negative_count:
            break
    if accepted != data.negative_count:
        raise ValueError(f'Only {accepted} unique readable FairFace images; need {data.negative_count}')
    if not any(r['label'] == 1 for r in rows):
        raise ValueError('No readable positive images')
    write_csv(raw, rows)
    write_json(selection, {'request': request, 'raw_sha256': sha256(raw), 'resolved_dataset_path': str(folder),
        'resolved_version': folder.name if folder.parent.name == 'versions' else None,
        'selected_files': [r['path'] for r in rows if r['label'] == 0],
        'counts': {'positive': len(rows) - accepted, 'negative': accepted},
        'exclusions': exclusions, 'license': 'FairFace CC BY 4.0',
        'source': 'https://github.com/joojs/fairface'})
    return raw
