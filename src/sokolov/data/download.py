"""Raw manifest v2: v1 positives + harvested web faces + local FairFace negatives.

Nothing is downloaded unless data.fairface_download=true and fairface_dir is missing.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps

from ..io import read_csv, read_json, sha256, write_csv, write_json

RAW_FIELDS = ['image_id', 'path', 'label', 'source_id', 'group_id', 'domain', 'dataset', 'person',
              'source_url', 'timestamp_seconds', 'partition', 'pixel_sha256', 'pinned_split']
HARVEST_FIELDS = ['image_id', 'path', 'label', 'source_id', 'group_id', 'domain', 'person', 'source_url',
                  'timestamp_seconds', 'det_score', 'face_px', 'sokolov_similarity', 'review_status', 'notes']
DOMAINS = ('lecture_webcam', 'lecture_room', 'talk_stage', 'interview', 'photo', 'fairface')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}


def image_info(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        image.load()
    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return image, digest


def _resolve(path, folder):
    """Harvest paths are relative to the repository root: folder/../../.. for data/raw/<name>, else cwd."""
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    candidates = [Path.cwd()/path]
    if len(folder.resolve().parents) > 2:
        candidates.insert(0, folder.resolve().parents[2]/path)
    return next((c.resolve() for c in candidates if c.exists()), candidates[0].resolve())


def harvest_rows(folder, dataset, pinned=False):
    """Accepted rows of a HARVEST CONTRACT manifest; (rows, exclusions, warnings).

    pinned: the manifest has an extra partition column (train/test) fixing whole groups.
    """
    folder = Path(folder)
    manifest = folder/'manifest.csv'
    if not manifest.is_file():
        return [], [], [f'{dataset}: {manifest} not found; dataset skipped']
    rows, exclusions, warnings, ids = [], [], [], set()
    table = read_csv(manifest)
    if table and list(table[0])[:len(HARVEST_FIELDS)] != HARVEST_FIELDS:
        raise ValueError(f'{manifest}: columns must start with {HARVEST_FIELDS}')
    for item in table:
        image_id = item['image_id']
        if image_id in ids:
            raise ValueError(f'{manifest}: duplicate image_id {image_id}')
        ids.add(image_id)
        if item['review_status'] != 'accepted':
            exclusions.append({'image_id': image_id, 'dataset': dataset, 'reason': 'review_status=' + item['review_status']})
            continue
        problem = None
        if item['label'] not in ('0', '1'):
            problem = 'invalid label'
        elif (item['label'] == '1') != (item['person'] == 'sokolov'):
            problem = 'label/person mismatch'
        elif item['domain'] not in DOMAINS or item['domain'] == 'fairface':
            problem = 'invalid domain'
        elif not item['source_id'] or not item['group_id'] or not image_id.isascii():
            problem = 'missing source_id/group_id or non-ASCII image_id'
        elif pinned and item.get('partition') not in ('train', 'test'):
            problem = 'pinned dataset needs partition=train/test'
        path = _resolve(item['path'], folder)
        if problem is None and not path.is_relative_to(folder.resolve()):
            problem = 'path outside dataset folder'
        elif problem is None and not path.is_file():
            problem = 'file missing'
        if problem:
            exclusions.append({'image_id': image_id, 'dataset': dataset, 'reason': problem})
            continue
        rows.append({'image_id': image_id, 'path': path, 'label': int(item['label']),
                     'source_id': item['source_id'], 'group_id': item['group_id'], 'domain': item['domain'],
                     'dataset': dataset, 'person': item['person'], 'source_url': item['source_url'],
                     'timestamp_seconds': item['timestamp_seconds'],
                     'pinned_split': item['partition'] if pinned else ''})
    if not rows:
        warnings.append(f'{dataset}: no accepted rows')
    return rows, exclusions, warnings


def _fairface_folder(data, override):
    local = override or data.fairface_dir
    folder = Path(local).resolve() if local else None
    if folder and folder.is_dir():
        return folder
    if not data.fairface_download:
        raise FileNotFoundError(f'FairFace directory not found: {local or "(empty)"}. Put FairFace images '
                                'there (see data/raw/fairface/SOURCE.md) or set data.fairface_download=true')
    import kagglehub  # optional fallback; large download, run only when requested
    print(f'FairFace: {local or "fairface_dir"} missing; downloading {data.fairface_handle} via kagglehub', flush=True)
    return Path(kagglehub.dataset_download(data.fairface_handle)).resolve()


def _request(config, fairface_dir):
    data = config.data
    def digest(folder):
        path = Path(folder)/'manifest.csv'
        return sha256(path) if path.is_file() else None
    return {'handle': data.fairface_handle, 'local_dir': str(fairface_dir or data.fairface_dir),
            'positive_dir': str(Path(data.positive_dir).resolve()), 'positive_manifest': digest(data.positive_dir),
            'web_positive_dir': str(Path(data.web_positive_dir).resolve()), 'web_positive_manifest': digest(data.web_positive_dir),
            'web_negative_dir': str(Path(data.web_negative_dir).resolve()), 'web_negative_manifest': digest(data.web_negative_dir),
            'user_dir': str(Path(data.user_dir).resolve()), 'user_manifest': digest(data.user_dir),
            'negative_count': data.negative_count, 'seed': config.split.seed, 'schema_version': 2}


def download_data(config, fairface_dir=None):
    """Build data/manifests/raw.csv (cached; changed inputs require a new raw path)."""
    data = config.data
    raw = Path(data.raw_manifest)
    selection = raw.parent / 'selection.local.json'
    request = _request(config, fairface_dir)
    if raw.exists():
        if config.split.mode == 'external':
            return raw
        if selection.exists() and read_json(selection)['request'] == request:
            if read_json(selection).get('raw_sha256') != sha256(raw):
                raise ValueError('Cached raw manifest was modified')
            return raw
        raise ValueError('Raw manifest exists with different inputs (harvest manifests, FairFace, seed); '
                         'delete it or set data.raw_manifest to a new path')
    rows, seen, exclusions, warnings = [], {}, [], []

    def add(path, label, image_id, source_id, group_id, domain, dataset, person, url='', timestamp='', pinned=''):
        try:
            _, digest = image_info(path)
        except (OSError, ValueError) as error:
            exclusions.append({'image_id': image_id, 'dataset': dataset, 'path': str(path),
                               'reason': f'decode: {type(error).__name__}'})
            return False
        if digest in seen:
            if seen[digest]['label'] != label:
                raise ValueError(f'Conflicting labels on identical pixels: {image_id} vs {seen[digest]["image_id"]}')
            exclusions.append({'image_id': image_id, 'dataset': dataset, 'path': str(path),
                               'reason': 'exact_duplicate', 'duplicate_of': seen[digest]['image_id']})
            return False
        row = {'image_id': image_id, 'path': str(Path(path).resolve()), 'label': label, 'source_id': source_id,
               'group_id': group_id, 'domain': domain, 'dataset': dataset, 'person': person,
               'source_url': url, 'timestamp_seconds': timestamp, 'partition': '', 'pixel_sha256': digest,
               'pinned_split': pinned}
        seen[digest] = row
        rows.append(row)
        return True

    positive = Path(data.positive_dir)
    for item in read_csv(positive / 'manifest.csv'):
        if item['label'] != '1':
            raise ValueError('Positive manifest must contain only label=1')
        path = (positive / item['file']).resolve()
        if not path.is_relative_to(positive.resolve()):
            raise ValueError('Positive path escapes dataset directory')
        add(path, 1, path.stem, item['source_video'], 'v1:' + item['source_video'], 'lecture_webcam',
            'v1', 'sokolov', '', item.get('timestamp_seconds', ''))
    for folder, dataset, pinned in ((data.web_positive_dir, 'web_positive_v2', False),
                                    (data.web_negative_dir, 'web_negative_v2', False), (data.user_dir, 'user_v3', True)):
        accepted, excluded, notes = harvest_rows(folder, dataset, pinned)
        exclusions += excluded
        warnings += notes
        for item in accepted:
            add(item.pop('path'), item.pop('label'), item.pop('image_id'), item.pop('source_id'),
                item.pop('group_id'), item.pop('domain'), item.pop('dataset'), item.pop('person'),
                item.pop('source_url'), item.pop('timestamp_seconds'), item.pop('pinned_split'))
    folder = _fairface_folder(data, fairface_dir)
    candidates = sorted(p for p in folder.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
    random.Random(config.split.seed).shuffle(candidates)
    accepted = 0
    for path in candidates:
        rel = path.relative_to(folder).as_posix()
        image_id = 'fairface_' + hashlib.sha256(rel.encode()).hexdigest()[:20]
        if add(path, 0, image_id, 'fairface:' + rel, 'fairface:' + rel, 'fairface', 'fairface', 'unknown'):
            accepted += 1
        if accepted == data.negative_count:
            break
    if accepted != data.negative_count:
        raise ValueError(f'Only {accepted} unique readable FairFace images in {folder}; need {data.negative_count}')
    if not any(r['label'] == 1 for r in rows):
        raise ValueError('No readable positive images')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('image_id collides across datasets')
    for warning in warnings:
        print('Warning:', warning, flush=True)
    write_csv(raw, rows, fields=RAW_FIELDS)
    counts = Counter((r['dataset'], r['domain'], r['label']) for r in rows)
    write_json(selection, {'request': request, 'raw_sha256': sha256(raw), 'resolved_dataset_path': str(folder),
        'resolved_version': folder.name if folder.parent.name == 'versions' else None,
        'selected_files': [r['path'] for r in rows if r['dataset'] == 'fairface'],
        'counts': {'positive': sum(r['label'] == 1 for r in rows), 'negative': sum(r['label'] == 0 for r in rows),
                   'fairface': accepted},
        'by_dataset_domain_label': [{'dataset': k[0], 'domain': k[1], 'label': k[2], 'images': v}
                                    for k, v in sorted(counts.items())],
        'groups': len({r['group_id'] for r in rows}), 'warnings': warnings,
        'exclusions': exclusions, 'license': 'FairFace CC BY 4.0',
        'source': 'https://github.com/joojs/fairface'})
    return raw
