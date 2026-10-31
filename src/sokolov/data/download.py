"""Raw manifest: v1 positives + imported manifests + FairFace negatives.

All additional data goes through one import path (import_manifest) that accepts two
formats: the simple extra CSV (image_id,path,label,source_id,source_url; data.extra_manifest,
docs/DOMAIN_REPAIR.md) and the HARVEST CONTRACT (docs/HARVEST.md; data.web_positive_dir,
data.web_negative_dir, data.user_dir with an extra partition column).
FairFace is read from data.fairface_dir; kagglehub is used only when fairface_dir is empty
(original behaviour) or when data.fairface_download=true.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps

from ..io import read_csv, read_json, sha256, write_csv, write_json

RAW_SCHEMA_VERSION = 3
RAW_FIELDS = ['image_id', 'path', 'label', 'source_id', 'group_id', 'domain', 'dataset', 'person',
              'source_url', 'timestamp_seconds', 'partition', 'pixel_sha256', 'pinned_split']
HARVEST_FIELDS = ['image_id', 'path', 'label', 'source_id', 'group_id', 'domain', 'person', 'source_url',
                  'timestamp_seconds', 'det_score', 'face_px', 'sokolov_similarity', 'review_status', 'notes']
EXTRA_FIELDS = ['image_id', 'path', 'label', 'source_id', 'source_url']
DOMAINS = ('lecture_webcam', 'lecture_room', 'talk_stage', 'interview', 'photo', 'fairface')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
EXTRA_DATASET = 'extra'


def image_info(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        image.load()
    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return image, digest


def legacy_fields(row):
    """dataset/domain of raw rows written before these columns existed (source_id prefix only)."""
    source = row.get('source_id', '')
    if source.startswith('fairface:'):
        return 'fairface', 'fairface'
    if source.startswith('web:'):
        return EXTRA_DATASET, 'photo'
    return '', ''


def _resolve(path, folder, prefer_cwd=False):
    """Relative paths: repository root (three levels above data/<kind>/<name>/manifest.csv) or cwd."""
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    candidates = [Path.cwd()/path]
    if len(folder.resolve().parents) > 2:
        candidates.insert(len(candidates) if prefer_cwd else 0, folder.resolve().parents[2]/path)
    return next((c.resolve() for c in candidates if c.exists()), candidates[0].resolve())


def import_manifest(manifest, dataset, *, folder=None, pinned=False):
    """Rows of an import manifest in either supported format; (rows, exclusions, warnings).

    HARVEST CONTRACT (columns start with HARVEST_FIELDS): only review_status=accepted rows,
    invalid rows become exclusions; folder (when given) must contain every image.
    Extra CSV (EXTRA_FIELDS, optional group_id/domain/person/timestamp_seconds): every row is
    a verified label, invalid label/source_id fails the import; group_id defaults to source_id.
    pinned: an extra partition column (train/test) fixes whole groups.
    """
    manifest = Path(manifest)
    if not manifest.is_file():
        return [], [], [f'{dataset}: {manifest} not found; dataset skipped']
    rows, exclusions, warnings, ids = [], [], [], set()
    table = read_csv(manifest)
    columns = list(table[0]) if table else []
    harvest = columns[:len(HARVEST_FIELDS)] == HARVEST_FIELDS
    if table and not harvest and not set(EXTRA_FIELDS) <= set(columns):
        raise ValueError(f'{manifest}: expected HARVEST CONTRACT columns {HARVEST_FIELDS} '
                         f'or extra manifest columns {EXTRA_FIELDS}')
    base = manifest.parent
    for item in table:
        image_id = item['image_id']
        if image_id in ids:
            raise ValueError(f'{manifest}: duplicate image_id {image_id}')
        ids.add(image_id)
        status = item.get('review_status') or 'accepted'
        if status != 'accepted':
            exclusions.append({'image_id': image_id, 'dataset': dataset, 'reason': 'review_status=' + status})
            continue
        if harvest:
            group, domain, person = item['group_id'], item['domain'], item['person']
        else:
            if item['label'] not in ('0', '1') or not item['source_id']:
                raise ValueError('Extra manifest requires verified label and source_id')
            group = item.get('group_id') or item['source_id']
            domain = item.get('domain') or ('photo' if item['source_id'].startswith('web:') else '')
            person = item.get('person') or ('sokolov' if item['label'] == '1' else 'unknown')
            if domain and (domain not in DOMAINS or domain == 'fairface'):
                raise ValueError(f'{manifest}: invalid domain {domain!r} for {image_id}')
        problem = None
        if item['label'] not in ('0', '1'):
            problem = 'invalid label'
        elif (item['label'] == '1') != (person == 'sokolov'):
            problem = 'label/person mismatch'
        elif harvest and (domain not in DOMAINS or domain == 'fairface'):
            problem = 'invalid domain'
        elif not item['source_id'] or not group or not image_id.isascii():
            problem = 'missing source_id/group_id or non-ASCII image_id'
        elif pinned and item.get('partition') not in ('train', 'test'):
            problem = 'pinned dataset needs partition=train/test'
        path = _resolve(item['path'], base, prefer_cwd=not harvest)
        if problem is None and folder is not None and not path.is_relative_to(Path(folder).resolve()):
            problem = 'path outside dataset folder'
        elif problem is None and not path.is_file():
            problem = 'file missing'
        if problem:
            exclusions.append({'image_id': image_id, 'dataset': dataset, 'reason': problem})
            continue
        rows.append({'image_id': image_id, 'path': path, 'label': int(item['label']),
                     'source_id': item['source_id'], 'group_id': group, 'domain': domain,
                     'dataset': dataset, 'person': person, 'source_url': item.get('source_url', ''),
                     'timestamp_seconds': item.get('timestamp_seconds', ''),
                     'pinned_split': item['partition'] if pinned else ''})
    if not rows:
        warnings.append(f'{dataset}: no accepted rows')
    return rows, exclusions, warnings


def harvest_rows(folder, dataset, pinned=False):
    """Accepted rows of a dataset folder with manifest.csv (HARVEST CONTRACT or extra CSV)."""
    folder = Path(folder)
    return import_manifest(folder/'manifest.csv', dataset, folder=folder, pinned=pinned)


def import_sources(data):
    """(manifest, dataset, folder, pinned) for every configured import, in priority order."""
    sources = []
    if data.extra_manifest:
        sources.append((Path(data.extra_manifest), EXTRA_DATASET, None, False))
    for folder, dataset, pinned in ((data.web_positive_dir, 'web_positive_v2', False),
                                    (data.web_negative_dir, 'web_negative_v2', False),
                                    (data.user_dir, 'user_v3', True)):
        if folder:
            sources.append((Path(folder)/'manifest.csv', dataset, Path(folder), pinned))
    return sources


def _fairface_folder(data, override):
    local = override or data.fairface_dir
    folder = Path(local).resolve() if local else None
    if folder and folder.is_dir():
        return folder
    if local and not data.fairface_download:
        raise FileNotFoundError(f'FairFace directory not found: {local}. Put FairFace images there '
                                '(see data/raw/fairface/SOURCE.md), pass --fairface-dir, or set '
                                'data.fairface_download=true (--download-fairface)')
    import kagglehub  # empty fairface_dir or explicit opt-in; large download
    print(f'FairFace: {local or "fairface_dir is empty"}; downloading {data.fairface_handle} via kagglehub', flush=True)
    return Path(kagglehub.dataset_download(data.fairface_handle)).resolve()


def _request(config, fairface_dir):
    data = config.data
    def digest(path):
        path = Path(path)
        return sha256(path) if path.is_file() else None
    request = {'handle': data.fairface_handle, 'local_dir': str(fairface_dir or data.fairface_dir),
               'positive_dir': str(Path(data.positive_dir).resolve()),
               'positive_manifest': digest(Path(data.positive_dir)/'manifest.csv'),
               'negative_count': data.negative_count, 'seed': config.split.seed,
               'schema_version': RAW_SCHEMA_VERSION}
    if data.extra_manifest:
        # An explicitly configured CSV must exist (harvest folders may be absent: warning only).
        if not Path(data.extra_manifest).is_file():
            raise FileNotFoundError(f'data.extra_manifest not found: {data.extra_manifest}. Build it with '
                                    'tools/build_domain_dataset.py or set extra_manifest = ""')
        request['extra_manifest_sha256'] = sha256(data.extra_manifest)
    for manifest, dataset, folder, _ in import_sources(data):
        if folder is not None:
            request[dataset] = {'dir': str(folder.resolve()), 'manifest_sha256': digest(manifest)}
    return request


def download_data(config, fairface_dir=None):
    """Build data.raw_manifest (cached; changed inputs require a new raw path)."""
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
        raise ValueError('Raw manifest exists with different inputs (import manifests, FairFace, seed, schema); '
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
    for manifest, dataset, folder, pinned in import_sources(data):
        accepted, excluded, notes = import_manifest(manifest, dataset, folder=folder, pinned=pinned)
        exclusions += excluded
        warnings += notes
        for item in accepted:
            add(item['path'], item['label'], item['image_id'], item['source_id'], item['group_id'],
                item['domain'], item['dataset'], item['person'], item['source_url'],
                item['timestamp_seconds'], item['pinned_split'])
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
