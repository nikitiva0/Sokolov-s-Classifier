"""manifest.csv и data/sources/<dataset>.json: идемпотентное дополнение под файловой блокировкой."""
from __future__ import annotations
import csv
import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

from . import COLUMNS, DOMAINS, REVIEW

SOURCE_FIELDS = ['source_id', 'url', 'platform', 'title', 'date', 'duration_s', 'domain', 'persons',
                 'identity_verification', 'frames_sampled', 'faces_accepted_label1',
                 'faces_accepted_label0', 'bytes_downloaded']


@contextmanager
def locked(path):
    """Эксклюзивная flock-блокировка. Не вкладывать: повторный flock того же файла в процессе зависнет."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_rows(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f'{path}: колонки {reader.fieldnames} не совпадают с контрактом')
        return list(reader)


def natural_key(row):
    """Одно и то же лицо одного кадра даёт тот же ключ при повторном запуске (детекция детерминирована)."""
    return (row['source_id'], row['timestamp_seconds'], row['source_url'], row['det_score'], row['face_px'])


def normalize(row):
    row = {c: '' if row.get(c) is None else str(row[c]) for c in COLUMNS}
    if not row['image_id'].isascii() or not row['image_id'] or ',' in row['image_id']:
        raise ValueError(f'image_id: {row["image_id"]!r}')
    if row['label'] not in ('0', '1'):
        raise ValueError(f'{row["image_id"]}: label {row["label"]!r}')
    if row['review_status'] not in REVIEW:
        raise ValueError(f'{row["image_id"]}: review_status {row["review_status"]!r}')
    if row['domain'] not in DOMAINS:
        raise ValueError(f'{row["image_id"]}: domain {row["domain"]!r}')
    if not row['source_id'] or not row['group_id'] or not row['person']:
        raise ValueError(f'{row["image_id"]}: пустые source_id/group_id/person')
    return row


def append_rows(manifest, lock, rows):
    """Добавить только новые строки (по image_id и natural_key). Возвращает добавленные."""
    rows = [normalize(r) for r in rows]
    with locked(lock):
        existing = read_rows(manifest)
        ids = {r['image_id']: natural_key(r) for r in existing}
        keys = set(ids.values())
        new = []
        for row in rows:
            key = natural_key(row)
            if row['image_id'] in ids:
                if ids[row['image_id']] != key:
                    raise ValueError(f'{row["image_id"]}: image_id уже занят другим лицом')
                continue
            if key in keys:
                continue
            ids[row['image_id']] = key
            keys.add(key)
            new.append(row)
        if new:
            fresh = not manifest.exists() or manifest.stat().st_size == 0
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with manifest.open('a', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, COLUMNS, lineterminator='\n')
                if fresh:
                    writer.writeheader()
                writer.writerows(new)
                stream.flush()
                os.fsync(stream.fileno())
    return new


def write_rows(manifest, rows):
    tmp = manifest.with_suffix('.csv.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(normalize(r) for r in rows)
    tmp.replace(manifest)


def update_rows(manifest, lock, change):
    """change(rows) правит строки на месте и возвращает число изменённых; запись атомарная."""
    with locked(lock):
        rows = read_rows(manifest)
        changed = change(rows)
        if changed:
            write_rows(manifest, rows)
    return changed


def next_index(rows, image_prefix):
    numbers = [int(r['image_id'][len(image_prefix):]) for r in rows
               if r['image_id'].startswith(image_prefix) and r['image_id'][len(image_prefix):].isdigit()]
    return max(numbers, default=0) + 1


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def refresh_sources(layout, updates=None):
    """Пересчитать счётчики accepted по manifest; updates: {source_id: {поле: значение}};
    числа frames_sampled/bytes_downloaded в updates прибавляются (накопительно)."""
    with locked(layout.lock):
        rows = read_rows(layout.manifest)
        sources = json.loads(layout.sources.read_text()) if layout.sources.exists() else []
        by_id = {s['source_id']: s for s in sources}
        for source_id, fields in (updates or {}).items():
            entry = by_id.get(source_id)
            if entry is None:
                entry = by_id[source_id] = {f: '' for f in SOURCE_FIELDS}
                entry.update(source_id=source_id, persons=[], frames_sampled=0, bytes_downloaded=0,
                             identity_verification='pending: visual audit against data/raw/sokolov_positive_v1')
                sources.append(entry)
            for key, value in fields.items():
                if key in ('frames_sampled', 'bytes_downloaded'):
                    entry[key] = int(entry.get(key) or 0) + int(value)
                elif key == 'persons':
                    entry[key] = sorted(set(entry.get(key) or []) | set(value))
                elif value not in (None, ''):
                    entry[key] = value
        for entry in sources:
            mine = [r for r in rows if r['source_id'] == entry['source_id'] and r['review_status'] == 'accepted']
            entry['faces_accepted_label1'] = sum(r['label'] == '1' for r in mine)
            entry['faces_accepted_label0'] = sum(r['label'] == '0' for r in mine)
            entry['persons'] = sorted(set(entry.get('persons') or []) | {r['person'] for r in mine})
        sources = [{f: s.get(f, '') for f in SOURCE_FIELDS} | {k: v for k, v in s.items() if k not in SOURCE_FIELDS}
                   for s in sorted(sources, key=lambda s: s['source_id'])]
        write_json(layout.sources, sources)
    return sources
