"""Small, dependency-free artifact and provenance helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def environment():
    packages = {}
    for name in ['torch', 'torchvision', 'numpy', 'scikit-learn', 'Pillow', 'ImageHash',
                 'matplotlib', 'insightface', 'onnxruntime', 'kagglehub', 'scipy',
                 'scikit-image', 'opencv-python-headless', 'Markdown', 'onnx']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    source_root = Path(__file__).resolve().parent
    source_digest = hashlib.sha256()
    for path in sorted(source_root.rglob('*.py')):
        source_digest.update(path.relative_to(source_root).as_posix().encode())
        source_digest.update(path.read_bytes())
    return {'source_sha256': source_digest.hexdigest(),
            'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'git_commit': git('rev-parse', 'HEAD'),
            'git_dirty': git('status', '--porcelain') != ''}


def download(url, destination, expected_sha256):
    """Atomic download with mandatory integrity check; never logs credentials."""
    import urllib.request
    destination = Path(destination)
    if destination.exists() and sha256(destination) == expected_sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + '.part')
    try:
        with urllib.request.urlopen(url, timeout=120) as src, tmp.open('wb') as dst:
            import shutil
            shutil.copyfileobj(src, dst)
        if sha256(tmp) != expected_sha256:
            raise ValueError('Downloaded file SHA-256 mismatch')
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)
    return destination
