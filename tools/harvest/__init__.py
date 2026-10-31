"""Сбор лиц для web_positive_v2 / web_negative_v2 по HARVEST CONTRACT (docs/HARVEST.md)."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASETS = {'web_positive_v2': 'wp2', 'web_negative_v2': 'wn2'}
DOMAINS = ('lecture_webcam', 'lecture_room', 'talk_stage', 'interview', 'photo', 'fairface')
REVIEW = ('accepted', 'excluded', 'ambiguous')
MODES = ('solo', 'multi', 'negative')
COLUMNS = ['image_id', 'path', 'label', 'source_id', 'group_id', 'domain', 'person', 'source_url',
           'timestamp_seconds', 'det_score', 'face_px', 'sokolov_similarity', 'review_status', 'notes']
REFERENCE = REPO / 'data/reference/sokolov_reference.npz'
REFERENCE_STATS = REPO / 'data/reference/sokolov_reference_stats.json'
THRESHOLDS = Path(__file__).with_name('thresholds.json')
SLUG = re.compile(r'[a-z0-9][a-z0-9_-]{0,63}')


def slug(value, what='slug'):
    if not isinstance(value, str) or not SLUG.fullmatch(value):
        raise ValueError(f'{what}: ожидается ASCII-slug [a-z0-9_-], до 64 символов: {value!r}')
    return value


@dataclass(frozen=True)
class Layout:
    """Пути одного датасета; root по умолчанию — корень репозитория (для тестов — временный)."""
    dataset: str
    root: Path = REPO

    @property
    def dir(self): return self.root / 'data/raw' / self.dataset
    @property
    def manifest(self): return self.dir / 'manifest.csv'
    @property
    def lock(self): return self.dir / '.manifest.lock'
    @property
    def sources(self): return self.root / 'data/sources' / f'{self.dataset}.json'
    @property
    def prefix(self): return DATASETS.get(self.dataset, 'tst')

    def faces(self, source_id): return self.dir / 'faces' / source_id
    def state(self, source_id): return self.dir / 'state' / f'{source_id}.json'
    def embeddings(self, source_id): return self.dir / 'embeddings' / f'{source_id}.npz'
    def sheet(self, name): return self.dir / 'sheets' / f'{name}.jpg'
    def rel(self, path): return Path(path).resolve().relative_to(self.root.resolve()).as_posix()
