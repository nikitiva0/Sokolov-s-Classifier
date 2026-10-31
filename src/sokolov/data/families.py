"""Acquisition family of a row: one definition for sampling, splits, metrics and audits.

lecture — v1 webcam lecture frames; web — internet/harvested/user photos;
fairface — FairFace. Rows without dataset/domain columns (raw manifests from
before the merge) fall back to the source_id prefix convention: web:/fairface:.
"""
from __future__ import annotations

WEB_DATASETS = ('web_positive_v2', 'web_negative_v2', 'user_v3')
FAMILIES = ('lecture', 'web', 'fairface')


def _get(item, key):
    value = item.get(key, '') if isinstance(item, dict) else getattr(item, key, '')
    return value or ''


def source_family(item):
    """'lecture', 'web' or 'fairface' for a raw/prepared row (dict) or a RawSample."""
    source, dataset = _get(item, 'source_id'), _get(item, 'dataset')
    if dataset == 'fairface' or _get(item, 'domain') == 'fairface' or source.startswith('fairface:'):
        return 'fairface'
    if source.startswith('web:') or dataset in WEB_DATASETS:
        return 'web'
    return 'lecture'
