"""Пороговая разметка по cosine similarity к центроиду Соколова (ArcFace buffalo_l)."""
from __future__ import annotations
import json

from . import MODES, THRESHOLDS, slug

KEYS = ('solo_min', 'pos', 'neg', 'flag', 'near_duplicate', 'v1_cos', 'v1_phash')


def load_thresholds(path=THRESHOLDS):
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f'{path} не найден: сначала reference + calibrate (docs/HARVEST.md)') from None
    values = data['thresholds']
    values.setdefault('v1_source_fraction', .25)
    if set(KEYS) - set(values):
        raise ValueError(f'{path}: нет порогов {sorted(set(KEYS) - set(values))}')
    if not values['neg'] < values['pos'] or not values['neg'] <= values['solo_min'] <= values['pos']:
        raise ValueError(f'{path}: ожидается neg < pos и neg <= solo_min <= pos')
    return values


def classify(mode, sim, t):
    """(label, review_status, note). label у ambiguous — предварительная гипотеза для аудита."""
    if mode not in MODES:
        raise ValueError(f'mode: {MODES}')
    if mode == 'solo':
        if sim >= t['solo_min']:
            return 1, 'accepted', ''
        return 1, 'ambiguous', f'solo: sim<{t["solo_min"]:g} (T_solo_min)'
    if mode == 'multi':
        if sim >= t['pos']:
            return 1, 'accepted', ''
        if sim <= t['neg']:
            return 0, 'accepted', 'bystander in Sokolov source'
        guess = int(sim >= (t['pos'] + t['neg']) / 2)
        return guess, 'ambiguous', f'multi: {t["neg"]:g}<sim<{t["pos"]:g}'
    if sim >= t['flag']:
        return 0, 'ambiguous', f'possible Sokolov: sim>={t["flag"]:g} (T_flag)'
    return 0, 'accepted', ''


def identity(label, source_id, person=None, primary=True):
    """(person, group_id): Соколов → source_id; известный человек → person:<slug>; прочие → source_id."""
    if int(label) == 1:
        return 'sokolov', source_id
    if person and primary:
        return slug(person, 'person'), f'person:{person}'
    return 'unknown', source_id
