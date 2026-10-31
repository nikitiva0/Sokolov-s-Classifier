"""One persisted split, before detection; explicit external test stays untouched.

group_stratified (default): whole groups (lecture/video, known person, FairFace
file) go to one split; groups are stratified by (group label, domain).
"""
from __future__ import annotations
import math
import random
from collections import Counter, defaultdict

from ..contracts import ParentAssignment

SPLITS = ('train', 'val', 'test')
WEB_POSITIVE = 'web_positive_v2'


def _round(value):
    return int(math.floor(value + .5))


def group_table(samples, group_ids):
    """Group label = 1 if it contains Sokolov; domain = most frequent (ties: name order).

    pinned: train/test fixed by a hand-sorted dataset; a group may not mix pins.
    """
    labels, domains, datasets, images = defaultdict(set), defaultdict(Counter), defaultdict(set), Counter()
    pins = defaultdict(set)
    for s in samples:
        g = group_ids[s.image_id]
        labels[g].add(s.label); domains[g][s.domain] += 1; datasets[g].add(s.dataset); images[g] += 1
        pins[g].add(s.pinned_split)
    if any(len(p) > 1 for p in pins.values()):
        raise ValueError('Group mixes pinned and unpinned/other pinned rows: '
                         + ', '.join(sorted(g for g, p in pins.items() if len(p) > 1)))
    return {g: {'label': int(1 in labels[g]),
                'domain': min(domains[g].items(), key=lambda item: (-item[1], item[0]))[0],
                'datasets': sorted(datasets[g]), 'images': images[g], 'pinned': next(iter(pins[g]))}
            for g in sorted(labels)}


def _move(assign, info, rank, predicate, target, sources=('train', 'val'), kind=None):
    """Move one matching group to target without emptying a class in its donor split.

    Prefers donors that keep another group of the same kind, then train, then seeded rank.
    """
    kind = kind or (lambda g: info[g]['label'])
    classes = Counter((s, info[g]['label']) for g, s in assign.items())
    kinds = Counter((s, kind(g)) for g, s in assign.items())
    options = []
    for order, source in enumerate(sources):
        for g, split in assign.items():
            if split != source or info[g]['pinned'] or not predicate(g) or classes[(source, info[g]['label'])] < 2:
                continue
            options.append((kinds[(source, kind(g))] < 2, order, rank[g], g))
    if not options:
        return None
    group = min(options)[3]
    assign[group] = target
    return group


def _group_stratified(info, config):
    order = sorted(info)
    random.Random(config.seed).shuffle(order)
    rank = {g: i for i, g in enumerate(order)}
    strata = defaultdict(list)
    for g in order:
        if not info[g]['pinned']:
            strata[(info[g]['label'], info[g]['domain'])].append(g)
    assign = {g: info[g]['pinned'] for g in order if info[g]['pinned']}  # never moved below
    for key in sorted(strata):
        members = strata[key]
        n = len(members)
        n_test = min(_round(n*config.test_fraction), max(n-1, 0))
        n_val = min(_round(n*config.validation_fraction), max(n-1-n_test, 0))
        for i, g in enumerate(members):
            assign[g] = 'test' if i < n_test else 'val' if i < n_test+n_val else 'train'
    in_split = lambda split, predicate: [g for g, s in assign.items() if s == split and predicate(g)]
    # Test: at least one v1 lecture (if a second one remains for training).
    v1 = lambda g: 'v1' in info[g]['datasets'] and info[g]['label'] == 1
    if sum(map(v1, info)) >= 2 and not in_split('test', v1):
        _move(assign, info, rank, v1, 'test', kind=lambda g: (info[g]['label'], v1(g)))
    # Test: web-positive sources from two different domains when available.
    web = lambda g: WEB_POSITIVE in info[g]['datasets'] and info[g]['label'] == 1
    web_domains = {info[g]['domain'] for g in info if web(g)}
    needed = min(2, len(web_domains)) if sum(map(web, info)) >= 2 else 0
    while len({info[g]['domain'] for g in in_split('test', web)}) < needed:
        present = {info[g]['domain'] for g in in_split('test', web)}
        if not _move(assign, info, rank, lambda g: web(g) and info[g]['domain'] not in present, 'test',
                     kind=lambda g: (info[g]['label'], info[g]['domain'])):
            break
    # Validation: at least one positive group for checkpoint/threshold selection.
    positive = lambda g: info[g]['label'] == 1
    if not in_split('val', positive):
        _move(assign, info, rank, positive, 'val', sources=('train',),
              kind=lambda g: (1, info[g]['domain']))
    # Every split must contain both classes.
    for split in ('test', 'val', 'train'):
        for label in (0, 1):
            if not in_split(split, lambda g: info[g]['label'] == label):
                donors = tuple(s for s in ('train', 'val', 'test') if s != split)
                _move(assign, info, rank, lambda g: info[g]['label'] == label, split, sources=donors,
                      kind=lambda g: (info[g]['label'], info[g]['domain']))
    return assign


def split_groups(samples, group_ids, config):
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    if len({s.image_id for s in samples}) != len(samples):
        raise ValueError('Duplicate image IDs')
    labels = [s.label for s in samples]
    if set(labels) != {0, 1}:
        raise ValueError('Both classes are required')
    indices = list(range(len(samples)))
    explicit = [s.partition is not None for s in samples]
    if any(explicit) and not all(explicit):
        raise ValueError('Mixed explicit and missing partitions')
    if config.mode != 'group_stratified' and any(s.pinned_split for s in samples):
        raise ValueError('Pinned datasets (user_dir) require split.mode = group_stratified')
    if all(explicit):
        groups_partitions = defaultdict(set)
        for s in samples:
            if s.partition not in ('train', 'test'):
                raise ValueError('partition must be train/test')
            groups_partitions[group_ids[s.image_id]].add(s.partition)
        if any(len(p) > 1 for p in groups_partitions.values()):
            raise ValueError('Source group overlaps external train/test')
        test = [i for i, s in enumerate(samples) if s.partition == 'test']
        train_pool = [i for i, s in enumerate(samples) if s.partition == 'train']
        if not test or not train_pool:
            raise ValueError('External manifest needs both train and test')
        y = np.array([labels[i] for i in train_pool])
        groups = [group_ids[samples[i].image_id] for i in train_pool]
        splitter = GroupShuffleSplit(n_splits=100, test_size=config.validation_fraction,
                                     random_state=config.seed)
        choices = []
        for tr, va in splitter.split(train_pool, y, groups):
            if set(y[tr]) == {0, 1} and set(y[va]) == {0, 1}:
                cost = abs(len(va)/len(y)-config.validation_fraction) + abs(y[va].mean()-y.mean())
                choices.append((cost, tr, va))
        if not choices:
            raise ValueError('Not enough independent external train groups for both classes in train/val')
        _, tr, va = min(choices, key=lambda item: item[0])
        train, val = [train_pool[i] for i in tr], [train_pool[i] for i in va]
    elif config.mode == 'external':
        raise ValueError('External mode requires partition=train/test for every row')
    elif config.mode == 'random_stratified':
        # Explicit legacy option: frames of one source may cross splits.
        train_val, test = train_test_split(indices, test_size=round(len(samples)*config.test_fraction),
                                          random_state=config.seed, stratify=labels)
        train, val = train_test_split(train_val,
            test_size=round(len(samples)*config.validation_fraction),
            random_state=config.seed, stratify=[labels[i] for i in train_val])
    elif config.mode == 'group_stratified':
        assign = _group_stratified(group_table(samples, group_ids), config)
        parts = {name: [i for i, s in enumerate(samples) if assign[group_ids[s.image_id]] == name]
                 for name in SPLITS}
        train, val, test = parts['train'], parts['val'], parts['test']
    else:
        raise ValueError(f'Unknown split mode: {config.mode}')
    mapping = {}
    for name, subset in [('train', train), ('val', val), ('test', test)]:
        if {labels[i] for i in subset} != {0, 1}:
            raise ValueError(f'Both classes required in {name}; not enough independent groups')
        mapping.update({i: name for i in subset})
    return [ParentAssignment(s.image_id, group_ids[s.image_id], s.partition, mapping[i],
                             s.label, s.domain, s.dataset) for i, s in enumerate(samples)]


def split_summary(samples, assignments, config):
    """Groups/images per split x label x domain, plus independent protocol checks."""
    by_id = {a.image_id: a for a in assignments}
    cells, splits_of = defaultdict(lambda: {'groups': set(), 'images': 0}), defaultdict(set)
    for s in samples:
        a = by_id[s.image_id]
        cell = cells[(a.split, s.label, s.domain)]
        cell['groups'].add(a.group_id); cell['images'] += 1
        splits_of[a.group_id].add(a.split)
    info = group_table(samples, {s.image_id: by_id[s.image_id].group_id for s in samples})
    split_of = {g: next(iter(v)) for g, v in splits_of.items() if len(v) == 1}
    groups_in = lambda split, predicate: sorted(g for g in info if split_of.get(g) == split and predicate(g))
    web = lambda g: WEB_POSITIVE in info[g]['datasets'] and info[g]['label'] == 1
    totals = {}
    for split in SPLITS:
        labels = Counter(s.label for s in samples if by_id[s.image_id].split == split)
        totals[split] = {'groups': len(groups_in(split, lambda g: True)),
                         'positive_groups': len(groups_in(split, lambda g: info[g]['label'] == 1)),
                         'images': sum(labels.values()), 'images_label_1': labels[1], 'images_label_0': labels[0]}
    return {'mode': config.mode, 'seed': config.seed, 'validation_fraction': config.validation_fraction,
        'test_fraction': config.test_fraction, 'unit': 'group_id',
        'cells': [{'split': k[0], 'label': k[1], 'domain': k[2], 'groups': len(v['groups']), 'images': v['images']}
                  for k, v in sorted(cells.items(), key=lambda item: (SPLITS.index(item[0][0]), -item[0][1], item[0][2]))],
        'totals': totals,
        'checks': {'groups_in_several_splits': sorted(g for g, v in splits_of.items() if len(v) > 1),
            'both_labels_every_split': all(totals[s]['images_label_1'] and totals[s]['images_label_0'] for s in SPLITS),
            'test_v1_lectures': groups_in('test', lambda g: 'v1' in info[g]['datasets'] and info[g]['label'] == 1),
            'test_web_positive_sources': groups_in('test', web),
            'test_web_positive_domains': sorted({info[g]['domain'] for g in groups_in('test', web)}),
            'web_positive_domains_available': sorted({info[g]['domain'] for g in info if web(g)}),
            'val_positive_groups': totals['val']['positive_groups'],
            'pinned_groups': sum(1 for g in info if info[g]['pinned']),
            'pinned_violations': sorted(g for g in info if info[g]['pinned'] and split_of.get(g) != info[g]['pinned'])}}
