"""One persisted split, before detection; explicit external test stays untouched."""
from __future__ import annotations
from collections import defaultdict

from ..contracts import ParentAssignment


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
    else:
        if config.mode == 'external':
            raise ValueError('External mode requires partition=train/test for every row')
        train_val, test = train_test_split(indices, test_size=round(len(samples)*config.test_fraction),
                                          random_state=config.seed, stratify=labels)
        train, val = train_test_split(train_val,
            test_size=round(len(samples)*config.validation_fraction),
            random_state=config.seed, stratify=[labels[i] for i in train_val])
    mapping = {}
    for name, subset in [('train', train), ('val', val), ('test', test)]:
        if {labels[i] for i in subset} != {0, 1}:
            raise ValueError(f'Both classes required in {name}')
        mapping.update({i: name for i in subset})
    return [ParentAssignment(s.image_id, group_ids[s.image_id], s.partition, mapping[i])
            for i, s in enumerate(samples)]
