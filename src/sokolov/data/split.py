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
    train_only_groups = {group_ids[s.image_id] for s in samples
                         if s.source_id in config.train_only_sources}
    if any(explicit) and not all(explicit):
        raise ValueError('Mixed explicit and missing partitions')
    if all(explicit):
        if any(s.partition == 'test' and group_ids[s.image_id] in train_only_groups for s in samples):
            raise ValueError('Train-only source conflicts with external test')
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
            if train_only_groups.intersection(groups[i] for i in va):
                continue
            if set(y[tr]) == {0, 1} and set(y[va]) == {0, 1}:
                cost = abs(len(va)/len(y)-config.validation_fraction) + abs(y[va].mean()-y.mean())
                choices.append((cost, tr, va))
        if not choices:
            raise ValueError('Not enough independent external train groups for both classes in train/val')
        _, tr, va = min(choices, key=lambda item: item[0])
        train, val = [train_pool[i] for i in tr], [train_pool[i] for i in va]
    elif config.mode == 'group_stratified':
        y = np.asarray(labels)
        groups = np.asarray([group_ids[s.image_id] for s in samples])
        web = np.asarray([s.label == 1 and s.source_id.startswith('web:') for s in samples])
        web_groups = set(groups[web])
        require_web = len(web_groups) >= 3
        min_web_held_out = 2 if len(web_groups) >= 5 and len(web_groups-train_only_groups) >= 4 else 1
        web_negative = np.asarray([s.label == 0 and s.source_id.startswith('web:') for s in samples])
        negative_groups = set(groups[web_negative])
        require_web_negative = len(negative_groups) >= 3
        min_negative_held_out = 2 if len(negative_groups) >= 5 and len(negative_groups-train_only_groups) >= 4 else 1
        choices = []
        outer = GroupShuffleSplit(n_splits=128, test_size=config.test_fraction,
                                  random_state=config.seed)
        for pool, te in outer.split(indices, y, groups):
            if train_only_groups.intersection(groups[te]):
                continue
            if set(y[te]) != {0, 1} or set(y[pool]) != {0, 1}:
                continue
            inner = GroupShuffleSplit(n_splits=24,
                test_size=config.validation_fraction/(1-config.test_fraction), random_state=config.seed)
            for tr_local, va_local in inner.split(pool, y[pool], groups[pool]):
                tr, va = pool[tr_local], pool[va_local]
                if train_only_groups.intersection(groups[va]):
                    continue
                folds = [tr, va, te]
                if any(set(y[fold]) != {0, 1} for fold in folds):
                    continue
                if require_web and any(len(set(groups[fold][web[fold]])) < minimum
                        for fold, minimum in zip(folds, [1, min_web_held_out, min_web_held_out])):
                    continue
                if require_web_negative and any(len(set(groups[fold][web_negative[fold]])) < minimum
                        for fold, minimum in zip(folds, [1, min_negative_held_out, min_negative_held_out])):
                    continue
                desired = [1-config.validation_fraction-config.test_fraction,
                           config.validation_fraction, config.test_fraction]
                cost = sum(abs(len(fold)/len(y)-fraction)+2*abs(y[fold].mean()-y.mean())
                           for fold, fraction in zip(folds, desired))
                choices.append((cost, tr, va, te))
        if not choices:
            raise ValueError('Need enough independent source groups for both classes in train/val/test')
        _, train, val, test = min(choices, key=lambda item: item[0])
    else:
        if train_only_groups:
            raise ValueError('train_only_sources requires group_stratified or external split')
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
