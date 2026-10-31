"""Binary metrics with explicit undefined reasons and validation-only thresholding."""
from __future__ import annotations
import numpy as np
from sklearn import metrics as m

from .data.families import source_family


def binary_metrics(labels, scores, threshold, detected=None):
    y, p = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if not len(y):
        return {'count': 0, 'reason': 'No eligible samples', 'metrics': None}
    if not set(y).issubset({0, 1}) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid binary predictions')
    pred = (p >= threshold).astype(int)
    if detected is not None:
        pred = pred * np.asarray(detected, dtype=bool)
    tn, fp, fn, tp = m.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, f1, support = m.precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    values = {'accuracy': float(m.accuracy_score(y, pred)),
        'balanced_accuracy': float(m.balanced_accuracy_score(y, pred)) if len(set(y)) == 2 else None,
        'macro_f1': float(f1.mean()), 'mcc': float(m.matthews_corrcoef(y, pred)) if len(set(y)) == 2 and len(set(pred)) == 2 else None,
        'specificity': float(tn/(tn+fp)) if tn+fp else None,
        'roc_auc': float(m.roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        'average_precision': float(m.average_precision_score(y, p)) if len(set(y)) == 2 else None,
        'log_loss': float(m.log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])),
        'brier_score': float(m.brier_score_loss(y, p))}
    undefined = {k: 'Insufficient true/predicted classes for this metric' for k, v in values.items() if v is None}
    classes = {}
    for i, name in enumerate(['not_sokolov', 'sokolov']):
        precision_defined = bool(np.sum(pred == i))
        recall_defined = bool(support[i])
        f1_defined = bool(np.sum(pred == i) + support[i])
        classes[name] = {'precision': float(precision[i]) if precision_defined else None,
                         'recall': float(recall[i]) if recall_defined else None,
                         'f1': float(f1[i]) if f1_defined else None, 'support': int(support[i])}
        for key, defined in [('precision', precision_defined), ('recall', recall_defined), ('f1', f1_defined)]:
            if not defined:
                undefined[f'{name}.{key}'] = 'No predictions and/or true samples for this class'
    return {'count': len(y), 'threshold': float(threshold), 'metrics': values, 'classes': classes,
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]], 'undefined': undefined}


def choose_threshold(labels, scores, detected=None, sample_weight=None):
    y, p = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if set(y) != {0, 1}:
        raise ValueError('Both validation classes are required to select threshold')
    # Include an all-negative threshold and 0; equality uses >= consistently.
    candidates = np.unique(np.r_[0., p, np.nextafter(1., 2.)])
    best = (-1., -1.)
    for threshold in candidates:
        pred = p >= threshold
        if detected is not None:
            pred &= np.asarray(detected, dtype=bool)
        score = float(m.f1_score(y, pred, zero_division=0, sample_weight=sample_weight))
        best = max(best, (score, float(threshold)))
    return {'threshold': best[1], 'validation_f1': best[0], 'selection_partition': 'val',
            'tie_break': 'largest_threshold',
            'weighting': 'equal_domains_and_sources_within_class' if sample_weight is not None else 'uniform'}


def positive_family_subsets(rows):
    """{family: rows} — positives of one acquisition family (lecture/web) plus all negatives.

    Each family is compared with the same negatives, so a large lecture family cannot hide
    failures on a few internet photos (source_macro_ap, by_positive_domain).
    """
    negatives = [r for r in rows if int(r['label']) == 0]
    families = {}
    for row in rows:
        if int(row['label']) == 1:
            families.setdefault(source_family(row), []).append(row)
    return {family: negatives+positives for family, positives in sorted(families.items())}


def validation_ap_by_domain(rows):
    """Each positive acquisition family is evaluated against the same negatives."""
    return {family: float(m.average_precision_score([int(r['label']) for r in items], [r['score'] for r in items]))
            for family, items in positive_family_subsets(rows).items()}


BOOTSTRAP_METRICS = ('roc_auc', 'average_precision', 'balanced_accuracy')


def group_bootstrap(labels, scores, threshold, detected, groups, *, resamples=1000, seed=42):
    """95% percentile CIs, resampling whole groups with replacement.

    Stratified by group label (a group with any Sokolov image is positive), so every
    resample keeps both classes. O(resamples * N log N); memory O(N).
    """
    y, p = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    pred = (p >= threshold) & np.asarray(detected if detected is not None else np.ones(len(y)), dtype=bool)
    index = {}
    for i, group in enumerate(groups):
        index.setdefault(group, []).append(i)
    members = {g: np.array(v) for g, v in index.items()}
    positive = sorted(g for g, v in members.items() if y[v].max() == 1)
    negative = sorted(g for g, v in members.items() if y[v].max() == 0)
    result = {'method': 'group bootstrap, stratified by group label, percentile 2.5/97.5',
              'resamples': resamples, 'seed': seed, 'positive_groups': len(positive),
              'negative_groups': len(negative), 'threshold': float(threshold)}
    if not positive or not negative or not (y == 0).any():
        return {**result, 'intervals': None, 'reason': 'Both positive and negative groups/images are required'}
    rng = np.random.default_rng(seed)
    values = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(resamples):
        chosen = [positive[i] for i in rng.integers(0, len(positive), len(positive))] + \
                 [negative[i] for i in rng.integers(0, len(negative), len(negative))]
        rows = np.concatenate([members[g] for g in chosen])
        yy, pp, dd = y[rows], p[rows], pred[rows]
        if len(set(yy)) < 2:
            continue
        values['roc_auc'].append(m.roc_auc_score(yy, pp))
        values['average_precision'].append(m.average_precision_score(yy, pp))
        values['balanced_accuracy'].append(m.balanced_accuracy_score(yy, dd.astype(int)))
    intervals = {}
    for name, items in values.items():
        intervals[name] = ({'low': float(np.percentile(items, 2.5)), 'high': float(np.percentile(items, 97.5)),
                            'valid_resamples': len(items)} if items else None)
    return {**result, 'intervals': intervals}


def _rate(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def breakdown(rows, threshold):
    """Test diagnostics: recall per positive source, FPR per negative domain/kind, per-domain AUC.

    rows need label, score, detected, domain, source_id. Negatives are split into
    in-domain hard negatives (everything except FairFace) and FairFace.
    """
    def decided(r):
        return bool(r['detected']) and float(r['score']) >= threshold
    by_domain, by_source, by_kind = {}, {}, {}
    for r in rows:
        by_domain.setdefault(r['domain'] or 'unknown', []).append(r)
        if int(r['label']) == 1:
            by_source.setdefault(r['source_id'], []).append(r)
        else:
            by_kind.setdefault('fairface' if r['domain'] == 'fairface' else 'in_domain', []).append(r)
    domains = {}
    for domain, items in sorted(by_domain.items()):
        pos = [r for r in items if int(r['label']) == 1]
        neg = [r for r in items if int(r['label']) == 0]
        y = [int(r['label']) for r in items]
        domains[domain] = {'positives': len(pos), 'negatives': len(neg),
            'recall': _rate(sum(map(decided, pos)), len(pos)),
            'false_positive_rate': _rate(sum(map(decided, neg)), len(neg)),
            'roc_auc': float(m.roc_auc_score(y, [float(r['score']) for r in items])) if pos and neg else None}
    sources = {source: {'domain': items[0]['domain'], 'positives': len(items),
                        'detected': sum(bool(r['detected']) for r in items),
                        'recall': _rate(sum(map(decided, items)), len(items)),
                        'median_score': float(np.median([float(r['score']) for r in items]))}
               for source, items in sorted(by_source.items())}
    kinds = {kind: {'negatives': len(items), 'false_positives': sum(map(decided, items)),
                    'false_positive_rate': _rate(sum(map(decided, items)), len(items))}
             for kind, items in sorted(by_kind.items())}
    return {'threshold': float(threshold), 'by_domain': domains, 'positive_sources': sources,
            'negatives_by_kind': kinds}
