"""Binary metrics with explicit undefined reasons and validation-only thresholding."""
from __future__ import annotations
import numpy as np
from sklearn import metrics as m


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


def choose_threshold(labels, scores, detected=None):
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
        score = float(m.f1_score(y, pred, zero_division=0))
        best = max(best, (score, float(threshold)))
    return {'threshold': best[1], 'validation_f1': best[0], 'selection_partition': 'val',
            'tie_break': 'largest_threshold'}
