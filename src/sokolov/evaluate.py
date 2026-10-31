"""Evaluate the frozen best checkpoint; threshold comes exclusively from validation."""
from __future__ import annotations
import time
from pathlib import Path

from .io import read_csv, read_json, write_csv, write_json, sha256
from .metrics import binary_metrics, breakdown, choose_threshold, group_bootstrap, positive_family_subsets
from .model import build_model
from .train import device_for, predict, training_fingerprint, validate_prepared

BOOTSTRAP_RESAMPLES = 1000


def select_threshold(config, checkpoint):
    import torch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build_model(config.model, pretrained=False)
    model.load_state_dict(saved['model'], strict=True)
    device = device_for(config.training.device)
    val = predict(config, model.to(device), device, 'val')
    weights = None
    if config.training.selection_metric == 'source_macro_ap':
        from .data.dataset import source_sampling_weights
        weights = source_sampling_weights(val)
    return choose_threshold([int(r['label']) for r in val], [r['score'] for r in val],
                            [r['detected'] for r in val], sample_weight=weights)


def plot_predictions(rows, threshold, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import metrics
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    y, p = np.array([int(r['label']) for r in rows]), np.array([float(r['score']) for r in rows])
    pred = (p >= threshold) & np.array([bool(r['detected']) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if len(set(y)) == 2:
        metrics.RocCurveDisplay.from_predictions(y, p, ax=axes[0])
        metrics.PrecisionRecallDisplay.from_predictions(y, p, ax=axes[1])
    else:
        for ax in axes:
            ax.text(.1, .5, 'Undefined: both classes required')
    fig.tight_layout(); fig.savefig(folder/'curves.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y, pred, labels=[0, 1]),
                                 display_labels=['not_sokolov', 'sokolov']).plot(ax=ax, colorbar=False)
    fig.tight_layout(); fig.savefig(folder/'confusion_matrix.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, name in [(0, 'not_sokolov'), (1, 'sokolov')]:
        ax.hist(p[y == label], bins=20, alpha=.5, label=name)
    ax.axvline(threshold, color='black', linestyle='--', label='validation threshold')
    ax.set(xlabel='Score', ylabel='Count'); ax.legend()
    fig.tight_layout(); fig.savefig(folder/'scores.png', dpi=140); plt.close(fig)


def benchmark(config, model, device):
    import torch
    from .data.dataset import build_dataset
    ds = build_dataset(config, 'test')
    batch = torch.stack([ds[i % len(ds)][0] for i in range(config.training.batch_size)]).to(device)
    model.eval()
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize(device)
        if device.type == 'mps': torch.mps.synchronize()
    with torch.inference_mode():
        for _ in range(5): model(batch)
        sync(); start = time.perf_counter()
        for _ in range(20): model(batch)
        sync(); seconds = time.perf_counter()-start
    n = len(batch)*20
    return {'device': str(device), 'batch_size': len(batch), 'warmup_batches': 5,
            'measured_batches': 20, 'images_per_second': n/seconds,
            'milliseconds_per_image': seconds*1000/n,
            'scope': 'classifier forward only; excludes detection, decoding and transfer'}


def evaluate_test(config, checkpoint, threshold=None):
    import torch
    from .contracts import EvaluationArtifacts
    folder = Path(checkpoint).parent
    status = read_json(folder/'status.json')
    if status['status'] not in ('trained', 'evaluated'):
        raise ValueError('Training must finish before final evaluation')
    metadata = validate_prepared(config)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['config_fingerprint'] != training_fingerprint(config) or saved['data_fingerprint'] != metadata:
        raise ValueError('Evaluation config/data differs from checkpoint')
    # A provided threshold cannot override validation selection.
    selected = select_threshold(config, checkpoint)
    if threshold is not None and threshold != selected['threshold']:
        raise ValueError('Threshold must match validation-only selection')
    threshold = selected['threshold']
    selected['checkpoint_sha256'] = sha256(checkpoint)
    write_json(folder/'threshold.json', selected)
    model = build_model(config.model, pretrained=False)
    model.load_state_dict(saved['model'], strict=True)
    device = device_for(config.training.device)
    rows = predict(config, model.to(device), device, 'test')
    usable = [r for r in rows if r['detected']]
    def measure(items):
        return binary_metrics([int(r['label']) for r in items], [r['score'] for r in items],
                              threshold, [r['detected'] for r in items])
    def bootstrap(items):
        return group_bootstrap([int(r['label']) for r in items], [r['score'] for r in items], threshold,
                               [r['detected'] for r in items], [r['group_id'] for r in items],
                               resamples=BOOTSTRAP_RESAMPLES, seed=config.split.seed)
    excluded = [r for r in read_csv(config.data.prepared_manifest)
                if r['split'] == 'test' and r['status'] == 'near_duplicate']
    # Positive acquisition families (lecture/web) against the same test negatives.
    families = positive_family_subsets(rows)
    results = {'image_level': measure(rows), 'face_level': measure(usable),
        'by_positive_domain': {family: measure(items) for family, items in families.items()},
        # One CI set per level of the main table; keys match the report level names.
        'bootstrap': {'image_level': bootstrap(rows), 'face_level': bootstrap(usable),
                      **{'positive_domain_'+family: bootstrap(items) for family, items in families.items()}},
        'breakdown': breakdown(rows, threshold),
        'coverage': {'total': len(rows), 'usable_faces': len(usable),
                     'fraction': len(usable)/len(rows) if rows else None,
                     'no_face': sum(r['status'] == 'no_face' for r in rows),
                     'alignment_failed': sum(r['status'] == 'alignment_failed' for r in rows),
                     'several_faces_primary_used': sum(r['detected'] and int(r['face_count']) > 1 for r in rows),
                     'near_duplicate_excluded': len(excluded),
                     'near_duplicate_excluded_positive': sum(r['label'] == '1' for r in excluded)},
        'test_groups': len({r['group_id'] for r in rows}),
        'benchmark': benchmark(config, model, device),
        'checkpoint_bytes': Path(checkpoint).stat().st_size,
        'checkpoint_sha256': sha256(checkpoint), 'technical_test': saved['technical_test']}
    predictions = []
    for r in rows:
        pred = int(r['detected'] and r['score'] >= threshold)
        predictions.append({**r, 'predicted_label': pred,
            'error_type': 'FP' if pred and int(r['label']) == 0 else
                          'FN' if not pred and int(r['label']) == 1 else ''})
    write_csv(folder/'test_predictions.csv', predictions)
    write_csv(folder/'errors.csv', [r for r in predictions if r['error_type']], fields=list(predictions[0]))
    write_json(folder/'test_metrics.json', results)
    plot_predictions(rows, threshold, folder/'plots')
    write_json(folder/'status.json', {'status': 'evaluated', 'technical_test': saved['technical_test']})
    return EvaluationArtifacts(folder/'test_metrics.json', folder/'test_predictions.csv',
                               folder/'plots/confusion_matrix.png', threshold)
