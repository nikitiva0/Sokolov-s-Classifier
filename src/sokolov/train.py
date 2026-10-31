"""Deterministic staged training with validation-only selection and resume."""
from __future__ import annotations
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from .io import environment, read_csv, read_json, sha256, write_csv, write_json
from .model import build_model, frozen_eval, set_training_stage


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name):
    import torch
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(name)
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA GPU unavailable')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable')
    return device


def fingerprint(config):
    import hashlib
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def training_fingerprint(config):
    # Training and data parameters, not wallclock/device-specific runtime state.
    return fingerprint(config)


def validate_prepared(config):
    path = Path(config.data.prepared_manifest)
    metadata = read_json(path.parent / 'prepared.json')
    if metadata['manifest_sha256'] != sha256(path):
        raise ValueError('Prepared manifest fingerprint mismatch')
    if metadata['split_sha256'] != sha256(path.parent / 'split.csv'):
        raise ValueError('Split fingerprint mismatch')
    signature = metadata['signature']
    if (signature['raw_sha256'] != sha256(config.data.raw_manifest) or
            signature['detector_name'] != config.data.detector_name or
            signature['detector_threshold'] != config.data.detector_threshold or
            signature['phash_max_distance'] != config.data.phash_max_distance):
        raise ValueError('Preparation config/input differs from training config')
    if metadata['signature']['split'] != asdict(config.split):
        raise ValueError('Prepared split settings differ from training config')
    for row in read_csv(path):
        if row['status'] == 'ok' and sha256(row['crop_path']) != row['crop_sha256']:
            raise ValueError('Prepared crop fingerprint mismatch')
    return metadata


RESNET_BLOCKS = ('layer1', 'layer2', 'layer3', 'layer4')


def apply_training_stage(model, config, finetune):
    """Head-only stage, then for ResNet18 the last training.finetune_blocks stages + head."""
    set_training_stage(model, config.model.architecture, finetune)
    if finetune and config.model.architecture == 'resnet18':
        unfrozen = RESNET_BLOCKS[len(RESNET_BLOCKS)-config.training.finetune_blocks:]
        for name in RESNET_BLOCKS:
            for parameter in getattr(model, name).parameters():
                parameter.requires_grad_(name in unfrozen)
        frozen_eval(model)


def evaluation_rows(config, split):
    """All audited rows of a split except near-duplicates of another split's crops."""
    return [r for r in read_csv(config.data.prepared_manifest)
            if r['split'] == split and r['status'] != 'near_duplicate']


def predict(config, model, device, split):
    import torch
    from torch.utils.data import DataLoader
    from .data.dataset import FaceDataset, transforms
    rows = evaluation_rows(config, split)
    usable = [r for r in rows if r['status'] == 'ok']
    dataset = FaceDataset(usable, transforms(config, train=False))
    loader = DataLoader(dataset, batch_size=config.training.batch_size,
                        num_workers=config.training.num_workers, shuffle=False)
    model.eval()
    predictions = {}
    with torch.inference_mode():
        for images, _, ids in loader:
            probabilities = model(images.to(device)).flatten().sigmoid().cpu().tolist()
            predictions.update(zip(ids, probabilities))
    return [{**r, 'score': predictions.get(r['image_id'], 0.), 'detected': r['status'] == 'ok'} for r in rows]


def train(config, run_dir, *, resume=False, pretrained=True, stop_after_epoch=None):
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score
    from torch.utils.data import DataLoader
    from .data.dataset import build_dataset
    from .contracts import TrainingArtifacts

    metadata = validate_prepared(config)
    folder = Path(run_dir) / config.model.architecture
    if folder.exists() and not resume:
        raise ValueError(f'Run already exists: {folder}. Use --resume or a new run directory')
    seed_everything(config.split.seed)
    device = device_for(config.training.device)
    checkpoint = None
    if resume:
        checkpoint = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        if checkpoint['config_fingerprint'] != training_fingerprint(config) or checkpoint['data_fingerprint'] != metadata:
            raise ValueError('Resume config/data fingerprint mismatch')
        if checkpoint['technical_test'] != (not pretrained):
            raise ValueError('Cannot mix technical tests and real runs')
    model = build_model(config.model, pretrained=pretrained and not resume).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
        model.pretrained_provenance = checkpoint['pretrained_provenance']
    folder.mkdir(parents=True, exist_ok=True)
    # Standalone train/evaluate/report commands must produce the same complete
    # run layout as run-all, including the shared data snapshot.
    import shutil
    data_snapshot = folder.parent/'data'
    data_snapshot.mkdir(exist_ok=True)
    if (data_snapshot/'prepared.json').exists() and read_json(data_snapshot/'prepared.json') != metadata:
        raise ValueError('Run already contains a different shared data snapshot')
    for name in ('prepared.json', 'split.csv', 'data_audit.json', 'split_summary.json'):
        shutil.copy2(Path(config.data.prepared_manifest).parent/name, data_snapshot/name)
    write_json(folder / 'config.json', asdict(config))
    env = environment()
    env['device'] = str(device)
    if device.type == 'cuda':
        env.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda)
    write_json(folder / 'environment.json', env)
    write_json(folder / 'status.json', {'status': 'training', 'technical_test': not pretrained})
    dataset = build_dataset(config, 'train')
    labels = [int(r['label']) for r in dataset.rows]
    if set(labels) != {0, 1}:
        raise ValueError('Both training classes required')
    pos_weight = labels.count(0)/labels.count(1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    history = checkpoint['history'] if checkpoint else []
    best_ap = checkpoint['best_ap'] if checkpoint else -1.
    start_epoch = checkpoint['epoch']+1 if checkpoint else 0
    total_epochs = config.training.head_epochs+config.training.finetune_epochs
    elapsed_before = checkpoint['training_seconds'] if checkpoint else 0.
    start = time.perf_counter()
    optimizer = None
    current_stage = None
    for epoch in range(start_epoch, total_epochs):
        seed_everything(config.split.seed + epoch)
        finetune = epoch >= config.training.head_epochs
        if current_stage != finetune:
            apply_training_stage(model, config, finetune)
            lr = config.training.finetune_lr if finetune else config.training.head_lr
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                         lr=lr, weight_decay=config.training.weight_decay)
            if checkpoint and checkpoint['stage'] == finetune:
                optimizer.load_state_dict(checkpoint['optimizer'])
            current_stage = finetune
        frozen_eval(model)
        generator = torch.Generator().manual_seed(config.split.seed+epoch)
        loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                            num_workers=config.training.num_workers, generator=generator,
                            drop_last=(config.model.architecture == 'arcface' and finetune))
        loss_sum = 0.
        count = 0
        for images, targets, _ in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images).flatten(), targets)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(images)
            count += len(images)
        if not count:
            raise ValueError('No training batches; reduce batch_size for this training stage')
        val = predict(config, model, device, 'val')
        ap = float(average_precision_score([int(r['label']) for r in val], [r['score'] for r in val]))
        history.append({'epoch': epoch+1, 'stage': 'finetune' if finetune else 'head',
                        'train_loss': loss_sum/count, 'validation_ap': ap,
                        'learning_rate': optimizer.param_groups[0]['lr']})
        improved = ap > best_ap
        if improved:
            best_ap = ap
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'epoch': epoch, 'stage': finetune, 'history': history, 'best_ap': best_ap,
            'config': asdict(config), 'config_fingerprint': training_fingerprint(config),
            'data_fingerprint': metadata, 'pretrained_provenance': model.pretrained_provenance,
            'pos_weight': pos_weight, 'label_mapping': {'0': 'not_sokolov', '1': 'sokolov'},
            'technical_test': not pretrained, 'environment': env,
            'training_seconds': elapsed_before + time.perf_counter()-start,
            'rng_state': {'python': random.getstate(), 'numpy': np.random.get_state(),
                          'torch': torch.get_rng_state(),
                          'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
        torch.save(payload, folder / 'last.tmp')
        (folder / 'last.tmp').replace(folder / 'last.pt')
        if improved:
            import shutil
            shutil.copyfile(folder / 'last.pt', folder / 'best.pt')
            torch.save(model.state_dict(), folder / 'best_weights.pt')
        write_csv(folder / 'training.csv', history)
        print(f'{config.model.architecture} epoch {epoch+1}/{total_epochs}: loss={loss_sum/count:.4f} val_AP={ap:.4f}', flush=True)
        if stop_after_epoch and epoch+1 >= stop_after_epoch and epoch+1 < total_epochs:
            return None
    last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    write_json(folder / 'training_summary.json', {'epochs': len(last['history']),
        'best_validation_ap': last['best_ap'], 'training_seconds': last['training_seconds'],
        'pos_weight': last['pos_weight'], 'parameters': sum(p.numel() for p in model.parameters()),
        'pretrained_provenance': last['pretrained_provenance'],
        'technical_test': last['technical_test'], 'data_fingerprint': metadata})
    write_json(folder / 'status.json', {'status': 'trained', 'technical_test': not pretrained})
    return TrainingArtifacts(folder/'best.pt', folder/'config.json', folder/'training.csv',
                             metadata['split_sha256'], str(model.pretrained_provenance))
