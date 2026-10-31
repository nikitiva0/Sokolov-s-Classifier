"""Activation maximization of a saved classifier: optimize pixels, never weights."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil

from .config import Config, DataConfig, GenerationConfig, ModelConfig, SplitConfig, TrainingConfig
from .contracts import SynthesisArtifacts
from .io import environment, read_json, sha256, write_csv, write_json
from .model import build_model, state_fingerprint
from .train import device_for


def checkpoint_config(saved):
    """Restore preprocessing from the checkpoint, without touching data or source weights."""
    raw = saved['config']
    generation = dict(raw['generation'])
    generation['seeds'] = tuple(generation['seeds'])
    return Config(DataConfig(**raw['data']), ModelConfig(**raw['model']),
                  SplitConfig(**raw['split']), TrainingConfig(**raw['training']),
                  GenerationConfig(**generation))


def select_best_model(run_dir):
    """Select by the configured validation criterion; ties prefer ResNet18."""
    import torch
    run = Path(run_dir)
    candidates = []
    for name in ('resnet18', 'arcface'):
        folder = run/name
        status = read_json(folder/'status.json')
        if status['status'] not in ('trained', 'evaluated'):
            raise ValueError(f'Finish training both models before selection: {name}')
        saved = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        ap = float(saved['best_ap'])
        if not math.isfinite(ap) or not 0 <= ap <= 1:
            raise ValueError('Invalid validation AP in best checkpoint')
        if saved['config']['model']['architecture'] != name:
            raise ValueError('Checkpoint architecture mismatch')
        candidates.append({'architecture': name, 'validation_ap': ap,
            'selection_metric': saved['config']['training']['selection_metric'],
            'epoch': int(saved['epoch'])+1, 'checkpoint_sha256': sha256(folder/'best.pt'),
            'technical_test': saved['technical_test'], 'data_fingerprint': saved['data_fingerprint']})
        del saved
    if candidates[0]['data_fingerprint'] != candidates[1]['data_fingerprint']:
        raise ValueError('Cannot compare models with different datasets or splits')
    if candidates[0]['technical_test'] != candidates[1]['technical_test']:
        raise ValueError('Cannot compare a technical test and a real trained model')
    winner = max(candidates, key=lambda c: c['validation_ap'])
    metadata = {k: v for k, v in winner.items() if k != 'data_fingerprint'}
    if len({c['selection_metric'] for c in candidates}) != 1:
        raise ValueError('Models must use the same validation selection criterion')
    metadata.update(selection_partition='val', selection_metric=winner['selection_metric'],
        tie_break='resnet18', candidates=[{k: v for k, v in c.items() if k != 'data_fingerprint'} for c in candidates])
    target = run/'best_model.pt'
    if not target.exists() or sha256(target) != winner['checkpoint_sha256']:
        temp = run/'best_model.tmp'
        shutil.copyfile(run/winner['architecture']/'best.pt', temp)
        temp.replace(target)
    write_json(run/'best_model.json', metadata)
    return target


def normalized_pixels(pixels, architecture, policy='rgb'):
    """Differentiable counterpart of dataset normalization; pixels remain in [0,1]."""
    from .preprocessing import normalized_face
    return normalized_face(pixels, architecture, policy)


def objective_terms(pixels, model, config):
    import torch
    logit = model(normalized_pixels(pixels, config.model.architecture,
                                   config.model.face_preprocessing)).reshape(-1)
    if logit.numel() != 1:
        raise ValueError('Synthesis expects exactly one binary logit')
    logit = logit[0]
    tv = (pixels[:, :, 1:, :]-pixels[:, :, :-1, :]).abs().mean() + \
         (pixels[:, :, :, 1:]-pixels[:, :, :, :-1]).abs().mean()
    l2 = ((pixels-.5)**2).mean()
    objective = logit-config.generation.tv_weight*tv-config.generation.l2_weight*l2
    return {'logit': logit, 'score': logit.sigmoid(), 'tv': tv, 'l2': l2, 'objective': objective}


def _verify_completed(folder, request):
    manifest = folder/'completed.json'
    if not manifest.exists():
        return False
    saved = read_json(manifest)
    if saved['request'] != request:
        raise ValueError('Synthesis settings/checkpoint changed; use a new output directory')
    for relative, digest in saved['files'].items():
        if sha256(folder/relative) != digest:
            raise ValueError('Synthesis artifact modified')
    return True


def synthesize(config: Config, checkpoint: Path, seed: int, *, output_dir=None) -> SynthesisArtifacts:
    import torch
    from PIL import Image
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    stored = checkpoint_config(saved)
    if (stored.model.architecture != config.model.architecture or stored.model.image_size != config.model.image_size
            or stored.model.face_preprocessing != config.model.face_preprocessing):
        raise ValueError('Synthesis config does not match checkpoint architecture/preprocessing')
    if type(seed) is not int or seed < 0:
        raise ValueError('Synthesis seed must be a nonnegative integer')
    root = Path(output_dir) if output_dir else checkpoint.parent/'synthesis'
    folder = root/f'seed_{seed}'
    device = device_for(config.training.device)
    request = {'checkpoint_sha256': sha256(checkpoint), 'seed': seed, 'device': str(device),
               'architecture': stored.model.architecture, 'image_size': stored.model.image_size,
               'generation': json.loads(json.dumps(asdict(config.generation))), 'schema_version': 1}
    artifacts = SynthesisArtifacts(seed, folder/'final.png', folder/'snapshots', folder/'trajectory.csv')
    if _verify_completed(folder, request):
        return artifacts
    if (folder/'request.json').exists() and read_json(folder/'request.json') != request:
        raise ValueError('Synthesis settings/checkpoint changed; use a new output directory')
    folder.mkdir(parents=True, exist_ok=True)
    artifacts.snapshots_dir.mkdir(exist_ok=True)
    write_json(folder/'request.json', request)
    model = build_model(stored.model, pretrained=False).to(device)
    model.load_state_dict(saved['model'], strict=True)
    technical_test = saved['technical_test']
    del saved
    model.eval()
    model.requires_grad_(False)
    before = state_fingerprint(model.state_dict())
    size = stored.model.image_size
    initial = torch.rand((1, 3, size, size), generator=torch.Generator().manual_seed(seed))
    pixels = initial.to(device).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pixels], lr=config.generation.lr)
    rows = []
    first_grad_norm = None
    for step in range(config.generation.steps+1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            terms = objective_terms(pixels, model, config)
            if not all(torch.isfinite(v).item() for v in terms.values()):
                raise ValueError('Non-finite synthesis objective')
            (-terms['objective']).backward()
            if pixels.grad is None or not torch.isfinite(pixels.grad).all():
                raise ValueError('Invalid input gradient')
            if first_grad_norm is None:
                first_grad_norm = float(pixels.grad.norm().item())
                if first_grad_norm == 0:
                    raise ValueError('Zero input gradient: cannot optimize pixels')
            optimizer.step()
            with torch.no_grad():
                pixels.clamp_(0, 1)
        # PNG uses 8-bit pixels. Score those exact pixels so saved images and curves agree.
        with torch.no_grad():
            quantized = (pixels.detach()*255).round().clamp(0, 255).to(torch.uint8)
            encoded_pixels = quantized.float()/255
            terms = objective_terms(encoded_pixels, model, config)
            if not all(torch.isfinite(v).item() for v in terms.values()):
                raise ValueError('Non-finite saved-image score')
            row = {'step': step, **{k: float(v.item()) for k, v in terms.items()}}
        snapshot = step == 0 or step % config.generation.snapshot_every == 0 or step == config.generation.steps
        row['snapshot'] = f'snapshots/step_{step:04d}.png' if snapshot else ''
        if snapshot:
            image = Image.fromarray(quantized[0].permute(1, 2, 0).cpu().numpy(), mode='RGB')
            image.save(folder/row['snapshot'])
            if step == config.generation.steps:
                image.save(artifacts.final_image)
        rows.append(row)
    if torch.equal(initial, pixels.detach().cpu()):
        raise ValueError('Synthesis did not change pixels')
    after = state_fingerprint(model.state_dict())
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Classifier weights changed during pixel optimization')
    write_csv(artifacts.trajectory, rows)
    write_json(folder/'metadata.json', {**request, 'technical_test': technical_test,
        'model_state_sha256_before': before, 'model_state_sha256_after': after,
        'environment': environment(), 'device': str(device),
        'first_input_gradient_norm': first_grad_norm,
        'pixel_min': float(pixels.detach().min().item()), 'pixel_max': float(pixels.detach().max().item()),
        'score_pixels': '8-bit PNG decoded as RGB / 255, then checkpoint normalization',
        'optimization_pixels': 'continuous RGB in [0,1]; Adam maximizes logit minus TV and L2 penalties'})
    write_json(folder/'completed.json', {'request': request, 'files': {
        p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and p.name != 'completed.json'}})
    return artifacts


def compare_seeds(config: Config, checkpoint: Path, *, output_dir=None) -> list[SynthesisArtifacts]:
    from .visualize import render_synthesis_report
    root = Path(output_dir) if output_dir else Path(checkpoint).parent/'synthesis'
    request = {'checkpoint_sha256': sha256(checkpoint), 'generation':
               json.loads(json.dumps(asdict(config.generation))), 'architecture': config.model.architecture,
               'device': str(device_for(config.training.device))}
    if (root/'request.json').exists() and read_json(root/'request.json') != request:
        raise ValueError('Synthesis experiment changed; choose a new output directory')
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'request.json', request)
    runs = []
    for seed in config.generation.seeds:
        print(f'Synthesis: {config.model.architecture}, seed={seed}', flush=True)
        runs.append(synthesize(config, checkpoint, seed, output_dir=root))
    render_synthesis_report(runs, root)
    summary = {'architecture': config.model.architecture, 'checkpoint_sha256': sha256(checkpoint),
        'generation': json.loads(json.dumps(asdict(config.generation))),
        'technical_test': read_json(root/f'seed_{runs[0].seed}'/'metadata.json')['technical_test'],
        'seeds': [r.seed for r in runs], 'selection_partition': 'val',
        'results': [],
        'interpretation': 'High classifier score does not guarantee a recognizable face; all seeds are shown.'}
    from .io import read_csv
    for run in runs:
        trajectory = read_csv(run.trajectory)
        meta = read_json(run.trajectory.parent/'metadata.json')
        summary['results'].append({'seed': run.seed, 'initial_score': float(trajectory[0]['score']),
            'final_score': float(trajectory[-1]['score']), 'final_logit': float(trajectory[-1]['logit']),
            'weights_unchanged': meta['model_state_sha256_before'] == meta['model_state_sha256_after'],
            'first_input_gradient_norm': meta['first_input_gradient_norm']})
    write_json(root/'summary.json', summary)
    write_json(root/'completed.json', {'request': request, 'files': {
        p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*'))
        if p.is_file() and p != root/'completed.json'}})
    return runs


def generate_best(run_dir, *, generation=None, device=None, output_dir=None):
    """Use the persisted winner, without requiring original datasets or pretrained files."""
    import torch
    from dataclasses import replace
    run = Path(run_dir)
    checkpoint = run/'best_model.pt'
    if not checkpoint.exists():
        checkpoint = select_best_model(run)
    selected = read_json(run/'best_model.json')
    if sha256(checkpoint) != selected['checkpoint_sha256']:
        raise ValueError('Saved best model fingerprint mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint_config(saved)
    del saved
    if generation is not None:
        config = replace(config, generation=generation)
    if device is not None:
        config = replace(config, training=replace(config.training, device=device))
    return compare_seeds(config, checkpoint, output_dir=output_dir)


def validate_synthesis(output_dir):
    root = Path(output_dir)
    completed = read_json(root/'completed.json')
    if read_json(root/'request.json') != completed['request']:
        raise ValueError('Synthesis request modified')
    for relative, digest in completed['files'].items():
        path = root/relative
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or sha256(path) != digest:
            raise ValueError('Synthesis artifact modified')
    summary = read_json(root/'summary.json')
    if summary['checkpoint_sha256'] != completed['request']['checkpoint_sha256']:
        raise ValueError('Synthesis checkpoint mismatch')
    return summary
