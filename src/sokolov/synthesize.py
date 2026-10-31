"""Activation maximization of a saved classifier: optimize the image, never weights.

Each [[generation.variants]] entry fixes a parametrization and constraints; every
variant x seed is one cached run whose trajectory is scored on its exact 8-bit pixels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cache
import json
import math
from pathlib import Path
import shutil
import time

from .config import Config, DataConfig, GenerationVariant, ModelConfig, SplitConfig, TrainingConfig, generation_config
from .contracts import SynthesisArtifacts
from .io import environment, read_csv, read_json, sha256, write_csv, write_json
from .model import build_model, state_fingerprint
from .train import device_for

REFERENCE_SIZE = 112  # pixel-valued variant parameters are given for a 112 px input
# Square root of the ImageNet RGB covariance used by Lucid (Olah et al., Distill 2017);
# an empirical ImageNet constant, not estimated on faces.
COLOR_CORRELATION_SVD_SQRT = ((.26, .09, .02), (.27, .00, -.05), (.27, -.09, .03))
FOURIER_SCALE = 4.0  # Lucid's constant divisor of the inverse FFT before colour/sigmoid


@dataclass(frozen=True)
class VariantRun(SynthesisArtifacts):
    variant: str = 'pixel'


def legacy_generation(raw):
    """Checkpoints saved before variants existed hold one pixel baseline (lr/tv/l2)."""
    raw = dict(raw)
    if 'variants' not in raw:
        raw['variants'] = [{'name': 'pixel', 'parametrization': 'pixel', 'lr': raw.pop('lr'),
                            'tv_weight': raw.pop('tv_weight'), 'l2_weight': raw.pop('l2_weight')}]
    return generation_config(raw)


def checkpoint_config(saved):
    """Restore preprocessing from the checkpoint, without touching data or source weights."""
    raw = saved['config']
    return Config(DataConfig(**raw['data']), ModelConfig(**raw['model']),
                  SplitConfig(**raw['split']), TrainingConfig(**raw['training']),
                  legacy_generation(raw['generation']))


def select_best_model(run_dir):
    """Select by validation AP only; ties prefer ResNet18. Persist a self-contained copy."""
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
            'epoch': int(saved['epoch'])+1, 'checkpoint_sha256': sha256(folder/'best.pt'),
            'technical_test': saved['technical_test'], 'data_fingerprint': saved['data_fingerprint']})
        del saved
    if candidates[0]['data_fingerprint'] != candidates[1]['data_fingerprint']:
        raise ValueError('Cannot compare models with different datasets or splits')
    if candidates[0]['technical_test'] != candidates[1]['technical_test']:
        raise ValueError('Cannot compare a technical test and a real trained model')
    winner = max(candidates, key=lambda c: c['validation_ap'])
    metadata = {k: v for k, v in winner.items() if k != 'data_fingerprint'}
    metadata.update(selection_partition='val', selection_metric='average_precision',
        tie_break='resnet18', candidates=[{k: v for k, v in c.items() if k != 'data_fingerprint'} for c in candidates])
    target = run/'best_model.pt'
    if not target.exists() or sha256(target) != winner['checkpoint_sha256']:
        temp = run/'best_model.tmp'
        shutil.copyfile(run/winner['architecture']/'best.pt', temp)
        temp.replace(target)
    write_json(run/'best_model.json', metadata)
    return target


def normalized_pixels(pixels, architecture):
    """Differentiable counterpart of dataset normalization; pixels remain in [0,1]."""
    if architecture == 'resnet18':
        mean, std = (.485, .456, .406), (.229, .224, .225)
    elif architecture == 'arcface':
        mean = std = (.5, .5, .5)
    else:
        raise ValueError('Unknown synthesis architecture')
    mean = pixels.new_tensor(mean).view(1, 3, 1, 1)
    std = pixels.new_tensor(std).view(1, 3, 1, 1)
    return (pixels-mean)/std


def objective_terms(pixels, model, variant, architecture, view=None):
    """Logit of `view` (transformed image) or of `pixels`; TV and L2 always on `pixels`."""
    logit = model(normalized_pixels(pixels if view is None else view, architecture)).reshape(-1)
    if logit.numel() != 1:
        raise ValueError('Synthesis expects exactly one binary logit')
    logit = logit[0]
    tv = (pixels[:, :, 1:, :]-pixels[:, :, :-1, :]).abs().mean() + \
         (pixels[:, :, :, 1:]-pixels[:, :, :, :-1]).abs().mean()
    l2 = ((pixels-.5)**2).mean()
    objective = logit-variant.tv_weight*tv-variant.l2_weight*l2
    return {'logit': logit, 'score': logit.sigmoid(), 'tv': tv, 'l2': l2, 'objective': objective}


def has_transforms(variant):
    return bool(variant.translate_px or variant.rotate_deg or variant.scale_min != 1 or variant.scale_max != 1)


def image_parametrization(variant, size, seed, device):
    """(parameter, render, generator). Noise and transform draws use one CPU generator per seed.

    pixel: uniform RGB noise, render is the identity (exactly the original baseline).
    fourier: N(0, init_std) real/imaginary spectrum scaled by 1/max(f, 1/size)^decay_power,
    inverse real FFT / 4, ImageNet colour decorrelation, sigmoid -> (0,1).
    """
    import torch
    generator = torch.Generator().manual_seed(seed)
    if variant.parametrization == 'pixel':
        initial = torch.rand((1, 3, size, size), generator=generator)
        return initial.to(device).clone().requires_grad_(True), (lambda p: p), generator
    frequencies = (torch.fft.fftfreq(size)[:, None]**2+torch.fft.rfftfreq(size)[None, :]**2).sqrt()
    scale = (1/frequencies.clamp(min=1/size)**variant.decay_power)[None, None, :, :, None].to(device)
    color = torch.tensor(COLOR_CORRELATION_SVD_SQRT)
    color = (color/color.norm(dim=0).max()).to(device)
    initial = torch.randn((1, 3, size, size//2+1, 2), generator=generator)*variant.init_std
    def render(parameter):
        spectrum = torch.view_as_complex((scale*parameter).contiguous())
        image = torch.fft.irfftn(spectrum, s=(size, size), norm='ortho')/FOURIER_SCALE
        return torch.einsum('bchw,dc->bdhw', image, color).sigmoid()
    return initial.to(device).clone().requires_grad_(True), render, generator


@cache
def _bilinear():
    import torch

    class Bilinear(torch.autograd.Function):
        """Gather on the model device; scatter-add backward on CPU (deterministic, also for MPS)."""
        @staticmethod
        def forward(ctx, image, index, weight, cpu_index, cpu_weight):
            ctx.save_for_backward(cpu_index, cpu_weight)
            ctx.shape = image.shape
            channels, n = image.shape[1], image.shape[2]*image.shape[3]
            values = image.reshape(channels, n).index_select(1, index).reshape(channels, 4, n)
            return (values*weight).sum(1).reshape(image.shape)

        @staticmethod
        def backward(ctx, grad):
            cpu_index, cpu_weight = ctx.saved_tensors
            channels, n = ctx.shape[1], ctx.shape[2]*ctx.shape[3]
            contribution = grad.detach().reshape(channels, 1, n).cpu()*cpu_weight
            flat = (cpu_index[None, :]+n*torch.arange(channels)[:, None]).reshape(-1)
            result = torch.zeros(channels*n, dtype=contribution.dtype).index_add_(0, flat, contribution.reshape(-1))
            return result.reshape(ctx.shape).to(grad.device), None, None, None, None

    return Bilinear


def sampling_grid(theta, size):
    """Bilinear sampling of F.affine_grid/grid_sample(align_corners=False, reflection), on CPU."""
    import torch
    import torch.nn.functional as F
    grid = F.affine_grid(theta, (1, 1, size, size), align_corners=False)[0]
    def reflect(coordinate):
        coordinate = (coordinate+.5).abs()
        flips = torch.floor(coordinate/size)
        extra = coordinate-flips*size
        return (torch.where(flips % 2 == 0, extra, size-extra)-.5).clamp(0, size-1)
    x, y = reflect(((grid[..., 0]+1)*size-1)/2), reflect(((grid[..., 1]+1)*size-1)/2)
    x0, y0 = x.floor(), y.floor()
    wx, wy = x-x0, y-y0
    x0, y0 = x0.long(), y0.long()
    x1, y1 = (x0+1).clamp(max=size-1), (y0+1).clamp(max=size-1)
    index = torch.stack([y0*size+x0, y0*size+x1, y1*size+x0, y1*size+x1]).reshape(-1)
    weight = torch.stack([(1-wx)*(1-wy), wx*(1-wy), (1-wx)*wy, wx*wy]).reshape(4, -1).float()
    return index, weight


def random_view(image, variant, generator):
    """Differentiable random shift/scale/rotation; returns the view and its parameters."""
    import torch
    size = image.shape[-1]
    u = torch.rand(4, generator=generator, dtype=torch.float64).tolist()
    shift = [(2*v-1)*variant.translate_px*size/REFERENCE_SIZE for v in u[:2]]
    scale = variant.scale_min+(variant.scale_max-variant.scale_min)*u[2]
    angle = math.radians((2*u[3]-1)*variant.rotate_deg)
    cos, sin = math.cos(angle)/scale, math.sin(angle)/scale
    theta = torch.tensor([[[cos, -sin, 2*shift[0]/size], [sin, cos, 2*shift[1]/size]]], dtype=torch.float64)
    index, weight = sampling_grid(theta, size)
    view = _bilinear().apply(image, index.to(image.device), weight.to(image.device), index, weight)
    return view, {'shift_x_px': shift[0], 'shift_y_px': shift[1], 'scale': scale, 'angle_deg': math.degrees(angle)}


def gaussian_blur(image, sigma):
    """Separable Gaussian blur with reflection padding; radius ceil(3 sigma)."""
    import torch
    import torch.nn.functional as F
    radius = max(1, math.ceil(3*sigma))
    x = torch.arange(-radius, radius+1, dtype=torch.float64)
    kernel = torch.exp(-x**2/(2*sigma**2))
    kernel = (kernel/kernel.sum()).float().to(image.device)
    channels = image.shape[1]
    image = F.pad(image, (radius,)*4, mode='reflect')
    image = F.conv2d(image, kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), groups=channels)
    return F.conv2d(image, kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), groups=channels)


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


def _variant(config, variant):
    if variant is None:
        return config.generation.variants[0]
    if isinstance(variant, GenerationVariant):
        return variant
    for candidate in config.generation.variants:
        if candidate.name == variant:
            return candidate
    raise ValueError(f'Unknown synthesis variant: {variant}')


def describe_variant(variant, size):
    """Russian one-line description with pixel values converted to this input size."""
    factor = size/REFERENCE_SIZE
    parts = ['RGB-пиксели, после шага ограничение [0,1]' if variant.parametrization == 'pixel' else
             f'спектр Фурье с масштабом 1/f^{variant.decay_power:g} (init std {variant.init_std:g}), '
             'декоррелированный цвет, sigmoid']
    if has_transforms(variant):
        parts.append(f'перед классификатором случайный сдвиг ±{variant.translate_px*factor:g} px, '
                     f'масштаб {variant.scale_min:g}–{variant.scale_max:g}, поворот ±{variant.rotate_deg:g}° '
                     '(билинейно, отражение на краях)')
    if variant.blur_every:
        parts.append(f'гауссово размытие параметра sigma={variant.blur_sigma_px*factor:g} px на каждом {variant.blur_every}-м шаге')
    parts.append(f'Adam lr={variant.lr:g}; цель = logit − {variant.tv_weight:g}·TV − {variant.l2_weight:g}·L2')
    return '; '.join(parts)


def synthesize(config: Config, checkpoint: Path, seed: int, *, variant=None, output_dir=None) -> VariantRun:
    import torch
    from PIL import Image
    checkpoint = Path(checkpoint)
    variant = _variant(config, variant)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    stored = checkpoint_config(saved)
    if stored.model.architecture != config.model.architecture or stored.model.image_size != config.model.image_size:
        raise ValueError('Synthesis config does not match checkpoint architecture/preprocessing')
    if type(seed) is not int or seed < 0:
        raise ValueError('Synthesis seed must be a nonnegative integer')
    root = Path(output_dir) if output_dir else checkpoint.parent/'synthesis'
    folder = root/variant.name/f'seed_{seed}'
    device = device_for(config.training.device)
    generation = config.generation
    request = {'checkpoint_sha256': sha256(checkpoint), 'seed': seed, 'device': str(device),
               'architecture': stored.model.architecture, 'image_size': stored.model.image_size,
               'steps': generation.steps, 'snapshot_every': generation.snapshot_every,
               'variant': asdict(variant), 'schema_version': 2}
    artifacts = VariantRun(seed, folder/'final.png', folder/'snapshots', folder/'trajectory.csv', variant.name)
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
    architecture = stored.model.architecture
    parameter, render, generator = image_parametrization(variant, size, seed, device)
    with torch.no_grad():
        initial = render(parameter).detach().cpu().clone()
    optimizer = torch.optim.Adam([parameter], lr=variant.lr)
    transforms, blur_sigma = has_transforms(variant), variant.blur_sigma_px*size/REFERENCE_SIZE
    rows = []
    first_grad_norm = None
    started = time.perf_counter()
    for step in range(generation.steps+1):
        optimized_logit, view_parameters = '', {}
        if step:
            optimizer.zero_grad(set_to_none=True)
            image = render(parameter)
            view, view_parameters = random_view(image, variant, generator) if transforms else (None, {})
            terms = objective_terms(image, model, variant, architecture, view=view)
            if not all(torch.isfinite(v).item() for v in terms.values()):
                raise ValueError('Non-finite synthesis objective')
            (-terms['objective']).backward()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError('Invalid input gradient')
            if first_grad_norm is None:
                first_grad_norm = float(parameter.grad.norm().item())
                if first_grad_norm == 0:
                    raise ValueError('Zero input gradient: cannot optimize pixels')
            optimizer.step()
            optimized_logit = float(terms['logit'].item())
            with torch.no_grad():
                if variant.parametrization == 'pixel':
                    parameter.clamp_(0, 1)
                    if variant.blur_every and step % variant.blur_every == 0:
                        parameter.copy_(gaussian_blur(parameter, blur_sigma))
        # PNG uses 8-bit pixels. Score those exact pixels so saved images and curves agree.
        with torch.no_grad():
            quantized = (render(parameter).detach()*255).round().clamp(0, 255).to(torch.uint8)
            encoded_pixels = quantized.float()/255
            terms = objective_terms(encoded_pixels, model, variant, architecture)
            if not all(torch.isfinite(v).item() for v in terms.values()):
                raise ValueError('Non-finite saved-image score')
            row = {'step': step, **{k: float(v.item()) for k, v in terms.items()}}
        row['optimized_logit'] = optimized_logit
        row.update({k: view_parameters.get(k, '') for k in ('shift_x_px', 'shift_y_px', 'scale', 'angle_deg')})
        snapshot = step == 0 or step % generation.snapshot_every == 0 or step == generation.steps
        row['snapshot'] = f'snapshots/step_{step:04d}.png' if snapshot else ''
        if snapshot:
            image = Image.fromarray(quantized[0].permute(1, 2, 0).cpu().numpy(), mode='RGB')
            image.save(folder/row['snapshot'])
            if step == generation.steps:
                image.save(artifacts.final_image)
        rows.append(row)
    seconds = time.perf_counter()-started
    with torch.no_grad():
        final = render(parameter).detach().cpu()
    if torch.equal(initial, final):
        raise ValueError('Synthesis did not change pixels')
    after = state_fingerprint(model.state_dict())
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Classifier weights changed during pixel optimization')
    write_csv(artifacts.trajectory, rows)
    write_json(folder/'metadata.json', {**request, 'technical_test': technical_test,
        'model_state_sha256_before': before, 'model_state_sha256_after': after,
        'environment': environment(), 'device': str(device), 'optimization_seconds': seconds,
        'first_input_gradient_norm': first_grad_norm,
        'gradient_parameter': 'RGB pixels' if variant.parametrization == 'pixel' else 'Fourier spectrum',
        'pixel_min': float(final.min().item()), 'pixel_max': float(final.max().item()),
        'score_pixels': '8-bit PNG decoded as RGB / 255, then checkpoint normalization; no random transform',
        'optimized_logit': 'logit of the continuous (randomly transformed) image used for the gradient, before the step',
        'optimization_pixels': describe_variant(variant, size)})
    write_json(folder/'completed.json', {'request': request, 'files': {
        p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and p.name != 'completed.json'}})
    return artifacts


def run_experiment(config: Config, checkpoint: Path, *, output_dir=None, run_dir=None) -> list[VariantRun]:
    """All variants x seeds, then independent analysis, figures and a verified summary."""
    from .analysis import analyze_experiment, interpretation
    from .visualize import render_synthesis_report
    root = Path(output_dir) if output_dir else Path(checkpoint).parent/'synthesis'
    generation = json.loads(json.dumps(asdict(config.generation)))
    request = {'checkpoint_sha256': sha256(checkpoint), 'generation': generation,
               'architecture': config.model.architecture,
               'device': str(device_for(config.training.device)), 'schema_version': 2}
    if (root/'request.json').exists() and read_json(root/'request.json') != request:
        raise ValueError('Synthesis experiment changed; choose a new output directory')
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'request.json', request)
    runs = []
    for variant in config.generation.variants:
        for seed in config.generation.seeds:
            started = time.perf_counter()
            runs.append(synthesize(config, checkpoint, seed, variant=variant, output_dir=root))
            print(f'Synthesis: {config.model.architecture}, {variant.name}, seed={seed}: '
                  f'{time.perf_counter()-started:.1f} s', flush=True)
    analysis = analyze_experiment(runs, root, config=config, checkpoint=checkpoint, run_dir=run_dir)
    render_synthesis_report(runs, root, analysis)
    results, trajectories = [], []
    for run in runs:
        trajectory = read_csv(run.trajectory)
        meta = read_json(run.trajectory.parent/'metadata.json')
        trajectories += [{'variant': run.variant, 'seed': run.seed, **row} for row in trajectory]
        results.append({'variant': run.variant, 'seed': run.seed, 'initial_score': float(trajectory[0]['score']),
            'final_score': float(trajectory[-1]['score']), 'final_logit': float(trajectory[-1]['logit']),
            'final_tv': float(trajectory[-1]['tv']),
            'weights_unchanged': meta['model_state_sha256_before'] == meta['model_state_sha256_after'],
            'first_input_gradient_norm': meta['first_input_gradient_norm'],
            'optimization_seconds': meta['optimization_seconds'], 'device': meta['device']})
    write_csv(root/'trajectories.csv', trajectories)
    runtime = {}
    for variant in config.generation.variants:
        own = [r for r in results if r['variant'] == variant.name]
        runtime[variant.name] = {'seconds_total': sum(r['optimization_seconds'] for r in own),
            'seconds_per_run_mean': sum(r['optimization_seconds'] for r in own)/len(own),
            'devices': sorted({r['device'] for r in own}), 'steps': config.generation.steps}
    technical = read_json(runs[0].trajectory.parent/'metadata.json')['technical_test']
    summary = {'architecture': config.model.architecture, 'image_size': config.model.image_size,
        'checkpoint_sha256': sha256(checkpoint), 'generation': generation, 'technical_test': technical,
        'seeds': list(config.generation.seeds), 'variants': [v.name for v in config.generation.variants],
        'variant_descriptions': {v.name: describe_variant(v, config.model.image_size) for v in config.generation.variants},
        'selection_partition': 'val', 'results': results, 'runtime': runtime, 'analysis': analysis,
        'interpretation': interpretation(analysis, results)}
    write_json(root/'summary.json', summary)
    write_json(root/'completed.json', {'request': request, 'files': {
        p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*'))
        if p.is_file() and p != root/'completed.json'}})
    return runs


compare_seeds = run_experiment


def override_generation(generation, *, variants=None, seeds=None, steps=None):
    """CLI overrides on top of a validated [generation]; the result is validated again."""
    raw = json.loads(json.dumps(asdict(generation)))
    if variants:
        known = {v['name']: v for v in raw['variants']}
        unknown = [name for name in variants if name not in known]
        if unknown:
            raise ValueError(f'Unknown synthesis variants {unknown}; configured: {sorted(known)}')
        raw['variants'] = [known[name] for name in variants]
    if seeds:
        raw['seeds'] = list(seeds)
    if steps is not None:
        raw['steps'], raw['snapshot_every'] = steps, max(1, min(raw['snapshot_every'], steps))
    return generation_config(raw)


def generate_best(run_dir, *, generation=None, device=None, output_dir=None, model='best',
                  variants=None, seeds=None, steps=None):
    """Use the persisted winner (or one model's best.pt), without original datasets or pretrained files."""
    import torch
    from dataclasses import replace
    run = Path(run_dir)
    if model == 'best':
        checkpoint = run/'best_model.pt'
        if not checkpoint.exists():
            checkpoint = select_best_model(run)
        selected = read_json(run/'best_model.json')
        if sha256(checkpoint) != selected['checkpoint_sha256']:
            raise ValueError('Saved best model fingerprint mismatch')
        default_output = run/'synthesis'
    elif model in ('resnet18', 'arcface'):
        checkpoint = run/model/'best.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'No trained checkpoint: {checkpoint}')
        default_output = run/f'synthesis_{model}'
    else:
        raise ValueError('model: best, resnet18 or arcface')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint_config(saved)
    del saved
    config = replace(config, generation=override_generation(generation or config.generation,
                                                            variants=variants, seeds=seeds, steps=steps))
    if device is not None:
        config = replace(config, training=replace(config.training, device=device))
    return run_experiment(config, checkpoint, output_dir=output_dir or default_output, run_dir=run)


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
