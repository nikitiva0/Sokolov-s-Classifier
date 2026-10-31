"""Progression sheets/GIFs, per-variant score/logit curves and the variant x seed grid.

Images are upscaled with NEAREST only, so displayed pixels are the optimized pixels.
"""
from pathlib import Path
from PIL import Image, ImageDraw
from .io import read_csv

TILE = 224
CAPTION = 44
# Fixed categorical order (validated reference palette); a seed keeps its colour across panels.
SEED_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
OTHER_COLOR = '#8a8984'


def _tile(path):
    with Image.open(path) as source:
        return source.convert('RGB').resize((TILE, TILE), Image.Resampling.NEAREST)


def _snapshots(run):
    return [r for r in read_csv(run.trajectory) if r['snapshot']]


def render_run(run):
    """Per-run contact sheet of every snapshot and an animated GIF of the same frames."""
    folder = run.trajectory.parent
    snapshots = _snapshots(run)
    columns = min(7, len(snapshots))
    sheet = Image.new('RGB', (columns*TILE, ((len(snapshots)+columns-1)//columns)*(TILE+36)), 'white')
    draw = ImageDraw.Draw(sheet)
    frames = []
    for i, row in enumerate(snapshots):
        image = _tile(folder/row['snapshot'])
        x, y = (i % columns)*TILE, (i//columns)*(TILE+36)
        sheet.paste(image, (x, y))
        label = f"step {row['step']}  score {float(row['score']):.4f}"
        draw.text((x+4, y+TILE+4), label, fill='black')
        frame = Image.new('RGB', (TILE, TILE+36), 'white')
        frame.paste(image, (0, 0))
        frame_draw = ImageDraw.Draw(frame)
        frame_draw.text((4, TILE+4), f'{run.variant}  seed {run.seed}', fill='black')
        frame_draw.text((4, TILE+18), label, fill='black')
        frames.append(frame)
    sheet.save(folder/'progression.png')
    # GIF palettes are 256 colours per frame: the PNG sheet is the exact record.
    durations = [250]*(len(frames)-1)+[1500]
    frames[0].save(folder/'progression.gif', save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False)
    return folder/'progression.png', folder/'progression.gif'


def _face_text(record):
    if record is None or record['faces'] is None:
        return 'faces N/A'
    confidence = record['max_confidence']
    text = f"faces {record['faces']}  det {confidence:.2f}" if confidence is not None else f"faces {record['faces']}  det <0.05"
    return text + (f"  cos {record['arcface_cosine']:.2f}" if record['arcface_cosine'] is not None else '')


def render_grid(runs, analysis, path):
    """Finals: rows are variants, columns are seeds; captions carry score and detector result."""
    variants = list(dict.fromkeys(r.variant for r in runs))
    seeds = list(dict.fromkeys(r.seed for r in runs))
    records = {(r['variant'], r['seed']): r for r in (analysis or {}).get('images', []) if r['kind'] == 'final'}
    label_width = 90
    grid = Image.new('RGB', (label_width+TILE*len(seeds), (TILE+CAPTION)*len(variants)), 'white')
    draw = ImageDraw.Draw(grid)
    for run in runs:
        row, column = variants.index(run.variant), seeds.index(run.seed)
        x, y = label_width+column*TILE, row*(TILE+CAPTION)
        grid.paste(_tile(run.final_image), (x, y))
        score = float(read_csv(run.trajectory)[-1]['score'])
        draw.text((x+4, y+TILE+4), f'seed {run.seed}  score {score:.4f}', fill='black')
        draw.text((x+4, y+TILE+20), _face_text(records.get((run.variant, run.seed))), fill='black')
    for row, name in enumerate(variants):
        draw.text((6, row*(TILE+CAPTION)+TILE//2), name, fill='black')
    grid.save(path)
    return path


def render_curves(runs, path):
    """One row per variant: score and logit over steps for every seed (shared axes per column)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants = list(dict.fromkeys(r.variant for r in runs))
    seeds = list(dict.fromkeys(r.seed for r in runs))
    figure, axes = plt.subplots(len(variants), 2, figsize=(11, 3.2*len(variants)), squeeze=False,
                                sharex=True, sharey='col')
    for run in runs:
        rows = read_csv(run.trajectory)
        steps = [int(r['step']) for r in rows]
        index = seeds.index(run.seed)
        color = SEED_COLORS[index] if index < len(SEED_COLORS) else OTHER_COLOR
        label = f'seed {run.seed}' if index < len(SEED_COLORS) else None
        row = variants.index(run.variant)
        axes[row, 0].plot(steps, [float(r['score']) for r in rows], color=color, linewidth=1.5, label=label)
        axes[row, 1].plot(steps, [float(r['logit']) for r in rows], color=color, linewidth=1.5, label=label)
    for row, name in enumerate(variants):
        axes[row, 0].set(ylabel=f'{name}: Sokolov score', ylim=(-.02, 1.02))
        axes[row, 1].set(ylabel=f'{name}: Sokolov logit')
        for axis in axes[row]:
            axis.grid(alpha=.2)
            axis.spines[['top', 'right']].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel('Optimization step (score of the saved 8-bit image)')
    axes[0, 1].legend(loc='best', frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def render_seed_sheets(runs, folder, columns=5):
    """For each seed: rows are variants, columns are evenly spaced snapshots (first and last included)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    variants = list(dict.fromkeys(r.variant for r in runs))
    paths = []
    for seed in dict.fromkeys(r.seed for r in runs):
        own = [r for r in runs if r.seed == seed]
        sheet = Image.new('RGB', (90+columns*TILE, len(variants)*(TILE+24)), 'white')
        draw = ImageDraw.Draw(sheet)
        for run in own:
            snapshots = _snapshots(run)
            chosen = sorted({round(i*(len(snapshots)-1)/(columns-1)) for i in range(columns)})
            y = variants.index(run.variant)*(TILE+24)
            draw.text((6, y+TILE//2), run.variant, fill='black')
            for column, index in enumerate(chosen):
                row = snapshots[index]
                sheet.paste(_tile(run.trajectory.parent/row['snapshot']), (90+column*TILE, y))
                draw.text((90+column*TILE+4, y+TILE+4), f"step {row['step']}  score {float(row['score']):.4f}", fill='black')
        path = folder/f'seed_{seed}.png'
        sheet.save(path)
        paths.append(path)
    return paths


def render_synthesis_report(runs, output_dir, analysis=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for run in runs:
        render_run(run)
    render_curves(runs, output/'scores.png')
    render_seed_sheets(runs, output/'by_seed')
    return render_grid(runs, analysis, output/'grid.png')
