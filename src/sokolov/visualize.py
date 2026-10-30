"""Contact sheets and shared-scale score/logit curves for activation maximization."""
from pathlib import Path
from PIL import Image, ImageDraw
from .io import read_csv


def render_synthesis_report(runs, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    finals = Image.new('RGB', (224*len(runs), 260), 'white')
    final_draw = ImageDraw.Draw(finals)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, run in enumerate(runs):
        rows = read_csv(run.trajectory)
        snapshots = [r for r in rows if r['snapshot']]
        columns = min(7, len(snapshots))
        sheet = Image.new('RGB', (columns*224, ((len(snapshots)+columns-1)//columns)*260), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(snapshots):
            with Image.open(run.trajectory.parent/row['snapshot']) as source:
                image = source.convert('RGB').resize((224, 224), Image.Resampling.NEAREST)
            x, y = (i % columns)*224, (i//columns)*260
            sheet.paste(image, (x, y))
            draw.text((x+4, y+228), f"step {row['step']}  score {float(row['score']):.4f}", fill='black')
        sheet.save(run.trajectory.parent/'progression.png')
        with Image.open(run.final_image) as source:
            finals.paste(source.resize((224, 224), Image.Resampling.NEAREST), (index*224, 0))
        final_draw.text((index*224+4, 228), f"seed {run.seed}  score {float(rows[-1]['score']):.4f}", fill='black')
        steps = [int(r['step']) for r in rows]
        axes[0].plot(steps, [float(r['score']) for r in rows], label=f'seed {run.seed}')
        axes[1].plot(steps, [float(r['logit']) for r in rows], label=f'seed {run.seed}')
    axes[0].set(xlabel='Optimization step', ylabel='Sokolov score', ylim=(-.02, 1.02))
    axes[1].set(xlabel='Optimization step', ylabel='Sokolov logit')
    for axis in axes:
        axis.legend(); axis.grid(alpha=.2)
    figure.tight_layout(); figure.savefig(output/'scores.png', dpi=140); plt.close(figure)
    finals.save(output/'seed_comparison.png')
    return output/'seed_comparison.png'
