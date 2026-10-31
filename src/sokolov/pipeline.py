"""End-to-end orchestration; model code stays reusable outside the notebook."""
from pathlib import Path
import shutil
from .io import read_json, write_json, environment


def run_all(configs, run_dir, *, resume=False):
    from .data.download import download_data
    from .data.prepare import prepare
    from .train import train
    from .evaluate import evaluate_test
    from .report import make_report
    from .synthesize import select_best_model, generate_best
    run = Path(run_dir)
    if run.exists() and not resume:
        raise ValueError('Run directory already exists. Use --resume or a new run ID')
    run.mkdir(parents=True, exist_ok=True)
    write_json(run/'status.json', {'status': 'running'})
    try:
        download_data(configs[0])
        prepared = prepare(configs[0])
        data = run/'data'
        data.mkdir(exist_ok=True)
        for name in ('data_audit.json', 'split.csv', 'split_summary.json', 'prepared.json'):
            shutil.copy2(prepared.parent/name, data/name)
        selection = Path(configs[0].data.raw_manifest).parent/'selection.local.json'
        if selection.is_file(): shutil.copy2(selection, data/selection.name)
        for config in configs:
            folder = run/config.model.architecture
            status = read_json(folder/'status.json')['status'] if (folder/'status.json').exists() else None
            if status not in ('trained', 'evaluated'):
                train(config, run, resume=resume and (folder/'last.pt').is_file())
            # evaluate validates checkpoint/config/data even on resumed complete runs.
            evaluate_test(config, folder/'best.pt')
        select_best_model(run)
        generate_best(run)
        result = make_report(run)
        write_json(run/'status.json', {'status': 'completed'})
        return result
    except Exception as error:
        write_json(run/'status.json', {'status': 'failed', 'error_type': type(error).__name__})
        raise
