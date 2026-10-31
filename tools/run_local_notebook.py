"""Execute the local MPS notebook, saving outputs and live stage after every cell."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sokolov.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', type=Path, default=ROOT/'notebooks/local_train_mps.ipynb')
    parser.add_argument('--output', type=Path, default=ROOT/'work/local-mps/local_train_mps_v3.executed.ipynb')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    os.environ['TORCH_HOME'] = str(ROOT/'work/torch-cache')
    os.environ['KAGGLEHUB_CACHE'] = str(ROOT/'work/kaggle-cache')
    os.environ['MPLCONFIGDIR'] = str(ROOT/'work/matplotlib')
    os.environ['ALBUMENTATIONS_OFFLINE'] = '1'
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    kernels = ROOT/'work/jupyter/kernels'
    kernel = kernels/'sokolov-mps'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Sokolov · MPS', 'language': 'python'}))
    notebook = nbformat.read(args.notebook, as_version=4)
    run_dir = (args.run_dir or Path(notebook.metadata.sokolov.run_dir)).resolve()
    os.environ['SOKOLOV_RUN_DIR'] = str(run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {'run_dir': str(run_dir),
             'prepared_manifest': str(ROOT/notebook.metadata.sokolov.prepared_manifest)}
    if sys.platform == 'darwin':
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])

    def save():
        temporary = args.output.with_suffix('.tmp.ipynb')
        nbformat.write(notebook, temporary)
        temporary.replace(args.output)

    def starting(cell, cell_index):
        if cell.cell_type != 'code':
            return
        stage = cell.metadata.get('progress', {})
        state.update(stage, updated=time.time(), cell_index=cell_index, error=None)
        write_json(ROOT/'work/local-mps/notebook-status.json', state)
        print(f"[{time.strftime('%H:%M:%S')}] {state.get('stage', 'Выполнение')}", flush=True)

    def executed(cell, cell_index, execute_reply):
        save()
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text, end='', flush=True)

    manager = KernelManager(kernel_name='sokolov-mps',
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    client = NotebookClient(notebook, km=manager, timeout=None, kernel_name='sokolov-mps',
        resources={'metadata': {'path': str(ROOT)}}, on_cell_start=starting,
        on_cell_executed=executed, record_timing=True)
    try:
        # Supplying a manager requires explicit cleanup; otherwise the kernel
        # (and its large MPS models) can outlive this process.
        client.execute(cleanup_kc=True)
        state.update(stage='Запуск завершён', stage_index=6, detail='Модели, генерация, отчёт и архив сохранены.',
                     updated=time.time())
        write_json(ROOT/'work/local-mps/notebook-status.json', state)
    except BaseException as error:
        state.update(error=f'{type(error).__name__}: {error}', updated=time.time())
        write_json(ROOT/'work/local-mps/notebook-status.json', state)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
