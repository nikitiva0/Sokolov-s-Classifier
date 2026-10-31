"""Start the local photo UI from any working directory."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'work/matplotlib'))
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
sys.path.insert(0, str(ROOT/'src'))

if __name__ == '__main__':
    from sokolov.__main__ import main
    raise SystemExit(main(['serve', *sys.argv[1:]]))
