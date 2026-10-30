"""Package only source/configs/notebook and the already published positive dataset."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
destination = root/'work/project_bundle.zip'
destination.parent.mkdir(exist_ok=True)
paths = [root/name for name in ['src', 'configs', 'docs', 'notebooks', 'tests', 'tools',
                               'data/raw/sokolov_positive_v1']]
with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ['pyproject.toml', 'requirements-kaggle.txt', 'README.md']:
        archive.write(root/name, name)
    for folder in paths:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                archive.write(path, path.relative_to(root))
print(destination)
