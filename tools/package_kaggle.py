"""Build a private Kaggle bundle with source and manifest-referenced photos only."""
import csv
from pathlib import Path
import zipfile


def package(root, destination):
    root = Path(root).resolve()
    paths = [root/name for name in ['pyproject.toml', 'requirements-kaggle.txt', 'README.md']]
    for name in ['src', 'configs', 'docs', 'notebooks', 'tests', 'tools',
                 'data/raw/sokolov_positive_v1', 'data/sources']:
        paths += [p for p in (root/name).rglob('*') if p.is_file() and
                  '__pycache__' not in p.parts and p.suffix != '.pyc']
    manifest = root/'data/manifests/domain_v3/web.csv'
    if not manifest.is_file():
        raise FileNotFoundError('Build extra photos first: python tools/build_domain_dataset.py')
    paths.append(manifest)
    with manifest.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            photo = root/row['path']
            if photo.is_symlink() or not photo.resolve().is_relative_to(root/'data/raw/sokolov_domain_v3'):
                raise ValueError('Extra photo must be inside data/raw/sokolov_domain_v3')
            if not photo.is_file():
                raise FileNotFoundError(photo)
            paths.append(photo)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.write(path, path.relative_to(root))
    return destination


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    print(package(root, root/'work/project_bundle.zip'))
