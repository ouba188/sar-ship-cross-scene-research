"""Copy only rebuilt M1/M2 inference assets; never train or reselect heads."""
import argparse
from pathlib import Path
import shutil


def export(legacy_root: Path, assets_root: Path):
    source = Path(legacy_root)/'regional_utility_pilot'/'upstream'
    destination = Path(assets_root)/'upstream'
    names = ['scales.json', 'models/reader_layer1.npz', 'models/reader_layer3.npz']
    names += [f'models/{stage}_seed{seed}.pt'
              for seed in (17, 29, 43) for stage in ('attention', 'full')]
    missing = [str(source/name) for name in names if not (source/name).is_file()]
    if missing:
        raise FileNotFoundError('Missing rebuilt assets: '+', '.join(missing))
    exported = []
    for name in names:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
        exported.append(target)
    return exported


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', required=True, type=Path)
    parser.add_argument('--assets-root', required=True, type=Path)
    args = parser.parse_args()
    for path in export(args.legacy_root, args.assets_root):
        print(path)
