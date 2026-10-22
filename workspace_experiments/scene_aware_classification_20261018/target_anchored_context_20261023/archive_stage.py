"""Archive completed anchored-reader artifacts, never feature or weight caches."""
import gzip
import shutil
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
ARCHIVE = Path('E:/临时会话/_sar_ship_cross_scene_incremental_git/workspace_experiments/scene_aware_classification_20261018')


def archive(folder, source=SOURCE):
    if not (folder / 'FINDINGS_ZH.md').exists():
        print('pending, not archived:', folder.name)
        return
    count = 0
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix not in {'.py', '.md', '.json', '.csv', '.png'}:
            continue
        out = ARCHIVE / source.name / path.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.csv' and path.stat().st_size > 1_000_000:
            with path.open('rb') as a, gzip.open(str(out) + '.gz', 'wb', compresslevel=6) as b:
                shutil.copyfileobj(a, b)
        else:
            shutil.copy2(path, out)
        count += 1
    print(folder.name, count, 'artifacts')


if __name__ == '__main__':
    archive(SOURCE)
    for name in ['centering_probe', 'partition_replication', 'controller_readout_probe', 'common_gain_diagnosis']:
        folder = SOURCE / name
        if folder.exists():
            archive(folder)
    eight = SOURCE.parent / 'eight_class_evidence_20261023'
    if eight.exists():
        archive(eight, source=eight)
