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
            compressed = Path(str(out) + '.gz')
            if compressed.exists():
                with gzip.open(compressed, 'rb') as previous:
                    if previous.read() == path.read_bytes():
                        count += 1
                        continue
            with path.open('rb') as a, gzip.open(str(out) + '.gz', 'wb', compresslevel=6) as b:
                shutil.copyfileobj(a, b)
        else:
            shutil.copy2(path, out)
        count += 1
    print(folder.name, count, 'artifacts')


if __name__ == '__main__':
    archive(SOURCE)
    for name in ['centering_probe', 'partition_replication', 'controller_readout_probe', 'common_gain_diagnosis',
                 'common_gain_training', 'score_scale_decoupling', 'linear_probe_control']:
        folder = SOURCE / name
        if folder.exists():
            archive(folder)
    eight = SOURCE.parent / 'eight_class_evidence_20261023'
    if eight.exists():
        archive(eight, source=eight)
        for name in ['baseline', 'gated_transfer']:
            folder = eight / name
            if (folder / 'FINDINGS_ZH.md').exists():
                archive(folder, source=eight)
                review = folder / 'review'
                if review.exists():
                    for path in review.glob('*.png'):
                        out = ARCHIVE / eight.name / path.relative_to(eight)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, out)
