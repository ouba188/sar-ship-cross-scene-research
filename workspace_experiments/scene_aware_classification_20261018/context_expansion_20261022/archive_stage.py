"""Copy only this completed stage's small research artifacts to the existing archive."""
import gzip
import shutil
from pathlib import Path
import pandas as pd

SOURCE = Path(__file__).resolve().parent
DEST = Path('E:/临时会话/_sar_ship_cross_scene_incremental_git/workspace_experiments/scene_aware_classification_20261018')/SOURCE.name
DEST.mkdir(parents=True, exist_ok=True)


def copy(path):
    out = DEST/path.relative_to(SOURCE)
    out.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.csv' and path.stat().st_size > 1_000_000:
        with path.open('rb') as src, gzip.open(str(out)+'.gz', 'wb', compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy2(path, out)


assert (SOURCE/'FINDINGS_ZH.md').exists()
for path in SOURCE.iterdir():
    if path.is_file() and path.suffix in {'.py', '.md', '.csv', '.json', '.png'}:
        copy(path)
for path in (SOURCE/'readout').iterdir():
    if path.is_file() and path.suffix in {'.py', '.md', '.csv', '.json'} and path.name != 'excluded.csv':
        copy(path)
for path in (SOURCE/'literature').glob('*.md'):
    copy(path)
for path in (SOURCE/'readout/review_panels').glob('*.png'):
    copy(path)
excluded = pd.read_csv(SOURCE/'readout/excluded.csv', low_memory=False)
if 'reason' in excluded:
    excluded.reason.value_counts(dropna=False).rename_axis('reason').reset_index(name='rows').to_csv(DEST/'readout/excluded_counts.csv', index=False)
files = [x for x in DEST.rglob('*') if x.is_file()]
assert not any(x.suffix in {'.npz', '.npy', '.pt', '.pth', '.tif'} for x in files)
print('archived', len(files), 'files', sum(x.stat().st_size for x in files), 'bytes')
