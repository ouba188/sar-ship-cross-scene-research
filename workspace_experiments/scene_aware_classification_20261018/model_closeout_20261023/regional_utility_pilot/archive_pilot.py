"""Publish this completed pilot's small evidence, excluding weights and arrays."""
from pathlib import Path
import gzip
import shutil

SOURCE = Path(__file__).resolve().parent
DEST = Path('E:/临时会话/_sar_ship_cross_scene_incremental_git/workspace_experiments/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot')


def main():
    assert (SOURCE/'FINDINGS_ZH.md').exists()
    count = 0
    for path in sorted(SOURCE.rglob('*')):
        rel = path.relative_to(SOURCE)
        if not path.is_file() or 'models' in rel.parts or '__pycache__' in rel.parts:
            continue
        if path.suffix not in {'.py', '.md', '.csv', '.json', '.png'}:
            continue
        out = DEST/rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.csv' and path.stat().st_size > 1_000_000:
            with path.open('rb') as src, gzip.open(str(out)+'.gz', 'wb', compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy2(path, out)
        count += 1
    print('Archived', count, 'pilot artifacts; no model/array caches.')


if __name__ == '__main__':
    main()
