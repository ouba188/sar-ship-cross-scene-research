"""Archive completed small evidence without datasets, projections or weights."""
from pathlib import Path
import gzip
import shutil
import argparse
SOURCE=Path(__file__).resolve().parent
DEST=Path('E:/临时会话/_sar_ship_cross_scene_incremental_git/workspace_experiments/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport')


def main(external=False):
    count=0
    for path in sorted(SOURCE.rglob('*')):
        rel=path.relative_to(SOURCE)
        if not path.is_file() or 'models' in rel.parts or '__pycache__' in rel.parts:continue
        if not external and 'external_confirmation' in rel.parts:continue
        if path.suffix not in {'.py','.md','.csv','.json','.png'}:continue
        out=DEST/rel;out.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix=='.csv' and path.stat().st_size>1_000_000:
            with path.open('rb') as a,gzip.open(str(out)+'.gz','wb',compresslevel=6) as b:
                shutil.copyfileobj(a,b)
        else:shutil.copy2(path,out)
        count+=1
    print('Archived',count,'small scoped files; no PT/NPZ/raw chips.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--external',action='store_true')
    main(parser.parse_args().external)
