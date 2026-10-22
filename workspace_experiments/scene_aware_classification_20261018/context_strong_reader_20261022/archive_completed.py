"""Archive completed reference and stronger-reader stages, excluding large caches."""
import gzip
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
ARCHIVE=Path('E:/临时会话/_sar_ship_cross_scene_incremental_git/workspace_experiments/scene_aware_classification_20261018')
STAGES=['context_reference_20261022','context_strong_reader_20261022']
for name in STAGES:
    source=ROOT/name
    assert (source/'FINDINGS_ZH.md').exists()
    selected=list(source.iterdir())
    if name=='context_reference_20261022':
        selected+=list((source/'moments').iterdir())+list((source/'moments/panels').glob('*.png'))
    else:
        selected+=list((source/'geometry_probe').iterdir())
    count=0
    for path in selected:
        if not path.is_file() or path.suffix not in {'.py','.md','.json','.csv','.png'}:
            continue
        out=ARCHIVE/name/path.relative_to(source)
        out.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix=='.csv' and path.stat().st_size>1_000_000:
            with path.open('rb') as a,gzip.open(str(out)+'.gz','wb',compresslevel=6) as b:
                shutil.copyfileobj(a,b)
        else:shutil.copy2(path,out)
        count+=1
    print(name,count,'files')
