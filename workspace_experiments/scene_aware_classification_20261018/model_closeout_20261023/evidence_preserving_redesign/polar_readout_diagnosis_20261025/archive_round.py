"""Archive scoped small research artifacts, excluding models and raw caches."""
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
DEST=Path('E:/临时会话/_sar_ship_cross_scene_incremental_git/workspace_experiments/scene_aware_classification_20261018/model_closeout_20261023/evidence_preserving_redesign/polar_readout_diagnosis_20261025')

def main():
    copied=[]
    for path in sorted(HERE.rglob('*')):
        rel=path.relative_to(HERE)
        if not path.is_file() or any(x in ('models','__pycache__') or x.startswith('.') for x in rel.parts):continue
        if path.suffix not in ('.py','.md','.csv','.json','.png','.gz'):continue
        if path.stat().st_size>8_000_000:raise ValueError(f'Inspect oversized artifact: {rel}')
        out=DEST/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
        copied.append((str(rel),path.stat().st_size))
    print('Scoped files',len(copied),'bytes',sum(s for _,s in copied))
    for name,size in copied:print(size,name)

if __name__=='__main__':main()
