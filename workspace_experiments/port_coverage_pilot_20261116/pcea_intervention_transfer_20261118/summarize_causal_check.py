"""Aggregate paired decisions and class changes, without exporting raw identifiers."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'causal_work_check_20261119'
CLASSES=['bulk_carrier','fishing','general_cargo','chemical_tanker',
         'container','crude_oil_tanker','tug','offshore']


def main():
    arrays={arm:np.load(ROOT/f'{arm}_scene_local.npz') for arm in ('baseline','split_ce','causal')}
    labels=arrays['baseline']['labels']
    rows=[]
    paired=[]
    for name,section in [('development160',slice(0,160)),('development198',slice(160,358))]:
        y=labels[section]
        predictions={arm:data['scores'][section].argmax(1) for arm,data in arrays.items()}
        for c,cls in enumerate(CLASSES):
            present=y==c
            rows.append(dict(cohort=name,category=cls,n=int(present.sum()),
                **{arm:int(((p==y)&present).sum()) for arm,p in predictions.items()}))
        for other in ('baseline','split_ce'):
            a=predictions[other]==y; b=predictions['causal']==y
            paired.append(dict(cohort=name,comparison='causal_vs_'+other,
                rescue=int((~a&b).sum()),harm=int((a&~b).sum()),
                both_correct=int((a&b).sum()),both_wrong=int((~a&~b).sum())))
    with (ROOT/'per_class.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    diagnostic={}
    for arm in arrays:
        records=json.loads((ROOT/f'{arm}_diagnostics.json').read_text(encoding='utf-8'))
        change=[r['changed_mask_entries'] for r in records[1:]]
        receipt=json.loads((ROOT/f'{arm}_s17/receipt.json').read_text(encoding='utf-8'))
        diagnostic[arm]=dict(mean_mask_changes=float(np.mean(change)),max_mask_changes=max(change),
            mean_mask_changes_last30=float(np.mean(change[-30:])),training_seconds=receipt['seconds'])
    (ROOT/'paired_summary.json').write_text(json.dumps(dict(comparisons=paired,optimization=diagnostic),indent=2),encoding='utf-8')
    print(json.dumps(dict(comparisons=paired,optimization=diagnostic),indent=2),flush=True)


if __name__=='__main__':
    main()
