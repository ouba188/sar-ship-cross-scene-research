"""Separate changed source learning from identical fixed-input retraining."""
from pathlib import Path
import json
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'geometry_targeted_fix_20261016'


def main():
    assert (HERE/'SOURCE_FROZEN.json').exists()
    records=[]
    for stage,family,arm in [('bounded','strict','geometry'),('recovered','optional','conditional_rank')]:
        for path in (HERE/stage/'trained').glob(f'*/{arm}_seed*.pt'):
            before=torch.load(OLD/stage/'trained'/path.parent.name/path.name,map_location='cpu',weights_only=False)
            after=torch.load(path,map_location='cpu',weights_only=False)
            previous_unavailable=before.get('unavailable',False)
            unavailable=after.get('unavailable',False)
            row=dict(pool=path.parent.name.replace('_',' '),family=family,seed=int(path.stem.split('_seed')[1]),
                old_unavailable=previous_unavailable,new_unavailable=unavailable,seconds=after.get('seconds',0),
                old_fit_tasks=before['inventory']['tasks'],new_fit_tasks=after['inventory']['tasks'])
            if not previous_unavailable and not unavailable:
                diff={k:float((after['state'][k]-v).abs().max()) for k,v in before['state'].items()}
                row.update(identical_state=all(v==0 for v in diff.values()),max_parameter_delta=max(diff.values()),
                           condition_delta=diff['condition.weight'])
                previous=path.with_name(path.stem+'_sampling.json')
                oldtrace=OLD/stage/'trained'/path.parent.name/previous.name
                row['same_sampling']=json.loads(previous.read_text())==json.loads(oldtrace.read_text())
            records.append(row)
    result=pd.DataFrame(records)
    result.to_csv(HERE/'diagnostics/training_comparison.csv',index=False)
    print(result.to_string(index=False))


if __name__=='__main__':
    main()
