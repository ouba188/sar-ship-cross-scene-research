"""Locate source-selection versus held-acquisition differences without refitting."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import HERE, old, op, setup, load, predict, restore, model_for
from analyze import result


@torch.no_grad()
def main():
    setup()
    chosen = json.loads((HERE/'source_selected.json').read_text())
    stage, gate = chosen['stage'], chosen['gate']
    table, arrays = load(stage)
    rows, summaries = [], []
    for seed in old.SEEDS:
        model = restore(stage, gate, seed)
        if stage == 1:
            baseline = restore(stage, 'none', seed)
        else:
            baseline = model_for(stage, 'none').eval()
            state = torch.load(old.HERE/'training'/f'none_{seed}'/'selected.pt',weights_only=True,map_location='cpu')['state']
            baseline.load_state_dict(state,strict=False)
        for role in ('fit_validation','evaluation'):
            ids = np.flatnonzero(table.role.eq(role))
            if role == 'evaluation':
                before, after = result(stage,'none',seed), result(stage,gate,seed)
                y, sb, sa = after['y'], before['score'], after['score']
            else:
                y = table.iloc[ids].class_id.to_numpy(int)
                sb, sa = predict(baseline,arrays,ids,stage), predict(model,arrays,ids,stage)
            cb, ca = sb.argmax(1)==y, sa.argmax(1)==y
            for arm, score in (('none',sb),(gate,sa)):
                summaries.append(dict(role=role,seed=seed,arm=arm,**op.metrics(y,score)))
            for c in range(8):
                use = y == c
                rows.append(dict(role=role,seed=seed,class_id=c,n=int(use.sum()),
                    before=float(cb[use].mean()),after=float(ca[use].mean()),
                    rescued=int((~cb[use] & ca[use]).sum()),harmed=int((cb[use] & ~ca[use]).sum())))
        del baseline, model
    pd.DataFrame(rows).to_csv(HERE/'source_gap_by_class.csv',index=False)
    pd.DataFrame(summaries).to_csv(HERE/'source_gap_metrics.csv',index=False)
    print(pd.DataFrame(rows).groupby(['role','class_id'])[['n','before','after','rescued','harmed']].mean().to_string())
    print(pd.DataFrame(summaries).groupby(['role','arm'])[['OA','BA','J']].mean().to_string())


if __name__ == '__main__':
    main()
