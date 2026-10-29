"""Diagnostic only: how much continuation acts like a query-common class shift?"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rows = []
for dataset in ('private','oss2'):
    for seed in (17,29,43):
        for arm in ('adapt_only','source_mix'):
            a = np.load(HERE/dataset/'predictions'/f'{arm}_s{seed}.npz')
            old, new, y = a['baseline'].astype(float), a['score'].astype(float), a['y']
            delta = new-old
            # Remove softmax-invisible all-class translation before measuring energy.
            delta -= delta.mean(1,keepdims=True)
            shared = delta.mean(0,keepdims=True)
            simple = old+shared
            actual, common, initial = new.argmax(1), simple.argmax(1), old.argmax(1)
            change = actual != initial
            ba = lambda p: 100*np.mean([np.mean(p[y==c]==c) for c in np.unique(y)])
            rows.append(dict(dataset=dataset,seed=seed,arm=arm,
                shared_logit_energy_fraction=float(len(y)*np.square(shared).sum()/np.square(delta).sum()),
                actual_changed=int(change.sum()), common_changed=int((common!=initial).sum()),
                same_prediction=int((actual==common).sum()), count=len(y),
                same_changed=int((change & (actual==common)).sum()),
                actual_oa=100*np.mean(actual==y), actual_ba=ba(actual),
                common_oa=100*np.mean(common==y),common_ba=ba(common)))
out = HERE/'diagnostic_summary'
out.mkdir(exist_ok=True)
with (out/'shared_shift.csv').open('w',encoding='utf-8',newline='') as f:
    w = csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader();w.writerows(rows)
lines = ['# Query-common shift diagnostic','',
         'Not a deployable candidate: the common offset is estimated on development predictions, without labels.',
         'Energy fraction concerns changes in centered logits, not physical information or explained classification accuracy.',
         'This measures the new training-induced change, not the entire original M3 contribution.','',
         '|Data|Seed|Arm|Shared energy|Same prediction|Same among changed|Actual OA/BA|Common-only OA/BA|',
         '|---|---|---|---|---|---|---|---|']
for r in rows:
    lines.append(f"|{r['dataset']}|{r['seed']}|{r['arm']}|{r['shared_logit_energy_fraction']:.6f}|"
                 f"{r['same_prediction']}/{r['count']}|{r['same_changed']}/{r['actual_changed']}|"
                 f"{r['actual_oa']:.6f}/{r['actual_ba']:.6f}|{r['common_oa']:.6f}/{r['common_ba']:.6f}|")
(out/'SHARED_SHIFT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
