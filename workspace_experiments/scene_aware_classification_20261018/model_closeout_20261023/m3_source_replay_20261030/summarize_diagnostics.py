"""Small CPU summaries of gradient and norm-matched controls."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'diagnostic_summary'
OUT.mkdir(exist_ok=True)
rows = []
for dataset in ('private','oss2'):
    for seed in (17,29,43):
        path = ROOT/dataset/'diagnostics'/f'norm_control_s{seed}.json'
        if not path.exists():
            continue
        rec = json.loads(path.read_text(encoding='utf-8'))
        for arm in ('saved_baseline','adapt_only','source_mix','norm_control'):
            m = rec['baseline'] if arm == 'saved_baseline' else rec['after'] if arm == 'norm_control' else json.loads(
                (ROOT/dataset/'training'/f'{arm}_s{seed}.json').read_text(encoding='utf-8'))['after']
            rows.append(dict(dataset=dataset,seed=seed,arm=arm,ratio=rec['ratio'] if arm=='norm_control' else '',**m))
with (OUT/'norm_control.csv').open('w',encoding='utf-8',newline='') as f:
    w = csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader();w.writerows(rows)
lines = ['# Gradient and update-magnitude diagnostic','',
         '|Dataset|Model|Orders|OA|BA|NLL|Brier|','|---|---|---|---|---|---|---|']
for dataset in ('private','oss2'):
    for arm in ('saved_baseline','adapt_only','source_mix','norm_control'):
        rr = [r for r in rows if r['dataset']==dataset and r['arm']==arm]
        if not rr:
            continue
        lines.append('|'+dataset+'|'+arm+'|'+str(len(rr))+'|'+
                     '|'.join(f'{np.mean([r[k] for r in rr]):.6f}' for k in ('oa','ba','nll','brier'))+'|')
lines += ['','Gradient comparisons are at seed17 complete-model endpoints; no development gradient enters training.',
          'A cosine is a local direction diagnostic, not a classification result or a causal explanation.']
for dataset in ('private','oss2'):
    g = json.loads((ROOT/dataset/'diagnostics/gradients_s17.json').read_text(encoding='utf-8'))
    lines += ['',dataset+':',json.dumps(g['total_cosines']),
              'Matched present classes: '+json.dumps(g['matched_class_cosines'])]
(OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
