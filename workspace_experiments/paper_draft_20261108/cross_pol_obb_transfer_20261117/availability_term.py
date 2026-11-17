"""Exact linear-head availability-term subtraction; no network forwards or fitting."""
import gc
import numpy as np
import torch
import run

ROOT=run.ROOT
recovery=dict(np.load(ROOT/'recovered_inputs.npz'))
pol={int(i):int(p) for i,p in zip(recovery['ids'],recovery['pol'])}
results=[]


def metrics(y,z):
    pred=z.argmax(1)
    recalls=[np.mean(pred[y==k]==k) for k in range(4)]
    return dict(oa=float(100*np.mean(pred==y)),ba=float(100*np.mean(recalls)))


for seed in (17,29,43):
    state=torch.load(ROOT.parent/'full_finetune_20261111'/f'selected_s{seed}'/'best.pt',
                     map_location='cpu',weights_only=True,mmap=True)
    w=state['net.guide.head.weight'][:,-2:].numpy().copy()
    del state;gc.collect()
    for split in ('validation','evaluation'):
        a=dict(np.load(ROOT/f'{split}_s{seed}.npz'))
        term=np.zeros_like(a['closed_full'])
        for j,idx in enumerate(a['ids']):
            if int(idx) in pol:term[j]=w[:,pol[int(idx)]]
        bit_only=a['closed_full']+term
        z={name:a[name]-term for name in ('global_full','transfer_full')}
        results.append(dict(seed=seed,split=split,
            head_availability_weights=w.tolist(),
            bit_only_metrics=metrics(a['label'],bit_only),
            bit_only_paired=run.paired(a['label'],bit_only,a['closed_full']),
            removed_metrics={k:metrics(a['label'],v) for k,v in z.items()},
            removed_paired={k:run.paired(a['label'],v,a['closed_full']) for k,v in z.items()},
            removed_top1_changes={k:int((v.argmax(1)!=a[k].argmax(1)).sum()) for k,v in z.items()},
            max_removed_logit=float(abs(term).max())))
run.write(ROOT/'availability_term.json',dict(results=results,
    note='Exact subtraction of the final linear head availability coefficient only. Other availability gating stays as in original evaluated policy. No retraining, no additional network forward.'))
for split in ('validation','evaluation'):
    part=[r for r in results if r['split']==split]
    print(split,'bit-only OA',np.mean([r['bit_only_metrics']['oa'] for r in part]),
          'without bit',[(k,np.mean([r['removed_metrics'][k]['oa'] for r in part])) for k in ('global_full','transfer_full')],flush=True)
