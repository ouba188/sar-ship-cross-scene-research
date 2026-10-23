"""Freeze the effective marginal reader and fit only extra dependence terms."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
from run import fit, UP, metrics
ARMS=('nonlinear','joint','shuffle101','shuffle211','shuffle307')


def main():
    tick=time.perf_counter()
    torch.set_num_threads(2)
    data=np.load(PARENT/'features.npz')
    idx,bank=data['idx'],data['bank']
    table=pd.read_csv(UP/'table.csv')
    role=table.iloc[idx].role.to_numpy()
    y=table.iloc[idx].class_id.to_numpy()
    tr,va,te=[np.flatnonzero(role==r) for r in ('utility_train','selection','evaluation')]
    rows,curves,receipts,pairs,classes=[],[],[],[],[]
    weights={}
    for seed in (17,29,43):
        old=np.load(PARENT.parent/f'query_evidence_pooling/predictions_{seed}.npz')
        base=old['joint'][idx]
        saved=dict(anchor=old['joint'],full=old['full'])
        for arm in ARMS:
            score,delta,w,states,history,step,error=fit(data[arm][...,6:],base,y,tr,va)
            weights[f'{arm}_{seed}']=w
            weights[f'{arm}_{seed}_checkpoints']=states
            saved[arm]=old['joint'].copy()
            saved[arm][idx]=score
            receipts.append(dict(seed=seed,arm=arm,step=step,parameters=len(w),replay_error=error))
            curves.extend(dict(seed=seed,arm=arm,**r) for r in history)
        correct0=saved['anchor'][idx[te]].argmax(-1)==y[te]
        for arm,s in saved.items():
            for r,select in [('utility_train',tr),('selection',va),('evaluation',te)]:
                rows.append(dict(seed=seed,arm=arm,role=r,**metrics(y[select],s[idx[select]])))
            correct=s[idx[te]].argmax(-1)==y[te]
            pairs.append(dict(seed=seed,arm=arm,rescued=int((correct&~correct0).sum()),harmed=int((~correct&correct0).sum()),net=int(correct.sum()-correct0.sum())))
            for c in range(8):
                mask=y[te]==c
                classes.append(dict(seed=seed,arm=arm,class_id=c,n=int(mask.sum()),correct=int(correct[mask].sum())))
        np.savez_compressed(HERE/f'predictions_{seed}.npz',idx=idx,bank=bank,**saved)
    np.savez_compressed(HERE/'weights.npz',**weights)
    for name,result in [('metrics',rows),('training_receipts',receipts),('selection_curve',curves),('paired_results',pairs),('class_results',classes)]:
        pd.DataFrame(result).to_csv(HERE/f'{name}.csv',index=False)
    summary=pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean()
    summary.to_csv(HERE/'summary.csv')
    receipt=dict(seconds=time.perf_counter()-tick,fits=len(receipts),frozen_marginal=True,new_parameters=7,
                 independent_test=False,source_only=True,max_replay_error=max(r['replay_error'] for r in receipts))
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(summary.to_string())
    print(pd.DataFrame(receipts).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
