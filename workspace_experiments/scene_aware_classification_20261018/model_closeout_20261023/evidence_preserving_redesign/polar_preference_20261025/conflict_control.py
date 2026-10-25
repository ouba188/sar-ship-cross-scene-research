"""One bounded source-conflict supervision correction; preserve the first round."""
import importlib.util
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('preference_train',HERE/'train.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
import time
import numpy as np
import pandas as pd
import torch

ARMS=('conflict_spatial','conflict_freq')


def main():
    torch.set_num_threads(2)
    data=run.Data();a,b=run.folds(data);y=data.table.iloc[:data.n].class_id.to_numpy(int)
    receipts,cv,proof=[] ,[],[]
    for seed in run.SEEDS:
        raw,z=run.inputs(data,seed)
        for name,ids in (('A',a),('B',b),('all',data.va)):
            yt=torch.tensor(y[ids]);v=(z['base']+z['delta_vv'])[ids].argmax(1)==yt
            h=(z['base']+z['delta_vh'])[ids].argmax(1)==yt
            mask=(v!=h)&z['available'][ids].all(1)
            assert ((v&mask).sum()>0) and ((h&mask).sum()>0)
            proof.append(dict(seed=seed,fold=name,paired_conflicts=int(mask.sum()),VV=int((v&mask).sum()),VH=int((h&mask).sum())))
        for arm in ARMS:
            start=time.perf_counter()
            oof={s:[np.zeros((data.n,8),np.float32),np.zeros((data.n,2),np.float32),
                    np.zeros(data.n,np.float32),np.zeros(data.n,np.float32)] for s in run.STEPS}
            for name,tr,va in (('A_to_B',a,b),('B_to_A',b,a)):
                _,_,snapshots,curve=run.train_steps(arm,seed,raw,z,y,tr,va)
                for step,values in snapshots.items():
                    for dest,value in zip(oof[step],values):dest[va]=value
                pd.DataFrame(curve).to_csv(HERE/'results'/f'curve_{arm}_{seed}_{name}.csv',index=False)
            chosen=0;best=run.prev.common.exact_j(y[data.va],oof[0][0][data.va].argmax(1))
            for step in run.STEPS:
                s=oof[step][0][data.va];j=run.prev.common.exact_j(y[data.va],s.argmax(1))
                cv.append(dict(seed=seed,arm=arm,step=step,**run.metrics(y[data.va],s)))
                if j>best:chosen,best=step,j
            model,scaler,_,_=run.train_steps(arm,seed,raw,z,y,data.va,data.va,total=chosen)
            receipt=dict(seed=seed,arm=arm,selected_step=chosen,source_cv_J=float(best),
                         parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start)
            torch.save(dict(state=model.state_dict(),scaler=scaler,receipt=receipt),HERE/'models'/f'{arm}_{seed}.pt')
            np.savez_compressed(HERE/'results'/f'source_cv_{arm}_{seed}.npz',
                **{k:v[data.va] for k,v in zip(('score','weight','strength','preference'),oof[chosen])},up_index=data.up[data.va])
            receipts.append(receipt);print('FIT',receipt,flush=True)
    pd.DataFrame(receipts).to_csv(HERE/'conflict_training_receipts.csv',index=False)
    pd.DataFrame(cv).to_csv(HERE/'conflict_source_cv.csv',index=False)
    run.write_json(HERE/'conflict_choices.json',dict(choices=receipts,training_targets=proof,
        new_development_metrics_computed=False,exploratory_after_initial_round=True))
    rows=[]
    for seed in run.SEEDS:
        raw,z=run.inputs(data,seed)
        for arm in ARMS:
            model,x,ck=run.restore(seed,arm,raw)
            with torch.no_grad():
                values=tuple(v.numpy() for v in run.prediction(model,x,z,np.arange(data.n)))
                diagnostics={mode:run.prediction(model,x,z,np.arange(data.n),mode)[0].numpy()
                             for mode in ('no_preference','fixed_strength')}
                if arm=='conflict_freq':
                    changed=x.clone();changed[...,41:]=0
                    diagnostics['no_frequency']=run.prediction(model,changed,z,np.arange(data.n))[0].numpy()
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',
                **dict(zip(('score','weight','strength','preference'),values)),up_index=data.up[:data.n])
            np.savez_compressed(HERE/'results'/f'diagnostics_{arm}_{seed}.npz',**diagnostics,up_index=data.up[:data.n])
            rows.append(dict(seed=seed,arm=arm,role='evaluation',**run.metrics(y[data.te],values[0][data.te])))
    frame=pd.DataFrame(rows);frame.to_csv(HERE/'conflict_metrics.csv',index=False)
    run.write_json(HERE/'conflict_complete.json',dict(status='complete',cv_fits=12,final_refits=6,
        new_models_frozen_before_new_dev_metrics=True,exploratory_not_independent=True))
    print(frame.groupby('arm')[['OA','BA','NLL','Brier']].mean().to_string(),flush=True)


if __name__=='__main__':
    main()
