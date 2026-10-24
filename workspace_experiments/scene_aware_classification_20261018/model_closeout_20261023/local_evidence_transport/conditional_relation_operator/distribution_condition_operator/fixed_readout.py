"""Use fixed conditional evidence directly, without subtracting its identity score."""
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
OUT=HERE/'fixed_readout'
spec=importlib.util.spec_from_file_location('fixed_distribution_run',HERE/'run.py')
run=importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
exp=run.exp
epi=run.prior.pool.epi


def feature(reader,ids):
    I=torch.eye(32,device='cuda')
    chunks=[]
    with torch.no_grad():
        for first in range(0,len(ids),128):
            chunks.append(reader.scores(ids[first:first+128],I,I)[0])
    return torch.cat(chunks)


def main():
    tick=time.perf_counter()
    OUT.mkdir(exist_ok=True)
    table,x,mass,old,idx,tr,va,te=exp.source_data()
    y=table.class_id.to_numpy(int)
    _,_,legal,_=exp.edge.edges(x,mass)
    banks,_=epi.diag.banks_from_source(table,mass,legal,old)
    schedule=epi.schedule(table,mass,old,legal)
    metrics,pairs,classes,curves,receipts,scales=[],[],[],[],[],[]
    weights={}
    for output in ('linear','rbf'):
        reader=run.Reader(x,mass,old,'excess',output_kernel=output)
        rms,count=reader.initialize(idx[tr])
        scales.extend(dict(output=output,group=g,divisor=float(reader.divisor[j]),rms=float(rms[j]),n=int(count[j])) for j,g in enumerate(exp.edge.GROUPS))
        training=[]
        for j,bank in enumerate(schedule):
            epi.reset_bank(reader,bank)
            training.append(feature(reader,idx[tr]))
            if (j+1)%50==0:
                print('CACHE',output,j+1,flush=True)
        features={}
        for name,bank in banks.items():
            epi.reset_bank(reader,bank)
            features[name]=feature(reader,idx)
        target=torch.tensor(y[idx[tr]],device='cuda')
        counts=np.bincount(y[idx[tr]],minlength=8)
        lossweight=torch.tensor(.5/len(tr)+.5/(8*counts[y[idx[tr]]]),device='cuda',dtype=torch.float32)
        for seed in exp.SEEDS:
            full=np.load(exp.HERE/f'predictions_{seed}.npz')['full'][idx]
            baseline=torch.tensor(full,device='cuda')
            a=torch.nn.Parameter(torch.zeros(4,device='cuda'))
            opt=torch.optim.Adam([a],lr=.01)
            best=exp.dep.exact_j(y[idx[va]],full[va].argmax(-1))
            stepbest=0; saved_a=a.detach().clone()
            for step in range(201):
                if step:
                    opt.zero_grad(set_to_none=True)
                    score=baseline[tr]+exp.delta(training[step-1],a)
                    loss=(F.cross_entropy(score,target,reduction='none')*lossweight).sum()+.01*a.square().sum()
                    loss.backward(); opt.step()
                if step not in exp.CHECKPOINTS:
                    continue
                with torch.no_grad():
                    predictions={name:(baseline[va]+exp.delta(z[va],a)).cpu().numpy() for name,z in features.items()}
                if step==0:
                    for score in predictions.values():
                        np.testing.assert_array_equal(score,full[va])
                criterion=sum(exp.dep.exact_j(y[idx[va]],s.argmax(-1)) for s in predictions.values())/len(predictions)
                curves.append(dict(output=output,seed=seed,step=step,selection_J=float(criterion)))
                if criterion>best:
                    best=criterion; stepbest=step; saved_a=a.detach().clone()
            weights[f'{output}_{seed}_a']=saved_a.cpu().numpy()
            saved=dict(idx=idx,y=y[idx],full=full)
            for name,z in features.items():
                with torch.no_grad():
                    score=(baseline+exp.delta(z,saved_a)).cpu().numpy()
                saved[name]=score
                if name=='original':
                    saved['parts']=(z*saved_a).cpu().numpy()
                for role,locs in [('utility_train',tr),('selection',va),('evaluation',te)]:
                    meta=dict(output=output,seed=seed,bank=name,role=role,step=stepbest)
                    metrics.append(dict(**meta,**exp.dep.metrics(y[idx[locs]],score[locs])))
                    before=full[locs].argmax(-1)==y[idx[locs]]
                    after=score[locs].argmax(-1)==y[idx[locs]]
                    pairs.append(dict(**meta,rescued=int((~before&after).sum()),harmed=int((before&~after).sum()),net=int(after.sum()-before.sum())))
                    for c in range(8):
                        mask=y[idx[locs]]==c
                        classes.append(dict(**meta,class_id=c,n=int(mask.sum()),correct=int(after[mask].sum()),full_correct=int(before[mask].sum())))
            np.savez_compressed(OUT/f'predictions_{output}_{seed}.npz',**saved)
            receipts.append(dict(output=output,seed=seed,step=stepbest,selection_J=float(best),parameters=4))
            print('SELECTED',output,seed,stepbest,exp.dep.metrics(y[idx[te]],saved['original'][te]),flush=True)
        del training,features
    for name,rows in [('metrics',metrics),('paired',pairs),('classes',classes),('curves',curves),('receipts',receipts),('scales',scales)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    np.savez_compressed(OUT/'weights.npz',**weights)
    frame=pd.DataFrame(metrics)
    summary=frame[frame.role.eq('evaluation')].groupby(['output','bank'])[['OA','BA','J','NLL','Brier']].mean()
    summary.to_csv(OUT/'summary.csv')
    (OUT/'execution.json').write_text(json.dumps(dict(fits=6,steps=200,parameters=4,initial_zero=True,
        identity_maps=True,source_selection='mean six-bank exact J',seconds=time.perf_counter()-tick),indent=2),encoding='utf-8')
    print(summary.to_string())


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    main()
