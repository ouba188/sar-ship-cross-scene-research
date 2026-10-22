"""Fixed-reader dependence on own foreground/background controller inputs."""
import numpy as np
import pandas as pd
import torch
from run_anchored import HERE,EXP,Anchored,SEEDS,metrics

torch.set_num_threads(2)
t=pd.read_csv(EXP/'readout/targets.csv')
y=t.class_id.to_numpy()
rows=[]
for fold in range(6):
    d=np.load(HERE/f'fold{fold}_inputs.npz')
    own=np.load(HERE/f'fold{fold}_own_input.npz')['env']
    train,test=d['train'],d['test']
    center=own[train].mean(0)
    q=torch.tensor(d['q'][test],dtype=torch.float32,device='cuda')
    valid=torch.ones(len(test),device='cuda')
    for seed in SEEDS:
        model=Anchored().cuda().eval()
        model.load_state_dict(torch.load(HERE/'models'/f'fold{fold}_own_regions_full_{seed}.pt',weights_only=True,map_location='cpu'))
        for arm,columns in [('original',[]),('foreground_mean',list(range(32))),('background_mean',list(range(32,64))),('both_mean',list(range(64)))]:
            env=own[test].copy()
            env[:,columns]=center[columns]
            with torch.no_grad():score=d['base'][test]+model(torch.tensor(env,dtype=torch.float32,device='cuda'),q,valid).cpu().numpy()
            for k,i in enumerate(test):rows.append(dict(fold=fold,seed=seed,arm=arm,instance_id=t.iloc[i].instance_id,
                class_id=int(y[i]),prediction=int(score[k].argmax()),base_prediction=int(d['base'][i].argmax()),
                **{f's{c}':float(score[k,c]) for c in range(3)}))
p=pd.DataFrame(rows)
p.to_csv(HERE/'role_intervention_predictions.csv',index=False)
ref=pd.read_csv(HERE/'control_predictions.csv').query('arm == "own_regions_full"').set_index(['seed','instance_id'])
original=p[p.arm.eq('original')].set_index(['seed','instance_id']).reindex(ref.index)
assert np.array_equal(original.prediction,ref.prediction)
assert np.allclose(original[['s0','s1','s2']],ref[['s0','s1','s2']],atol=1e-5)
summary=[]
for (arm,seed),q0 in p.groupby(['arm','seed']):
    summary.append(dict(arm=arm,seed=seed,**metrics(q0.class_id.to_numpy(),q0[['s0','s1','s2']].to_numpy()),
        net=int(q0.prediction.eq(q0.class_id).sum()-q0.base_prediction.eq(q0.class_id).sum())))
pd.DataFrame(summary).to_csv(HERE/'role_intervention_metrics.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False),flush=True)
