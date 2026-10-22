"""Compare own/external controls and isolate the added class-bias path."""
import json
import numpy as np
import pandas as pd
import torch
from run_anchored import HERE, EXP, STRONG, metrics, SEEDS
from controls import ARMS, WithBias
from analyze_anchored import render, clustered_ci

torch.set_num_threads(2)
t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
edges=pd.read_csv(EXP/'readout/neighbors.csv')
p=pd.read_csv(HERE/'control_predictions.csv')
a=pd.read_csv(HERE/'predictions.csv')
p['net']=p.prediction.eq(p.class_id).astype(int)-p.base_prediction.eq(p.class_id).astype(int)
p['kind']=np.select([p.net.eq(1),p.net.eq(-1),p.prediction.eq(p.class_id)],['rescued','harmed','correct'],default='wrong')
q=p[p.arm.eq('own_regions_full')&p.seed.eq(29)&~p.has_external]
chosen=[]
for kind in ['rescued','harmed','correct','wrong']:
    chosen.extend(q[q.kind.eq(kind)].sort_values('instance_id').head(1).instance_id)
empty=pd.DataFrame(columns=['instance_id','seed','margin_increment'])
render(t,edges,p,chosen,empty,arms=ARMS,prefix='control_case')
p[p.instance_id.isin(chosen)].to_csv(HERE/'control_review_cases.csv',index=False)
p.groupby(['arm','seed','class_id']).agg(n=('instance_id','size'),net=('net','sum')).reset_index().to_csv(HERE/'control_class_effects.csv',index=False)
intervals=[]
for arm,reference in [('own_regions_full','base'),('own_regions_matched','anchored'),('anchored_plus_bias','anchored')]:
    aa=p[p.arm.eq(arm)].set_index(['seed','instance_id'])
    if reference=='base':diff=aa.net
    else:
        bb=a[a.arm.eq(reference)].set_index(['seed','instance_id']).reindex(aa.index)
        diff=aa.prediction.eq(aa.class_id).astype(int)-bb.prediction.eq(bb.class_id).astype(int)
    avg=diff.groupby('instance_id').mean().reindex(t.instance_id).to_numpy()
    for block in ['pass_group','mmsi']:intervals.append(dict(arm=arm,reference=reference,block=block,**clustered_ci(t,avg,block)))
pd.DataFrame(intervals).to_csv(HERE/'control_intervals.csv',index=False)
rows=[]
for fold in range(6):
    d=np.load(HERE/f'fold{fold}_inputs.npz');test=d['test']
    env=torch.tensor(d['env'][test],dtype=torch.float32,device='cuda')
    qq=torch.tensor(d['q'][test],dtype=torch.float32,device='cuda')
    has=torch.tensor(d['has'][test],dtype=torch.float32,device='cuda')
    for seed in SEEDS:
        model=WithBias().cuda().eval()
        model.load_state_dict(torch.load(HERE/'models'/f'fold{fold}_anchored_plus_bias_{seed}.pt',weights_only=True,map_location='cpu'))
        with torch.no_grad():
            full=d['base'][test]+model(env,qq,has).cpu().numpy()
            bias=(model.bias_head(model.net[1](model.net[0](env)))*has[:,None]).cpu().numpy()
        removed=full-bias
        for k,i in enumerate(test):
            rows.append(dict(fold=fold,seed=seed,instance_id=t.iloc[i].instance_id,class_id=int(t.iloc[i].class_id),
                base_prediction=int(d['base'][i].argmax()),full_prediction=int(full[k].argmax()),prediction=int(removed[k].argmax()),
                **{f's{c}':float(removed[k,c]) for c in range(3)},**{f'bias{c}':float(bias[k,c]) for c in range(3)}))
rp=pd.DataFrame(rows);rp.to_csv(HERE/'removed_class_bias_predictions.csv',index=False)
rr=[]
for seed,q in rp.groupby('seed'):
    rr.append(dict(seed=seed,**metrics(q.class_id.to_numpy(),q[['s0','s1','s2']].to_numpy()),
        net_vs_base=int(q.prediction.eq(q.class_id).sum()-q.base_prediction.eq(q.class_id).sum()),
        net_vs_full=int(q.prediction.eq(q.class_id).sum()-q.full_prediction.eq(q.class_id).sum())))
pd.DataFrame(rr).to_csv(HERE/'removed_class_bias_metrics.csv',index=False)
(HERE/'control_analysis_summary.json').write_text(json.dumps(dict(review_ids=chosen,
    removal_note='Fixed already trained plus-bias model, no refit; gates differ from the separately trained anchored model.'),indent=2),encoding='utf-8')
print(pd.DataFrame(rr).to_string(index=False),flush=True)
print(pd.DataFrame(intervals).to_string(index=False),flush=True)
