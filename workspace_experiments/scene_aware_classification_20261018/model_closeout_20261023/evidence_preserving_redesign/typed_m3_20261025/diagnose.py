"""Locate risk-versus-decision changes; review active-head CC/WW and metric flips."""
import analyze as a
import train as run
import json
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
HERE=run.HERE

def risk(s,y):
    lp=s.astype(float)-logsumexp(s.astype(float),axis=1,keepdims=True)
    return np.stack((-lp[np.arange(len(y)),y],((np.exp(lp)-np.eye(8)[y])**2).sum(1)),1)

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=run.Data();ids=data.te;f=data.table.iloc[ids];y=f.class_id.to_numpy(int)
    records=[];flips=[];stats=[];scores={};sw=np.load(HERE/'results/switches.npz')
    for seed in run.SEEDS:
        old=np.load(HERE/'results'/f'old_full_{seed}.npz')['score']
        new=np.load(HERE/'results'/f'typed_both_{seed}.npz')['score'];scores[seed]=(old,new)
        ap,bp=new[ids].argmax(1)==y,old[ids].argmax(1)==y
        r=risk(new[ids],y)-risk(old[ids],y)
        for group,mask in (('CC',ap&bp),('WW',~ap&~bp),('rescue',ap&~bp),('harm',~ap&bp),('all',np.ones(len(ids),bool))):
            if mask.any():stats.append(dict(seed=seed,group=group,n=int(mask.sum()),NLL_delta=r[mask,0].mean(),Brier_delta=r[mask,1].mean(),
                mean_old_margin=a.margin(old[ids][mask],y[mask]).mean(),mean_new_margin=a.margin(new[ids][mask],y[mask]).mean()))
        initial=sw[f'{seed}_initial'];changed=np.flatnonzero(new[ids].argmax(1)!=initial.argmax(1))
        for k in changed:
            flips.append(dict(seed=seed,obs_id=int(f.iloc[k].obs_id),class_id=int(y[k]),class_name=f.iloc[k].class_name,
                              port=f.iloc[k].port,pass_group=f.iloc[k].pass_group,old_prediction=int(old[ids[k]].argmax()),
                              initial_prediction=int(initial[k].argmax()),learned_prediction=int(new[ids[k]].argmax()),
                              initial_margin=float(a.margin(initial[[k]],y[[k]])[0]),new_margin=float(a.margin(new[ids[[k]]],y[[k]])[0])))
    pd.DataFrame(stats).to_csv(HERE/'risk_transitions.csv',index=False);pd.DataFrame(flips).to_csv(HERE/'metric_flips.csv',index=False)
    geo=run.load('extra_typed_geo',run.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up=pd.read_csv(run.BASE/'regional_utility_pilot/upstream/table.csv')
    valid=np.load(run.BASE/'regional_utility_pilot/upstream/summaries.npz')['valid']
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs=np.load(run.LOCAL/'scores_layer1.npz')['refs'];selected=[]
    # Existing median cases used seed29's zero-update model; inspect active seed17 too.
    old,new=scores[17];ap,bp=new[ids].argmax(1)==y,old[ids].argmax(1)==y
    for label,mask in (('active_CC',ap&bp),('active_WW',~ap&~bp)):
        options=np.flatnonzero(mask);options=options[np.argsort(f.iloc[options].obs_id)]
        k=int(options[(len(options)-1)//2]);selected.append(dict(seed=17,obs_id=int(f.iloc[k].obs_id),index=int(ids[k]),eval_index=k,case=label,comparison='old_full'))
    seen={x['obs_id'] for x in json.loads((HERE/'analysis/cases.json').read_text())}
    for x in flips:
        if x['learned_prediction']==x['class_id'] and x['initial_prediction']!=x['class_id'] and x['obs_id'] not in seen:
            j=int(np.flatnonzero(data.table.obs_id.eq(x['obs_id']))[0]);k=int(np.flatnonzero(ids==j)[0])
            selected.append(dict(seed=x['seed'],obs_id=x['obs_id'],index=j,eval_index=k,case='metric_rescue_extra',comparison='fixed_initial_metric'))
    for case in selected:
        seed,j,k=case['seed'],case['index'],case['eval_index'];old,new=scores[seed]
        before=old[j] if case['comparison']=='old_full' else sw[f'{seed}_initial'][k];after=new[j]
        path,prov,true,rival=a.panel(case,data,before,after,geo,up,valid,snapshots,refs)
        records.append(dict(**case,true=true,rival=rival,before=before,after=after,old_full=old[j],
            diagnostics=a.case_diagnostics(data,case,true,rival),figure=str(path),provenance=prov))
        print('EXTRA',case,flush=True)
    run.write_json(HERE/'analysis/extra_cases.json',records)
    print(pd.DataFrame(flips).to_string(index=False),flush=True)
    print(pd.DataFrame(stats).to_string(index=False),flush=True)

if __name__=='__main__':main()
