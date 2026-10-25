"""Frozen-input diagnostics: does the selected policy use action responses?"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train as run

HERE=Path(__file__).resolve().parent
analysis=run.load('selector_own_analysis',HERE/'analyze.py')


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    data=run.Data(); y=data.table.iloc[:data.n].class_id.to_numpy(int)
    rows=[]; events=[]; cases=[]
    for seed in run.SEEDS:
        arrays=analysis.load_inputs(seed)
        x=torch.tensor(arrays['x'],device='cuda')
        extra=torch.tensor(arrays['extra'],device='cuda')
        valid=torch.tensor(arrays['valid'],device='cuda')
        scaler=(arrays['response_mu'],arrays['response_sd'])
        cand=np.load(HERE/'results'/f'candidates_{seed}.npz')['score']
        response,_=run.response_features(cand,data.tr,scaler)
        nochange=np.repeat(cand[:,:1],19,axis=1)
        zero_delta,_=run.response_features(nochange,data.tr,scaler)
        for arm in run.LEARNED:
            ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
            model=run.make_model(arm,seed);model.load_state_dict(ck['state']);model.eval()
            original=np.load(HERE/'results'/f'{arm}_{seed}.npz')
            for mode,xx,rr in (('actual',x,response),('no_action_delta',x,zero_delta),
                                ('no_summary',x*0,response)):
                for role,ids in (('selection',data.va),('evaluation',data.te)):
                    s,a,prob=run.choose(model,xx,rr,extra,valid,cand,ids)
                    old=cand[ids,0]
                    rows.append(dict(seed=seed,arm=arm,mode=mode,role=role,
                        selected_step=ck['receipt']['selected_step'],
                        action_changed_vs_actual=int((a!=original['action'][ids]).sum()),
                        prediction_changed_vs_actual=int((s.argmax(1)!=original['score'][ids].argmax(1)).sum()),
                        mean_max_probability=float(prob.max(1).mean()),
                        mean_noop_probability=float(prob[:,0].mean()),
                        **(analysis.stats.effect(s,old,y[ids])|run.metrics(y[ids],s))))
                    if role=='evaluation' and seed==43:
                        for obs in (29687,7085,23397,16424,33503):
                            loc=np.flatnonzero(data.table.iloc[ids].obs_id.to_numpy(int)==obs)
                            if len(loc):
                                k=int(loc[0]);j=int(ids[k])
                                cases.append(dict(seed=seed,arm=arm,mode=mode,obs_id=obs,
                                    action=int(a[k]),action_name=analysis.action_name(a[k]),
                                    true=int(y[j]),prediction=int(s[k].argmax()),
                                    margin=float(analysis.stats.margin(s[k:k+1],y[j:j+1])[0])))
            ids=data.te
            changed=original['score'][ids].argmax(1)!=cand[ids,0].argmax(1)
            for j in ids[changed]:
                row=data.table.iloc[j]
                a=int(original['action'][j]);s=original['score'][j];base=cand[j,0]
                events.append(dict(seed=seed,arm=arm,obs_id=int(row.obs_id),mmsi=row.mmsi,
                    port=row.port,pass_group=row.pass_group,class_id=int(y[j]),class_name=row.class_name,
                    action=a,action_name=analysis.action_name(a),before_prediction=int(base.argmax()),
                    after_prediction=int(s.argmax()),rescued=bool(s.argmax()==y[j]),harmed=bool(base.argmax()==y[j]),
                    before_margin=float(analysis.stats.margin(base[None],y[j:j+1])[0]),
                    after_margin=float(analysis.stats.margin(s[None],y[j:j+1])[0]),
                    true_score_change=float(s[y[j]]-base[y[j]])))
    pd.DataFrame(rows).to_csv(HERE/'frozen_input_diagnostics.csv',index=False)
    pd.DataFrame(cases).to_csv(HERE/'case_input_diagnostics.csv',index=False)
    pd.DataFrame(events).to_csv(HERE/'changed_queries.csv',index=False)
    print(pd.DataFrame(rows).query("role=='evaluation' and seed==43").to_string(index=False))
    print(pd.DataFrame(events).to_string(index=False))


if __name__=='__main__':main()
