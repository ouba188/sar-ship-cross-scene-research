"""Paired system effects and actual support-path traces for the frozen probe."""
import json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from run_probe import HERE,Study,ConvReader

KEY=['task','seed','draw','row']
METKEY=['task','line','budget','seed','draw']
COLS=['oa','ba','brier','nll','map_r']
COMPARISONS=[('conv_first','base'),('conv_first','pool_first'),('conv_first','point_first'),
             ('point_first','pool_first'),('pool_first','base'),
             ('conv_first','conv_scramble'),('conv_drop_high','conv_drop_control')]


def main():
    p=pd.read_csv(HERE/'predictions.csv');m=pd.read_csv(HERE/'metrics.csv')
    events=[];deltas=[];selected=[]
    for arm,comp in COMPARISONS:
        a=p[p.arm.eq(arm)].set_index(KEY).sort_index()
        b=p[p.arm.eq(comp)].set_index(KEY).sort_index()
        assert a.index.equals(b.index)
        e=a[['line','budget','truth','prediction','margin','top_same','top_wrong','same_support','readout_valid']].copy()
        e['arm']=arm;e['comparator']=comp;e['old_prediction']=b.prediction
        e['delta_margin']=a.margin-b.margin;e['delta_brier']=a.brier-b.brier
        e['rescue']=a.correct&~b.correct;e['harm']=~a.correct&b.correct
        e['persistent_correct']=a.correct&b.correct;e['persistent_wrong']=~a.correct&~b.correct
        e=e.reset_index();events.append(e)
        d=m[m.arm.eq(arm)].set_index(METKEY)[COLS]-m[m.arm.eq(comp)].set_index(METKEY)[COLS]
        d=d.reset_index();d['arm']=arm;d['comparator']=comp;deltas.append(d)
        if arm=='conv_first' and comp in ('base','pool_first','point_first'):
            for category in ['rescue','harm','persistent_correct','persistent_wrong']:
                # Fixed B16 cross-pass diagnostic; median event, not maximum benefit.
                group=e[e.line.eq('cross_pass')&e.budget.eq(16)&e[category]].sort_values(['delta_margin','row','seed','draw'])
                if len(group):
                    case=group.iloc[(len(group)-1)//2].to_dict();case['category']=category
                    selected.append(case)
    events=pd.concat(events);delta=pd.concat(deltas)
    events.to_csv(HERE/'paired_events.csv',index=False)
    delta.to_csv(HERE/'paired_metrics.csv',index=False)
    seeds=delta.groupby(['arm','comparator','line','budget','seed'])[COLS].mean().reset_index()
    seeds.to_csv(HERE/'paired_seed.csv',index=False)
    summary=[]
    for keys,g in seeds.groupby(['arm','comparator','line','budget']):
        arm,comp,line,budget=keys
        ev=events[events.arm.eq(arm)&events.comparator.eq(comp)&events.line.eq(line)&events.budget.eq(budget)]
        summary.append(dict(arm=arm,comparator=comp,line=line,budget=budget,
            oa_pp=100*g.oa.mean(),ba_pp=100*g.ba.mean(),oa_seed_sd_pp=100*g.oa.std(),ba_seed_sd_pp=100*g.ba.std(),
            positive_oa_seeds=int((g.oa>1e-10).sum()),positive_ba_seeds=int((g.ba>1e-10).sum()),
            brier_delta=g.brier.mean(),map_delta=g.map_r.mean(),
            rescue_events=int(ev.rescue.sum()),harm_events=int(ev.harm.sum()),
            rescued_rows=ev[ev.rescue].row.nunique(),harmed_rows=ev[ev.harm].row.nunique()))
    summary=pd.DataFrame(summary);summary.to_csv(HERE/'paired_summary.csv',index=False)
    events.groupby(['arm','comparator','line','budget','truth'])[['rescue','harm','delta_margin']].agg(['sum','mean']).to_csv(HERE/'class_effects.csv')
    pd.DataFrame(selected).to_csv(HERE/'selected_cases.csv',index=False)
    print(summary.to_string(index=False))

    study=Study();cache={};paths=[];contributions=[]
    for case in pd.DataFrame(selected).itertuples():
        task=next(t for t in study.tasks['tasks'] if t['name']==case.task)
        ct=study.context(task['source'],task['supports'][case.draw],task['query'])
        j=list(ct['q']).index(case.row);ns=ct['ns'];qi=ns+j;out={}
        for arm in ['conv_first',case.comparator]:
            if arm=='base':out[arm]=(ct['base_prob'],ct['base'],ct['base_alpha'],ct['base_score'])
            else:
                key=(arm,case.seed)
                if key not in cache:
                    r=ConvReader(arm).cuda().eval()
                    r.load_state_dict(torch.load(HERE/f'{arm}_seed{case.seed}.pt',weights_only=True));cache[key]=r
                with torch.no_grad():out[arm]=study.predict(cache[key],ct)[:4]
        pp,k,alpha,score=[v.cpu().numpy() for v in out['conv_first']]
        p0,k0,a0,s0=[v.cpu().numpy() for v in out[case.comparator]]
        recorded=p[p.arm.eq('conv_first')&p.task.eq(case.task)&p.seed.eq(case.seed)&p.draw.eq(case.draw)&p.row.eq(case.row)]
        np.testing.assert_allclose(pp[j],recorded[[f'p{i}' for i in range(7)]].to_numpy()[0],atol=2e-8)
        truth=case.truth;rivals=p0[j].copy();rivals[truth]=-np.inf;rival=int(rivals.argmax())
        aa=alpha[:,truth]-alpha[:,rival];ab=a0[:,truth]-a0[:,rival]
        ka,kb=k[qi,:ns],k0[qi,:ns]
        direct=.5*(ka-kb)*(aa+ab);refit=.5*(ka+kb)*(aa-ab);total=direct+refit
        observed=(score[qi,truth]-score[qi,rival])-(s0[qi,truth]-s0[qi,rival])
        np.testing.assert_allclose(total.sum(),observed,atol=1e-10)
        tags=dict(task=case.task,seed=case.seed,draw=case.draw,row=case.row,comparator=case.comparator,category=case.category)
        paths.append(dict(**tags,truth=truth,rival=rival,direct=float(direct.sum()),refit=float(refit.sum()),
            branch_margin_change=float(observed),same_class_sum=float(total[study.y[ct['s']]==truth].sum()),
            old_final_margin=float(p0[j,truth]-np.delete(p0[j],truth).max()),new_final_margin=float(pp[j,truth]-np.delete(pp[j],truth).max())))
        for t,s in enumerate(ct['s']):
            contributions.append(dict(**tags,support_row=int(s),support_label=int(study.y[s]),
                old_similarity=float(kb[t]),new_similarity=float(ka[t]),direct=float(direct[t]),refit=float(refit[t]),total=float(total[t])))
    pd.DataFrame(paths).to_csv(HERE/'case_paths.csv',index=False)
    pd.DataFrame(contributions).to_csv(HERE/'case_support_contributions.csv',index=False)
    print('CASE PATHS',pd.DataFrame(paths).to_string(index=False))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=1):main()
