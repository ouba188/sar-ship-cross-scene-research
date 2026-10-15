"""Posthoc frozen-logit path neutralization, not a trained or selected policy."""

import numpy as np
import pandas as pd
from run_probe import OUT, PREV, read


def main():
    events=pd.read_csv(OUT/'evaluation_events.csv')
    units=read(PREV/'PREDICTIONS_FROZEN.json')['units']
    refs=pd.read_csv(PREV/'metrics.csv').set_index(['uid','arm'])
    rows,classes=[] ,[]
    masks={'original':[], 'neutral_count':['support_count'],
           'neutral_oof':['support_oof'], 'neutral_both':['support_count','support_oof']}
    for u in units:
        f=events.loc[events.uid==u['uid']]
        with np.load(u['path'],allow_pickle=False) as z:
            n=len(z['query_ids'])
        initial=refs.loc[(u['uid'],'initial')]
        key={k:u[k] for k in ('family','task','cohort','uid')}
        for arm,columns in masks.items():
            logit=f.logit-f[columns].sum(axis=1)
            choose=logit>0
            delta=np.where(choose,0,-f.gain)
            ba=float(100*f.weight@delta);oa=float(100*delta.sum()/n)
            old_choose=f.logit>0
            count=int(((f.gain==1)&(~old_choose)&choose).sum())
            new_harm=int(((f.gain==-1)&(~old_choose)&choose).sum())
            if arm=='original':
                np.testing.assert_allclose([oa,ba],refs.loc[(u['uid'],'expanded__contextual'),['versus_initial_oa','versus_initial_ba']].to_numpy(float),atol=1e-10)
            rows.append(dict(**key,arm=arm,oa=float(initial.oa+oa),ba=float(initial.ba+ba),
                versus_initial_oa=oa,versus_initial_ba=ba,
                versus_selector_oa=oa-refs.loc[(u['uid'],'expanded__contextual'),'versus_initial_oa'],
                versus_selector_ba=ba-refs.loc[(u['uid'],'expanded__contextual'),'versus_initial_ba'],
                old_lost_rescues_recovered=count,old_avoided_harms_reintroduced=new_harm))
            for cls in range(8):
                keep=f.class_id==cls
                classes.append(dict(**key,arm=arm,class_id=cls,oa_contribution=float(100*delta[keep].sum()/n),
                    ba_contribution=float(100*f.loc[keep,'weight']@delta[keep]),
                    old_lost_rescues_recovered=int(((f.gain==1)&(~old_choose)&choose&keep).sum()),
                    old_avoided_harms_reintroduced=int(((f.gain==-1)&(~old_choose)&choose&keep).sum())))
    data=pd.DataFrame(rows);data.to_csv(OUT/'neutralization.csv',index=False)
    df=data.groupby(['family','task','cohort','arm']).mean(numeric_only=True).reset_index()
    df.to_csv(OUT/'neutralization_task.csv',index=False)
    df['scope']=np.where(df.family=='external',df.task,df.family)
    df.groupby(['scope','cohort','arm']).mean(numeric_only=True).reset_index().to_csv(OUT/'neutralization_summary.csv',index=False)
    cf=pd.DataFrame(classes);cf.to_csv(OUT/'neutralization_classes.csv',index=False)
    cf.groupby(['family','task','cohort','arm','class_id']).mean(numeric_only=True).reset_index().to_csv(OUT/'neutralization_class_summary.csv',index=False)
    print(pd.read_csv(OUT/'neutralization_summary.csv').to_string(index=False),flush=True)


if __name__=='__main__':
    main()
