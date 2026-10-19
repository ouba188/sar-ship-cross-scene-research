"""Summarize paired residual effects without treating seeds as new ships."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def met(g):
    return dict(n=len(g),oa=g.correct.mean(),ba=g.groupby('class_id').correct.mean().mean())


def main():
    p=pd.read_csv(ROOT/'predictions.csv',dtype={'mmsi':str})
    summaries=[]
    for (arm,selection,seed),g in p.groupby(['arm','selection','seed']):
        summaries.append(dict(arm=arm,selection=selection,seed=seed,**met(g)))
    s=pd.DataFrame(summaries); s.to_csv(ROOT/'summary_by_seed.csv',index=False)
    s.groupby(['arm','selection']).agg(n=('n','first'),oa_mean=('oa','mean'),oa_sd=('oa','std'),ba_mean=('ba','mean'),ba_sd=('ba','std')).to_csv(ROOT/'summary.csv')
    pairs=[]; merged_all=[]
    for arm,g in p[p.selection.eq('source_selected')].groupby('arm'):
        base=p[p.arm.eq(arm)&p.selection.eq('base')]
        joined=g.merge(base,on=['obs_id','seed','fold','class_id','mmsi','port'],suffixes=('_new','_base'),validate='one_to_one')
        joined['delta']=joined.correct_new.astype(int)-joined.correct_base.astype(int); joined['arm']=arm
        merged_all.append(joined)
        for seed,a in joined.groupby('seed'):
            pairs.append(dict(arm=arm,seed=seed,n=len(a),rescues=int(a.delta.eq(1).sum()),harms=int(a.delta.eq(-1).sum()),oa_gain=a.delta.mean(),ba_gain=a.groupby('class_id').delta.mean().mean()))
    pd.DataFrame(pairs).to_csv(ROOT/'paired.csv',index=False)
    merged=pd.concat(merged_all)
    for field in ('class_id','port'):
        rows=[]
        for (arm,key,seed),g in merged.groupby(['arm',field,'seed']):
            rows.append(dict(arm=arm,group=key,seed=seed,n=len(g),base_oa=g.correct_base.mean(),new_oa=g.correct_new.mean(),delta=g.delta.mean()))
        pd.DataFrame(rows).to_csv(ROOT/f'by_{field}.csv',index=False)
    alignment=pd.read_csv(ROOT/'crossview_alignment.csv')[['obs_id','status']]
    aligned=merged.merge(alignment,on='obs_id',validate='many_to_one')
    rows=[]; flips=[]
    for (arm,status,seed),g in aligned.groupby(['arm','status','seed']):
        rows.append(dict(arm=arm,status=status,seed=seed,n=len(g),base_oa=g.correct_base.mean(),new_oa=g.correct_new.mean(),delta=g.delta.mean(),ba_delta=g.groupby('class_id').delta.mean().mean()))
    pd.DataFrame(rows).to_csv(ROOT/'by_alignment.csv',index=False)
    labels=pd.read_csv(ROOT.parent/'conditional_evidence_pilot/cohort.csv').set_index('obs_id').class_id
    rows=[]
    for seed in (20261019,20261020,20261021):
        for name in ('drop_target','drop_other','drop_rest','wrong_condition'):
            batches=[]
            for fold in range(3):
                data=np.load(ROOT/f'intervention_f{fold}_s{seed}_{name}.npz')
                yy=labels.loc[data['obs_id']].to_numpy(); pred=data['scores'].argmax(1)
                batches.append(pd.DataFrame(dict(class_id=yy,correct=pred==yy)))
                if name=='wrong_condition':
                    ref=p[p.arm.eq('pre')&p.selection.eq('source_selected')&p.seed.eq(seed)&p.fold.eq(fold)].set_index('obs_id').loc[data['obs_id']]
                    flips.append(dict(seed=seed,fold=fold,n=len(yy),changed=int((pred!=ref.prediction.to_numpy()).sum()),rescues=int(((pred==yy)&~ref.correct.to_numpy()).sum()),harms=int(((pred!=yy)&ref.correct.to_numpy()).sum())))
            rows.append(dict(seed=seed,intervention=name,**met(pd.concat(batches))))
    pd.DataFrame(rows).to_csv(ROOT/'intervention_summary_by_seed.csv',index=False)
    pd.DataFrame(flips).to_csv(ROOT/'condition_flip_counts.csv',index=False)
    # Vessel bootstrap on seed-mean paired outcomes, conditional on the trained models.
    u=merged[merged.arm.eq('pre')].groupby(['obs_id','mmsi','class_id']).delta.mean().reset_index()
    ships=u.groupby(['class_id','mmsi']).agg(n=('delta','size'),d=('delta','sum')).reset_index()
    groups=[g[['n','d']].to_numpy() for _,g in ships.groupby('class_id')]
    rng=np.random.default_rng(20261019); values=[]
    for _ in range(500):
        sampled=[g[rng.integers(len(g),size=len(g))] for g in groups]
        values.append([sum(g[:,1].sum() for g in sampled)/sum(g[:,0].sum() for g in sampled),np.mean([g[:,1].sum()/g[:,0].sum() for g in sampled])])
    ci=np.quantile(values,[.025,.975],axis=0)
    pd.DataFrame(dict(metric=['oa','ba'],low=ci[0],high=ci[1])).to_csv(ROOT/'conditional_ci.csv',index=False)
    print(pd.read_csv(ROOT/'summary.csv').to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))
    print('selected steps',pd.read_csv(ROOT/'selected_states.csv').groupby('arm').selected_step.apply(list).to_dict())


if __name__=='__main__': main()
