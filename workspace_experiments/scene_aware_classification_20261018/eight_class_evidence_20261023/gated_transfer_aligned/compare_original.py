"""Paired original/fixed results and exact same-case evidence; no model fitting."""
import difflib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_transfer import HERE, PARENT, ARMS, SEEDS, per_row

OLD=PARENT/'gated_transfer'


def main():
    old=pd.read_csv(OLD/'predictions.csv',dtype={'mmsi':str})
    new=pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    keys=['arm','seed','instance_id']
    p=new.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(p)==4744*9
    for col in ('class_id','row_index','mmsi','pass_group','fold','vh_valid'):
        np.testing.assert_array_equal(p[col+'_new'],p[col+'_old'])
    np.testing.assert_array_equal(p[[f'b{c}_new' for c in range(8)]],p[[f'b{c}_old' for c in range(8)]])
    p['fix_net']=p.prediction_new.eq(p.class_id_new).astype(int)-p.prediction_old.eq(p.class_id_old).astype(int)
    p.to_csv(HERE/'original_vs_fixed_predictions.csv',index=False)
    rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            a=pd.read_csv(HERE/'pooled.csv').set_index(['arm','seed']).loc[(arm,seed)]
            b=pd.read_csv(OLD/'pooled.csv').set_index(['arm','seed']).loc[(arm,seed)]
            part=p[p.arm.eq(arm)&p.seed.eq(seed)]
            rows.append(dict(arm=arm,seed=seed,old_net=int(b.net),fixed_net=int(a.net),
                fix_rescued=int(part.fix_net.eq(1).sum()),fix_harmed=int(part.fix_net.eq(-1).sum()),fix_net=int(part.fix_net.sum()),
                **{'old_'+k:b[k] for k in ('OA','BA','NLL','balanced_NLL','Brier','balanced_Brier')},
                **{'fixed_'+k:a[k] for k in ('OA','BA','NLL','balanced_NLL','Brier','balanced_Brier')},
                **{'delta_'+k:a[k]-b[k] for k in ('OA','BA','NLL','balanced_NLL','Brier','balanced_Brier')}))
    pd.DataFrame(rows).to_csv(HERE/'original_vs_fixed_metrics.csv',index=False)
    for label,group in [('class',['class_id_new','class_name_new']),('acquisition',['port_new','pass_group_new'])]:
        p.groupby(['arm','seed']+group).agg(n=('fix_net','size'),fix_net=('fix_net','sum'),
            fix_rescued=('fix_net',lambda v:v.eq(1).sum()),fix_harmed=('fix_net',lambda v:v.eq(-1).sum())).reset_index().to_csv(HERE/f'original_vs_fixed_{label}.csv',index=False)
    anchor=p[p.arm.eq(ARMS[0])&p.seed.eq(SEEDS[0])].sort_values('row_index_new')
    ids=anchor.instance_id.to_numpy();y=anchor.class_id_new.to_numpy()
    intervals=[]
    for block in ('mmsi','pass_group'):
        groups,inverse=np.unique(anchor[block+'_new'].to_numpy(),return_inverse=True)
        counts=np.zeros((len(groups),8));np.add.at(counts,(inverse,y),1)
        draws=np.random.default_rng(20261023).multinomial(len(groups),np.ones(len(groups))/len(groups),size=2000)
        denominator=draws@counts;valid=(denominator>0).all(1)
        for arm in ARMS:
            deltas=[]
            for seed in SEEDS:
                part=p[p.arm.eq(arm)&p.seed.eq(seed)].set_index('instance_id').loc[ids]
                ns=part[[f's{c}_new' for c in range(8)]].to_numpy()
                os=part[[f's{c}_old' for c in range(8)]].to_numpy()
                deltas.append(np.stack(per_row(y,ns),axis=1)-np.stack(per_row(y,os),axis=1))
            difference=np.mean(deltas,axis=0)
            for j,plain,balanced,factor in [(0,'OA','BA',100.),(1,'NLL','balanced_NLL',1.),(2,'Brier','balanced_Brier',1.)]:
                sums=np.zeros_like(counts);np.add.at(sums,(inverse,y),difference[:,j])
                numerator=draws@sums
                dist=numerator.sum(1)/denominator.sum(1)*factor
                bdist=(numerator[valid]/denominator[valid]).mean(1)*factor
                for metric,d,point in [(plain,dist,difference[:,j].mean()*factor),
                    (balanced,bdist,np.mean([difference[y==c,j].mean() for c in range(8)])*factor)]:
                    lo,hi=np.quantile(d,[.025,.975])
                    intervals.append(dict(arm=arm,block=block,metric=metric,fixed_minus_original=point,
                        low=lo,high=hi,resamples=len(d),clusters=len(groups)))
    pd.DataFrame(intervals).to_csv(HERE/'original_vs_fixed_intervals.csv',index=False)
    # Keep both pre-fixed image sets and join scores on exactly the same case.
    cases=[]
    for source,folder in [('original_selected',OLD),('fixed_selected',HERE)]:
        rc=pd.read_csv(folder/'review_cases.csv').drop_duplicates(['arm','seed','instance_id'])
        for r in rc.itertuples():
            part=p[p.arm.eq(r.arm)&p.seed.eq(r.seed)&p.instance_id.eq(r.instance_id)].iloc[0]
            truth=int(part.class_id_new)
            a=part[[f's{c}_old' for c in range(8)]].to_numpy(dtype=float)
            b=part[[f's{c}_new' for c in range(8)]].to_numpy(dtype=float)
            margin=lambda s:float(s[truth]-np.max(np.delete(s,truth)))
            cases.append(dict(case_source=source,arm=r.arm,seed=r.seed,instance_id=r.instance_id,truth=truth,
                old_prediction=int(part.prediction_old),fixed_prediction=int(part.prediction_new),
                old_true_margin=margin(a),fixed_true_margin=margin(b),old_outcome=part.outcome_old,
                fixed_outcome=part.outcome_new,panel_path=r.panel_path))
    pd.DataFrame(cases).to_csv(HERE/'original_vs_fixed_cases.csv',index=False)
    old_code=(OLD/'run_transfer.py').read_text(encoding='utf-8')
    new_code=(HERE/'run_transfer.py').read_text(encoding='utf-8')
    diff=''.join(difflib.unified_diff(old_code.splitlines(keepends=True),new_code.splitlines(keepends=True),
        fromfile='original/run_transfer.py',tofile='aligned/run_transfer.py'))
    (HERE/'implementation_change.diff').write_text(diff,encoding='utf-8')
    pd.DataFrame(rows).groupby('arm').mean(numeric_only=True).reset_index().to_csv(HERE/'original_vs_fixed_mean.csv',index=False)
    summary=dict(paired_rows=len(p),heads_original=27,heads_fixed=27,baseline_identical=True,
        ids_labels_folds_identical=True,only_training_input_change='oof_whole64[train] to whole64[train]',
        old_outputs_untouched=True,scope='OOF reader, shared unlabeled outer-source condition; not full-pipeline OOF')
    (HERE/'comparison_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows)[['arm','seed','old_net','fixed_net','fix_net','delta_OA','delta_BA','delta_NLL','delta_Brier']].to_string(index=False))
    print(pd.DataFrame(intervals).loc[lambda d:d.metric.isin(['OA','BA'])].to_string(index=False))
    print(diff)


if __name__=='__main__':
    main()
