"""Summarize objective effects without another fit or policy choice."""

from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
metrics=pd.read_csv(OUT/'metrics.csv')
classes=pd.read_csv(OUT/'classes.csv')
keys=['family','task','cohort','uid']
joined=classes.groupby(keys+['arm'])[['oa_contribution','ba_contribution']].sum().reset_index().merge(
    metrics[keys+['arm','versus_initial_oa','versus_initial_ba']],on=keys+['arm'],validate='one_to_one')
np.testing.assert_allclose(joined.oa_contribution,joined.versus_initial_oa,atol=1e-10,rtol=0)
np.testing.assert_allclose(joined.ba_contribution,joined.versus_initial_ba,atol=1e-10,rtol=0)

records=[]
for arm in ('joint','micro'):
    frame=metrics[metrics.arm==arm].merge(metrics[metrics.arm=='ba'],on=keys,suffixes=('_new','_ba'),validate='one_to_one')
    for _,row in frame.iterrows():
        records.append(dict(**{k:row[k] for k in keys},arm=arm,
            oa_delta=row.oa_new-row.oa_ba,ba_delta=row.ba_new-row.ba_ba,
            versus_initial_oa=row.versus_initial_oa_new,versus_initial_ba=row.versus_initial_ba_new,
            both_positive_vs_initial=bool(row.versus_initial_oa_new>1e-10 and row.versus_initial_ba_new>1e-10)))
effects=pd.DataFrame(records)
effects.to_csv(OUT/'objective_effects.csv',index=False)
effects.groupby(['family','task','cohort','arm']).agg(oa_delta=('oa_delta','mean'),ba_delta=('ba_delta','mean'),
    both_positive_vs_initial=('both_positive_vs_initial','sum'),units=('uid','size')).reset_index().to_csv(
        OUT/'objective_effect_summary.csv',index=False)
classes.groupby(['family','task','cohort','arm','class_id'])[['n','recall','oa_contribution','ba_contribution']].mean().reset_index().to_csv(
    OUT/'class_summary.csv',index=False)

old=pd.read_csv(OUT.parent/'correction_choice_probe_20261013/metrics.csv')
comp=[]
for reference in ('combined','confidence'):
    frame=metrics[metrics.arm=='joint'].merge(old[old.arm==reference],on=keys,suffixes=('_new','_old'),validate='one_to_one')
    for _,row in frame.iterrows():
        comp.append(dict(**{k:row[k] for k in keys},reference=reference,
            oa_delta=row.oa_new-row.oa_old,ba_delta=row.ba_new-row.ba_old))
cf=pd.DataFrame(comp)
cf.to_csv(OUT/'strong_comparators.csv',index=False)
cf.groupby(['family','task','cohort','reference'])[['oa_delta','ba_delta']].mean().reset_index().to_csv(
    OUT/'strong_comparator_summary.csv',index=False)
print('Class-sum identities passed',len(joined),'rows /',len(joined)*2,'OA/BA checks')
print(pd.read_csv(OUT/'objective_effect_summary.csv').to_string(index=False))
