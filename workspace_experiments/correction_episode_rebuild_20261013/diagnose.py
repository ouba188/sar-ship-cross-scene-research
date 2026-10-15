"""Paired effects and coverage of the frozen episode-rebuild experiment."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
PREV=OUT.parent/'correction_choice_probe_20261013'
meta=pd.read_csv(OUT.parent/'same_vessel_condition_study_20261010/observations.csv',dtype={'mmsi':str})
events=pd.read_csv(OUT/'training_events.csv',dtype={'ship':str})
events=events[events.context=='pooled']
events.groupby(['recipe','class_id']).agg(exposures=('gain','size'),observations=('obs_id','nunique'),
    ships=('ship','nunique'),rescues=('gain',lambda x:int((x>0).sum())),
    harms=('gain',lambda x:int((x<0).sum())),weight=('weight','sum')).reset_index().to_csv(
        OUT/'event_class_coverage.csv',index=False)

metrics=pd.read_csv(OUT/'metrics.csv')
keys=['family','task','cohort','uid']
paired=[]
contrasts=[('coverage','expanded__contextual','old__contextual'),
           ('locality','expanded_local__contextual','expanded_matched_mixed__contextual'),
           ('local_extra','expanded_local__contextual','expanded__contextual'),
           ('support_features','expanded__contextual','expanded__score_only')]
for name,a,b in contrasts:
    joined=metrics[metrics.arm==a].merge(metrics[metrics.arm==b],on=keys,validate='one_to_one',suffixes=('_a','_b'))
    for _,row in joined.iterrows():
        paired.append(dict(**{k:row[k] for k in keys},contrast=name,
            oa_delta=row.oa_a-row.oa_b,ba_delta=row.ba_a-row.ba_b))
p=pd.DataFrame(paired)
p.to_csv(OUT/'paired_contrasts.csv',index=False)
p.groupby(['family','task','cohort','contrast']).agg(mean_ba=('ba_delta','mean'),sd_ba=('ba_delta','std'),
    mean_oa=('oa_delta','mean'),positive=('ba_delta',lambda x:int((x>1e-10).sum())),
    units=('uid','size')).reset_index().to_csv(OUT/'paired_contrast_summary.csv',index=False)

# Full-query weights distinguish support-count effects from changing class proportions.
source_units=json.loads((OUT/'SOURCE_READY.json').read_text(encoding='utf-8'))['units']
source=[]
for u in source_units:
    with np.load(u['path']) as z:
        y=z['labels'];g=(z['initial'].argmax(1)==y).astype(int)-(z['raw'].argmax(1)==y)
        counts=z['profile_counts'];n=np.bincount(y,minlength=8)
        w=1./((n>0).sum()*n[y])
        for k in range(3):
            use=np.minimum(counts[y],2)==k
            source.append(dict(context=u['context'],kind=u['kind'],key=u['key'],stratum=k,
                ba_mass=float(100*w[use].sum()),initial_ba_contribution=float(100*w[use]@g[use]),
                n=int(use.sum())))
s=pd.DataFrame(source)
s.to_csv(OUT/'source_support_strata.csv',index=False)
s.groupby(['kind','stratum']).mean(numeric_only=True).reset_index().to_csv(OUT/'source_support_strata_summary.csv',index=False)

# Report the strongest existing simple comparator without using it to select a new policy.
old=pd.read_csv(PREV/'metrics.csv')
comparison=[]
for arm in ('expanded__contextual','expanded_local__contextual','expanded_matched_mixed__contextual'):
    current=metrics[metrics.arm==arm]
    for baseline in ('confidence','combined'):
        joined=current.merge(old[old.arm==baseline],on=keys,suffixes=('_new','_old'),validate='one_to_one')
        for _,row in joined.iterrows():
            comparison.append(dict(**{k:row[k] for k in keys},arm=arm,baseline=baseline,
                oa_delta=row.oa_new-row.oa_old,ba_delta=row.ba_new-row.ba_old))
ref=pd.DataFrame(comparison)
ref.to_csv(OUT/'strong_comparators.csv',index=False)
ref.groupby(['family','task','cohort','arm','baseline'])[['oa_delta','ba_delta']].mean().reset_index().to_csv(
    OUT/'strong_comparator_summary.csv',index=False)

print('Paired contrasts')
print(p.groupby(['family','cohort','contrast'])[['oa_delta','ba_delta']].mean().to_string())
print('Source support-count contribution')
print(pd.read_csv(OUT/'source_support_strata_summary.csv').to_string(index=False))
