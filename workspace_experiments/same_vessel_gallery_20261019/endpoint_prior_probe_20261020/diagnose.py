"""Bounded train/validation/generalization diagnosis using saved heads only."""
import numpy as np
import pandas as pd
from probe_endpoints import ROOT,PREV,old

f=pd.read_csv(old.prior.ROOT/'cohort.csv',dtype={'mmsi':str}); y=f.class_id.to_numpy()
d=np.load(PREV/'descriptors.npz'); extras=np.load(ROOT/'endpoint_descriptors.npz')
axis=np.concatenate((d['base'],d['axis']),axis=1)
exclude=pd.read_csv(old.LOCAL/'EXCLUSIONS.csv').obs_id.to_numpy()
rows=[]
for fold in range(3):
    partitions=old.prior.indices(f,fold)
    for arm in ('axis','hard_strong','soft_strong'):
        raw=axis if arm=='axis' else np.concatenate((axis,extras[arm]),axis=1)
        h=np.load(ROOT/f'head_f{fold}_{arm}.npz')
        for name,ii in zip(('train','validation','test'),partitions):
            ii=ii[~f.iloc[ii].obs_id.isin(exclude).to_numpy()]
            x=(raw[ii]-h['mean'])/h['std']/np.sqrt(raw.shape[1]); scores=x@h['coef'].T+h['intercept']
            rows.append(dict(fold=fold,arm=arm,partition=name,**old.prior.metrics(y[ii],scores)))
pd.DataFrame(rows).to_csv(ROOT/'generalization_diagnosis.csv',index=False)

# Reliability strata are source-fixed median thresholds, not optimized test gates.
p=pd.read_csv(ROOT/'predictions.csv')
strength=pd.read_csv(ROOT/'endpoint_strength.csv').query('pol=="vv"').set_index('obs_id')
group_rows=[]
for fold in range(3):
    train,val,test=old.prior.indices(f,fold)
    train,val,test=[ii[~f.iloc[ii].obs_id.isin(exclude).to_numpy()] for ii in (train,val,test)]
    threshold=float(strength.loc[f.iloc[val].obs_id,'contrast'].abs().median())
    for arm in ('hard_strong','soft_strong'):
        a=p.query('fold==@fold and arm==@arm').set_index('obs_id')
        b=p.query('fold==@fold and arm=="axis"').set_index('obs_id').loc[a.index]
        s=strength.loc[a.index]
        for name,mask in [('low_contrast',s.contrast.abs()<=threshold),('high_contrast',s.contrast.abs()>threshold),
                          ('peak_tie',s.peak_tie),('distinct_peak',~s.peak_tie)]:
            z=a.loc[mask]; base=b.loc[mask]
            group_rows.append(dict(fold=fold,arm=arm,group=name,n=len(z),source_validation_median=threshold,
                rescue=int((z.correct&~base.correct).sum()),harm=int((~z.correct&base.correct).sum())))
pd.DataFrame(group_rows).to_csv(ROOT/'strength_groups.csv',index=False)
print(pd.DataFrame(rows).groupby(['partition','arm'])[['oa','ba']].mean().to_string())
print(pd.DataFrame(group_rows).groupby(['arm','group'])[['n','rescue','harm']].sum().to_string())
