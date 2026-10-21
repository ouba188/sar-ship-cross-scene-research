"""Describe held-out gains across identities, classes and acquisition groups."""
import json
import numpy as np
import pandas as pd
from run import OUT,ROOT

p=pd.read_csv(OUT/'predictions.csv',dtype={'mmsi':str})
c=pd.read_csv(ROOT/'multitemporal_20261021/cohort.csv',dtype={'mmsi':str})
f=p[p.variant.eq('all')]
wide=f.pivot(index=['obs_id','pol','metric','mmsi','split_fold'],columns='arm',values='abs_log_error').reset_index()
wide=wide.merge(c[['obs_id','overpass','class_id','port']],on='obs_id',validate='many_to_one')
rows=[];clusters=[]
rng=np.random.default_rng(20261021)
for (pol,metric),g in wide.groupby(['pol','metric']):
    for arm in ['own_geometry_fixedscale','transfer_geometry','transfer_direction']:
        g=g.copy();g['gain']=g.history_mean-g[arm]
        for field in ['split_fold','class_id','port','overpass']:
            for level,part in g.groupby(field):
                rows.append(dict(pol=pol,metric=metric,arm=arm,field=field,level=level,ships=len(part),
                    mean_gain=part.gain.mean(),relative_reduction_pct=100*part.gain.mean()/part.history_mean.mean()))
        groups=g.groupby('overpass').gain.agg(['sum','size'])
        draws=rng.integers(0,len(groups),size=(10000,len(groups)))
        boot=groups['sum'].to_numpy()[draws].sum(1)/groups['size'].to_numpy()[draws].sum(1)
        loo=(g.gain.sum()-groups['sum'])/(len(g)-groups['size'])
        clusters.append(dict(pol=pol,metric=metric,arm=arm,query_passes=len(groups),
            positive_passes=int((groups['sum']>0).sum()),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),
            minimum_leave_one_pass_out_gain=loo.min(),maximum_leave_one_pass_out_gain=loo.max()))
pd.DataFrame(rows).to_csv(OUT/'stratified_gains.csv',index=False,lineterminator='\n')
pd.DataFrame(clusters).to_csv(OUT/'query_pass_uncertainty.csv',index=False,lineterminator='\n')
print(pd.DataFrame(clusters).query('metric == "r3_length_m"').to_string(index=False))
