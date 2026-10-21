"""Check acquisition counting, cached measurements, and vessel-level estimates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run import pass_key

root=Path(__file__).resolve().parent
a='S1A_IW_GRDH_1SDV_20250820T095615_20250820T095640_060618_078A74_3D5A'
b='S1A_IW_GRDH_1SDV_20250820T095640_20250820T095705_060618_078A74_ABCD'
c='S1A_IW_GRDH_1SDV_20250901T095615_20250901T095640_060793_079999_3D5A'
assert pass_key(a)==pass_key(b)=='S1A_060618_078A74'
assert pass_key(a)!=pass_key(c)
cohort=pd.read_csv(root/'cohort.csv',dtype={'mmsi':str})
assert not cohort.duplicated(['mmsi','overpass']).any()
assert cohort.groupby('mmsi').size().min()>=4
m=pd.read_csv(root/'measurements.csv',dtype={'mmsi':str})
assert len(m)==2*len(cohort) and not m.duplicated(['obs_id','pol']).any()
assert np.isfinite(m[['length_m','width_m','incidence','sin2phi','cos2phi']]).all().all()
assert (m.length_m>=m.width_m).all() and (m.width_m>0).all()
old=pd.read_csv(root.parent/'condition_morphometry_20261021/measurements.csv',dtype={'mmsi':str})
joined=m.merge(old,on=['obs_id','pol'],suffixes=('_new','_old'))
for key in ['length_m','width_m','contrast','incidence','sin2phi','cos2phi']:
    np.testing.assert_allclose(joined[key+'_new'],joined[key+'_old'],rtol=1e-12,atol=1e-12,equal_nan=True)
p=pd.read_csv(root/'condition_pairs.csv',dtype={'mmsi':str})
summary=pd.read_csv(root/'condition_comparison.csv')
for row in summary.itertuples():
    f=p[p.pol.eq(row.pol)].groupby(['mmsi','group'])[row.metric].mean().unstack().dropna(subset=['near','far'])
    assert len(f)==row.ships
    np.testing.assert_allclose((f.far-f.near).mean(),row.far_minus_near,atol=1e-12)
checks=dict(pass_grouping=True,unique_passes=True,matched_vv_vh=True,finite_dimensions=True,
            cached_views_equal=len(joined),vessel_weighted_comparisons_replayed=len(summary),
            ships=cohort.mmsi.nunique(),observations=len(cohort))
(root/'VERIFICATION.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
