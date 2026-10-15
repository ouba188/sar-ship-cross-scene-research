"""Support-draw sensitivity and class/scene accounting of the oracle headroom."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
draws=pd.read_csv(OUT/'curve_by_support_draw.csv')
draws=draws[(draws['mode']=='full') & (draws.cell=='other_acquisition')]
records=[]
for (stage,alpha),g in draws.groupby(['stage','alpha']):
    best=g.ba_delta.idxmax()
    records.append(dict(stage=stage,alpha=alpha,ba_gain=g.ba_delta.mean(),
        ba_draw_sd=g.ba_delta.std(),positive_draws=int((g.ba_delta>1e-12).sum()),
        negative_draws=int((g.ba_delta < -1e-12).sum()),
        best_draw=int(g.loc[best,'draw']),best_draw_gain=g.loc[best,'ba_delta'],
        mean_without_best_draw=g.drop(index=best).ba_delta.mean()))
pd.DataFrame(records).to_csv(OUT/'draw_sensitivity.csv',index=False)
events=pd.read_csv(OUT/'class_events.csv')
events=events[(events.cell=='other_acquisition') & events.alpha.eq(.25)].copy()
keys=['stage','port','episode','draw','seed']
events['query_class_count']=events.groupby(keys).class_id.transform('size')
events['scene_count']=events.groupby(['stage','port']).episode.transform('nunique')
events['global_ba_contribution']=100*(events.rescues-events.harms)/events.query_n/events.query_class_count/events.scene_count/3/5/3
effects=events.groupby(['stage','port','episode','class_id'])[['global_ba_contribution','rescues','harms']].sum().reset_index()
effects.sort_values('global_ba_contribution',ascending=False).to_csv(OUT/'scene_class_contributions.csv',index=False)
effects.groupby(['stage','class_id']).global_ba_contribution.sum().reset_index().to_csv(OUT/'class_contributions.csv',index=False)
effects.groupby(['stage','port']).global_ba_contribution.sum().reset_index().to_csv(OUT/'port_contributions.csv',index=False)
allcurve=pd.read_csv(OUT/'curve.csv')
expected=allcurve[(allcurve['mode']=='full') & allcurve.cell.eq('other_acquisition') & allcurve.alpha.eq(.25)].set_index('stage').ba_delta
np.testing.assert_allclose(effects.groupby('stage').global_ba_contribution.sum().loc[expected.index],expected,atol=1e-12)
# Query events retain identity: repeated seeds/draws are not counted as new ships.
manifest=pd.read_csv(OUT.parent/'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv',
                     usecols=['port','mmsi','class_id'],dtype={'mmsi':str},keep_default_na=False)
enrollment=json.loads((OUT.parent/'class_reference_metric_pilot_20261014/ENROLLMENT.json').read_text(encoding='utf-8'))
rows=[]
for held in sorted({e['held'] for e in enrollment}):
    tasks=[e for e in enrollment if e['held']==held and e['role']=='held']
    for ti,e in enumerate(tasks):
        if e['cell']!='other_acquisition':continue
        for stage in ['initial','matched']:
            with np.load(OUT/'predictions'/f"{stage}_{held.replace(' ','_')}_{ti:03d}.npz") as a:
                y=a['y'];q=a['q'];i=int(np.flatnonzero(a['alphas']==.25)[0])
                b=a['full'][:,0].argmax(2);p=a['full'][:,i].argmax(2)
                for si,seed in enumerate(a['seeds']):
                    for j in np.flatnonzero((b[si]==y)!=(p[si]==y)):
                        rows.append(dict(stage=stage,port=held,episode=e['episode'],draw=e['draw'],seed=int(seed),
                            row=int(q[j]),ship=manifest.iloc[q[j]].mmsi,class_id=int(y[j]),
                            event='rescue' if p[si,j]==y[j] else 'harm'))
cases=pd.DataFrame(rows)
cases.to_csv(OUT/'changed_cases_alpha025.csv',index=False)
cases.groupby(['stage','event']).agg(exposures=('row','size'),unique_rows=('row','nunique'),unique_ships=('ship','nunique')).reset_index().to_csv(OUT/'changed_case_counts.csv',index=False)
print('DRAW SENSITIVITY')
print(pd.DataFrame(records).query('alpha in [.25,.5]').to_string(index=False))
print('TOP SCENE CLASS CONTRIBUTIONS')
print(effects.sort_values('global_ba_contribution',ascending=False).groupby('stage').head(5).to_string(index=False))
