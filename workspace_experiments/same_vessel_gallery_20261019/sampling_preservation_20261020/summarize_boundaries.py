"""Availability/class boundaries, without new training or selecting a policy."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
p=pd.read_csv(ROOT/'mse_aligned/predictions.csv')
c=pd.read_csv(ROOT/'coverage.csv').pivot(index='obs_id',columns='pol',values='radar_valid').fillna(0)
p=p.join(c.rename(columns={'vv':'radar_vv','vh':'radar_vh'}),on='obs_id')
p['base_correct']=p.base_prediction.eq(p.class_id)
p['delta']=p.correct.astype(float)-p.base_correct.astype(float)
rows=[]
for (arm,seed,valid),g in p.groupby(['arm','seed','radar_vh']):
    rows.append(dict(arm=arm,seed=seed,radar_vh=valid,n=len(g),classes=g.class_id.nunique(),
        oa=g.correct.mean(),base_oa=g.base_correct.mean(),ba=g.groupby('class_id').correct.mean().mean(),
        base_ba=g.groupby('class_id').base_correct.mean().mean(),net=g.delta.sum()))
pd.DataFrame(rows).to_csv(ROOT/'mse_aligned/availability_metrics.csv',index=False)
p.groupby(['arm','seed','radar_vh','class_id']).agg(n=('correct','size'),oa=('correct','mean'),base_oa=('base_correct','mean'),net=('delta','sum')).to_csv(ROOT/'mse_aligned/availability_classes.csv')
summary=pd.DataFrame(rows).groupby(['arm','radar_vh'])[['n','oa','base_oa','ba','base_ba','net']].mean()
print(summary.to_string())
