"""Final class decomposition and saved angle-control verification."""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from study import OUT,metadata,write

warnings.filterwarnings('once',message='y_pred contains classes not in y_true',category=UserWarning)

d=metadata(); labels=d.class_id.to_numpy()
primary=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
angle=json.loads((OUT/'ANGLE_ENROLLMENT.json').read_text(encoding='utf-8'))
stored=pd.read_csv(OUT/'angle_metrics.csv')
stored=stored[stored.policy=='angle_matched'].set_index(['episode','draw','arm'])
rows=[];checked=0
for e in angle:
    origin=primary[e['episode']]
    with np.load(OUT/'angle_predictions'/f"episode_{e['episode']:03d}.npz") as z:
        q=z['query_rows'];p=z['probabilities'].argmax(-1);y=labels[q]
        np.testing.assert_allclose(z['probabilities'].sum(-1),1,atol=1e-12)
        for draw in range(5):
            for ai,arm in enumerate(['SS','OS']):
                metric=[100*accuracy_score(y,p[draw,ai]),100*balanced_accuracy_score(y,p[draw,ai])]
                np.testing.assert_allclose(metric,stored.loc[(e['episode'],draw,arm),['oa','ba']].to_numpy(float),atol=1e-10)
                checked+=1
            for c in np.unique(y):
                mask=y==c
                same=p[draw,0][mask]==c;other=p[draw,1][mask]==c
                rows.append(dict(episode=e['episode'],budget=origin['budget'],query_port=origin['query_port'],support_port=origin['support_port'],
                    acquisition=origin['acquisition'],draw=draw,class_id=int(c),n=int(mask.sum()),lost=int((same&~other).sum()),rescued=int((~same&other).sum()),
                    delta_recall=100*float(other.mean()-same.mean())))
ac=pd.DataFrame(rows);ac.to_csv(OUT/'angle_class_effects.csv',index=False)
ac.groupby(['budget','query_port','support_port','class_id']).delta_recall.mean().groupby(['budget','class_id']).mean().to_csv(OUT/'angle_class_summary.csv')
c=pd.read_csv(OUT/'classes.csv')
index=['episode','budget','query_port','support_port','draw','mode','class_id']
s=c[c.cell=='SS'].set_index(index);o=c[c.cell=='OS'].set_index(index)
delta=(o.recall-s.recall).rename('delta_recall').reset_index()
delta.to_csv(OUT/'port_class_effects.csv',index=False)
delta.groupby(['budget','query_port','support_port','mode','class_id']).delta_recall.mean().groupby(['budget','mode','class_id']).mean().to_csv(OUT/'port_class_summary.csv')
source=np.load(OUT/'source_rows.npy')
targetrows=np.unique(np.concatenate([np.array(e['query']+sum([a[c] for a in e['draws'] for c in ['SS','OS','SO','OO']],[])) for e in primary]))
assert not set(d.iloc[source].port)&set(d.iloc[targetrows].port)
assert not set(d.iloc[source].mmsi)&set(d.iloc[targetrows].mmsi)
assert not set(d.iloc[source].acquisition)&set(d.iloc[targetrows].acquisition)
coverage=[]
for kind,entries in [('primary',primary),('angle',angle)]:
    for budget in [16,64]:
        ee=[e for e in entries if (e['budget'] if kind=='primary' else primary[e['episode']]['budget'])==budget]
        q=np.unique(np.concatenate([np.array(e['query']) for e in ee]))
        acq={e['acquisition'] if kind=='primary' else primary[e['episode']]['acquisition'] for e in ee}
        coverage.append(dict(kind=kind,budget=budget,episodes=len(ee),unique_queries=len(q),
            unique_vessels=d.iloc[q].mmsi.nunique(),unique_acquisitions=len(acq),classes=sorted(d.iloc[q].class_id.unique().tolist())))
cond=pd.read_csv(OUT/'angle_support_conditions.csv')
gap=(cond[cond.arm=='OS'].set_index(['episode','draw']).mean_angle-cond[cond.arm=='SS'].set_index(['episode','draw']).mean_angle).abs()
assert gap.max()<2
write('FINAL_CHECK.json',dict(status='complete',source_target_port_overlap=0,source_target_known_vessel_overlap=0,
    source_target_acquisition_overlap=0,angle_metric_rows=checked,angle_support_mean_difference_max=float(gap.max()),coverage=coverage,
    scope='Source fitted stages only; inherited encoder exposure not erased'))
print(json.dumps(coverage,ensure_ascii=False,indent=2))
print('angle classes',ac.groupby(['budget','query_port','support_port','class_id']).delta_recall.mean().groupby(['budget','class_id']).mean().to_string())
