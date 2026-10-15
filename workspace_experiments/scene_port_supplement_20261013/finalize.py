"""Record actual cohort coverage and verify the reported class decomposition."""
import json
import numpy as np
import pandas as pd
from run import OUT, frame, write

d=frame()
plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
coverage=[]
for policy in ['class_matched','class_angle_matched']:
    for budget in [16,64]:
        selected=[e for e in plans if e['policy']==policy and e['budget']==budget]
        q=np.unique(np.concatenate([e['query'] for e in selected]))
        f=d.iloc[q]
        counts=[d.iloc[e['query']].class_id.value_counts().to_dict() for e in selected]
        coverage.append(dict(policy=policy,budget=budget,episodes=len(selected),ports=sorted(f.port.unique()),
            acquisitions=int(f.acquisition.nunique()),unique_queries=len(q),unique_query_vessels=int(f.mmsi.nunique()),
            query_class_counts={int(c):int(n) for c,n in f.class_id.value_counts().sort_index().items()},
            minimum_scene_class_n=min(min(x.values()) for x in counts),
            start_min=str(f.start.min()),start_max=str(f.start.max()),
            complete_products=int(f['product'].nunique())))
write('SCOPE.json',coverage)
tr=pd.read_csv(OUT/'transitions.csv');ef=pd.read_csv(OUT/'effects.csv')
keys=['episode','budget','policy','query_port','support_port','acquisition','draw','mode']
class_sums=tr.groupby(keys+['contrast']).ba_delta_contribution.sum().reset_index()
class_sums['contrast']=class_sums.contrast.map({'OS-SS':'port_same_acq','SO-SS':'acq_same_port','OO-source_only':'OO_minus_source'})
joined=class_sums.merge(ef,on=keys+['contrast'],validate='one_to_one')
assert len(joined)==len(class_sums)
np.testing.assert_allclose(joined.ba_delta_contribution,joined.delta_ba,atol=1e-10)
condition=pd.read_csv(OUT/'conditions.csv').merge(pd.DataFrame(plans)[['episode','policy','budget']],on='episode',validate='many_to_one')
angle=condition[condition.policy=='class_angle_matched'].pivot(index=['episode','draw'],columns='cell',values='mean_angle')
gap=float((angle['OS']-angle['SS']).abs().max())
write('COMPLETE.json',dict(status='complete',scope=coverage,class_effect_sum_checks=len(joined),
    max_angle_matched_SS_OS_mean_gap_deg=gap,main_model_changed=False,manuscript_changed=False,
    no_new_encoder_training=True,new_source_fits=0,
    protocol='Frozen-feature diagnostic; acquisition/port/known-identity isolation at fitted-readout stages, not end-to-end new-data confirmation'))
print(json.dumps(coverage,ensure_ascii=False,indent=2))
print('CLASS_SUM_CHECKS',len(joined),'MAX_ANGLE_GAP',gap)
