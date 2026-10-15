"""Matched factorial outcomes, data strata, and independent reader checks."""
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import OUT, metadata, metrics, CELLS

enrollment=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
frame=metadata()
inc=pd.read_csv('E:/Hermes/paper_tracking/pixel_correction_review/chip_incidence_v1/incidence_by_instance.csv',
    usecols=['instance_id','product','status','incidence_deg'])
angles=frame[['instance_id','product']].merge(inc,on=['instance_id','product'],how='left',validate='one_to_one')
angles.loc[angles.status.ne('ok'),'incidence_deg']=np.nan
angle=angles.incidence_deg.to_numpy()
rows=[]; classes=[]; conditions=[]; transitions=[]
checked=0; records=[]
for e in enrollment:
    with np.load(OUT/'predictions'/f"episode_{e['episode']:03d}.npz") as z:
        q=z['query_rows']; supports=z['support_rows']; probs=z['probabilities']; modes=list(z['modes'])
    y=frame.class_id.to_numpy()[q]
    np.testing.assert_allclose(probs.sum(-1),1,atol=1e-12)
    p=probs.argmax(-1)
    records.append(dict(episode=e['episode'],budget=e['budget'],query_port=e['query_port'],support_port=e['support_port'],
        acquisition=e['acquisition'],queries=len(q),classes=len(np.unique(y)),query_angle_known=int(np.isfinite(angle[q]).sum())))
    for draw in range(5):
        for ci,cell in enumerate(CELLS):
            s=supports[draw,ci]
            same_product=float(np.mean(np.isin(frame.iloc[s]['product'],frame.iloc[q]['product'])))
            delta=[]
            for c in np.unique(y):
                sq=s[frame.class_id.to_numpy()[s]==c]; qq=q[y==c]
                aq=angle[qq]; ass=angle[sq]
                if np.isfinite(aq).any() and np.isfinite(ass).any():
                    delta.append(abs(float(np.nanmedian(aq)-np.nanmedian(ass))))
            startq=pd.to_datetime(frame.iloc[q].start,format='%Y%m%dT%H%M%S').median()
            starts=pd.to_datetime(frame.iloc[s].start,format='%Y%m%dT%H%M%S')
            timegap=float((starts-startq).dt.total_seconds().abs().median())
            conditions.append(dict(episode=e['episode'],budget=e['budget'],draw=draw,cell=cell,
                query_port=e['query_port'],support_port=e['support_port'],support_angle_known=int(np.isfinite(angle[s]).sum()),
                same_product_fraction=same_product,start_time_gap_seconds=timegap,
                class_matched_median_angle_gap=float(np.mean(delta)) if delta else np.nan))
            for mi,mode in enumerate(modes):
                got=metrics(y,p[draw,ci,mi])
                independent=[100*accuracy_score(y,p[draw,ci,mi]),100*recall_score(y,p[draw,ci,mi],labels=np.unique(y),average='macro',zero_division=0)]
                np.testing.assert_allclose([got['oa'],got['ba']],independent,atol=1e-10)
                checked+=1
                rows.append(dict(episode=e['episode'],budget=e['budget'],query_port=e['query_port'],support_port=e['support_port'],
                    acquisition=e['acquisition'],draw=draw,cell=cell,mode=mode,**got))
                for c in np.unique(y):
                    mask=y==c
                    classes.append(dict(episode=e['episode'],budget=e['budget'],query_port=e['query_port'],support_port=e['support_port'],
                        acquisition=e['acquisition'],draw=draw,cell=cell,mode=mode,class_id=int(c),n=int(mask.sum()),
                        recall=100*float(np.mean(p[draw,ci,mi][mask]==c)),support_n=int(e['quota'][c])))
        for mi,mode in enumerate(modes):
            a=p[draw,0,mi]==y; b=p[draw,1,mi]==y
            transitions.append(dict(episode=e['episode'],budget=e['budget'],draw=draw,mode=mode,
                query_port=e['query_port'],support_port=e['support_port'],rescued=int((~a&b).sum()),lost=int((a&~b).sum()),n=len(y)))
df=pd.DataFrame(rows);df.to_csv(OUT/'metrics.csv',index=False)
pd.DataFrame(classes).to_csv(OUT/'classes.csv',index=False)
pd.DataFrame(records).to_csv(OUT/'query_coverage.csv',index=False)
cond=pd.DataFrame(conditions);cond.to_csv(OUT/'conditions.csv',index=False)
pd.DataFrame(transitions).to_csv(OUT/'transitions.csv',index=False)
ix=['episode','budget','query_port','support_port','acquisition','draw','mode']
tables={c:df[df.cell==c].set_index(ix)[['oa','ba']] for c in CELLS}
effects=[]
for name,value in [('port_same_acquisition',tables['OS']-tables['SS']),
    ('acquisition_same_port',tables['SO']-tables['SS']),('port_other_acquisition',tables['OO']-tables['SO']),
    ('acquisition_other_port',tables['OO']-tables['OS']),('both_changes',tables['OO']-tables['SS']),
    ('interaction',tables['OO']-tables['OS']-tables['SO']+tables['SS'])]:
    effects.append(value.rename(columns={'oa':'delta_oa','ba':'delta_ba'}).reset_index().assign(contrast=name))
ef=pd.concat(effects,ignore_index=True);ef.to_csv(OUT/'effects.csv',index=False)
scene=ef.groupby(['episode','budget','query_port','support_port','acquisition','mode','contrast'])[['delta_oa','delta_ba']].mean().reset_index()
pair=scene.groupby(['budget','query_port','support_port','mode','contrast'])[['delta_oa','delta_ba']].mean().reset_index()
pair.to_csv(OUT/'direction_effects.csv',index=False)
summary=pair.groupby(['budget','mode','contrast']).agg(directions=('query_port','size'),delta_oa=('delta_oa','mean'),delta_ba=('delta_ba','mean'),
    negative_ba_directions=('delta_ba',lambda x:int((x<0).sum()))).reset_index()
summary.to_csv(OUT/'summary.csv',index=False)
absolute=df.groupby(['episode','budget','query_port','support_port','mode','cell'])[['oa','ba']].mean().groupby(
    ['budget','query_port','support_port','mode','cell']).mean().groupby(['budget','mode','cell']).mean()
absolute.to_csv(OUT/'absolute_summary.csv')

# Independent sklearn dual solver checks weighting, residual targets and probability fusion.
z=np.load(OUT/'encoded.npz'); qrows=z['rows']; labels=frame.class_id.to_numpy()
maxerror=0.
for e in [enrollment[0],next(e for e in enrollment if e['budget']==64)]:
    saved=np.load(OUT/'predictions'/f"episode_{e['episode']:03d}.npz")
    q=np.searchsorted(qrows,saved['query_rows'])
    for ci,cell in enumerate(CELLS):
        sr=saved['support_rows'][0,ci];s=np.searchsorted(qrows,sr)
        sy=labels[sr];w=compute_sample_weight('balanced',sy)
        target=2*np.eye(8)[sy]-1
        outputs={}
        for kind,phi,prior in [('base',z['b'],z['pb']),('expert',np.c_[z['r'],z['t']],z['pd'])]:
            kss=phi[s]@phi[s].T+1;kqs=phi[q]@phi[s].T+1
            fitted=KernelRidge(alpha=8.,kernel='precomputed').fit(kss,target-prior[s],sample_weight=w)
            outputs[kind]=softmax(prior[q]+fitted.predict(kqs),axis=1)
        actual=.5*(outputs['base']+outputs['expert'])
        error=float(np.max(np.abs(actual-saved['probabilities'][0,ci,2])))
        maxerror=max(maxerror,error)
        np.testing.assert_allclose(actual,saved['probabilities'][0,ci,2],atol=1e-9,rtol=1e-9)
del z

from plot import main as render_plot
render_plot()
receipt=dict(status='complete',independent_metric_rows=checked,independent_sklearn_solver_cases=8,solver_max_probability_error=maxerror,
    model='Common compact diagnostic reader, not the frozen full legacy model',
    primary_summary=summary[(summary['mode']=='dual_late')&(summary.contrast=='port_same_acquisition')].to_dict('records'),
    no_population_ci='Only three target ports and reciprocal/shared acquisitions; support draws are not independent ports')
(OUT/'VERIFIED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(summary[summary['mode']=='dual_late'].to_string(index=False))
print(absolute.loc[(slice(None),'dual_late',slice(None)),:].to_string())
print('INDEPENDENT CHECK',maxerror,checked)
