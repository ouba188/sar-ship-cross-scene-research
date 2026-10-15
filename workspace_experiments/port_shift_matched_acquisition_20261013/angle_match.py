"""Predefined two-degree class-angle support match, same-acquisition only."""
import json
import pickle
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from sklearn.metrics import recall_score
from study import OUT,FEATURES,metadata,operator,encode,SEED,write

d=metadata()
inc=pd.read_csv('E:/Hermes/paper_tracking/pixel_correction_review/chip_incidence_v1/incidence_by_instance.csv',
    usecols=['instance_id','product','status','incidence_deg'])
a=d[['instance_id','product']].merge(inc,on=['instance_id','product'],how='left',validate='one_to_one')
a.loc[a.status.ne('ok'),'incidence_deg']=np.nan
d['angle']=a.incidence_deg.to_numpy()
d['angle_bin']=np.floor(d.angle/2).fillna(-1).astype(int)
original=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
available=np.load(FEATURES/'rows.npy')
pool=d.iloc[available]; pool=pool[pool.known_ship&pool.angle.notna()]
plans=[];eligibility=[]
for e in original:
    q=np.array([i for i in e['query'] if np.isfinite(d.angle.iloc[i])])
    if len(q)<5 or d.iloc[q].class_id.nunique()<2:
        eligibility.append(dict(episode=e['episode'],eligible=False,reason='query_angle_coverage',query_n=len(q)));continue
    p=pool[(pool.acquisition==e['acquisition'])&pool.port.isin([e['query_port'],e['support_port']])]
    both=p.groupby('mmsi').port.nunique();p=p[~p.mmsi.isin(both[both>1].index)]
    p=p[~p.mmsi.isin(d.iloc[e['query']].mmsi)]
    groups=[p[p.port==name].drop_duplicates('mmsi') for name in [e['query_port'],e['support_port']]]
    counts=[g.groupby(['class_id','angle_bin']).size() for g in groups]
    cap=pd.concat(counts,axis=1).fillna(0).min(axis=1).astype(int)
    cap=cap[cap>0]
    if cap.sum()<e['budget'] or cap.index.get_level_values(0).nunique()<2:
        eligibility.append(dict(episode=e['episode'],eligible=False,reason='matched_capacity',capacity=int(cap.sum()),query_n=len(q)));continue
    # Ensure each available class has a quota, then allocate by common stratum capacity.
    quota=np.zeros(len(cap),int)
    for c in cap.index.get_level_values(0).unique():
        ix=np.flatnonzero(cap.index.get_level_values(0)==c)
        quota[ix[np.argmax(cap.iloc[ix].to_numpy())]]=1
    while quota.sum()<e['budget']:
        priority=cap.to_numpy()/cap.sum()*e['budget']-quota
        priority[quota>=cap.to_numpy()]=-np.inf
        quota[int(priority.argmax())]+=1
    supports=[]
    for draw in range(5):
        rng=np.random.default_rng(SEED+500000+e['episode']*100+draw)
        arms=[]
        for g in groups:
            rows=[]
            for (c,b),n in zip(cap.index,quota):
                if n:rows.extend(rng.choice(g.index[(g.class_id==c)&(g.angle_bin==b)],n,replace=False).tolist())
            arms.append(rows)
        supports.append(arms)
    plans.append(dict(episode=e['episode'],query=q.tolist(),supports=supports,
        strata=[dict(class_id=int(c),angle_bin=int(b),n=int(n)) for (c,b),n in zip(cap.index,quota) if n]))
    eligibility.append(dict(episode=e['episode'],eligible=True,reason='eligible',capacity=int(cap.sum()),query_n=len(q)))
write('ANGLE_PROTOCOL.json',dict(bins='floor(deg/2), fixed origin0',missing='exclude, no imputation',
    query='Original query subset with known angle, same for both arms; all original query ships excluded from supports',
    sampling='Exact class-angle-bin counts in same-acq same/other port; same B;5draws; original enrollment only, no outcome-based expansion',
    scope='Incomplete-angle subset; within-bin residuals/other conditions persist; not pure causal port isolation',
    source_refit=False,selection_before_predictions=True))
write('ANGLE_ENROLLMENT.json',plans)
pd.DataFrame(eligibility).to_csv(OUT/'angle_eligibility.csv',index=False)
print('ANGLE ELIGIBLE',len(plans),'/',len(original),flush=True)
if not plans:raise SystemExit(0)
ev=operator()
with (OUT/'fits/geometry.pkl').open('rb') as f:bundle=pickle.load(f)
arrays={k:np.load(FEATURES/(k+'.npy'),mmap_mode='r') for k in ['raw','c4','extra']}
rows=np.unique(np.concatenate([np.array(e['query']+sum([x for draw in e['supports'] for x in draw],[])) for e in plans]))
out=OUT/'angle_predictions';out.mkdir(exist_ok=True)
records=[]; conditions=[]
with threadpool_limits(limits=2):
    blocks=[encode(ev,arrays,rows[i:i+512],bundle) for i in range(0,len(rows),512)]
    f={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    for e in plans:
        origin=original[e['episode']];q=np.array(e['query']);qi=np.searchsorted(rows,q);y=d.iloc[q].class_id.to_numpy()
        probability=np.empty((5,2,len(q),8),float)
        saved=np.load(OUT/'predictions'/f"episode_{e['episode']:03d}.npz")
        oldloc=pd.Index(saved['query_rows']).get_indexer(q);assert (oldloc>=0).all()
        for draw,arms in enumerate(e['supports']):
            for ai,sr in enumerate(arms):
                s=np.array(sr);si=np.searchsorted(rows,s)
                assert not set(d.iloc[s].mmsi)&set(d.iloc[q].mmsi)
                assert d.iloc[s].mmsi.nunique()==origin['budget']
                actual=d.iloc[s].groupby(['class_id','angle_bin']).size().to_dict()
                expected={(t['class_id'],t['angle_bin']):t['n'] for t in e['strata']}
                assert actual==expected
                target=d.iloc[s].class_id.to_numpy()
                kb=f['b'][si]@f['b'][si].T+1
                kt=f['r'][si]@f['r'][si].T+f['t'][si]@f['t'][si].T+1
                ab=ev.coefficients(kb,f['pb'][si],target,8.)
                at=ev.coefficients(kt,f['pd'][si],target,8.)
                bp=softmax(f['pb'][qi]+(f['b'][qi]@f['b'][si].T+1)@ab,axis=1)
                ep=softmax(f['pd'][qi]+(f['r'][qi]@f['r'][si].T+f['t'][qi]@f['t'][si].T+1)@at,axis=1)
                prob=.5*(bp+ep);probability[draw,ai]=prob
                conditions.append(dict(episode=e['episode'],draw=draw,arm=['SS','OS'][ai],mean_angle=d.iloc[s].angle.mean()))
                for policy,p in [('angle_matched',prob),('original_on_same_query',saved['probabilities'][draw,ai,2][oldloc])]:
                    pred=p.argmax(1)
                    records.append(dict(episode=e['episode'],budget=origin['budget'],query_port=origin['query_port'],
                        support_port=origin['support_port'],draw=draw,arm=['SS','OS'][ai],policy=policy,n=len(y),
                        oa=100*float(np.mean(pred==y)),ba=100*recall_score(y,pred,labels=np.unique(y),average='macro',zero_division=0)))
        np.savez_compressed(out/f"episode_{e['episode']:03d}.npz",probabilities=probability,query_rows=q,supports=np.array(e['supports']))
df=pd.DataFrame(records);df.to_csv(OUT/'angle_metrics.csv',index=False)
pd.DataFrame(conditions).to_csv(OUT/'angle_support_conditions.csv',index=False)
ix=['episode','budget','query_port','support_port','draw','policy']
delta=df[df.arm=='OS'].set_index(ix)[['oa','ba']]-df[df.arm=='SS'].set_index(ix)[['oa','ba']]
delta=delta.rename(columns={'oa':'delta_oa','ba':'delta_ba'}).reset_index()
delta.to_csv(OUT/'angle_effects.csv',index=False)
pair=delta.groupby(['budget','query_port','support_port','policy'])[['delta_oa','delta_ba']].mean().reset_index()
pair.to_csv(OUT/'angle_direction_effects.csv',index=False)
summary=pair.groupby(['budget','policy']).agg(directions=('query_port','size'),delta_oa=('delta_oa','mean'),delta_ba=('delta_ba','mean')).reset_index()
summary.to_csv(OUT/'angle_summary.csv',index=False)
write('ANGLE_COMPLETE.json',dict(episodes=len(plans),pairs=len(plans)*5,source_fits=0,neural_training=0,
    summary=summary.to_dict('records')))
print(pair.to_string(index=False));print(summary.to_string(index=False))
