"""Frozen reference-memory expansion over all eligible held-port acquisitions."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='2'
import argparse
import json
import pickle
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
PILOT=ROOT/'scene_applicability_design_20261013/pilot'
sys.path.insert(0,str(PILOT))
from run_pilot import metadata,SEED,FEATURES,sample,quota,operator,encode
from core import ARMS,RHO,relation,kernel,map_at_r


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def inventory():
    d=metadata(); initial=np.load(ROOT/'port_shift_matched_acquisition_20261013/source_rows.npy')
    source=np.load(PILOT/'source_rows.npy')
    previous=json.loads((PILOT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    old={(e['port'],e['acquisition']):e for e in previous}
    ports=sorted({e['port'] for e in previous})
    oldq=np.concatenate([e['query'] for e in previous]); oldships=set(d.iloc[oldq].mmsi)
    conflicts=d[d.known_ship].groupby('mmsi').class_id.nunique()
    p=d.loc[initial];p=p[p.known_ship & ~p.mmsi.isin(conflicts[conflicts>1].index)]
    plans=[];coverage=[]
    # Keep pilot RNG order, including per-port gallery selection, for exact replay.
    for pi,(port,g) in enumerate(p.groupby('port',sort=True)):
        if port not in ports:continue
        rng=np.random.default_rng(SEED+pi);qships=[]
        for _,group in g.groupby('class_id',sort=True):
            ships=group.mmsi.unique()
            qships.extend(rng.choice(ships,max(1,int(np.ceil(.3*len(ships)))),replace=False))
        qp=g[g.mmsi.isin(qships)];sp=g[~g.mmsi.isin(qships)];eligible=[]
        for acq,qg in qp.groupby('acquisition',sort=True):
            if qg.mmsi.nunique()<8 or qg.class_id.nunique()<2:continue
            pools=[sp[sp.acquisition==acq],sp[sp.acquisition!=acq]]
            cap=np.minimum(*[a.groupby('class_id').mmsi.nunique().reindex(range(8),fill_value=0).to_numpy() for a in pools])
            if cap.sum()<64 or (cap>0).sum()<2:continue
            q=[int(rng.choice(qg.index[qg.mmsi==s])) for s in qg.mmsi.unique()]
            if len(q)>128:q=rng.choice(q,128,replace=False).tolist()
            if d.iloc[q].class_id.nunique()<2:continue
            gallery_pool=pools[1];gs=gallery_pool.mmsi.unique()
            if len(gs)>512:gs=rng.choice(gs,512,replace=False)
            gallery=[int(rng.choice(gallery_pool.index[gallery_pool.mmsi==s])) for s in gs]
            draws=[];q64,q16=quota(cap,64),quota(cap,16)
            for draw in range(5):
                rr=np.random.default_rng(SEED+pi*1000+len(eligible)*10+draw)
                draws.append({str(b):[sample(a,qq,rr) for a in pools] for b,qq in [(16,q16),(64,q64)]})
            e=dict(port=port,acquisition=acq,start=str(qg.start.min()),query=q,gallery=gallery,draws=draws,
                   quotas={'16':q16.tolist(),'64':q64.tolist()})
            if (port,acq) in old:
                for key in e:assert e[key]==old[port,acq][key],key
            e['pilot_episode']=old.get((port,acq),{}).get('episode',-1)
            e['cohort']='pilot' if e['pilot_episode']>=0 else 'new_acquisition'
            e['pilot_unseen_ship_mask']=[s not in oldships for s in d.iloc[q].mmsi]
            eligible.append(e)
        for e in sorted(eligible,key=lambda a:(a['start'],a['acquisition'])):
            e['episode']=len(plans);plans.append(e)
            q=d.iloc[e['query']];mask=np.array(e['pilot_unseen_ship_mask'])
            coverage.append(dict(episode=e['episode'],port=port,acquisition=e['acquisition'],cohort=e['cohort'],
                query_n=len(q),query_ships=q.mmsi.nunique(),new_ship_queries=int(mask.sum()),
                class_counts=json.dumps(q.class_id.value_counts().sort_index().to_dict()),
                new_ship_class_counts=json.dumps(q[mask].class_id.value_counts().sort_index().to_dict())))
    assert sum(e['cohort']=='pilot' for e in plans)==6
    taskrows=np.unique(np.concatenate([e['query']+e['gallery']+sum([s for a in e['draws'] for ss in a.values() for s in ss],[]) for e in plans]))
    assert not set(d.iloc[source].mmsi)&set(d.iloc[taskrows].mmsi)
    assert not set(d.iloc[source].acquisition)&set(d.iloc[taskrows].acquisition)
    assert not set(d.iloc[source].port)&set(ports)
    write('ENROLLMENT.json',plans);pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    write('PROTOCOL.json',dict(date='2026-10-14',status='Frozen before expanded predictions',ports=ports,
        source_rows=len(source),source_fitted_overlap=0,neural_training=0,new_geometry_fits=0,new_memory_fits=0,
        candidates='Original and all3 relative-profile memory controls from previous pilot, exact reused calibration',
        rho=RHO,ridge=8.,budgets=[16,64],draws=5,probability_temperature=1.,fusion=.5,
        primary='All newly enrolled acquisitions, average draws then scenes within ports then equal ports',
        secondary='Query ships not observed in pilot queries, evaluated as fixed subset without resampling supports',
        pooled='Also pool repeated query/support exposures; not independent ships or independent training seeds',
        cohort='All metadata-eligible acquisitions of same3source-held ports, retains original6 for exact replay',
        scope='Compact reader, historical encoder/data exposure remains; NOT new independent dataset',
        old_artifacts_modified=False,external_datasets=False,repair_rule='Preserve frozen first results; diagnose before atmostone targeted correction'))
    print(pd.DataFrame(coverage).groupby(['port','cohort'])[['query_n','new_ship_queries']].agg(['count','sum']).to_string(),flush=True)


def predict():
    started=time.perf_counter();d=metadata();ev=operator()
    plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    with (PILOT/'fits/geometry.pkl').open('rb') as h:bundle=pickle.load(h)
    with (PILOT/'memories.pkl').open('rb') as h:memories=pickle.load(h)
    calibration=json.loads((PILOT/'relative_repair/PROTOCOL.json').read_text(encoding='utf-8'))['calibration']
    old=np.load(PILOT/'encoded.npz');oldrel=np.load(PILOT/'relative_repair/relations.npz')
    rows=np.unique(np.concatenate([e['query']+e['gallery']+sum([s for a in e['draws'] for ss in a.values() for s in ss],[]) for e in plans]))
    known=np.isin(rows,old['rows']);fresh=rows[~known]
    arrays={k:np.load(FEATURES/(k+'.npy'),mmap_mode='r') for k in ['raw','c4','extra']}
    np.testing.assert_array_equal(np.load(FEATURES/'ids.npy')[rows],d.iloc[rows].instance_id)
    assert np.isin(rows,np.load(FEATURES/'rows.npy')).all()
    blocks=[encode(ev,arrays,fresh[i:i+512],bundle) for i in range(0,len(fresh),512)]
    f={}
    for key in ['r','t','b','pr','pd','pb']:
        cached=old[key];f[key]=np.empty((len(rows),cached.shape[1]),dtype=cached.dtype)
        f[key][known]=cached[np.searchsorted(old['rows'],rows[known])]
        if blocks:f[key][~known]=np.concatenate([b[key] for b in blocks])
    z=np.c_[f['r'],f['t']];rel={}
    for arm in ARMS[1:]:
        r=relation(z,memories[arm]);c=calibration[arm]
        rel[arm]=(r/np.maximum(np.linalg.norm(r,axis=1,keepdims=True),1e-15)-np.array(c['center']))*c['scale']
        np.testing.assert_allclose(rel[arm][known],oldrel[arm][np.searchsorted(oldrel['rows'],rows[known])],atol=1e-10,rtol=1e-10)
        rel[arm][known]=oldrel[arm][np.searchsorted(oldrel['rows'],rows[known])]
    np.savez_compressed(OUT/'encoded.npz',rows=rows,**f,**{'rel_'+k:v for k,v in rel.items()})
    (OUT/'predictions').mkdir(exist_ok=True);replays=[];retrieval=[]
    for e in plans:
        q=np.array(e['query']);qi=np.searchsorted(rows,q);y=d.iloc[q].class_id.to_numpy(int)
        g=np.array(e['gallery']);gi=np.searchsorted(rows,g);gy=d.iloc[g].class_id.to_numpy(int)
        assert not set(d.iloc[g].mmsi)&set(d.iloc[q].mmsi)
        for arm in ARMS:
            r=None if arm=='original' else rel[arm]
            sim=kernel(z[qi],z[gi],None if r is None else r[qi],None if r is None else r[gi])
            ap=map_at_r(sim,y,gy)
            for row,c,v,ns in zip(q,y,ap,e['pilot_unseen_ship_mask']):
                retrieval.append(dict(episode=e['episode'],port=e['port'],cohort=e['cohort'],row=int(row),
                    class_id=int(c),arm=arm,map_at_r=float(v),new_ship=ns))
        for b in [16,64]:
            probs=np.empty((5,2,4,2,len(q),8));supports=[]
            for draw,a in enumerate(e['draws']):
                supports.append(a[str(b)])
                for ci,s in enumerate(a[str(b)]):
                    s=np.array(s);si=np.searchsorted(rows,s);sy=d.iloc[s].class_id.to_numpy(int)
                    assert d.iloc[s].mmsi.nunique()==b
                    assert not set(d.iloc[s].mmsi)&set(d.iloc[q].mmsi)
                    np.testing.assert_array_equal(np.bincount(sy,minlength=8),e['quotas'][str(b)])
                    assert (d.iloc[s].acquisition.eq(e['acquisition']) if ci==0 else d.iloc[s].acquisition.ne(e['acquisition'])).all()
                    ab=ev.coefficients(f['b'][si]@f['b'][si].T+1,f['pb'][si],sy,8.)
                    bp=softmax(f['pb'][qi]+(f['b'][qi]@f['b'][si].T+1)@ab,axis=1)
                    for ai,arm in enumerate(ARMS):
                        r=None if arm=='original' else rel[arm]
                        ks=kernel(z[si],z[si],None if r is None else r[si],None if r is None else r[si])
                        kq=kernel(z[qi],z[si],None if r is None else r[qi],None if r is None else r[si])
                        alpha=ev.coefficients(ks,f['pd'][si],sy,8.)
                        ep=softmax(f['pd'][qi]+kq@alpha,axis=1)
                        probs[draw,ci,ai]=[ep,.5*(bp+ep)]
            if e['pilot_episode']>=0:
                oi=e['pilot_episode']
                raw=np.load(PILOT/'predictions'/f'episode_{oi:02d}_b{b}.npz')['probabilities']
                fixed=np.load(PILOT/'relative_repair'/f'episode_{oi:02d}_b{b}.npz')['probabilities']
                expected=np.concatenate([raw[:,:,:1],fixed],axis=2)
                np.testing.assert_array_equal(probs.argmax(-1),expected.argmax(-1))
                np.testing.assert_allclose(probs,expected,atol=1e-10,rtol=1e-10)
                replays.append(dict(episode=e['episode'],budget=b,max_error=float(np.max(np.abs(probs-expected)))))
            np.savez_compressed(OUT/'predictions'/f"episode_{e['episode']:02d}_b{b}.npz",probabilities=probs,
                query_rows=q,support_rows=np.array(supports),arms=np.array(ARMS),modes=np.array(['expert','compact_full']))
        print('PREDICTED',e['episode'],e['port'],e['cohort'],len(q),flush=True)
    pd.DataFrame(retrieval).to_csv(OUT/'retrieval.csv',index=False)
    write('REPLAY.json',replays)
    write('PREDICTIONS_COMPLETE.json',dict(episodes=len(plans),expert_solves=len(plans)*2*5*2*4,
        source_geometry_fits=0,neural_training=0,seconds=time.perf_counter()-started,
        cached_rows=int(known.sum()),newly_encoded_rows=len(fresh),pilot_replays=len(replays)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['inventory','predict'])
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        inventory() if args.stage=='inventory' else predict()
