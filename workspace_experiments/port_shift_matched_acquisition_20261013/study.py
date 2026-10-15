"""Cross port x acquisition support experiment with a shared held-port reader."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
PARENT=OUT.parent
FEATURES=PARENT/'ports_root_cause_20261013/features/original'
sys.path.insert(0,str(PARENT/'scene_shift_probe_20261013'))
from analyze import metadata, metrics
SEED=20261014
CELLS=('SS','OS','SO','OO')


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def inventory():
    d=metadata()
    available=np.load(FEATURES/'rows.npy')
    np.testing.assert_array_equal(np.load(FEATURES/'ids.npy')[available],d.instance_id.to_numpy()[available])
    s=d.groupby('acquisition').port.nunique()
    shared=s[s>1].index
    held=sorted(d[d.acquisition.isin(shared)].port.unique())
    known=d[d.port.isin(held)&d.known_ship].mmsi.unique()
    source=available[~d.iloc[available].port.isin(held)&~d.iloc[available].mmsi.isin(known)]
    assert np.unique(d.class_id.to_numpy()[source]).size==8
    assert not set(d.iloc[source].port)&set(held)
    assert not set(d.iloc[source].mmsi)&set(known)
    np.save(OUT/'source_rows.npy',source)
    write('PROTOCOL.json',dict(seed=SEED,held_ports=held,source_n=len(source),
        source='Shared frozen original R50/C4/R18 caches, available rows only; exclude all five shared-acquisition ports AND their known vessels',
        reader='Existing compact dual-prior expert + C4 base equal probability fusion; NOT legacy frequency/stable/incidence full model',
        fitting='One new common source geometry and source heads; no encoder training; no target outcomes for hyperparameter selection',
        source_ridge=1.,support_ridge=8.,fusion=.5,budgets=[16,64],draws=5,
        cells=dict(SS='same port same acquisition',OS='other port same acquisition',SO='same port other acquisitions',OO='other port other acquisitions'),
        acquisition='platform+absoluteorbit+datatake; product and incidence tracked separately',
        query='One random observation per known vessel;30percent vessel holdout; all four cells use identical queries',
        support='Exact B/perclasscounts across4cells; known unique ships; exclude all queryMMSIs; one random observation per support ship',
        selection='At most3 eligible chronologically evenly spaced acquisitions per directed port pair/B; labels used only for matched diagnostic sampling',
        scene_filter='Exclude vessels appearing in multiple ports within the same acquisition from same-acquisition query/support pools',
        required='At least5query vessels,2queryclasses and2classes available in all4support pools',
        primary='OS-SS; average draws then scenes within directedpair; show bothdirections and unorderedpairs',
        secondary='SO-SS,OO-SO,OO-OS and interaction OO-OS-SO+SS; no additive causal attribution',
        scope='Frozen pretrained exposure remains; same datatake is not equal incidence/background; three China ports expected, no broad population inference',
        main_model_changed=False))
    target=d.iloc[available]
    target=target[target.port.isin(held)&target.known_ship].copy()
    conflict=target.groupby('mmsi').class_id.nunique()
    target=target[~target.mmsi.isin(conflict[conflict>1].index)]
    overlap=target.groupby(['acquisition','mmsi']).port.nunique()
    overlap=set(overlap[overlap>1].index)
    ambiguous=np.array([(a,m) in overlap for a,m in zip(target.acquisition,target.mmsi)])
    target=target[~ambiguous]
    records=[]; availability=[]
    for pi,(qp,sp) in enumerate(itertools.permutations(held,2)):
        pair=target[target.port.isin([qp,sp])]
        groups=sorted(set(pair[pair.port==qp].acquisition)&set(pair[pair.port==sp].acquisition))
        eligible={16:[],64:[]}
        for si,acq in enumerate(groups):
            current=pair[(pair.port==qp)&(pair.acquisition==acq)]
            ships=current.mmsi.unique()
            if len(ships)<8: continue
            rng=np.random.default_rng(SEED+pi*1000+si)
            qship=rng.choice(ships,max(4,int(np.ceil(.3*len(ships)))),replace=False)
            q=np.array([rng.choice(current.index[current.mmsi==m]) for m in qship])
            if len(q)<5 or d.iloc[q].class_id.nunique()<2: continue
            pool=pair[~pair.mmsi.isin(qship)]
            pools=dict(SS=pool[(pool.port==qp)&(pool.acquisition==acq)],
                OS=pool[(pool.port==sp)&(pool.acquisition==acq)],
                SO=pool[(pool.port==qp)&(pool.acquisition!=acq)],
                OO=pool[(pool.port==sp)&(pool.acquisition!=acq)])
            capacity=np.min([p.groupby('class_id').mmsi.nunique().reindex(range(8),fill_value=0).to_numpy() for p in pools.values()],axis=0)
            if (capacity>0).sum()<2: continue
            for budget in (16,64):
                if capacity.sum()<budget: continue
                quota=(capacity>0).astype(int)
                while quota.sum()<budget:
                    score=capacity/capacity.sum()*budget-quota
                    score[quota>=capacity]=-np.inf
                    quota[int(score.argmax())]+=1
                draws=[]
                for draw in range(5):
                    rr=np.random.default_rng(SEED+pi*10000+si*100+budget+draw)
                    cell={}
                    for name,p in pools.items():
                        selected=[]
                        for c in range(8):
                            if not quota[c]: continue
                            sub=p[p.class_id==c]
                            selected_ships=rr.choice(sub.mmsi.unique(),quota[c],replace=False)
                            selected.extend(int(rr.choice(sub.index[sub.mmsi==m])) for m in selected_ships)
                        cell[name]=selected
                    draws.append(cell)
                eligible[budget].append(dict(query_port=qp,support_port=sp,acquisition=acq,start=current.start.min(),
                    budget=budget,query=q.tolist(),quota=quota.tolist(),draws=draws,
                    pool_sizes={k:int(v.mmsi.nunique()) for k,v in pools.items()}))
        for budget,items in eligible.items():
            items=sorted(items,key=lambda x:(x['start'],x['acquisition']))
            selected=[]
            if items:
                ix=np.unique(np.rint(np.linspace(0,len(items)-1,min(3,len(items)))).astype(int))
                selected=[items[k] for k in ix]
                records.extend(selected)
            availability.append(dict(query_port=qp,support_port=sp,budget=budget,shared_acquisitions=len(groups),
                eligible=len(items),selected=len(selected)))
    for i,e in enumerate(records): e['episode']=i
    write('ENROLLMENT.json',records)
    pd.DataFrame(availability).to_csv(OUT/'availability.csv',index=False)
    write('INVENTORY.json',dict(held_ports=held,source_n=len(source),source_class_counts=d.iloc[source].class_id.value_counts().sort_index().to_dict(),
        available_feature_rows=len(available),missing_cache_rows=len(d)-len(available),
        excluded_conflicting_identity_n=int((conflict>1).sum()),excluded_multiport_acquisition_observations=int(ambiguous.sum()),
        excluded_multiport_acquisition_identity_groups=len(overlap),episodes=len(records)))
    print(pd.DataFrame(availability).query('selected>0').to_string(index=False),flush=True)
    print('source',len(source),'episodes',len(records),flush=True)


def operator():
    sys.path.insert(0,str(PARENT/'public_oss_comparison_20261012'))
    import evaluate as ev
    return ev


def encode(ev,arrays,rows,bundle):
    model,stats,heads=(bundle[k] for k in ('model','stats','heads'))
    raw=arrays['raw'][rows]; extra=arrays['extra'][rows]
    rr,rd=ev.r50_views(raw,model,stats)
    x=ev._expert_inputs(raw,extra,model)
    x4=ev.pca_coordinates(arrays['c4'][rows],model)
    return dict(r=np.c_[rr,rd]*np.sqrt(.5),t=ev.ssl_map(extra,model,stats),
        b=(x4-model['residual_center'])/model['residual_scale']/np.sqrt(128),
        pr=heads['r50'].decision_function(x[:,:128]).astype(np.float32).astype(float),
        pd=heads['dual'].decision_function(x).astype(np.float32).astype(float),
        pb=heads['base'].decision_function(x4).astype(np.float32).astype(float))


def predict():
    from threadpoolctl import threadpool_limits
    from scipy.special import softmax
    ev=operator(); source=np.load(OUT/'source_rows.npy')
    arrays={k:np.load(FEATURES/(k+'.npy'),mmap_mode='r') for k in ['raw','c4','extra']}
    ev.fit_dir=lambda regime,name: OUT/'fits'
    ev.source_arrays=lambda regime,name:(source,arrays['raw'],arrays['c4'],arrays['extra'][source])
    with threadpool_limits(limits=2):
        started=time.perf_counter()
        bundle=ev.prepare('shared_frozen','held_shared_ports')
        np.testing.assert_array_equal(bundle['source_rows'],source)
        enrollment=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
        rows=np.unique(np.concatenate([np.array(e['query']+sum([a[c] for a in e['draws'] for c in CELLS],[])) for e in enrollment]))
        assert not np.intersect1d(source,rows).size
        blocks=[]
        for lo in range(0,len(rows),512): blocks.append(encode(ev,arrays,rows[lo:lo+512],bundle))
        f={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
        (OUT/'predictions').mkdir(exist_ok=True)
        d=metadata(); labels=d.class_id.to_numpy()
        checks=[]
        for e in enrollment:
            q=np.array(e['query']); qi=np.searchsorted(rows,q)
            pred=np.empty((5,4,3,len(q),8),dtype=float)
            for draw,supports in enumerate(e['draws']):
                for ci,cell in enumerate(CELLS):
                    s=np.array(supports[cell]); si=np.searchsorted(rows,s)
                    assert not set(d.iloc[q].mmsi)&set(d.iloc[s].mmsi)
                    assert d.iloc[s].mmsi.nunique()==e['budget']
                    np.testing.assert_array_equal(np.bincount(labels[s],minlength=8),e['quota'])
                    assert d.iloc[s].port.eq(e['query_port'] if cell[0]=='S' else e['support_port']).all()
                    assert (d.iloc[s].acquisition.eq(e['acquisition']) if cell[1]=='S' else d.iloc[s].acquisition.ne(e['acquisition'])).all()
                    kr=f['r'][si]@f['r'][si].T+1
                    kt=kr+f['t'][si]@f['t'][si].T
                    kb=f['b'][si]@f['b'][si].T+1
                    ab=ev.coefficients(kb,f['pb'][si],labels[s],8.)
                    ar=ev.coefficients(kr,f['pr'][si],labels[s],8.)
                    at=ev.coefficients(kt,f['pd'][si],labels[s],8.)
                    b=softmax(f['pb'][qi]+(f['b'][qi]@f['b'][si].T+1)@ab,axis=1)
                    rk=f['r'][qi]@f['r'][si].T+1
                    r=softmax(f['pr'][qi]+rk@ar,axis=1)
                    t=softmax(f['pd'][qi]+(rk+f['t'][qi]@f['t'][si].T)@at,axis=1)
                    pred[draw,ci]=np.array([b,.5*(b+r),.5*(b+t)])
                    checks.append(dict(episode=e['episode'],draw=draw,cell=cell,support=e['budget'],
                        query=len(q),support_query_mmsi_overlap=0,class_quota_exact=True))
            np.savez_compressed(OUT/'predictions'/f"episode_{e['episode']:03d}.npz",probabilities=pred,
                query_rows=q,support_rows=np.array([[a[c] for c in CELLS] for a in e['draws']]),
                modes=np.array(['base','r50_late','dual_late']),cells=np.array(CELLS))
            print('PREDICTED',e['episode'],e['query_port'],'<-',e['support_port'],e['budget'],flush=True)
        pd.DataFrame(checks).to_csv(OUT/'design_checks.csv',index=False)
        np.savez_compressed(OUT/'encoded.npz',rows=rows,**f)
        write('PREDICTIONS_COMPLETE.json',dict(episodes=len(enrollment),probability_arrays=len(enrollment),
            compact_head_pairs=len(enrollment)*5*4,source_geometry_fits=1,source_linear_heads=3,
            neural_training=0,query_labels_used_for_fit=False,seconds=time.perf_counter()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['inventory','predict'])
    args=parser.parse_args()
    if args.stage=='inventory':inventory()
    else:predict()
