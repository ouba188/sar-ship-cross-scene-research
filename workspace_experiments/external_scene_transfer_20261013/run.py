"""Frozen SAR cross-dataset and acquisition support-transfer supplement."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import json
import pickle
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=Path('E:/Hermes/paper_tracking/sensitivity')
SOURCE=OUT.parent/'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916'
sys.path.insert(0,str(OUT.parent/'port_shift_matched_acquisition_20261013'))
from study import operator,encode
DATASETS=['f0921','oss2']
MODES=['base','r50_late','dual_late']
SEED=20261013


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def original(name):
    return ROOT/('f0921_detector_recrop_20261007' if name=='f0921' else 'opensarship2_transfer_20261007')


def metadata(name):
    if name=='f0921':
        return pd.read_csv(OUT/'f0921_metadata.csv',dtype={'identity':str},keep_default_na=False)
    target=pd.read_csv(original(name)/'target_manifest.csv',dtype={'mmsi_or_ship_id':str},keep_default_na=False)
    public=pd.read_csv(OUT.parent/'ports_root_cause_20261013/dataset_audit/public_metadata.csv',
        usecols=['dataset','sample_id','class_id','identity','product','datatake_key','product_type','incidence_xml'],
        dtype={'identity':str},keep_default_na=False)
    public=public[public.dataset==name].set_index('sample_id').loc[target.sample_id].reset_index()
    np.testing.assert_array_equal(public.class_id.to_numpy(int),target.class_id.to_numpy(int))
    public['acquisition']=public.datatake_key
    public['known_ship']=public.identity.str.fullmatch('[1-9][0-9]{8}')
    public['acquisition_known']=public.acquisition.ne('')
    public.to_csv(OUT/'oss2_metadata.csv',index=False)
    return public


def cleaned(d):
    p=d[d.known_ship.astype(bool)].copy()
    conflict=p.groupby('identity').class_id.nunique()
    return p[~p.identity.isin(conflict[conflict>1].index)]


def sample_ships(pool,n,rng):
    selected=rng.choice(pool.identity.unique(),n,replace=False)
    groups=pool.groupby('identity',sort=False).indices
    rows=pool.index.to_numpy()
    return [int(rows[rng.choice(groups[s])]) for s in selected]


def quota(cap,b):
    q=(cap>0).astype(int)
    while q.sum()<b:
        score=cap/cap.sum()*b-q;score[q>=cap]=-np.inf
        q[int(score.argmax())]+=1
    return q


def matched_sample(pool,q,rng):
    return sum([sample_ships(pool[pool.class_id==c],int(n),rng) for c,n in enumerate(q) if n],[])


def scene_sample(pool,q,rng):
    return sum([sample_ships(pool[pool.support_stratum==c],int(n),rng) for c,n in enumerate(q) if n],[])


def inventory():
    frames={name:metadata(name) for name in DATASETS}
    pools={name:cleaned(d) for name,d in frames.items()}
    plans=[];capacities=[];global_q={}
    for ni,name in enumerate(DATASETS):
        rng=np.random.default_rng(SEED+ni)
        q=[]
        for c,g in pools[name].groupby('class_id',sort=True):
            q.extend(sample_ships(g,max(1,int(np.ceil(.3*g.identity.nunique()))),rng))
        global_q[name]=q
    all_query_ships=set().union(*(set(frames[n].iloc[global_q[n]].identity) for n in DATASETS))
    global_pools={n:pools[n][~pools[n].identity.isin(all_query_ships)] for n in DATASETS}
    for ni,name in enumerate(DATASETS):
        draws=[]
        for draw in range(5):
            s=sample_ships(global_pools[name],64,np.random.default_rng(SEED+100+ni*10+draw))
            draws.append(s)
        for b in [16,64]:
            plans.append(dict(study='dataset_adaptation',target=name,budget=b,acquisition='GLOBAL',
                query=global_q[name],arms=['target_support'],support_datasets=[name],
                support_rows=[[s[:b]] for s in draws],quota=None))
    cap=np.minimum(*[global_pools[n].groupby('class_id').identity.nunique().reindex(range(8),fill_value=0).to_numpy() for n in DATASETS])
    for b in [16,64]:
        assert cap.sum()>=b
        q=quota(cap,b)
        draws=[]
        for draw in range(5):
            rng=np.random.default_rng(SEED+200+b+draw)
            draws.append([matched_sample(global_pools[n],q,rng) for n in DATASETS])
        for name in DATASETS:
            plans.append(dict(study='support_exchange',target=name,budget=b,acquisition='GLOBAL',
                query=global_q[name],arms=['f0921_support','oss2_support'],support_datasets=DATASETS,
                support_rows=draws,quota=q.tolist()))
    scene_pools={}
    for ni,name in enumerate(DATASETS):
        p=pools[name];p=p[p.acquisition_known.astype(bool)].copy()
        p['support_stratum']=p.class_id.to_numpy(int)
        bins=8
        if name=='oss2':
            assert p.product_type.isin(['GRDH','SLC']).all()
            p['support_stratum']+=8*p.product_type.eq('SLC').astype(int)
            bins=16
        vi=p.groupby('identity').agg(class_id=('class_id','first'),acquisitions=('acquisition','nunique'))
        global_counts=vi.groupby('class_id').size().reindex(range(8),fill_value=0).to_numpy()
        eligible={16:[],64:[]}
        for ai,(acq,g) in enumerate(p.groupby('acquisition',sort=True)):
            n=g.identity.nunique()
            if n<8:continue
            rng=np.random.default_rng(SEED+1000+ni*1000+ai)
            q=sample_ships(g,max(4,min(256,int(np.ceil(.3*n)))),rng)
            if len(q)<5 or frames[name].iloc[q].class_id.nunique()<2:continue
            qships=frames[name].iloc[q].identity
            if name=='f0921':
                inside=g[~g.identity.isin(qships)].groupby('class_id').identity.nunique().reindex(range(8),fill_value=0).to_numpy()
                only_here=vi.loc[g.identity.unique()]
                lost=only_here[only_here.acquisitions==1].groupby('class_id').size().reindex(range(8),fill_value=0).to_numpy()
                query_info=vi.loc[qships]
                query_outside=query_info[query_info.acquisitions>1].groupby('class_id').size().reindex(range(8),fill_value=0).to_numpy()
                cap=np.minimum(inside,global_counts-lost-query_outside)
            else:
                allowed=p[~p.identity.isin(qships)]
                arms=[allowed[allowed.acquisition==acq],allowed[allowed.acquisition!=acq]]
                arms=[a.iloc[rng.permutation(len(a))].drop_duplicates('identity') for a in arms]
                scene_pools[name,acq]=[a.index.to_numpy() for a in arms]
                cap=np.minimum(*[a.groupby('support_stratum').identity.nunique().reindex(range(bins),fill_value=0).to_numpy() for a in arms])
            capacities.append(dict(dataset=name,acquisition=acq,query_n=len(q),capacity=int(cap.sum()),classes=int((cap>0).sum())))
            if len(set(np.flatnonzero(cap>0)%8))<2:continue
            for b in [16,64]:
                if cap.sum()<b:continue
                qu=quota(cap,b)
                eligible[b].append(dict(study='acquisition_transfer',target=name,budget=b,acquisition=acq,query=q,
                    start=str(g['product'].str.extract(r'_(\d{8}T\d{6})_')[0].min()),
                    arms=['same_acquisition','other_acquisition'],support_datasets=[name,name],
                    sampling_seed=SEED+10000+ni*10000+ai*100+b,
                    quota=(qu if bins==8 else qu[:8]+qu[8:]).tolist(),stratum_quota=qu.tolist()))
        hi=sorted(eligible[64],key=lambda e:(e['start'],e['acquisition']))
        picked=[]
        if hi:
            selected=np.unique(np.rint(np.linspace(0,len(hi)-1,min(5,len(hi)))).astype(int))
            picked=[hi[i] for i in selected];plans.extend(picked)
        acqs={e['acquisition'] for e in picked}
        lo={e['acquisition']:e for e in eligible[16]}
        plans.extend(lo[a] for a in sorted(acqs))
        rest=sorted([e for a,e in lo.items() if a not in acqs],key=lambda e:(e['start'],e['acquisition']))
        slots=max(0,5-len(acqs))
        if rest and slots:
            ix=np.unique(np.rint(np.linspace(0,len(rest)-1,min(slots,len(rest)))).astype(int))
            plans.extend(rest[i] for i in ix)
    for i,e in enumerate(plans):
        e['episode']=i
        if e['study']!='acquisition_transfer':continue
        name=e['target'];p=pools[name];p=p[p.acquisition_known.astype(bool)].copy()
        p['support_stratum']=p.class_id.to_numpy(int)+(8*p.product_type.eq('SLC').to_numpy(int) if name=='oss2' else 0)
        key=(name,e['acquisition'])
        if key not in scene_pools:
            allowed=p[~p.identity.isin(frames[name].iloc[e['query']].identity)]
            arms=[allowed[allowed.acquisition==e['acquisition']],allowed[allowed.acquisition!=e['acquisition']]]
            rng=np.random.default_rng(e['sampling_seed'])
            scene_pools[key]=[a.iloc[rng.permutation(len(a))].drop_duplicates('identity').index.to_numpy() for a in arms]
        arms=[p.loc[rr] for rr in scene_pools[key]]
        e['support_rows']=[]
        for draw in range(5):
            rng=np.random.default_rng(e['sampling_seed']+draw)
            e['support_rows'].append([scene_sample(a,e['stratum_quota'],rng) for a in arms])
    pd.DataFrame(capacities).to_csv(OUT/'acquisition_capacity.csv',index=False)
    write('ENROLLMENT.json',plans)
    src=pd.read_csv(SOURCE/'manifest.csv',usecols=['mmsi','class_id','product'],dtype={'mmsi':str},keep_default_na=False)
    splits=[np.load(ROOT/(('f0921' if n=='f0921' else 'opensarship2')+'_transfer_20261007')/'source_split.npz')['source_rows'] for n in DATASETS]
    rows=np.intersect1d(*splits)
    known=set().union(*(set(d.loc[d.known_ship.astype(bool),'identity']) for d in frames.values()))
    acqs=set().union(*(set(d.loc[d.acquisition_known.astype(bool),'acquisition']) for d in frames.values()))
    tokens=src['product'].str.split('_');source_acq=tokens.map(lambda s:'|'.join([s[0],s[6],s[7]]))
    rows=rows[~src.iloc[rows].mmsi.isin(known).to_numpy() & ~source_acq.iloc[rows].isin(acqs).to_numpy()]
    assert src.iloc[rows].class_id.nunique()==8
    np.save(OUT/'source_rows.npy',rows)
    write('PROTOCOL.json',dict(date='2026-10-13',seed=SEED,datasets=DATASETS,budgets=[16,64],draws=5,
        source='One common self24 source-row intersection, additionally exclude known external identities and recovered external acquisitions',
        source_n=len(rows),source_ridge=1.,support_ridge=8.,fusion=.5,
        model='Frozen R50+C4 and e206 R18, common new source PCA/scales/3 heads; compact reader, NOT historical full model',
        source_geometry='Existing compact recipe only; no method tuning or new encoder training',
        exposure='F0921 corpus was used historically for R18 SSL; prior development remains, not independent encoder validation',
        dataset_adaptation='24port source-only vs nested random16/64 unique-vessel target supports; fixed queries, query identities excluded',
        global_query='Seeded30percent of known consistent-label vessels per class, one random observation each; samequeriesacrossB/arms',
        support_exchange='Same24source model + F0921 or OSS2 support; exactsameB/classcounts andsamequeries; NOT pure F-to-OSS2 source-only training',
        acquisition_transfer='Atmost5 metadata-selected acquisitions per target/B, fixedqueryvessels; F matches classes, OSS2 matches class x GRDH/SLC; same/other acquisition',
        acquisition_query='30percent vessels, cap256peracquisition, oneobservationeach; B64-capable queries reusedB16',
        public_scope='OSS2 actualportunknown; datatake groups GRDH/SLC together, product strata reported, notpureportchange',
        sampling='Acquisition/exchange arms are labelmatcheddiagnostics, not labelblindB-label acquisition',
        main_model_changed=False,query_tuning=False))
    write('SOURCE_ISOLATION.json',dict(source_n=len(rows),source_class_counts=src.iloc[rows].class_id.value_counts().sort_index().to_dict(),
        external_identity_overlap=len(set(src.iloc[rows].mmsi)&known),known_acquisition_overlap=len(set(source_acq.iloc[rows])&acqs),
        external_queries={n:len(global_q[n]) for n in DATASETS},eligible_known_consistent_rows={n:len(pools[n]) for n in DATASETS}))
    print(pd.DataFrame(plans).groupby(['study','target','budget']).size().to_string(),flush=True)
    print('SOURCE',len(rows),'QUERIES',{n:len(global_q[n]) for n in DATASETS},flush=True)


def arrays_for(name,d):
    dest=original(name)
    if name=='f0921':
        raw=np.load(dest/'paired_r50.npy',mmap_mode='r');extra=np.load(dest/'vh_ssl.npy',mmap_mode='r')
        ids=pd.read_csv(dest/'target_manifest.csv',usecols=['sample_id']).sample_id.to_numpy()
    else:
        with np.load(dest/'features.npz') as z:
            raw=z['paired_r50'];extra=z['vh_ssl'];ids=z['sample_ids']
    c4dir=ROOT/'cross_dataset_c4_20261009'/name
    np.testing.assert_array_equal(ids,d.sample_id.to_numpy())
    np.testing.assert_array_equal(np.load(c4dir/'sample_ids.npy'),ids)
    assert raw.shape==(len(d),2048) and extra.shape==(len(d),512)
    return dict(raw=raw,c4=np.load(c4dir/'pooled_c4.npy',mmap_mode='r'),extra=extra)


def predict():
    from scipy.special import softmax
    from threadpoolctl import threadpool_limits
    ev=operator();plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    frames={n:metadata(n) for n in DATASETS}
    source=np.load(OUT/'source_rows.npy')
    raw=np.load(SOURCE/'pooled_features.npy',mmap_mode='r')
    c4=np.load(ROOT/'c4_base_trial_20261009/pooled_c4.npy',mmap_mode='r')
    extra=np.load(ROOT/'neural_complement_pilot_20261007/frozen_replay/source_extra.npy',mmap_mode='r')
    ev.fit_dir=lambda regime,name:OUT/'fits'
    ev.source_arrays=lambda regime,name:(source,raw,c4,extra[source])
    (OUT/'predictions').mkdir(exist_ok=True)
    started=time.perf_counter();checks=[];encoded={}
    with threadpool_limits(limits=2):
        bundle=ev.prepare('shared_frozen','external_common')
        np.testing.assert_array_equal(bundle['source_rows'],source)
        for name in DATASETS:
            ref=[]
            for e in plans:
                if e['target']==name:ref.extend(e['query'])
                for draw in e['support_rows']:
                    for sn,rows in zip(e['support_datasets'],draw):
                        if sn==name:ref.extend(rows)
            rows=np.unique(ref);a=arrays_for(name,frames[name])
            blocks=[encode(ev,a,rows[i:i+512],bundle) for i in range(0,len(rows),512)]
            f={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
            np.savez_compressed(OUT/(name+'_encoded.npz'),rows=rows,**f)
            f['rows']=rows;f['dual']=np.c_[f['r'],f['t']];encoded[name]=f
            del blocks,a
            print('ENCODED',name,len(rows),flush=True)
        for e in plans:
            name=e['target'];d=frames[name];fq=encoded[name];q=np.array(e['query']);qi=np.searchsorted(fq['rows'],q)
            p0=[softmax(fq[k][qi],axis=1) for k in ['pb','pr','pd']]
            prior=np.array([p0[0],.5*(p0[0]+p0[1]),.5*(p0[0]+p0[2])])
            prob=np.empty((5,len(e['arms']),3,len(q),8),float)
            for draw,supports in enumerate(e['support_rows']):
                for ai,(sn,srows) in enumerate(zip(e['support_datasets'],supports)):
                    ds=frames[sn];fs=encoded[sn];s=np.array(srows);si=np.searchsorted(fs['rows'],s);sy=ds.iloc[s].class_id.to_numpy(int)
                    assert len(s)==e['budget'] and ds.iloc[s].identity.nunique()==len(s)
                    assert not set(d.iloc[q].identity)&set(ds.iloc[s].identity)
                    if e['quota'] is not None:np.testing.assert_array_equal(np.bincount(sy,minlength=8),e['quota'])
                    if e['study']=='acquisition_transfer':
                        assert (ds.iloc[s].acquisition.eq(e['acquisition']) if ai==0 else ds.iloc[s].acquisition.ne(e['acquisition'])).all()
                        strata=sy+(8*ds.iloc[s].product_type.eq('SLC').to_numpy(int) if sn=='oss2' else 0)
                        np.testing.assert_array_equal(np.bincount(strata,minlength=len(e['stratum_quota'])),e['stratum_quota'])
                    outs=[]
                    for key,pk in [('b','pb'),('r','pr'),('dual','pd')]:
                        phi=fs[key][si];coef=ev.coefficients(phi@phi.T+1,fs[pk][si],sy,8.)
                        outs.append(softmax(fq[pk][qi]+(fq[key][qi]@phi.T+1)@coef,axis=1))
                    prob[draw,ai]=[outs[0],.5*(outs[0]+outs[1]),.5*(outs[0]+outs[2])]
                    checks.append(dict(episode=e['episode'],draw=draw,arm=e['arms'][ai],support_n=len(s),identity_overlap=0))
            np.savez_compressed(OUT/'predictions'/f"episode_{e['episode']:03d}.npz",probabilities=prob,
                source_only=prior,query_rows=q,support_rows=np.array(e['support_rows']),arms=np.array(e['arms']),modes=np.array(MODES))
            print('PREDICTED',e['episode'],e['study'],name,e['budget'],len(q),flush=True)
    pd.DataFrame(checks).to_csv(OUT/'design_checks.csv',index=False)
    write('PREDICTIONS_COMPLETE.json',dict(episodes=len(plans),support_conditions=len(checks),solver_calls=3*len(checks),
        seconds=time.perf_counter()-started,new_encoder_training=0,new_source_geometry=1,new_source_linear_heads=3))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['inventory','predict']);args=ap.parse_args()
    if args.stage=='inventory':inventory()
    else:predict()
