"""Source-held scene tasks; frozen features, four fixed support kernels."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from core import ARMS, RHO, build_memories, relation, kernel, map_at_r

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
FEATURES = ROOT/'ports_root_cause_20261013/features/original'
SEED = 20261013
sys.path.insert(0, str(ROOT/'port_shift_matched_acquisition_20261013'))
from study import operator, encode
sys.path.insert(0, str(ROOT/'scene_shift_probe_20261013'))
from analyze import metadata


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def quota(capacity, budget):
    q = (capacity > 0).astype(int)
    while q.sum() < budget:
        score = capacity/capacity.sum()*budget-q
        score[q >= capacity] = -np.inf
        q[np.argmax(score)] += 1
    return q


def sample(pool, counts, rng):
    result = []
    for c, count in enumerate(counts):
        g = pool[pool.class_id == c]
        if count:
            for ship in rng.choice(g.mmsi.unique(), int(count), replace=False):
                result.append(int(rng.choice(g.index[g.mmsi == ship])))
    return result


def inventory():
    d = metadata()
    available = np.load(FEATURES/'rows.npy')
    np.testing.assert_array_equal(np.load(FEATURES/'ids.npy')[available], d.iloc[available].instance_id)
    initial = np.load(ROOT/'port_shift_matched_acquisition_20261013/source_rows.npy')
    conflicts = d[d.known_ship].groupby('mmsi').class_id.nunique()
    p = d.loc[initial]
    p = p[p.known_ship & ~p.mmsi.isin(conflicts[conflicts > 1].index)]
    byport, capacities = {}, []
    for pi, (port, g) in enumerate(p.groupby('port', sort=True)):
        rng = np.random.default_rng(SEED+pi)
        qships = []
        for c, group in g.groupby('class_id', sort=True):
            ships = group.mmsi.unique()
            qships.extend(rng.choice(ships, max(1, int(np.ceil(.3*len(ships)))), replace=False))
        qp = g[g.mmsi.isin(qships)]
        sp = g[~g.mmsi.isin(qships)]
        eligible = []
        for acq, qg in qp.groupby('acquisition', sort=True):
            if qg.mmsi.nunique() < 8 or qg.class_id.nunique() < 2:
                continue
            pools = [sp[sp.acquisition == acq], sp[sp.acquisition != acq]]
            cap = np.minimum(*[a.groupby('class_id').mmsi.nunique().reindex(range(8), fill_value=0).to_numpy() for a in pools])
            capacities.append(dict(port=port, acquisition=acq, query_ships=qg.mmsi.nunique(),
                                   query_classes=qg.class_id.nunique(), matched_capacity=int(cap.sum())))
            if cap.sum() < 64 or (cap > 0).sum() < 2:
                continue
            q = [int(rng.choice(qg.index[qg.mmsi == s])) for s in qg.mmsi.unique()]
            if len(q) > 128:
                q = rng.choice(q, 128, replace=False).tolist()
            if d.iloc[q].class_id.nunique() < 2:
                continue
            gallery_pool = pools[1]
            gs = gallery_pool.mmsi.unique()
            if len(gs) > 512:
                gs = rng.choice(gs, 512, replace=False)
            gallery = [int(rng.choice(gallery_pool.index[gallery_pool.mmsi == s])) for s in gs]
            draws = []
            q64, q16 = quota(cap, 64), quota(cap, 16)
            # Quotas for16 need not be nested; query cohort remains identical.
            for draw in range(5):
                rr = np.random.default_rng(SEED+pi*1000+len(eligible)*10+draw)
                draws.append({str(b): [sample(a, qq, rr) for a in pools] for b, qq in [(16, q16), (64, q64)]})
            eligible.append(dict(port=port, acquisition=acq, start=str(qg.start.min()), query=q,
                                 gallery=gallery, draws=draws, quotas={'16':q16.tolist(), '64':q64.tolist()}))
        if len(eligible) >= 2:
            byport[port] = sorted(eligible, key=lambda e: (e['start'], e['acquisition']))
    assert len(byport) >= 3, 'Insufficient source-port capacity for prespecified small study'
    ports = sorted(byport, key=lambda port: (-len(byport[port]), port))[:3]
    plans = []
    for port in ports:
        eligible = byport[port]
        for index in [0, len(eligible)-1]:
            e = eligible[index]; e['episode'] = len(plans); plans.append(e)
    held = d[d.port.isin(ports)]
    source = p[~p.port.isin(ports) & ~p.mmsi.isin(held[held.known_ship].mmsi) & ~p.acquisition.isin(held.acquisition)].index.to_numpy()
    assert d.iloc[source].class_id.nunique() == 8
    taskrows = np.unique(np.concatenate([e['query']+e['gallery']+sum([s for draw in e['draws'] for ss in draw.values() for s in ss], []) for e in plans]))
    assert not set(d.iloc[source].mmsi)&set(d.iloc[taskrows].mmsi)
    assert not set(d.iloc[source].acquisition)&set(d.iloc[taskrows].acquisition)
    assert not set(d.iloc[source].port)&set(ports)
    pd.DataFrame(capacities).to_csv(OUT/'capacity.csv', index=False)
    np.save(OUT/'source_rows.npy', source)
    write('ENROLLMENT.json', plans)
    write('PROTOCOL.json', dict(date='2026-10-13', seed=SEED, arms=ARMS, rho=RHO, budgets=[16,64], draws=5,
        task='Source-development held ports, same/other acquisition support within each port',
        enrollment='Top3 ports by number metadata-eligible scenes; earliest/latest eligible scene per port; no predictions used',
        candidate_ports={k:len(v) for k,v in byport.items()}, held_ports=ports,
        query='30percent known consistent-label ships per class reserved across each whole port; max128 per scene',
        support='Known unique ships, class quotas matched between same/other acquisition; same queries at B16/64, nonnested supports',
        gallery='Fixed up to512 unique non-query ships from other acquisitions in same port; classes not forced balanced',
        source_rows=len(source), source_classes=d.iloc[source].class_id.value_counts().sort_index().to_dict(),
        source_port_identity_acquisition_overlap=0, max_prototypes_per_class=8, kmeans_n_init=5,
        relation='RBF to source references, common source pair-distance bandwidth; class-capacity scaling; source mean-diagonal matching',
        model='New common compact PCA/stats/prior fit, frozen R50/C4+e206 SSL, not canonical full model',
        priors='Identical across arms, base branch unchanged; expert coefficients re-solved for every kernel',
        exposure='Historical frozen encoder and development exposure remains; not independent new data',
        calibration='softmax T=1 frozen across arms, descriptive probability risks not calibrated-probability claim',
        target_external_tested=False, neural_training=False, main_model_changed=False))
    print('SOURCE', len(source), 'PORTS', ports, 'EPISODES', len(plans), flush=True)
    print(pd.DataFrame([{k:e[k] for k in ['episode','port','acquisition']} | {'query':len(e['query'])} for e in plans]).to_string(index=False), flush=True)


def predict():
    started = time.perf_counter()
    d = metadata(); source = np.load(OUT/'source_rows.npy')
    plans = json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    arrays = {k:np.load(FEATURES/(k+'.npy'), mmap_mode='r') for k in ['raw','c4','extra']}
    ev = operator()
    ev.fit_dir = lambda regime, name: OUT/'fits'
    ev.source_arrays = lambda regime, name: (source, arrays['raw'], arrays['c4'], arrays['extra'][source])
    bundle = ev.prepare('shared_frozen', 'source_scene_pilot')
    np.testing.assert_array_equal(bundle['source_rows'], source)
    taskrows = np.unique(np.concatenate([e['query']+e['gallery']+sum([s for draw in e['draws'] for ss in draw.values() for s in ss], []) for e in plans]))
    rows = np.union1d(source, taskrows)
    blocks = [encode(ev, arrays, rows[i:i+512], bundle) for i in range(0, len(rows), 512)]
    f = {k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    z = np.c_[f['r'],f['t']]
    si = np.searchsorted(rows, source)
    memories, coverage, unique = build_memories(z[si], d.iloc[source].class_id.to_numpy(),
        d.iloc[source].mmsi.to_numpy(), d.iloc[source].acquisition.to_numpy(), SEED)
    rel = {name:relation(z, m) for name,m in memories.items()}
    with (OUT/'memories.pkl').open('wb') as h:
        pickle.dump(memories, h)
    write('MEMORY.json', dict(coverage=coverage, source_unique_ships=len(unique),
        tau2=memories['class_mean']['tau2'], scales={k:v['scale'] for k,v in memories.items()}))
    np.savez_compressed(OUT/'encoded.npz', rows=rows, **f, **{'rel_'+k:v for k,v in rel.items()})
    retrieval, diagnostics = [], []
    (OUT/'predictions').mkdir(exist_ok=True)
    for e in plans:
        q=np.array(e['query']); qi=np.searchsorted(rows,q); y=d.iloc[q].class_id.to_numpy(int)
        gallery=np.array(e['gallery']); gi=np.searchsorted(rows,gallery)
        assert not set(d.iloc[q].mmsi)&set(d.iloc[gallery].mmsi)
        assert d.iloc[gallery].acquisition.ne(e['acquisition']).all()
        gy=d.iloc[gallery].class_id.to_numpy(int)
        for arm in ARMS:
            r=None if arm=='original' else rel[arm]
            sim=kernel(z[qi],z[gi],None if r is None else r[qi],None if r is None else r[gi])
            ap=map_at_r(sim,y,gy)
            for row,c,v in zip(q,y,ap):
                retrieval.append(dict(episode=e['episode'],port=e['port'],acquisition=e['acquisition'],arm=arm,row=int(row),class_id=int(c),map_at_r=float(v),gallery=len(gallery)))
        for budget in [16,64]:
            probs=np.empty((5,2,len(ARMS),2,len(q),8))
            supports=[]
            for draw, selections in enumerate(e['draws']):
                supports.append(selections[str(budget)])
                for cell,s in enumerate(selections[str(budget)]):
                    s=np.array(s); ix=np.searchsorted(rows,s); sy=d.iloc[s].class_id.to_numpy(int)
                    assert d.iloc[s].mmsi.nunique()==budget
                    assert not set(d.iloc[s].mmsi)&set(d.iloc[q].mmsi)
                    np.testing.assert_array_equal(np.bincount(sy,minlength=8),e['quotas'][str(budget)])
                    assert (d.iloc[s].acquisition.eq(e['acquisition']) if cell==0 else d.iloc[s].acquisition.ne(e['acquisition'])).all()
                    ab=ev.coefficients(f['b'][ix]@f['b'][ix].T+1,f['pb'][ix],sy,8.)
                    bp=softmax(f['pb'][qi]+(f['b'][qi]@f['b'][ix].T+1)@ab,axis=1)
                    for ai,arm in enumerate(ARMS):
                        r=None if arm=='original' else rel[arm]
                        ks=kernel(z[ix],z[ix],None if r is None else r[ix],None if r is None else r[ix])
                        kq=kernel(z[qi],z[ix],None if r is None else r[qi],None if r is None else r[ix])
                        alpha=ev.coefficients(ks,f['pd'][ix],sy,8.)
                        logits=f['pd'][qi]+kq@alpha
                        ep=softmax(logits,axis=1)
                        probs[draw,cell,ai]=[ep,.5*(bp+ep)]
                        diag=dict(episode=e['episode'],port=e['port'],budget=budget,draw=draw,cell=cell,arm=arm,
                            support_condition=float(np.linalg.cond(ks+8*np.eye(budget))),
                            correction_rms=float(np.sqrt(np.mean((kq@alpha)**2))),
                            source_tau2=memories['class_mean']['tau2'])
                        if r is not None:
                            gm=r[qi]@r[ix].T
                            diag['memory_kernel_std']=float(gm.std())
                            diag['memory_kernel_mean']=float(gm.mean())
                        diagnostics.append(diag)
            np.savez_compressed(OUT/'predictions'/f"episode_{e['episode']:02d}_b{budget}.npz",probabilities=probs,
                query_rows=q,support_rows=np.array(supports),arms=np.array(ARMS),modes=np.array(['expert','compact_full']),
                source_only=np.stack([softmax(f['pd'][qi],axis=1),.5*(softmax(f['pb'][qi],axis=1)+softmax(f['pd'][qi],axis=1))]))
        print('PREDICTED',e['episode'],e['port'],len(q),flush=True)
    pd.DataFrame(retrieval).to_csv(OUT/'retrieval.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'kernel_diagnostics.csv',index=False)
    write('PREDICTIONS_COMPLETE.json',dict(episodes=len(plans),support_cells=len(plans)*2*5*2,
        expert_solves=len(plans)*2*5*2*len(ARMS),new_source_geometry=1,new_source_heads=3,neural_training=0,
        seconds=time.perf_counter()-started,query_labels_used_for_fit=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['inventory','predict'])
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        inventory() if args.stage=='inventory' else predict()
