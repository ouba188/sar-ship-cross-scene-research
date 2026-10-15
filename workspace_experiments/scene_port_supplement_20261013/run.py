"""Frozen-reader acquisition-held and acquisition-controlled port supplement."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import itertools
import json
import pickle
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'port_shift_matched_acquisition_20261013'
sys.path.insert(0, str(OLD))
from study import FEATURES, metadata, operator, encode, CELLS
SEED = 20261013
MODES = ['base', 'r50_late', 'dual_late']
PORTS = ['Ningbo-Zhoushan', 'Qingdao', 'Shanghai']


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def frame():
    d = metadata()
    inc = pd.read_csv('E:/Hermes/paper_tracking/pixel_correction_review/chip_incidence_v1/incidence_by_instance.csv',
                      usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    joined = d[['instance_id', 'product']].merge(inc, on=['instance_id', 'product'], how='left', validate='one_to_one')
    d['angle'] = joined.incidence_deg.where(joined.status.eq('ok')).to_numpy()
    d['angle_bin'] = np.floor(d.angle / 2).fillna(-1).astype(int)
    return d


def quotas(capacity, budget):
    cap = capacity.to_numpy()
    classes = capacity.index.get_level_values(0)
    q = np.zeros(len(cap), int)
    for c in classes.unique():
        loc = np.flatnonzero(classes == c)
        q[loc[np.argmax(cap[loc])]] = 1
    while q.sum() < budget:
        priority = cap / cap.sum() * budget - q
        priority[q >= cap] = -np.inf
        q[int(priority.argmax())] += 1
    return q


def inventory():
    d = frame()
    available = np.load(FEATURES / 'rows.npy')
    pool = d.iloc[available]
    pool = pool[pool.port.isin(PORTS) & pool.known_ship].copy()
    conflict = pool.groupby('mmsi').class_id.nunique()
    pool = pool[~pool.mmsi.isin(conflict[conflict > 1].index)]
    cross = pool.groupby(['acquisition', 'mmsi']).port.nunique()
    cross = set(cross[cross > 1].index)
    pool = pool.loc[[(a, m) not in cross for a, m in zip(pool.acquisition, pool.mmsi)]]
    queries = {}
    for i, ((port, acq), group) in enumerate(pool.groupby(['port', 'acquisition'], sort=True)):
        vessels = group.mmsi.unique()
        if len(vessels) < 8:
            continue
        rng = np.random.default_rng(SEED + i)
        ships = rng.choice(vessels, max(4, int(np.ceil(.3 * len(vessels)))), replace=False)
        rows = [int(rng.choice(group.index[group.mmsi == m])) for m in ships]
        if len(rows) >= 5 and d.iloc[rows].class_id.nunique() >= 2:
            queries[port, acq] = rows
    plans, capacities = [], []
    for direction, (qp, sp) in enumerate(itertools.permutations(PORTS, 2)):
        shared = sorted(set(pool[pool.port == qp].acquisition) & set(pool[pool.port == sp].acquisition))
        for policy in ['class_matched', 'class_angle_matched']:
            eligible = {16: [], 64: []}
            for ai, acq in enumerate(shared):
                original_q = queries.get((qp, acq))
                if original_q is None:
                    continue
                q = [r for r in original_q if policy == 'class_matched' or np.isfinite(d.angle.iloc[r])]
                if len(q) < 5 or d.iloc[q].class_id.nunique() < 2:
                    continue
                p = pool[pool.port.isin([qp, sp]) & ~pool.mmsi.isin(d.iloc[original_q].mmsi)]
                if policy == 'class_angle_matched':
                    p = p[p.angle.notna()]
                arms = dict(SS=p[(p.port == qp) & (p.acquisition == acq)],
                            OS=p[(p.port == sp) & (p.acquisition == acq)],
                            SO=p[(p.port == qp) & (p.acquisition != acq)],
                            OO=p[(p.port == sp) & (p.acquisition != acq)])
                keys = ['class_id'] if policy == 'class_matched' else ['class_id', 'angle_bin']
                # One seeded observation per vessel prevents multi-bin vessels being counted twice.
                groups = {}
                for ci, (cell, g) in enumerate(arms.items()):
                    rr = np.random.default_rng(SEED + 100000 + direction * 1000 + ai * 4 + ci)
                    groups[cell] = g.iloc[rr.permutation(len(g))].drop_duplicates('mmsi')
                counts = [g.groupby(keys).size() for g in groups.values()]
                cap = pd.concat(counts, axis=1).fillna(0).min(axis=1).astype(int)
                cap = cap[cap > 0].sort_index()
                capacities.append(dict(query_port=qp, support_port=sp, acquisition=acq, policy=policy,
                                       capacity=int(cap.sum()), support_classes=cap.index.get_level_values(0).nunique(), query_n=len(q)))
                if cap.index.get_level_values(0).nunique() < 2:
                    continue
                for budget in [16, 64]:
                    if cap.sum() < budget:
                        continue
                    quota = quotas(cap, budget)
                    draws = []
                    for draw in range(5):
                        rr = np.random.default_rng(SEED + 200000 + direction * 10000 + ai * 100 + budget + draw)
                        cells = {}
                        for cell, g in groups.items():
                            picked = []
                            for stratum, n in zip(cap.index, quota):
                                if not n:
                                    continue
                                values = stratum if isinstance(stratum, tuple) else (stratum,)
                                mask = np.ones(len(g), bool)
                                for key, value in zip(keys, values):
                                    mask &= g[key].to_numpy() == value
                                picked.extend(int(x) for x in rr.choice(g.index[mask], n, replace=False))
                            cells[cell] = picked
                        draws.append(cells)
                    eligible[budget].append(dict(query_port=qp, support_port=sp, acquisition=acq,
                        start=d.iloc[q].start.min(), policy=policy, budget=budget, query=q,
                        all_query_ships=d.iloc[original_q].mmsi.tolist(), draws=draws,
                        strata=[dict(values=[int(v) for v in (s if isinstance(s, tuple) else (s,))], count=int(n))
                                for s, n in zip(cap.index, quota) if n], stratum_keys=keys))
            # Enroll B64-capable scenes first; exactly the same queries also receive B16.
            high = sorted(eligible[64], key=lambda x: (x['start'], x['acquisition']))
            selected_high = set()
            if high:
                ix = np.unique(np.rint(np.linspace(0, len(high)-1, min(5, len(high)))).astype(int))
                chosen = [high[i] for i in ix]
                selected_high = {e['acquisition'] for e in chosen}
                plans.extend(chosen)
            low = {e['acquisition']: e for e in eligible[16]}
            plans.extend(low[a] for a in sorted(selected_high))
            extra = sorted([e for a, e in low.items() if a not in selected_high], key=lambda x: (x['start'], x['acquisition']))
            slots = max(0, 5-len(selected_high))
            if slots and extra:
                ix = np.unique(np.rint(np.linspace(0, len(extra)-1, min(slots, len(extra)))).astype(int))
                plans.extend(extra[i] for i in ix)
    for i, e in enumerate(plans):
        e['episode'] = i
    source = np.load(OLD / 'source_rows.npy')
    targets = np.unique(np.concatenate([np.array(e['query'] + sum([s[c] for s in e['draws'] for c in CELLS], [])) for e in plans]))
    assert not set(d.iloc[source].port) & set(d.iloc[targets].port)
    assert not set(d.iloc[source].mmsi) & set(d.iloc[targets].mmsi)
    assert not set(d.iloc[source].acquisition) & set(d.iloc[targets].acquisition)
    write('PROTOCOL.json', dict(seed=SEED, reader='Unchanged common compact reader from port_shift_matched_acquisition_20261013; NOT legacy full',
        source_fit_reused=str(OLD / 'fits/geometry.pkl'), encoder_training=0, new_source_fits=0,
        source_n=len(source), source_exclusion='Target ports, known target vessels and target acquisitions disjoint at PCA/statistics/linear-head fitting stages',
        encoder_scope='Historical frozen encoder exposure is not reset; not an end-to-end unseen-acquisition training claim',
        budgets=[16,64], support_ridge=8., draws=5, fusion=.5,
        enrollment='At most5 metadata-selected scenes per direction/policy/B; same B64-eligible queries also evaluated at B16; no target score selection',
        policies=dict(class_matched='exact class counts across four cells', class_angle_matched='exact class x floor(angle/2deg) counts across four cells'),
        source_only='Scores before support adaptation, same readers/queries, recorded once per query context',
        scene_held='SO/OO use no query-acquisition support; SS/OS intentionally allow distinct labeled vessels from query acquisition as comparators',
        sampling='Known vessels only; one seeded observation per vessel per pool; all original query vessels excluded; label-aware matching not label-blind acquisition',
        metrics='Query-present-class BA and OA; draws then acquisitions then directions, no independence claim for reciprocal directions',
        target_history='Development data including previously studied ports/acquisitions; not a new independent confirmation dataset',
        manuscript_changed=False, main_model_changed=False))
    write('ENROLLMENT.json', plans)
    pd.DataFrame(capacities).to_csv(OUT / 'capacity.csv', index=False)
    write('ISOLATION.json', dict(source_n=len(source), target_rows=len(targets),
        source_target_port_overlap=0, source_target_known_ship_overlap=0, source_target_acquisition_overlap=0))
    print(pd.DataFrame(plans).groupby(['policy','budget']).agg(episodes=('episode','size'), acquisitions=('acquisition','nunique')).to_string(), flush=True)


def predict():
    from scipy.special import softmax
    from threadpoolctl import threadpool_limits
    started = time.perf_counter()
    ev = operator()
    with (OLD / 'fits/geometry.pkl').open('rb') as f:
        bundle = pickle.load(f)
    plans = json.loads((OUT / 'ENROLLMENT.json').read_text(encoding='utf-8'))
    arrays = {k: np.load(FEATURES / (k+'.npy'), mmap_mode='r') for k in ['raw','c4','extra']}
    rows = np.unique(np.concatenate([np.array(e['query'] + sum([s[c] for s in e['draws'] for c in CELLS], [])) for e in plans]))
    d = frame()
    labels = d.class_id.to_numpy()
    (OUT / 'predictions').mkdir(exist_ok=True)
    checks, condition = [], []
    with threadpool_limits(limits=2):
        blocks = [encode(ev, arrays, rows[i:i+512], bundle) for i in range(0,len(rows),512)]
        f = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
        feature_views = [(f['b'],'pb'),(f['r'],'pr'),(np.c_[f['r'],f['t']],'pd')]
        for e in plans:
            q = np.array(e['query']); qi = np.searchsorted(rows, q)
            p0b, p0r, p0t = [softmax(f[k][qi], axis=1) for k in ['pb','pr','pd']]
            prior = np.array([p0b, .5*(p0b+p0r), .5*(p0b+p0t)])
            prob = np.empty((5,4,3,len(q),8), float)
            for draw, cells in enumerate(e['draws']):
                for ci, cell in enumerate(CELLS):
                    s = np.array(cells[cell]); si = np.searchsorted(rows,s)
                    assert len(s) == e['budget'] and d.iloc[s].mmsi.nunique() == len(s)
                    assert not set(d.iloc[s].mmsi) & set(e['all_query_ships'])
                    assert d.iloc[s].port.eq(e['query_port'] if cell[0]=='S' else e['support_port']).all()
                    assert (d.iloc[s].acquisition.eq(e['acquisition']) if cell[1]=='S' else d.iloc[s].acquisition.ne(e['acquisition'])).all()
                    actual = d.iloc[s].groupby(e['stratum_keys']).size().to_dict()
                    expected = {(t['values'][0] if len(t['values'])==1 else tuple(t['values'])):t['count'] for t in e['strata']}
                    assert actual == expected
                    outputs = []
                    for phi, prior_key in feature_views:
                        coef = ev.coefficients(phi[si] @ phi[si].T+1, f[prior_key][si], labels[s], 8.)
                        outputs.append(softmax(f[prior_key][qi]+(phi[qi]@phi[si].T+1)@coef,axis=1))
                    b,r,t = outputs
                    prob[draw,ci] = np.array([b,.5*(b+r),.5*(b+t)])
                    checks.append(dict(episode=e['episode'],draw=draw,cell=cell,support_n=len(s),valid=True))
                    condition.append(dict(episode=e['episode'],draw=draw,cell=cell,
                        angle_known=int(d.iloc[s].angle.notna().sum()), mean_angle=float(d.iloc[s].angle.mean()),
                        acquisitions=int(d.iloc[s].acquisition.nunique())))
            np.savez_compressed(OUT/'predictions'/f"episode_{e['episode']:03d}.npz",
                probabilities=prob, source_only=prior, query_rows=q,
                support_rows=np.array([[s[c] for c in CELLS] for s in e['draws']]),
                modes=np.array(MODES), cells=np.array(CELLS))
            print('PREDICTED',e['episode'],e['policy'],e['query_port'],e['support_port'],e['budget'],flush=True)
        np.savez_compressed(OUT/'encoded.npz',rows=rows,**f)
    pd.DataFrame(checks).to_csv(OUT/'design_checks.csv',index=False)
    pd.DataFrame(condition).to_csv(OUT/'conditions.csv',index=False)
    write('PREDICTIONS_COMPLETE.json',dict(episodes=len(plans),support_cells=len(checks),solver_calls=len(checks)*3,
        neural_training=0,new_source_fits=0,seconds=time.perf_counter()-started))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['inventory','predict'])
    stage=parser.parse_args().stage
    if stage=='inventory': inventory()
    else: predict()
