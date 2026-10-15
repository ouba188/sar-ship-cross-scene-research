"""Frozen class-reference external transfer with source-only fusion selection."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
OLD = PARENT/'external_scene_transfer_20261013'
spec = importlib.util.spec_from_file_location('external_old', OLD/'run.py')
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
from study import encode, operator
SEED = 20261013
ARMS = ['original', 'class_mean', 'shrink_control']
GRID = np.round(np.arange(11)/10, 1)
RHO = .25


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def source_metadata():
    d = pd.read_csv(ext.SOURCE/'manifest.csv', usecols=['instance_id', 'port', 'mmsi', 'class_id', 'product'],
                    dtype={'mmsi': str}, keep_default_na=False)
    d['identity'] = d.mmsi
    d['known_ship'] = d.mmsi.str.fullmatch('[1-9][0-9]{8}')
    d['acquisition'] = d['product'].str.split('_').map(lambda s: '|'.join([s[0], s[6], s[7]]))
    return d


def inventory():
    d = source_metadata(); source = np.load(OLD/'source_rows.npy')
    pool = ext.cleaned(d.iloc[source])
    counts = pool.groupby('port').agg(classes=('class_id', 'nunique'), ships=('identity', 'nunique'))
    held = counts[counts.classes == 8].sort_values(['ships'], ascending=False).head(3).index.tolist()
    assert len(held) == 3
    held_all = d[d.port.isin(held)]
    allowed = ~d.iloc[source].identity.isin(held_all.identity) & ~d.iloc[source].acquisition.isin(held_all.acquisition)
    fit = source[allowed & ~d.iloc[source].port.isin(held)]
    assert d.iloc[fit].class_id.nunique() == 8
    plans = []; query = {}
    for pi, port in enumerate(held):
        p = pool[pool.port == port]; rng = np.random.default_rng(SEED+pi)
        q = []
        for _, g in p.groupby('class_id', sort=True):
            q += ext.sample_ships(g, max(1, int(np.ceil(.3*g.identity.nunique()))), rng)
        query[port] = q
    all_query_ships = set(d.loc[sum(query.values(), []), 'identity'])
    for pi, port in enumerate(held):
        p = pool[(pool.port == port) & ~pool.identity.isin(all_query_ships)]
        assert p.identity.nunique() >= 64
        supports = [ext.sample_ships(p, 64, np.random.default_rng(SEED+100+pi*10+i)) for i in range(5)]
        for b in [16, 64]:
            plans.append(dict(episode=len(plans), port=port, budget=b, query=query[port], supports=[s[:b] for s in supports]))
    np.save(OUT/'validation_fit_rows.npy', fit)
    write(OUT/'VALIDATION_ENROLLMENT.json', plans)
    target = [e for e in json.loads((OLD/'ENROLLMENT.json').read_text(encoding='utf-8'))
              if e['study'] in ['dataset_adaptation', 'acquisition_transfer']]
    write(OUT/'EXTERNAL_ENROLLMENT.json', target)
    counts.to_csv(OUT/'source_port_capacity.csv')
    write(OUT/'PROTOCOL.json', dict(date='2026-10-14', status='Frozen before new source/target outcomes',
        arms=ARMS, kernel='original or .75 original + .25 relative-class-reference; shrink_control=.75 original',
        memory='8 source class centers, one seeded observation per source ship; median source distance bandwidth; response L2, source centering and energy match',
        parameters=dict(rho=RHO, support_ridge=8., source_ridge=1., temperature=1., seed=SEED),
        source_validation_ports=held, selection='Top3 eight-class source ports by known consistent ship capacity, metadata only',
        validation_fit_n=len(fit), deployment_source_n=len(source),
        source_validation='One shared fit excluding full held ports, their known identities and acquisitions; 30percent per-class ships query; nested random16/64 supports,5 draws',
        memory_identity='Known consistent-label source ships only; exclude unknown identities instead of treating them as one ship',
        weight_grid=GRID.tolist(), weight_meaning='expert probability weight; base weight=1-weight',
        choose='Separate per arm and B; mean draw then equal source ports BA; exact ties closest to.5 then smaller weight',
        deployment='Refit source class memory to existing common external geometry; shared chosen weights across both targets, no external outcome selection',
        selection_vs_deployment='Held source ports excluded during weight-selection fitting; deployment reuses full allowed source including validation ports. This is source-selected hyperparameter transfer, not a permanently held-source model.',
        primary='Existing global one-query-per-vessel cohorts, identical historical queries/supports',
        secondary='Existing5acq per external dataset, same/other matched class supports; OSS2 also product-type matched',
        report='Expert alone, equal fusion, each arm own source-chosen fusion; full grid descriptive only, no target oracle selection',
        diagnostic='Fixed shrink-only control distinguishes reference geometry from original-kernel weakening; no extra parameter tuning',
        scope='Compact readout, NOT legacy full-model integration; historical encoder and development exposure persists',
        neural_training=0, main_model_changed=False))
    write(OUT/'SOURCE_ISOLATION.json', dict(fit_n=len(fit), fit_classes=np.bincount(d.iloc[fit].class_id, minlength=8).tolist(),
        held_ports=held, source_vs_validation_identity_overlap=len(set(d.iloc[fit].identity)&set(held_all.identity)),
        source_vs_validation_acquisition_overlap=len(set(d.iloc[fit].acquisition)&set(held_all.acquisition))))
    print('ENROLLED', held, 'fit', len(fit), 'source_tasks', len(plans), 'external_tasks', len(target), flush=True)


def source_arrays():
    return dict(raw=np.load(ext.SOURCE/'pooled_features.npy', mmap_mode='r'),
        c4=np.load(ext.ROOT/'c4_base_trial_20261009/pooled_c4.npy', mmap_mode='r'),
        extra=np.load(ext.ROOT/'neural_complement_pilot_20261007/frozen_replay/source_extra.npy', mmap_mode='r'))


def encoded(ev, arrays, rows, bundle):
    blocks = [encode(ev, arrays, rows[i:i+512], bundle) for i in range(0, len(rows), 512)]
    f = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    f['rows'] = rows
    f['z'] = np.c_[f.pop('r'), f.pop('t')]
    return f


def memory(ev, arrays, source, bundle, destination):
    d = source_metadata(); rng = np.random.default_rng(SEED)
    # Unknown identities cannot be treated as repeated observations of one vessel.
    source = ext.cleaned(d.iloc[source]).index.to_numpy()
    ships = d.iloc[source].identity.to_numpy()
    chosen = np.array([rng.choice(np.flatnonzero(ships == s)) for s in np.unique(ships)])
    selected = source[chosen]
    f = encoded(ev, arrays, selected, bundle); z = f['z']; y = d.iloc[selected].class_id.to_numpy(int)
    centers = np.stack([z[y == c].mean(0) for c in range(8)])
    left = rng.choice(len(z), 2048, replace=True); right = rng.choice(len(z), 2048, replace=True)
    dist = np.sum((z[left]-z[right])**2, axis=1); tau2 = float(np.median(dist[dist > 0]))
    unit = np.exp(-cdist(z, centers, 'sqeuclidean')/tau2)
    unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-15)
    center = unit.mean(0); energy = np.mean(np.sum(z*z, axis=1)+1)
    scale = np.sqrt(energy/np.mean(np.sum((unit-center)**2, axis=1)))
    bank = dict(centers=centers, tau2=tau2, center=center, scale=scale, source_rows=selected)
    with destination.open('wb') as h: pickle.dump(bank, h)
    return bank


def relation(z, bank):
    r = np.exp(-cdist(z, bank['centers'], 'sqeuclidean')/bank['tau2'])
    r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-15)
    return (r-bank['center'])*bank['scale']


def predict(ev, f, q, s, sy, verify=False):
    qi = np.searchsorted(f['rows'], q); si = np.searchsorted(f['rows'], s)
    np.testing.assert_array_equal(f['rows'][qi], q); np.testing.assert_array_equal(f['rows'][si], s)
    ks = f['z'][si]@f['z'][si].T+1; kq = f['z'][qi]@f['z'][si].T+1
    kb = f['b'][si]@f['b'][si].T+1
    ab = ev.coefficients(kb, f['pb'][si], sy, 8.)
    bp = softmax(f['pb'][qi]+(f['b'][qi]@f['b'][si].T+1)@ab, axis=1)
    eps = []; errors = []
    for arm in ARMS:
        ss, qs = ks, kq
        if arm != 'original':
            ss, qs = .75*ks, .75*kq
            if arm == 'class_mean':
                ss = ss+.25*(f['rel'][si]@f['rel'][si].T)
                qs = qs+.25*(f['rel'][qi]@f['rel'][si].T)
        coef = ev.coefficients(ss, f['pd'][si], sy, 8.)
        ep = softmax(f['pd'][qi]+qs@coef, axis=1); eps.append(ep)
        if verify:
            other = KernelRidge(alpha=8., kernel='precomputed').fit(ss, 2*np.eye(8)[sy]-1-f['pd'][si],
                sample_weight=compute_sample_weight('balanced', sy))
            check = softmax(f['pd'][qi]+other.predict(qs), axis=1)
            np.testing.assert_allclose(check, ep, atol=1e-10, rtol=1e-10)
            errors.append(float(np.max(np.abs(check-ep))))
    return bp, np.stack(eps), errors


def scores(y, p):
    hit = p.argmax(1) == y; present = np.unique(y)
    br = np.sum((p-np.eye(8)[y])**2, axis=1)
    nl = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
    return dict(oa=100*hit.mean(), ba=100*np.mean([hit[y == c].mean() for c in present]),
        brier=br.mean(), macro_brier=np.mean([br[y == c].mean() for c in present]),
        nll=nl.mean(), macro_nll=np.mean([nl[y == c].mean() for c in present]))


def validation():
    start = time.perf_counter(); ev = operator(); a = source_arrays(); d = source_metadata()
    source = np.load(OUT/'validation_fit_rows.npy')
    ev.fit_dir = lambda regime, name: OUT/'validation_fit'
    ev.source_arrays = lambda regime, name: (source, a['raw'], a['c4'], a['extra'][source])
    bundle = ev.prepare('shared_frozen', 'class_reference_validation')
    np.testing.assert_array_equal(bundle['source_rows'], source)
    bank = memory(ev, a, source, bundle, OUT/'validation_memory.pkl')
    plans = json.loads((OUT/'VALIDATION_ENROLLMENT.json').read_text(encoding='utf-8'))
    rows = np.unique(sum([e['query']+sum(e['supports'], []) for e in plans], []))
    assert not set(d.iloc[source].identity)&set(d.iloc[rows].identity)
    f = encoded(ev, a, rows, bundle); f['rel'] = relation(f['z'], bank)
    records = []; errors = []; dst = OUT/'source_predictions'; dst.mkdir(exist_ok=True)
    for e in plans:
        q = np.array(e['query']); y = d.iloc[q].class_id.to_numpy(int); bs = []; es = []
        for draw, sr in enumerate(e['supports']):
            s = np.array(sr); sy = d.iloc[s].class_id.to_numpy(int)
            assert not set(d.iloc[q].identity)&set(d.iloc[s].identity)
            bp, eps, err = predict(ev, f, q, s, sy, verify=draw == 0)
            bs.append(bp); es.append(eps); errors.extend(err)
            for ai, arm in enumerate(ARMS):
                for w in GRID:
                    records.append(dict(port=e['port'], budget=e['budget'], draw=draw, arm=arm,
                        weight=float(w), **scores(y, (1-w)*bp+w*eps[ai])))
        np.savez_compressed(dst/f"episode_{e['episode']:02d}.npz", base=bs, expert=es, query_rows=q, support_rows=e['supports'])
        print('SOURCE_PREDICTED', e['port'], e['budget'], len(q), flush=True)
    df = pd.DataFrame(records); df.to_csv(OUT/'source_grid_metrics.csv', index=False)
    curve = df.groupby(['budget', 'arm', 'weight', 'port']).ba.mean().groupby(['budget', 'arm', 'weight']).mean().reset_index()
    curve.to_csv(OUT/'source_selection_curve.csv', index=False); selected = {}
    for (b, arm), g in curve.groupby(['budget', 'arm']):
        candidates = g[g.ba == g.ba.max()].copy(); candidates['distance'] = abs(candidates.weight-.5)
        best = candidates.sort_values(['distance', 'weight']).iloc[0]
        selected[f'{b}:{arm}'] = dict(weight=float(best.weight), source_ba=float(best.ba))
    write(OUT/'SELECTED_WEIGHTS.json', selected)
    write(OUT/'SOURCE_COMPLETE.json', dict(seconds=time.perf_counter()-start, source_tasks=len(plans),
        geometry_fits=1, heads=3, memory_fits=1, neural_training=0, expert_solves=len(plans)*5*len(ARMS),
        independent_krr_cases=len(errors), max_error=max(errors)))
    print('SOURCE_SELECTED', selected, flush=True)


def external():
    start = time.perf_counter(); ev = operator()
    selected = json.loads((OUT/'SELECTED_WEIGHTS.json').read_text(encoding='utf-8'))
    source = np.load(OLD/'source_rows.npy')
    with (OLD/'fits/geometry.pkl').open('rb') as h: bundle = pickle.load(h)
    np.testing.assert_array_equal(bundle['source_rows'], source)
    bank = memory(ev, source_arrays(), source, bundle, OUT/'deployment_memory.pkl')
    frames = {n: pd.read_csv(OLD/f'{n}_metadata.csv', dtype={'identity': str}, keep_default_na=False) for n in ext.DATASETS}
    plans = json.loads((OUT/'EXTERNAL_ENROLLMENT.json').read_text(encoding='utf-8'))
    dst = OUT/'predictions'; dst.mkdir(exist_ok=True)
    errors = []; replays = []; conditions = []
    for name in ext.DATASETS:
        with np.load(OLD/f'{name}_encoded.npz') as old: f = {k: old[k] for k in old.files}
        f['z'] = np.c_[f.pop('r'), f.pop('t')]; f['rel'] = relation(f['z'], bank)
        for e in [e for e in plans if e['target'] == name]:
            q = np.array(e['query']); d = frames[name]
            bp = np.empty((5, len(e['arms']), len(q), 8))
            ep = np.empty((5, len(e['arms']), len(ARMS), len(q), 8))
            with np.load(OLD/'predictions'/f"episode_{e['episode']:03d}.npz") as prev: oldp = prev['probabilities']
            for draw, supports in enumerate(e['support_rows']):
                for ci, sr in enumerate(supports):
                    s = np.array(sr); sy = d.iloc[s].class_id.to_numpy(int)
                    assert e['support_datasets'][ci] == name
                    assert not set(d.iloc[q].identity)&set(d.iloc[s].identity)
                    assert len(s) == e['budget'] == d.iloc[s].identity.nunique()
                    bb, ee, err = predict(ev, f, q, s, sy, verify=draw == 0 and ci == 0)
                    bp[draw, ci] = bb; ep[draw, ci] = ee; errors.extend(err)
                    replay = .5*(bb+ee[0]); reference = oldp[draw, ci, 2]
                    np.testing.assert_allclose(replay, reference, atol=1e-10, rtol=1e-10)
                    np.testing.assert_array_equal(replay.argmax(1), reference.argmax(1))
                    replays.append(float(np.max(np.abs(replay-reference))))
                    conditions.append(dict(episode=e['episode'], dataset=name, study=e['study'], cell=e['arms'][ci],
                        budget=e['budget'], draw=draw, support_counts=json.dumps(np.bincount(sy, minlength=8).tolist()),
                        query_n=len(q), query_ships=d.iloc[q].identity.nunique(),
                        query_counts=json.dumps(np.bincount(d.iloc[q].class_id, minlength=8).tolist())))
            np.savez_compressed(dst/f"episode_{e['episode']:03d}.npz", base=bp, expert=ep, query_rows=q,
                support_rows=e['support_rows'], arms=np.array(ARMS), cells=np.array(e['arms']))
            print('EXTERNAL_PREDICTED', name, e['episode'], e['study'], e['budget'], len(q), flush=True)
    pd.DataFrame(conditions).to_csv(OUT/'conditions.csv', index=False)
    write(OUT/'EXTERNAL_COMPLETE.json', dict(seconds=time.perf_counter()-start, tasks=len(plans),
        conditions=len(conditions), expert_solves=len(conditions)*len(ARMS), new_source_geometry=0,
        new_source_heads=0, new_memory_fits=1, neural_training=0, weights_frozen=selected,
        old_prediction_replays=len(replays), max_replay_error=max(replays),
        independent_krr_cases=len(errors), max_solver_error=max(errors)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['inventory', 'validation', 'external'])
    args = ap.parse_args()
    with threadpool_limits(limits=2):
        {'inventory': inventory, 'validation': validation, 'external': external}[args.stage]()
