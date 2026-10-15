"""Frozen-model acquisition coverage and support-budget verification."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
PREV = OUT.parent / 'class_contrast_core_controls_20261014'
sys.path.insert(0, str(PREV))
import run_controls as controls
run = controls.run
ext = run.old.ext
BUDGETS = (16, 32, 64)
ARMS = ['original', 'class_mean', 'class_fixed'] + [f'class_utility_{i}' for i in range(3)] + [f'class_metric_{i}' for i in range(3)]
KEY = ['dataset', 'episode', 'port', 'cohort', 'budget', 'draw', 'cell']


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def task(dataset, episode, port, cohort, acquisition, budget, draw, cell, query, support, archive=None):
    return dict(dataset=dataset, episode=int(episode), port=port, cohort=cohort, acquisition=acquisition,
                budget=int(budget), draw=int(draw), cell=cell, query=[int(v) for v in query],
                support=[int(v) for v in support], archive=archive)


def enroll(frames):
    prior = json.loads((run.PREV / 'EXTERNAL_ENROLLMENT.json').read_text(encoding='utf-8'))
    lookup = {(e['target'], e['acquisition'], e['budget']): e for e in prior if e['study'] == 'acquisition_transfer'}
    capacity = pd.read_csv(run.old.OLD / 'acquisition_capacity.csv')
    tasks, scenes = [], []
    for ni, name in enumerate(('f0921', 'oss2')):
        meta = frames[name]
        pool = ext.cleaned(meta)
        pool = pool[pool.acquisition_known.astype(bool)].copy()
        bins = 16 if name == 'oss2' else 8
        pool['support_stratum'] = pool.class_id.to_numpy(int)
        if name == 'oss2':
            pool['support_stratum'] += 8 * pool.product_type.eq('SLC').to_numpy(int)
        selected = set(capacity.loc[(capacity.dataset == name) & (capacity.capacity >= 64), 'acquisition'])
        episode = 0
        for ai, (acq, g) in enumerate(pool.groupby('acquisition', sort=True)):
            if acq not in selected:
                continue
            rng = np.random.default_rng(20261013 + 1000 + ni * 1000 + ai)
            q = ext.sample_ships(g, max(4, min(256, int(np.ceil(.3 * g.identity.nunique())))), rng)
            allowed = pool[~pool.identity.isin(meta.loc[q].identity)]
            pools = [allowed[allowed.acquisition == acq], allowed[allowed.acquisition != acq]]
            pools = [a.iloc[rng.permutation(len(a))].drop_duplicates('identity') for a in pools]
            cap = np.minimum(*[a.groupby('support_stratum').size().reindex(range(bins), fill_value=0).to_numpy() for a in pools])
            if cap.sum() < 64 or len(set(np.flatnonzero(cap) % 8)) < 2:
                continue
            legacy = (name, acq, 64) in lookup
            cohort = 'legacy5' if legacy else 'expanded'
            if legacy:
                np.testing.assert_array_equal(q, lookup[name, acq, 64]['query'])
            supports = {}
            for b in (16, 64):
                if legacy:
                    supports[b] = lookup[name, acq, b]['support_rows']
                else:
                    quota = ext.quota(cap, b)
                    supports[b] = []
                    for draw in range(5):
                        rr = np.random.default_rng(20261014 + 10000 + ni * 1000000 + ai * 100 + b + draw)
                        supports[b].append([ext.scene_sample(a, quota, rr) for a in pools])
            supports[32] = []
            for draw, paired in enumerate(supports[64]):
                count = np.bincount(pool.loc[paired[0]].support_stratum.to_numpy(int), minlength=bins)
                quota = ext.quota(count, 32)
                rr = np.random.default_rng(20261014 + 2000000 + ni * 1000000 + ai * 10 + draw)
                supports[32].append([ext.scene_sample(pool.loc[s], quota, rr) for s in paired])
            for b in BUDGETS:
                for draw, paired in enumerate(supports[b]):
                    counts = [np.bincount(pool.loc[s].support_stratum.to_numpy(int), minlength=bins) for s in paired]
                    np.testing.assert_array_equal(*counts)
                    for ci, s in enumerate(paired):
                        cell = ('same_acquisition', 'other_acquisition')[ci]
                        archive = None
                        if legacy and b != 32:
                            archive = f"{name}_{lookup[name, acq, b]['episode']:03d}_{b}_{draw}_{cell}.npz"
                        tasks.append(task(name, episode, name, cohort, acq, b, draw, cell, q, s, archive))
            scenes.append(dict(dataset=name, episode=episode, cohort=cohort, port=name, acquisition=acq,
                               query_n=len(q), query_ships=meta.loc[q].identity.nunique(),
                               classes=meta.loc[q].class_id.nunique(), capacity=int(cap.sum())))
            episode += 1
        print('ENROLLED', name, episode, flush=True)
    ports = json.loads((run.EXPAND / 'ENROLLMENT.json').read_text(encoding='utf-8'))
    meta = frames['ports']
    for e in ports:
        q = e['query']
        acq = str(meta.loc[q].acquisition.iloc[0])
        assert meta.loc[q].acquisition.nunique() == 1
        for draw, existing in enumerate(e['draws']):
            supports = {b: existing[str(b)] for b in (16, 64)}
            quota = ext.quota(np.bincount(meta.loc[supports[64][0]].class_id.to_numpy(int), minlength=8), 32)
            rng = np.random.default_rng(20261014 + 4000000 + e['episode'] * 10 + draw)
            supports[32] = [ext.matched_sample(meta.loc[s], quota, rng) for s in supports[64]]
            for b in BUDGETS:
                np.testing.assert_array_equal(*[np.bincount(meta.loc[s].class_id.to_numpy(int), minlength=8) for s in supports[b]])
                for ci, s in enumerate(supports[b]):
                    cell = ('same_acquisition', 'other_acquisition')[ci]
                    archive = f"ports_{e['episode']:03d}_64_{draw}_{cell}.npz" if b == 64 and ci == 1 else None
                    tasks.append(task('ports', e['episode'], e['port'], 'legacy48', acq, b, draw, cell, q, s, archive))
        scenes.append(dict(dataset='ports', episode=e['episode'], cohort='legacy48', port=e['port'], acquisition=acq,
                           query_n=len(q), query_ships=meta.loc[q].identity.nunique(), classes=meta.loc[q].class_id.nunique(), capacity=64))
    write('ENROLLMENT.json', tasks)
    pd.DataFrame(scenes).to_csv(OUT / 'scenes.csv', index=False)
    return tasks


def main():
    start = time.perf_counter()
    write('PROTOCOL.json', dict(status='Frozen before expanded outcomes', budgets=BUDGETS, draws=5,
        models='All three already-frozen balanced class-pair models and all three matched free-class metrics; no new fits or selection',
        geometry='Existing validation_fit geometry used identically for relation training and deployment; no source refit',
        scope='All external scenes in prior metadata capacity inventory with B64 feasible; original five preserved; all48 held-port scenes',
        comparators=ARMS, fixed=dict(rho=.25, ridge=8, fusion=.5),
        sampling='Same/other acquisition class quotas exactly matched; OSS2 additionally GRDH/SLC matched. Label-matched diagnostic, not label-blind labeling policy.',
        budgets_note='Original16/64 supports unchanged, not nested. New32 subsampled within each64 with matched class/product quotas; no monotonicity claim.',
        selection='No target-based checkpoint, amplitude, model or winning-scene selection',
        aggregation='Scene means within each port, then equal ports. External datasets have one port placeholder; actual ports unknown.',
        uncertainty='Training seeds and support draws separate; repeated ships/acquisitions not independent evidence',
        exposure='Development coverage expansion, not new independent data. Historical backbone exposure unchanged.',
        timing='Cached-feature relation mapping and adaptation only; not end-to-end image processing or measured labeling cost.',
        main_model_changed=False))
    frames = {n: pd.read_csv(run.old.OLD / f'{n}_metadata.csv', dtype={'identity': str}, keep_default_na=False) for n in ('f0921', 'oss2')}
    frames['ports'] = run.old.source_metadata()
    tasks = (json.loads((OUT / 'ENROLLMENT.json').read_text(encoding='utf-8'))
             if (OUT / 'ENROLLMENT.json').exists() else enroll(frames))
    with (PREV / 'specs.pkl').open('rb') as h:
        specs = pickle.load(h)
    with (PREV / 'models.pkl').open('rb') as h:
        saved = pickle.load(h)
    models = {arm: saved[arm] for arm in ARMS}
    used_specs = sorted(set(m['spec'] for m in models.values() if m['spec'] is not None))
    with (run.PREV / 'validation_fit/geometry.pkl').open('rb') as h:
        geometry = pickle.load(h)
    ev = run.old.operator()
    train = frames['ports'].loc[np.load(run.PREV / 'validation_fit_rows.npy')]
    records, timings, checks, map_times, coverage, events = [], [], [], [], [], []
    (OUT / 'predictions').mkdir(exist_ok=True)
    for name in ('f0921', 'oss2', 'ports'):
        active = [t for t in tasks if t['dataset'] == name]
        rows = np.unique(np.concatenate([t['query'] + t['support'] for t in active]))
        meta = frames[name]
        assert not set(train.identity) & set(meta.loc[rows].identity)
        assert not set(train.acquisition) & set(meta.loc[rows].acquisition)
        if name == 'ports':
            assert not set(train.port) & set(meta.loc[rows].port)
        arrays = run.old.source_arrays() if name == 'ports' else ext.arrays_for(name, meta)
        f = run.old.encoded(ev, arrays, rows, geometry)
        maps = {k: controls.project(f['z'], specs[k]) for k in used_specs}
        # Include source-calibrated relation-map overhead, which solver-only timers omit.
        test = f['z'][:min(1024, len(rows))]
        for k in used_specs:
            for repeat in range(6):
                t0 = time.perf_counter(); controls.project(test, specs[k]); elapsed = time.perf_counter() - t0
                if repeat:
                    map_times.append(dict(dataset=name, spec=k, rows=len(test), repeat=repeat, seconds=elapsed))
        allq = np.unique(np.concatenate([t['query'] for t in active]))
        coverage.append(dict(dataset=name, scenes=len(set(t['episode'] for t in active)),
                             query_rows=len(allq), query_ships=meta.loc[allq].identity.nunique(), support_cells=len(active),
                             query_counts=np.bincount(meta.loc[allq].class_id.to_numpy(int), minlength=8).tolist()))
        print('EVALUATE', name, 'cells', len(active), 'encoded', len(rows), flush=True)
        for ti, t in enumerate(active):
            q, s = np.asarray(t['query']), np.asarray(t['support'])
            qi, si = np.searchsorted(rows, q), np.searchsorted(rows, s)
            y, sy = meta.loc[q].class_id.to_numpy(int), meta.loc[s].class_id.to_numpy(int)
            assert len(s) == t['budget'] == meta.loc[s].identity.nunique()
            assert not set(meta.loc[q].identity) & set(meta.loc[s].identity)
            assert (meta.loc[s].acquisition.eq(t['acquisition']) if t['cell'] == 'same_acquisition' else meta.loc[s].acquisition.ne(t['acquisition'])).all()
            query, support = run.select(f, q), run.select(f, s)
            t0 = time.perf_counter()
            kss = support['z'] @ support['z'].T + 1
            kqs = query['z'] @ support['z'].T + 1
            bss = support['b'] @ support['b'].T + 1
            bqs = query['b'] @ support['b'].T + 1
            base = softmax(query['pb'] + bqs @ ev.coefficients(bss, support['pb'], sy, 8.), axis=1)
            common_seconds = time.perf_counter() - t0
            key = {k: t[k] for k in KEY}
            outputs = {}
            for arm, model in models.items():
                t0 = time.perf_counter()
                if model['spec'] is None:
                    ss, qs = kss, kqs
                else:
                    v = maps[model['spec']]
                    ss = .75 * kss + .25 * (v[si] @ model['metric'] @ v[si].T)
                    qs = .75 * kqs + .25 * (v[qi] @ model['metric'] @ v[si].T)
                alpha = ev.coefficients(ss, support['pd'], sy, 8.)
                full = .5 * (base + softmax(query['pd'] + qs @ alpha, axis=1))
                outputs[arm] = full
                timings.append(dict(**key, arm=arm, common_seconds=common_seconds,
                                    operator_seconds=time.perf_counter() - t0, query_n=len(q), support_n=len(s)))
                records.append(dict(**key, arm=arm, **run.old.scores(y, full)))
            if t['archive']:
                replay_error = 0.
                with np.load(PREV / 'predictions' / t['archive']) as old:
                    np.testing.assert_array_equal(q, old['query_rows'])
                    np.testing.assert_array_equal(s, old['support_rows'])
                    for arm in ARMS:
                        expected = old['full'][list(old['arms']).index(arm)]
                        replay_error = max(replay_error, float(abs(outputs[arm] - expected).max()))
                        # Frozen source scores are float32-rounded; batch layout may change last bits.
                        np.testing.assert_allclose(outputs[arm], expected, atol=2e-7, rtol=1e-6)
                        np.testing.assert_array_equal(outputs[arm].argmax(1), expected.argmax(1))
                checks.append(dict(kind='archived_probability_and_exact_decision_replay', dataset=name, episode=t['episode'],
                                   budget=t['budget'], draw=t['draw'], cell=t['cell'], max_probability_error=replay_error))
            nc = np.bincount(y, minlength=8)
            for seed in range(3):
                pred = outputs[f'class_utility_{seed}'].argmax(1)
                for comparator in ('class_mean', 'class_fixed', f'class_metric_{seed}', 'original'):
                    ref = outputs[comparator].argmax(1)
                    hit = (pred == y).astype(int) - (ref == y).astype(int)
                    for j in np.flatnonzero(hit):
                        events.append(dict(**key, seed=seed, comparator=comparator, row=int(q[j]), ship=str(meta.loc[q[j]].identity),
                                           class_id=int(y[j]), net_hit=int(hit[j]), oa_contribution=100 * hit[j] / len(q),
                                           ba_contribution=100 * hit[j] / (len(np.unique(y)) * nc[y[j]]),
                                           support_class_count=int(np.sum(sy == y[j]))))
            tag = f"{name}_{t['episode']:03d}_{t['budget']}_{t['draw']}_{t['cell']}"
            np.savez_compressed(OUT / 'predictions' / f'{tag}.npz', full=np.stack([outputs[a] for a in ARMS]),
                                arms=np.array(ARMS), query_rows=q, support_rows=s, labels=y)
            if ti % 300 == 0:
                print('PROGRESS', name, ti, '/', len(active), flush=True)
        pd.DataFrame(records).to_csv(OUT / 'metrics.csv', index=False)
        pd.DataFrame(events).to_csv(OUT / 'events.csv', index=False)
        pd.DataFrame(timings).to_csv(OUT / 'timings.csv', index=False)
        write('CHECKS.json', checks)
        print('DONE', name, flush=True)
    pd.DataFrame(map_times).to_csv(OUT / 'mapping_timings.csv', index=False)
    write('COVERAGE.json', coverage)
    write('COMPLETE.json', dict(seconds=time.perf_counter() - start, support_cells=len(tasks), solves=len(tasks) * len(ARMS),
                               frozen_training_seeds=controls.SEEDS, new_training=0, new_source_geometry=0, main_model_changed=False))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
