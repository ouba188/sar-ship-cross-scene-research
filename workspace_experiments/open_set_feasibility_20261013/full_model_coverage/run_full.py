"""Fixed support-coverage repair in the exact frozen eight-class full model."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
PILOT = OUT.parent
WORK = PILOT.parent
ARCHIVE = WORK/'full_model_explainability_20261013/tasks'
sys.path.insert(0, str(PILOT))
import study as compact
sys.path.insert(0, str(WORK/'full_model_explainability_20261013'))
from run_paths import Geometry
from unified_head_run import base_kernel
from unified_head_core import coefficients

spec = importlib.util.spec_from_file_location('coverage_sampling', PILOT/'coverage_intervention/run.py')
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)

ARMS = ('random', 'cover1', 'sham1')
MODES = ('base', 'expert', 'full')


def write(name, value):
    compact.write(OUT/name, value)


def metadata(tasks):
    source = pd.read_csv(compact.SOURCE/'manifest.csv', usecols=['instance_id','mmsi','class_id','port'],
                         dtype={'mmsi': str}, keep_default_na=False)
    source = compact.clean(source, 'mmsi', 'instance_id')
    old = json.loads((PILOT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    frames, plans, inventory = {}, [], []
    for task in tasks:
        name = task['task']
        directory = ARCHIVE/name
        rows = np.load(directory/'input/rows.npy', mmap_mode='r')
        ids = np.load(directory/'input/ids.npy', mmap_mode='r')
        if name.startswith('self24ports/'):
            frame = source.iloc[rows].reset_index(drop=True)
            alias = 'ports/'+name.split('/', 1)[1]
        else:
            dataset = name.split('/')[0]
            folder = compact.ARCH/('f0921_detector_recrop_20261007' if dataset == 'f0921' else 'opensarship2_transfer_20261007')
            idcol = 'mmsi' if dataset == 'f0921' else 'mmsi_or_ship_id'
            frame = compact.clean(pd.read_csv(folder/'target_manifest.csv', dtype={idcol:str}, keep_default_na=False), idcol, 'sample_id')
            alias = dataset
        np.testing.assert_array_equal(frame.sample_id.to_numpy(), ids)
        good = frame[frame.eligible]
        ships = np.sort(good.identity.unique())
        rng = np.random.default_rng(compact.SEED)
        shuffled = rng.permutation(ships)
        pool_n = max(64, int(np.ceil(.3*len(shuffled))))
        if pool_n >= len(shuffled):
            inventory.append(dict(task=name, usable=False, ships=len(ships), reason='No disjoint queries after 64 support ships'))
            continue
        pool = good[good.identity.isin(set(shuffled[:pool_n]))]
        queries = good[~good.identity.isin(set(shuffled[:pool_n]))]
        q = []
        for c in range(8):
            sub = queries[queries.class_id == c]
            selected = rng.permutation(sub.identity.unique())[:128]
            q.extend(int(rng.choice(sub.index[sub.identity == ship])) for ship in selected)
        q = np.array(sorted(q), int)
        if alias in old:
            np.testing.assert_array_equal(q, old[alias]['query'])
        counts = pool.groupby('class_id').identity.nunique().reindex(range(8), fill_value=0).to_numpy()
        qcounts = np.bincount(frame.class_id.iloc[q], minlength=8)
        inventory.append(dict(task=name, usable=True, ships=len(ships), pool_ships=pool_n,
            query_ships=len(q), query_classes=int((qcounts > 0).sum()), pool_classes=int((counts > 0).sum()),
            query_class_without_pool=int(((qcounts > 0)&(counts == 0)).sum()),
            pool_counts=counts.tolist(), query_counts=qcounts.tolist(), previous_query_replayed=alias in old))
        for draw in range(5):
            rr = np.random.default_rng(compact.SEED+90001+draw)
            sids = rr.choice(pool.identity.unique(), 64, replace=False)
            original = np.array([int(rr.choice(pool.index[pool.identity == ship])) for ship in sids])
            seed = compact.SEED+19001+draw
            covered, edits, _ = sampling.repair(frame, pool, original, np.arange(8), 1, seed)
            sham = original.copy()
            used = set(frame.identity.iloc[original])
            srng = np.random.default_rng(seed+30001)
            sham_ok = True
            for pos in edits:
                c = int(frame.class_id.iloc[original[pos]])
                available = pool[(pool.class_id == c)&~pool.identity.isin(used)]
                if available.empty:
                    sham_ok = False
                    break
                row = sampling.pick_row(pool, c, used, srng)
                sham[pos] = row
                used.add(frame.identity.iloc[row])
            variants = dict(random=original, cover1=covered)
            if sham_ok:
                variants['sham1'] = sham
                np.testing.assert_array_equal(np.bincount(frame.class_id.iloc[sham], minlength=8),
                                              np.bincount(frame.class_id.iloc[original], minlength=8))
                assert np.sum(sham != original) == len(edits)
            for arm, selected in variants.items():
                selected_ids = set(frame.identity.iloc[selected])
                assert len(selected_ids) == len(selected) == 64
                assert not selected_ids & set(frame.identity.iloc[q])
                assert set(selected).issubset(set(pool.index))
                sy = frame.class_id.iloc[selected].to_numpy()
                sc = np.bincount(sy, minlength=8)
                if arm == 'cover1':
                    assert np.all(sc[counts > 0] > 0)
                plans.append(dict(task=name, draw=draw, arm=arm, supports=selected.tolist(),
                    counts=sc.tolist(), old_counts=np.bincount(frame.class_id.iloc[original], minlength=8).tolist(),
                    changed=int(np.sum(selected != original)), sham_feasible=sham_ok))
        frames[name] = (frame, q)
    write('SUPPORT_PLAN.json', plans)
    pd.DataFrame(inventory).to_csv(OUT/'inventory.csv', index=False)
    write('QUERY_PLAN.json', {name:q.tolist() for name, (_,q) in frames.items()})
    print(pd.DataFrame(inventory).drop(columns=['pool_counts','query_counts']).to_string(index=False), flush=True)
    return frames, plans


def execute(task, frame, q, plan):
    name = task['task']; start = time.perf_counter()
    dest = OUT/'tasks'/name; dest.mkdir(parents=True, exist_ok=True)
    cache = ARCHIVE/name
    data = {key:np.load(cache/'input'/f'{key}.npy', mmap_mode='r')
            for key in ('rows','ids','support','support_labels','phi','mean','conditions','prior')}
    emb = np.load(cache/'embedding/embedding.npy', mmap_mode='r')
    np.testing.assert_array_equal(np.load(cache/'embedding/rows.npy'), data['rows'])
    with np.load(task['coefficients']) as z:
        co = {k:z[k] for k in z.files}
    geo = Geometry(task, data, co, dest)
    reference = np.unique(np.r_[data['support'].ravel(), np.array([r['supports'] for r in plan]).ravel()])
    union = np.unique(np.r_[reference, q])
    take = np.searchsorted(union, reference)
    qp = np.searchsorted(union, q)
    # All score paths use precisely the frozen full-model kernels.
    encoded = geo.encode(union)
    refs = {key:None if value is None else value[take] for key,value in encoded.items()}
    kr, kbextra = geo.kernels(encoded, refs)
    ke = kr+kbextra
    local = {key:data[key][union] for key in ('phi','mean','conditions')}
    kb = base_kernel(local, emb[union].astype(float), take, geo.ctx.info['eta'])
    pe = encoded['pt']; pb = np.asarray(data['prior'][union], float)
    labels = frame.class_id.to_numpy()
    with np.load(task['predictions']) as z:
        old = z['predictions']; oldmodes = list(z['modes'])
    replays = []
    for draw, supports in enumerate(data['support']):
        sp = np.searchsorted(union, supports); cols = np.searchsorted(reference, supports)
        sy = labels[supports]
        np.testing.assert_array_equal(sy, data['support_labels'][draw])
        a = coefficients(kb[np.ix_(sp,cols)], pb[sp], sy, 8.)
        b = coefficients(ke[np.ix_(sp,cols)], pe[sp], sy, 8.)
        np.testing.assert_allclose(a, co[f'{draw}_base'], atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(b, co[f'{draw}_dual_refit_expert'], atol=1e-7, rtol=1e-7)
        pbase = softmax(pb[qp]+kb[np.ix_(qp,cols)]@a, axis=1)
        pexpert = softmax(pe[qp]+ke[np.ix_(qp,cols)]@b, axis=1)
        for mode, prob in [('base',pbase),('dual_refit_expert',pexpert),('dual_refit_late',.5*(pbase+pexpert))]:
            expected = old[draw,oldmodes.index(mode),q]
            np.testing.assert_array_equal(prob.argmax(1), expected)
            replays.append(dict(task=name, draw=draw, mode=mode, flips=0, query_n=len(q)))
    probability = np.full((5,3,3,len(q),8), np.nan, float)
    support = np.full((5,3,64), -1, int)
    timing = []
    for rec in plan:
        draw = rec['draw']; ai = ARMS.index(rec['arm']); rows = np.array(rec['supports'])
        sp = np.searchsorted(union, rows); cols = np.searchsorted(reference, rows)
        t0 = time.perf_counter()
        a = coefficients(kb[np.ix_(sp,cols)], pb[sp], labels[rows], 8.)
        b = coefficients(ke[np.ix_(sp,cols)], pe[sp], labels[rows], 8.)
        pbase = softmax(pb[qp]+kb[np.ix_(qp,cols)]@a, axis=1)
        pexpert = softmax(pe[qp]+ke[np.ix_(qp,cols)]@b, axis=1)
        probability[draw,ai] = np.stack((pbase,pexpert,.5*(pbase+pexpert)))
        support[draw,ai] = rows
        timing.append(dict(task=name, draw=draw, arm=rec['arm'], support_solve_predict_seconds=time.perf_counter()-t0))
    selected = np.unique(np.r_[q,support[support>=0]])
    frame.iloc[selected].assign(local_row=selected).to_csv(dest/'manifest.csv', index=False)
    np.savez_compressed(dest/'predictions.npz', probability=probability, supports=support,
        query=q, labels=labels[q], identities=frame.identity.iloc[q].to_numpy(dtype=str),
        support_labels=np.where(support >= 0, labels[np.maximum(support,0)], -1),
        modes=np.array(MODES), arms=np.array(ARMS))
    duration = time.perf_counter()-start
    print('FULL COMPLETE', name, 'query', len(q), 'seconds', round(duration,2), flush=True)
    del geo, encoded, refs, kr, kbextra, ke, kb, data, old
    gc.collect()
    return replays, timing, dict(task=name, seconds=duration, query_n=len(q), reference_n=len(reference))


def main():
    start = time.perf_counter()
    tasks = json.loads((WORK/'paper_core_freeze_20261013/FROZEN_REFERENCE.json').read_text(encoding='utf-8'))['tasks']
    frames, plan = metadata(tasks)
    replays, costs, task_costs = [], [], []
    with threadpool_limits(limits=1):
        for task in tasks:
            if task['task'] not in frames:
                continue
            r,c,t = execute(task, *frames[task['task']], [x for x in plan if x['task'] == task['task']])
            replays.extend(r); costs.extend(c); task_costs.append(t)
            pd.DataFrame(replays).to_csv(OUT/'replay.csv', index=False)
            pd.DataFrame(costs).to_csv(OUT/'timing.csv', index=False)
            pd.DataFrame(task_costs).to_csv(OUT/'task_timing.csv', index=False)
    write('PREDICTIONS_COMPLETE.json', dict(tasks=len(frames), units=len(plan), replays=len(replays),
        seconds=time.perf_counter()-start, new_source_fits=0, new_neural_fits=0,
        closed_set=True, unknown_detection_evaluated=False, main_reference_modified=False))


if __name__ == '__main__':
    main()
