"""Minimal class-coverage repairs at a fixed support budget, no neural training."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
sys.path.insert(0, str(BASE))
import study as core

ARMS = ('random', 'cover1', 'sham1', 'cover3')


def pool_rows(frame):
    good = frame[frame.eligible]
    ships = np.sort(good.identity.unique())
    rng = np.random.default_rng(core.SEED)
    support_ids = set(rng.permutation(ships)[:int(np.ceil(.3 * len(ships)))])
    return good[good.identity.isin(support_ids)]


def pick_row(pool, cl, excluded, rng):
    eligible = pool[(pool.class_id == cl) & ~pool.identity.isin(excluded)]
    ship = rng.choice(eligible.identity.unique())
    return int(rng.choice(eligible.index[eligible.identity == ship]))


def repair(frame, pool, original, classes, floor, seed):
    rng = np.random.default_rng(seed)
    new = original.copy()
    used = set(frame.identity.iloc[original])
    capacity = pool.groupby('class_id').identity.nunique().to_dict()
    minimum = {int(c): min(floor, capacity.get(c, 0)) for c in classes}
    edits = []
    for cl in classes:
        while np.count_nonzero(frame.class_id.iloc[new].to_numpy() == cl) < minimum[cl]:
            labels = frame.class_id.iloc[new].to_numpy()
            counts = np.bincount(labels, minlength=8)
            donor_classes = [c for c in classes if counts[c] > minimum[c]]
            top = max(counts[c] for c in donor_classes)
            donor = int(rng.choice([c for c in donor_classes if counts[c] == top]))
            positions = np.flatnonzero(labels == donor)
            positions = np.array([p for p in positions if p not in edits])
            pos = int(rng.choice(positions))
            row = pick_row(pool, cl, used, rng)
            used.add(frame.identity.iloc[row])
            new[pos] = row
            edits.append(pos)
    return new, edits, minimum


def main():
    start = time.perf_counter()
    core.OUT = OUT / 'input_replay'
    core.OUT.mkdir(parents=True, exist_ok=True)
    source, source_rows, arrays, tasks, enrollment = core.inputs()
    assert enrollment == json.loads((BASE / 'ENROLLMENT.json').read_text(encoding='utf-8'))
    np.testing.assert_array_equal(source_rows, np.load(BASE / 'source_rows.npy'))
    source_ids = set(source.identity.iloc[source_rows])
    plan = []
    capacities = []
    for task, t in tasks.items():
        frame = t['frame']
        pool = pool_rows(frame)
        for c in range(8):
            capacities.append(dict(task=task, class_id=c, ships=pool[pool.class_id == c].identity.nunique()))
        for trial in enrollment[task]['trials']:
            unknown = trial['unknown']
            classes = np.delete(np.arange(8), unknown)
            p = pool[pool.class_id != unknown]
            for draw, rows in enumerate(trial['supports']):
                original = np.array(rows, int)
                seed = core.SEED + 19001 + 101 * unknown + draw
                one, edits, minima = repair(frame, p, original, classes, 1, seed)
                three, edits3, minima3 = repair(frame, p, original, classes, 3, seed)
                sham = original.copy()
                rng = np.random.default_rng(seed + 30001)
                used = set(frame.identity.iloc[original])
                for pos in edits:
                    row = pick_row(p, int(frame.class_id.iloc[original[pos]]), used, rng)
                    sham[pos] = row
                    used.add(frame.identity.iloc[row])
                variants = dict(random=original, cover1=one, sham1=sham, cover3=three)
                oldcounts = np.bincount(frame.class_id.iloc[original], minlength=8)
                np.testing.assert_array_equal(np.bincount(frame.class_id.iloc[sham], minlength=8), oldcounts)
                assert np.sum(one != original) == np.sum(sham != original) == len(edits)
                for arm, selected in variants.items():
                    selected_ids = set(frame.identity.iloc[selected])
                    assert len(selected) == len(selected_ids) == 64
                    assert not selected_ids & source_ids
                    assert not selected_ids & set(frame.identity.iloc[t['query']])
                    assert set(selected).issubset(set(p.index))
                    labels = frame.class_id.iloc[selected].to_numpy()
                    assert unknown not in labels
                    counts = np.bincount(labels, minlength=8)
                    plan.append(dict(task=task, unknown=unknown, draw=draw, arm=arm,
                        supports=selected.tolist(), counts=counts.tolist(),
                        old_counts=oldcounts.tolist(), changed=int(np.sum(selected != original)),
                        original_missing=int(np.sum(oldcounts[classes] == 0)),
                        missing_after=int(np.sum(counts[classes] == 0)),
                        full_coverage_feasible=all(minima[c] > 0 for c in classes),
                        floor3_feasible=all(minima3[c] == 3 for c in classes)))
        union = np.unique(np.r_[t['query'], np.array([r['supports'] for r in plan if r['task'] == task]).ravel()])
        t['reference_union'] = t['union'].copy()
        t['reference_values'] = core.raw_take(t['arrays'], frame, t['reference_union'])
        t['union'] = union
        t['values'] = core.raw_take(t['arrays'], frame, union)
        destination = OUT / 'tasks' / task
        destination.mkdir(parents=True, exist_ok=True)
        frame.iloc[union].assign(local_row=union).to_csv(destination / 'manifest.csv', index=False)
    core.write(OUT / 'SUPPORT_PLAN.json', plan)
    pd.DataFrame(capacities).to_csv(OUT / 'pool_capacity.csv', index=False)
    print('CAPACITIES', pd.DataFrame(capacities).pivot(index='task', columns='class_id', values='ships').to_string(), flush=True)
    print('PLANNED', len(plan), 'arms; outcomes not inspected', flush=True)
    replay = []
    with threadpool_limits(limits=1):
        for unknown in range(8):
            with (BASE / 'fits' / f'without_{unknown}.pkl').open('rb') as handle:
                model = pickle.load(handle)
            for task, t in tasks.items():
                archive = np.load(BASE / 'tasks' / task / f'unknown_{unknown}.npz')
                f = core.encode(t['values'], model)
                # Preserve the original float32 PCA batch path on all archived rows.
                reference = core.encode(t['reference_values'], model)
                reference_positions = np.searchsorted(t['union'], t['reference_union'])
                for field in f:
                    f[field][reference_positions] = reference[field]
                q = np.searchsorted(t['union'], t['query'])
                yy = t['frame'].class_id.to_numpy()
                np.testing.assert_array_equal(archive['query'], t['query'])
                records = [r for r in plan if r['task'] == task and r['unknown'] == unknown]
                # Reloaded float32 transforms can differ at roundoff. Pair all arms
                # within this execution and require unchanged archived decisions.
                replayed = {}
                for draw in range(5):
                    ref = next(r for r in records if r['arm'] == 'random' and r['draw'] == draw)
                    ss = np.searchsorted(t['union'], ref['supports'])
                    pp, cc = core.predict(f, ss, q, yy[ref['supports']], model['classes'], loo=True)
                    scores, cs = core.anomaly(f, ss, q, pp, cc, model['source_bank'])
                    threshold = np.quantile(cs, .95, axis=0)
                    np.testing.assert_array_equal(pp.argmax(1), archive['probability'][draw].argmax(1))
                    np.testing.assert_array_equal(scores > threshold, archive['score'][draw] > archive['thresholds'][draw, 1])
                    replayed[draw] = (pp, scores, cs, threshold)
                    replay.append(dict(task=task, unknown=unknown, draw=draw,
                        prob_error=float(abs(pp-archive['probability'][draw]).max()),
                        score_error=float(abs(scores-archive['score'][draw]).max()),
                        threshold_error=float(abs(threshold-archive['thresholds'][draw, 1]).max()),
                        classification_flips=0, rejection_flips=0))
                for arm in ARMS:
                    probabilities, anomalies, calibrations, thresholds, selected = [], [], [], [], []
                    for draw in range(5):
                        rec = next(r for r in records if r['arm'] == arm and r['draw'] == draw)
                        rows = rec['supports']
                        if rec['changed'] == 0:
                            prob, score, cal_score, threshold = replayed[draw]
                        else:
                            s = np.searchsorted(t['union'], rows)
                            prob, cal = core.predict(f, s, q, yy[rows], model['classes'], loo=True)
                            score, cal_score = core.anomaly(f, s, q, prob, cal, model['source_bank'])
                            threshold = np.quantile(cal_score, .95, axis=0)
                            # This query score uses the frozen source bank, not target supports.
                            np.testing.assert_allclose(score[:, 1], replayed[draw][1][:, 1], atol=1e-10, rtol=0)
                        probabilities.append(prob); anomalies.append(score)
                        calibrations.append(cal_score); thresholds.append(threshold); selected.append(rows)
                    np.savez_compressed(OUT / 'tasks' / task / f'unknown_{unknown}_{arm}.npz',
                        probability=np.array(probabilities), score=np.array(anomalies),
                        calibration_score=np.array(calibrations), thresholds=np.array(thresholds),
                        supports=np.array(selected), support_labels=yy[np.array(selected)],
                        query=t['query'], labels=yy[t['query']], classes=model['classes'])
                print('DONE', task, unknown, round(time.perf_counter()-start, 1), flush=True)
    pd.DataFrame(replay).to_csv(OUT / 'replay.csv', index=False)
    core.write(OUT / 'PREDICTIONS_COMPLETE.json', dict(contexts=200, arms=4, source_fits=0, neural_training=False,
        seconds=time.perf_counter()-start, reference_replays=len(replay),
        changed_support_arms=sum(r['changed'] > 0 for r in plan)))


if __name__ == '__main__':
    main()
