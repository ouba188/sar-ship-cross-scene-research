"""Ship-cluster uncertainty and scene reporting for fixed completed predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
import raw_current as raw

ROOT = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)


def report(folder, rows, scores, full):
    y = np.array([r['label'] for r in rows])
    ship = np.array([str(r['mmsi']) for r in rows])
    scene = np.array([raw.task.orbit(r) for r in rows])
    correct = {name: np.stack([z.argmax(1) == y for z in runs]).mean(0)
               for name, runs in scores.items()}
    groups = []
    for cls in np.unique(y):
        units = [np.flatnonzero(ship == s) for s in np.unique(ship[y == cls])]
        assert all(np.all(y[u] == cls) for u in units), 'Mixed-label ship'
        groups.append(units)
    intervals, scenes = [], []
    for control, base in correct.items():
        if control == full:
            continue
        delta = correct[full] - base
        rng = np.random.default_rng(20261109)
        total, count, class_gain = [], [], []
        for units in groups:
            sums = np.array([delta[u].sum() for u in units])
            sizes = np.array([len(u) for u in units])
            draw = rng.integers(0, len(units), (3000, len(units)))
            t, n = sums[draw].sum(1), sizes[draw].sum(1)
            total.append(t); count.append(n); class_gain.append(t / n)
        oa = 100 * np.sum(total, axis=0) / np.sum(count, axis=0)
        ba = 100 * np.mean(class_gain, axis=0)
        intervals.append(dict(control=control, n=len(y), ships=len(set(ship)),
            oa_delta=100*delta.mean(), oa_low=np.quantile(oa,.025), oa_high=np.quantile(oa,.975),
            ba_delta=100*np.mean([delta[y==c].mean() for c in np.unique(y)]),
            ba_low=np.quantile(ba,.025), ba_high=np.quantile(ba,.975),
            scope='class-stratified paired ship bootstrap; conditional on stored fits and observed scenes'))
        for value in np.unique(scene):
            mask = scene == value
            scenes.append(dict(control=control, scene=value, n=int(mask.sum()),
                classes=','.join(map(str,np.unique(y[mask]))), oa_delta=100*delta[mask].mean(),
                ba_delta_present=100*np.mean([delta[mask & (y==c)].mean() for c in np.unique(y[mask])])))
    pd.DataFrame(intervals).to_csv(folder/'paired_intervals.csv',index=False)
    pd.DataFrame(scenes).to_csv(folder/'per_scene_differences.csv',index=False)
    print(folder.name, pd.DataFrame(intervals).to_string(index=False), flush=True)


def main():
    records, groups = raw.prev.records()
    folder = ROOT/'common_init'
    scores = {}
    for arm in ('plain','full','no_m1','no_m2','no_m3'):
        runs = []
        for seed in SEEDS:
            d = np.load(folder/f'{arm}_s{seed}_360/evaluation.npz')
            assert np.array_equal(d['ids'],groups['evaluation'])
            runs.append(d['score'])
        scores[arm] = runs
    report(folder,[records[i] for i in groups['evaluation']],scores,'full')
    folder = ROOT/'oss1_transfer'
    rows = raw.task.data.read_manifest(folder/'target_manifest.json')
    data = np.load(folder/'predictions.npz')
    assert data['ids'].tolist() == [r['id'] for r in rows]
    scores = {arm:[data[f'{arm}_s{s}'] for s in SEEDS]
              for arm in ('class','mean','content','no_m3_refit')}
    report(folder,rows,scores,'class')


if __name__ == '__main__':
    main()
