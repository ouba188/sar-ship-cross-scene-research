"""Separate source geometry from class-dependent metadata availability."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from diagnose_feature_readability import SOURCE, geometry_features, probe, score


ROOT = Path('E:/Hermes/paper_tracking/sensitivity/information_loss_diagnosis_20261008')


def main():
    out = ROOT/'geometry_followup'
    out.mkdir(exist_ok=True)
    frame = pd.read_csv(SOURCE/'manifest.csv', low_memory=False)
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    geometry = geometry_features(frame, source=True)
    labels = frame.class_id.to_numpy(int)
    complete = geometry[:, 5:].sum(1) == 0
    pd.crosstab(frame.class_id, complete).to_csv(out/'availability_by_class.csv')
    results = []
    with threadpool_limits(limits=4):
        for task in ['ports24_within', 'ports24_leaveport']:
            with np.load(ROOT/'feature_probes'/task/'split.npz') as split:
                fit, query = split['ordered_fit_rows'], split['query_rows']
            bx, bq = np.asarray(base[fit]), np.asarray(base[query])
            views = {
                'r50_plus_missing_flags': (
                    np.column_stack((bx, geometry[fit, 5:])),
                    np.column_stack((bq, geometry[query, 5:])), fit, query),
                'missing_flags_only': (geometry[fit, 5:], geometry[query, 5:], fit, query),
            }
            # Complete-case arms share exactly the same fit/query rows. They
            # describe only classes 0/1/2, not the eight-class task.
            cf, cq = fit[complete[fit]], query[complete[query]]
            cx, cy = np.asarray(base[cf]), np.asarray(base[cq])
            views['complete3_r50'] = (cx, cy, cf, cq)
            views['complete3_r50_plus_geometry'] = (
                np.column_stack((cx, geometry[cf, :5])),
                np.column_stack((cy, geometry[cq, :5])), cf, cq)
            for name, (x, qx, f, q) in views.items():
                pred = probe(x, qx, labels[f], labels[q])
                metrics = score(pred, labels[q])
                row = dict(task=task, view=name, fit_rows=len(f), query_rows=len(q),
                    query_class_counts=np.bincount(labels[q], minlength=8).tolist(), **metrics)
                results.append(row)
                print(f'{task} {name} OA={100*metrics["acc"]:.3f} BA={100*metrics["ba"]:.3f}', flush=True)
                np.savez_compressed(out/f'{task}_{name}.npz', fit_rows=f, query_rows=q,
                    truth=labels[q], prediction=pred)
    with (out/'results.json').open('w', encoding='utf-8') as stream:
        json.dump(dict(purpose='diagnostic decomposition, not a deployable method',
            caveat='Complete geometry exists exclusively for classes 0/1/2; original nested4096 rows restricted without replenishment.',
            results=results), stream, indent=2)
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in results]).to_csv(out/'results.csv', index=False)


if __name__ == '__main__':
    main()
