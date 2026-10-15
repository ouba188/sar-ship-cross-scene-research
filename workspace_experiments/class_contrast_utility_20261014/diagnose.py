"""Task coverage and same-geometry held-port transfer, without retraining."""
import json
import pickle
import time
import numpy as np
import pandas as pd
from train_and_test import OUT, run, probe, weighted_bank, SEEDS


def main():
    start = time.perf_counter()
    meta = run.old.source_metadata()
    tasks = json.loads((OUT / 'TRAIN_ENROLLMENT.json').read_text(encoding='utf-8'))
    roles = []
    for i, task in enumerate(tasks):
        for role in ('query', 'support'):
            frame = meta.loc[task[role]]
            counts = frame.class_id.value_counts()
            for c, n in counts.items():
                roles.append(dict(task=i, role=role, port=task['port'], budget=task['budget'],
                                  class_id=int(c), observations=int(n)))
    pd.DataFrame(roles).to_csv(OUT / 'training_class_exposure.csv', index=False)
    rows = []
    for role in ('query', 'support'):
        ids = np.unique(sum([t[role] for t in tasks], []))
        for c in range(8):
            selected = meta.loc[ids]
            selected = selected[selected.class_id == c]
            count = sum(c in set(meta.loc[t[role]].class_id) for t in tasks)
            rows.append(dict(role=role, class_id=c, task_presence=count, ships=selected.identity.nunique(),
                             observations=len(selected), acquisitions=selected.acquisition.nunique()))
    pd.DataFrame(rows).to_csv(OUT / 'training_class_coverage.csv', index=False)
    learned = np.load(OUT / 'frozen_weights.npy')
    write = lambda name, value: (OUT / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    write('WEIGHT_DIAGNOSTICS.json', [dict(seed=seed, maximum=float(w.max()),
        maximum_pair=probe.PAIRS[w.argmax()].tolist(), top2_mass=float(np.sort(w)[-2:].sum() / w.sum()),
        effective_pairs=float(w.sum() ** 2 / np.sum(w * w))) for seed, w in zip(SEEDS, learned)])
    with (OUT / 'training_bank.pkl').open('rb') as h:
        linear = pickle.load(h)
    with (run.PREV / 'validation_fit/geometry.pkl').open('rb') as h:
        geometry = pickle.load(h)
    plans = json.loads((run.EXPAND / 'ENROLLMENT.json').read_text(encoding='utf-8'))
    rows = np.unique(sum([e['query'] + sum([d['64'][1] for d in e['draws']], []) for e in plans], []))
    training_ids = np.load(run.PREV / 'validation_fit_rows.npy')
    for field in ('identity', 'acquisition', 'port'):
        assert not set(meta.loc[training_ids, field]) & set(meta.loc[rows, field])
    ev = run.old.operator()
    f = run.old.encoded(ev, run.old.source_arrays(), rows, geometry)
    results = []
    for e in plans:
        q = np.array(e['query'])
        query = run.select(f, q)
        y = meta.loc[q].class_id.to_numpy(int)
        for draw, d in enumerate(e['draws']):
            s = np.array(d['64'][1])
            support = run.select(f, s)
            sy = meta.loc[s].class_id.to_numpy(int)
            for arm, bank in [('linear', linear)] + [(f'utility_{i}', weighted_bank(linear, w)) for i, w in enumerate(learned)]:
                model = probe.ContrastModel(ev.coefficients, bank, rho=.25).fit_support(support, sy)
                p = model.predict(query)['full']
                results.append(dict(port=e['port'], episode=e['episode'], cohort=e['cohort'], draw=draw, arm=arm,
                                    **run.old.scores(y, p)))
    metrics = pd.DataFrame(results)
    metrics.to_csv(OUT / 'same_geometry_held_metrics.csv', index=False)
    keys = ['port', 'episode', 'draw', 'cohort']
    ref = metrics[metrics.arm == 'linear'][keys + probe.analysis.VALUES]
    paired = metrics.merge(ref, on=keys, suffixes=('', '_linear'), validate='many_to_one')
    values = []
    for metric in probe.analysis.VALUES:
        col = metric + '_delta'
        paired[col] = paired[metric] - paired[metric + '_linear']
        values.append(col)
    draw = paired.groupby(['port', 'draw', 'arm'])[values].mean().groupby(['draw', 'arm']).mean().reset_index()
    draw.to_csv(OUT / 'same_geometry_held_draws.csv', index=False)
    summary = draw.groupby('arm')[values].mean().reset_index()
    summary.to_csv(OUT / 'same_geometry_held_summary.csv', index=False)
    write('DIAGNOSTICS_COMPLETE.json', dict(seconds=time.perf_counter() - start, same_geometry_support_cells=240,
        same_geometry_solves=960, same_geometry_query_rows=len(rows), retraining=0,
        purpose='Separate coordinate-bank refit from held-task generalization; fixed learned weights, all seeds',
        limitation='Changed source heads/geometry relative to previous target run, so not an isolated single-variable causal comparison'))
    print(pd.DataFrame(roles).groupby(['role', 'class_id']).observations.sum().to_string(), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
