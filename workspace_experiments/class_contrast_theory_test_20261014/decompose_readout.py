"""Locate gains in changed query relations versus support-coefficient refitting."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
OLD = ROOT / 'class_contrast_scope_validation_20261014'
sys.path.insert(0, str(ROOT / 'class_contrast_core_controls_20261014'))
import run_controls as core
run, ext = core.run, core.run.old.ext


def main():
    with (ROOT / 'class_contrast_core_controls_20261014/specs.pkl').open('rb') as h:
        spec = pickle.load(h)['class_edge']
    with (ROOT / 'class_contrast_core_controls_20261014/models.pkl').open('rb') as h:
        models = pickle.load(h)
    with (run.PREV / 'validation_fit/geometry.pkl').open('rb') as h:
        geometry = pickle.load(h)
    ev = run.old.operator()
    plans = json.loads((OLD / 'ENROLLMENT.json').read_text(encoding='utf-8'))
    events, checks = [], []
    for name, budget in [('f0921', 64), ('oss2', 64), ('ports', 32)]:
        active = [t for t in plans if t['dataset'] == name and t['budget'] == budget and t['cell'] == 'other_acquisition']
        meta = run.old.source_metadata() if name == 'ports' else pd.read_csv(run.old.OLD / f'{name}_metadata.csv', dtype={'identity': str}, keep_default_na=False)
        rows = np.unique(np.concatenate([t['query'] + t['support'] for t in active]))
        arrays = run.old.source_arrays() if name == 'ports' else ext.arrays_for(name, meta)
        f = run.old.encoded(ev, arrays, rows, geometry)
        pair = core.project(f['z'], spec)
        for task in active:
            q, s = np.array(task['query']), np.array(task['support'])
            qi, si = np.searchsorted(rows, q), np.searchsorted(rows, s)
            query, support = run.select(f, q), run.select(f, s)
            y, sy = meta.loc[q].class_id.to_numpy(int), meta.loc[s].class_id.to_numpy(int)
            bss = support['b'] @ support['b'].T + 1
            bqs = query['b'] @ support['b'].T + 1
            base = softmax(query['pb'] + bqs @ ev.coefficients(bss, support['pb'], sy, 8.), axis=1)
            vis_ss = .75 * (support['z'] @ support['z'].T + 1)
            vis_qs = .75 * (query['z'] @ support['z'].T + 1)
            ps, pq = pair[si], pair[qi]
            k0ss = vis_ss + .25 * (ps @ ps.T)
            k0qs = vis_qs + .25 * (pq @ ps.T)
            alpha0 = ev.coefficients(k0ss, support['pd'], sy, 8.)
            full0 = .5 * (base + softmax(query['pd'] + k0qs @ alpha0, axis=1))
            tag = f"{name}_{task['episode']:03d}_{budget}_{task['draw']}_other_acquisition.npz"
            with np.load(OLD / 'predictions' / tag) as archive:
                archived = archive['full']
                arms = list(archive['arms'])
                np.testing.assert_allclose(full0, archived[arms.index('class_fixed')], atol=3e-7, rtol=1e-6)
                other = full0.copy(); other[np.arange(len(y)), y] = -np.inf
                competitor = other.argmax(1)
                margin0 = full0[np.arange(len(y)), y] - full0[np.arange(len(y)), competitor]
                for seed in range(3):
                    weights = np.diag(models[f'class_utility_{seed}']['metric'])
                    ss = vis_ss + .25 * ((ps * weights) @ ps.T)
                    qs = vis_qs + .25 * ((pq * weights) @ ps.T)
                    relation = .5 * (base + softmax(query['pd'] + qs @ alpha0, axis=1))
                    alpha1 = ev.coefficients(ss, support['pd'], sy, 8.)
                    new = .5 * (base + softmax(query['pd'] + qs @ alpha1, axis=1))
                    np.testing.assert_allclose(new, archived[arms.index(f'class_utility_{seed}')], atol=3e-7, rtol=1e-6)
                    m1 = relation[np.arange(len(y)), y] - relation[np.arange(len(y)), competitor]
                    m2 = new[np.arange(len(y)), y] - new[np.arange(len(y)), competitor]
                    for j in range(len(q)):
                        events.append(dict(dataset=name, budget=budget, episode=task['episode'], draw=task['draw'], seed=seed,
                                           row=int(q[j]), ship=str(meta.loc[q[j]].identity), class_id=int(y[j]),
                                           relation_gain=float(m1[j] - margin0[j]), refit_gain=float(m2[j] - m1[j]),
                                           total_gain=float(m2[j] - margin0[j]),
                                           net_hit=int(new[j].argmax() == y[j]) - int(full0[j].argmax() == y[j])))
            checks.append(dict(dataset=name, episode=task['episode'], draw=task['draw'], budget=budget))
        print('DONE', name, len(active), flush=True)
        pd.DataFrame(events).to_csv(OUT / 'readout_paths.csv', index=False)
    data = pd.DataFrame(events)
    np.testing.assert_allclose(data.relation_gain + data.refit_gain, data.total_gain, atol=1e-12)
    data.groupby('dataset').agg(cells=('episode', 'size'), unique_ships=('ship', 'nunique'),
        direct_margin=('relation_gain', 'mean'), refit_margin=('refit_gain', 'mean'),
        total_margin=('total_gain', 'mean'), rescues=('net_hit', lambda x: int((x > 0).sum())),
        harms=('net_hit', lambda x: int((x < 0).sum()))).reset_index().to_csv(OUT / 'readout_summary.csv', index=False)
    data.groupby(['dataset', 'class_id'])[['relation_gain', 'refit_gain', 'total_gain', 'net_hit']].mean().reset_index().to_csv(OUT / 'readout_by_class.csv', index=False)
    ship = data.groupby(['dataset', 'ship'])[['relation_gain', 'refit_gain', 'total_gain']].mean().reset_index()
    paired = pd.read_csv(OUT / 'ship_outcome_link.csv', dtype={'ship': str})
    ship = ship.merge(paired[['dataset', 'ship', 'relative_shift']], on=['dataset', 'ship'], how='inner', validate='one_to_one')
    out = []
    for name, g in ship.groupby('dataset'):
        for field in ('relation_gain', 'refit_gain', 'total_gain'):
            corr = spearmanr(g.relative_shift, g[field])
            out.append(dict(dataset=name, ships=len(g), field=field, mean=float(g[field].mean()),
                            spearman=float(corr.statistic)))
    pd.DataFrame(out).to_csv(OUT / 'readout_shift_links.csv', index=False)
    (OUT / 'READOUT_CHECKS.json').write_text(json.dumps(dict(archived_task_replays=len(checks),
        verified_candidate_probabilities=3 * len(checks), max_path_additivity_error=float(abs(data.relation_gain + data.refit_gain - data.total_gain).max())), indent=2), encoding='utf-8')
    print(pd.read_csv(OUT / 'readout_summary.csv').to_string(index=False), flush=True)
    print(pd.DataFrame(out).to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
