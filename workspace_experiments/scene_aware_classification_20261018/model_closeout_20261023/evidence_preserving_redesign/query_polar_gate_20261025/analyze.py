"""Gate effects, frequency dependence, availability strata and native cases."""
import os
from pathlib import Path
import importlib.util
import sys
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplcache')
spec = importlib.util.spec_from_file_location('train', HERE / 'train.py')
run = importlib.util.module_from_spec(spec)
sys.modules['train'] = run
spec.loader.exec_module(run)
import json
import numpy as np
import pandas as pd
import torch

stats = run.load('gate_stats', HERE.parent / 'typed_m3_20261025/analyze.py')
stats.HERE = HERE
stats.PAIRS = [(a, 'hybrid_equal') for a in run.ARMS] + [
    ('frequency', 'spatial'), ('frequency', 'spatial_wide'), ('frequency', 'score_wide'),
    ('frequency', 'constant'), ('frequency', 'prior_late'), ('frequency', 'old_full'),
    ('frequency', 'hybrid_source'), ('frequency', 'no_frequency'), ('frequency', 'mean_weights')]


def population(data, scores, weights):
    rows, flips, wr = [], [], []
    for seed in run.SEEDS:
        raw, z = run.inputs(data, seed)
        available = z['available'].numpy().astype(bool)
        for role, ids in (('source_refit', data.va), ('evaluation', data.te)):
            y = data.table.iloc[ids].class_id.to_numpy(int)
            for arm, comp in stats.PAIRS:
                a, b = scores[seed, arm][ids], scores[seed, comp][ids]
                for group, take in (('all', np.ones(len(ids), bool)), ('paired', available[ids].all(1)),
                                    ('VH_missing', ~available[ids, 1])):
                    rows.append(dict(seed=seed, role=role, arm=arm, comparator=comp, group=group)
                                | stats.effect(a[take], b[take], y[take]))
                if role == 'evaluation':
                    for k in np.flatnonzero((a.argmax(1) == y) != (b.argmax(1) == y)):
                        row = data.table.iloc[ids[k]]
                        flips.append(dict(seed=seed, arm=arm, comparator=comp, obs_id=int(row.obs_id),
                            mmsi=row.mmsi, port=row.port, pass_group=row.pass_group, class_id=int(y[k]),
                            class_name=row.class_name, VH_present=bool(available[ids[k], 1]),
                            case='rescue' if a[k].argmax() == y[k] else 'harm',
                            before=int(b[k].argmax()), after=int(a[k].argmax())))
            for arm in run.ARMS:
                for p, pol in enumerate(('VV', 'VH')):
                    val = weights[seed, arm][ids, p][available[ids, p]]
                    wr.append(dict(seed=seed, role=role, arm=arm, polarization=pol, n=len(val),
                                   mean=val.mean(), std=val.std(), min=val.min(), max=val.max()))
    pd.DataFrame(rows).to_csv(HERE / 'population_pairs.csv', index=False)
    pd.DataFrame(flips).to_csv(HERE / 'flips.csv', index=False)
    pd.DataFrame(wr).to_csv(HERE / 'weight_summary.csv', index=False)


def select_cases(data, scores):
    ids = data.te; y = data.table.iloc[ids].class_id.to_numpy(int)
    result, used = [], set()
    for comp, names in (('spatial_wide', ('rescue', 'harm', 'CC', 'WW')),
                        ('no_frequency', ('rescue', 'harm')), ('mean_weights', ('rescue', 'harm'))):
        for name in names:
            options = []
            for seed in run.SEEDS:
                a = scores[seed, 'frequency'][ids].argmax(1) == y
                b = scores[seed, comp][ids].argmax(1) == y
                take = {'rescue': a & ~b, 'harm': ~a & b, 'CC': a & b, 'WW': ~a & ~b}[name]
                for j in ids[take]:
                    obs = int(data.table.iloc[j].obs_id)
                    if obs not in used:
                        options.append((seed, obs, int(j)))
            if options:
                options.sort()
                seed, obs, j = options[0] if name in ('rescue', 'harm') else options[len(options)//2]
                used.add(obs)
                result.append(dict(seed=seed, obs_id=obs, index=j, case=name, arm='frequency', comparator=comp))
    for obs, seed in ((22572, 17), (2080, 43), (5585, 29), (24640, 43)):
        if obs in used:
            continue
        j = int(np.flatnonzero(data.table.obs_id.eq(obs) & data.table.role.eq('evaluation'))[0])
        yy = int(data.table.iloc[j].class_id)
        a, b = (scores[seed, arm][j].argmax() == yy for arm in ('frequency', 'hybrid_equal'))
        name = 'CC' if a and b else 'WW' if not a and not b else 'rescue' if a else 'harm'
        result.append(dict(seed=seed, obs_id=obs, index=j, case=name, arm='frequency', comparator='hybrid_equal', historical=True))
    for obs, seed in ((26176, 29), (11655, 43)):
        if obs in used:
            continue
        j = int(np.flatnonzero(data.table.obs_id.eq(obs) & data.table.role.eq('evaluation'))[0])
        result.append(dict(seed=seed, obs_id=obs, index=j, case='rescue',
                           arm='frequency', comparator='no_frequency'))
    return result


def main():
    assert (HERE / 'complete.json').exists()
    data = run.Data(); scores, weights = {}, {}
    for seed in run.SEEDS:
        for arm in run.ARMS + run.COMPARATORS:
            with np.load(HERE / 'results' / f'{arm}_{seed}.npz') as z:
                np.testing.assert_array_equal(z['up_index'], data.up[:data.n])
                scores[seed, arm] = z['score'].copy()
                if arm in run.ARMS:
                    weights[seed, arm] = z['weight'].copy()
        with np.load(HERE / 'results' / f'diagnostics_{seed}.npz') as z:
            for arm in ('no_frequency', 'mean_weights'):
                scores[seed, arm] = z[arm].copy()
    receipt = stats.statistics(data, scores)
    population(data, scores, weights)
    native = run.load('gate_native_panels', HERE.parent / 'null_centered_readout_20261025/analyze.py')
    native.HERE = HERE
    geo = run.load('gate_geometry', run.BASE.parent / 'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up = pd.read_csv(run.BASE / 'regional_utility_pilot/upstream/table.csv')
    valid = np.load(run.BASE / 'regional_utility_pilot/upstream/summaries.npz')['valid']
    snapshots = {int(s['obs_id']): s for s in json.loads((geo.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs = np.load(run.LOCAL / 'scores_layer1.npz')['refs']
    records = []
    for case in select_cases(data, scores):
        path, provenance, true, rival = native.panel(case, data, scores, geo, up, valid, snapshots, refs)
        seed, j = case['seed'], case['index']
        raw, z = run.inputs(data, seed)
        dv = float(z['delta_vv'][j, true]-z['delta_vv'][j, rival])
        dh = float(z['delta_vh'][j, true]-z['delta_vh'][j, rival])
        records.append(case | dict(true=true, rival=rival, figure=str(path), provenance=provenance,
            branch_delta_margin=dict(VV=dv, VH=dh),
            weights={a: weights[seed, a][j].tolist() for a in run.ARMS},
            comparisons={a: dict(prediction=int(scores[seed, a][j].argmax()),
                fixed_margin=float(scores[seed, a][j, true]-scores[seed, a][j, rival]))
                for a in run.ARMS+run.COMPARATORS+('no_frequency', 'mean_weights')}))
        print('CASE', case, flush=True)
    run.write_json(HERE / 'analysis/cases.json', records)
    receipt.update(cases=len(records), note='Frozen-head input neutralization and source-mean gate replacement are diagnostics, not raw-image causal interventions.')
    run.write_json(HERE / 'analysis_receipt.json', receipt)
    print(pd.read_csv(HERE / 'paired.csv').groupby(['arm', 'comparator'])[['OA_pp', 'BA_pp', 'rescued', 'harmed']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
