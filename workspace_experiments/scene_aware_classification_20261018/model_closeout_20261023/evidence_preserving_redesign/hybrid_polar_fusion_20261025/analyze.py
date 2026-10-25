"""Paired fixed-weight effects and native query/reference review."""
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

stats = run.load('hybrid_stats', HERE.parent / 'typed_m3_20261025/analyze.py')
stats.HERE = HERE
stats.PAIRS = [
    ('hybrid_equal', 'prior_late'), ('hybrid_equal', 'center_late'),
    ('hybrid_source', 'hybrid_equal'), ('hybrid_source', 'ordinary_source'),
    ('ordinary_source', 'prior_late'), ('hybrid_source', 'prior_late'),
    ('hybrid_source', 'old_full'), ('hybrid_source', 'strong'),
    ('hybrid_equal', 'old_full'), ('hybrid_source', 'ordinary_at_hybrid_weight'),
    ('hybrid_at_ordinary_weight', 'ordinary_source'),
    ('ordinary_at_hybrid_weight', 'ordinary_source')]


def population(data, scores):
    rows, flips = [], []
    for seed in run.SEEDS:
        z = np.load(run.previous.PREVIOUS / 'results' / f'inputs_{seed}.npz')
        for role, ids in (('selection', data.va), ('evaluation', data.te)):
            y = data.table.iloc[ids].class_id.to_numpy(int)
            for arm, comp in stats.PAIRS:
                a, b = scores[seed, arm][ids], scores[seed, comp][ids]
                for group, take in (('all', np.ones(len(ids), bool)),
                    ('paired', z['query_valid'][ids].all(1)), ('VH_missing', ~z['query_valid'][ids, 1])):
                    rows.append(dict(seed=seed, role=role, arm=arm, comparator=comp, group=group)
                                | stats.effect(a[take], b[take], y[take]))
                if role != 'evaluation':
                    continue
                changed = (a.argmax(1) == y) != (b.argmax(1) == y)
                for k in np.flatnonzero(changed):
                    r = data.table.iloc[ids[k]]
                    flips.append(dict(seed=seed, arm=arm, comparator=comp, obs_id=int(r.obs_id),
                        mmsi=r.mmsi, port=r.port, pass_group=r.pass_group, class_id=int(y[k]),
                        class_name=r.class_name, case='rescue' if a[k].argmax() == y[k] else 'harm',
                        VH_present=bool(z['query_valid'][ids[k], 1]), before=int(b[k].argmax()), after=int(a[k].argmax())))
    pd.DataFrame(rows).to_csv(HERE / 'population_pairs.csv', index=False)
    pd.DataFrame(flips).to_csv(HERE / 'flips.csv', index=False)


def cases(data, scores):
    ids = data.te
    y = data.table.iloc[ids].class_id.to_numpy(int)
    out, used = [], set()
    for arm, comp, names in (
        ('hybrid_equal', 'prior_late', ('rescue', 'harm', 'CC', 'WW')),
        ('hybrid_source', 'ordinary_source', ('rescue', 'harm'))):
        for name in names:
            choices = []
            for seed in run.SEEDS:
                a = scores[seed, arm][ids].argmax(1) == y
                b = scores[seed, comp][ids].argmax(1) == y
                take = {'rescue': a & ~b, 'harm': ~a & b, 'CC': a & b, 'WW': ~a & ~b}[name]
                for j in ids[take]:
                    obs = int(data.table.iloc[j].obs_id)
                    if obs not in used:
                        choices.append((seed, obs, int(j)))
            if choices:
                choices.sort()
                seed, obs, j = choices[0] if name in ('rescue', 'harm') else choices[len(choices)//2]
                out.append(dict(seed=seed, obs_id=obs, index=j, case=name, arm=arm, comparator=comp))
                used.add(obs)
    for obs, seed in ((22572, 17), (2080, 43)):
        if obs in used:
            continue
        j = int(np.flatnonzero(data.table.obs_id.eq(obs) & data.table.role.eq('evaluation'))[0])
        yy = int(data.table.iloc[j].class_id)
        arm, comp = 'hybrid_source', 'prior_late'
        a, b = (scores[seed, x][j].argmax() == yy for x in (arm, comp))
        name = 'CC' if a and b else 'WW' if not a and not b else 'rescue' if a else 'harm'
        out.append(dict(seed=seed, obs_id=obs, index=j, case=name, arm=arm, comparator=comp, historical=True))
    # Equal-weight-at-deployment comparisons isolate the actual VH replacement.
    for obs, name in ((24640, 'rescue'), (7944, 'harm')):
        j = int(np.flatnonzero(data.table.obs_id.eq(obs) & data.table.role.eq('evaluation'))[0])
        out.append(dict(seed=43, obs_id=obs, index=j, case=name, arm='hybrid_source',
                        comparator='ordinary_at_hybrid_weight', diagnostic='same alpha=0.1'))
    return out


def main():
    assert (HERE / 'complete.json').exists()
    (HERE / 'analysis').mkdir(exist_ok=True)
    data = run.Data()
    scores = {}
    for seed in run.SEEDS:
        for arm in ('old_full', *run.ARMS):
            with np.load(HERE / 'results' / f'{arm}_{seed}.npz') as z:
                np.testing.assert_array_equal(z['up_index'], data.up[:data.n])
                scores[seed, arm] = z['score'].copy()
    receipt = stats.statistics(data, scores)
    population(data, scores)
    native = run.load('hybrid_native', run.PREVIOUS / 'analyze.py')
    native.HERE = HERE
    geo = run.load('hybrid_geometry', run.BASE.parent / 'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up = pd.read_csv(run.BASE / 'regional_utility_pilot/upstream/table.csv')
    valid = np.load(run.BASE / 'regional_utility_pilot/upstream/summaries.npz')['valid']
    snapshots = {int(s['obs_id']): s for s in json.loads((geo.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs = np.load(run.LOCAL / 'scores_layer1.npz')['refs']
    choices = {r['seed']: r for r in json.loads((HERE / 'source_choices.json').read_text())['choices']}
    records = []
    for case in cases(data, scores):
        path, provenance, true, rival = native.panel(case, data, scores, geo, up, valid, snapshots, refs)
        seed, j = case['seed'], case['index']
        comparisons = {a: dict(prediction=int(scores[seed, a][j].argmax()),
            fixed_margin=float(scores[seed, a][j, true] - scores[seed, a][j, rival])) for a in ('old_full', *run.ARMS)}
        bm = comparisons['base']['fixed_margin']
        delta = {a: comparisons[a]['fixed_margin'] - bm for a in ('prior_vv', 'prior_vh', 'center_vh')}
        records.append(case | dict(true=true, rival=rival, figure=str(path), provenance=provenance,
            comparisons=comparisons, base_margin=bm, branch_delta_margin=delta, source_weights=choices[seed]))
        print('CASE', case, flush=True)
    run.write_json(HERE / 'analysis/cases.json', records)
    receipt.update(cases=len(records), refits=0, interpretation='Frozen-score combination on reused development data; scalar weights chosen only on source selection.')
    run.write_json(HERE / 'analysis_receipt.json', receipt)
    print(pd.read_csv(HERE / 'paired.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
