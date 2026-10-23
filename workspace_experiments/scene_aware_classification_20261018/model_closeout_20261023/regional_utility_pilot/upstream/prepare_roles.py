"""One fixed source-only split; no model predictions enter membership."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'eight_class_evidence_20261023'
ROLES = ['strategy_fit', 'utility_train', 'selection', 'evaluation']
FRACTIONS = np.array([.55, .20, .125, .125])


def materialize(frame, assignments, names):
    out = frame.copy()
    out['role'] = out.pass_group.map(assignments)
    seen, keep = set(), np.ones(len(out), bool)
    for role in reversed(names):
        ix = np.flatnonzero(out.role.eq(role))
        keep[ix] = ~out.iloc[ix].mmsi.isin(seen).to_numpy()
        seen.update(out.iloc[ix[keep[ix]]].mmsi)
    return out[keep].copy(), out[~keep].copy()


def counts(frame, names):
    return frame.groupby(['role', 'class_id']).size().unstack(fill_value=0).reindex(index=names, columns=range(8), fill_value=0)


def split(frame, names, fractions, seed):
    groups = np.array(sorted(frame.pass_group.unique()))
    groups = np.random.default_rng(seed).permutation(groups)
    cut = np.r_[0, np.floor(np.cumsum(fractions)[:-1] * len(groups)).astype(int), len(groups)]
    assignments = {g: r for j, r in enumerate(names) for g in groups[cut[j]:cut[j+1]]}
    chosen, excluded = materialize(frame, assignments, names)
    changes = []
    # Repair only genuinely absent classes, using class/pass coverage, never model scores.
    while (counts(chosen, names).to_numpy() == 0).any():
        before = int((counts(chosen, names).to_numpy() == 0).sum())
        best = None
        for group in sorted(assignments):
            old = assignments[group]
            for role in names:
                if role == old:
                    continue
                candidate = dict(assignments, **{group: role})
                f, e = materialize(frame, candidate, names)
                missing = int((counts(f, names).to_numpy() == 0).sum())
                sizes = f.role.value_counts().reindex(names, fill_value=0).to_numpy()
                score = (missing, float(abs(sizes / max(len(f), 1) - fractions).sum()), group, role)
                if missing < before and (best is None or score < best[0]):
                    best = (score, candidate, f, e, old)
        if best is None:
            break
        score, assignments, chosen, excluded, old = best
        changes.append(dict(pass_group=score[2], old_role=old, new_role=score[3], missing_after=score[0]))
    return chosen, excluded, assignments, changes


def main():
    with np.load(ROOT / 'baseline/fold0_inputs.npz') as z:
        source = np.sort(np.r_[z['train'], z['validation']])
    all_test = np.unique(np.concatenate([np.load(ROOT / 'baseline' / f'fold{f}_inputs.npz')['test'] for f in range(3)]))
    assert len(source) == 4702 and not np.intersect1d(source, all_test).size
    allowed_lines = set((source + 1).tolist())
    frame = pd.read_csv(ROOT / 'cohort.csv', skiprows=lambda i: i != 0 and i not in allowed_lines,
                        dtype={'mmsi': str, 'instance_id': str})
    frame.insert(0, 'row_index', source)
    frame['pass_group'] = frame['product'].map(lambda s: '_'.join([s.split('_')[0], s.split('_')[6], s.split('_')[7]]))
    chosen, excluded, assignments, changes = split(frame, ROLES, FRACTIONS, 20261023)
    chosen = chosen.sort_values('row_index').reset_index(drop=True)
    fit = chosen[chosen.role.eq('strategy_fit')]
    inner, inner_excluded, inner_assignment, inner_changes = split(fit, ['fit_train', 'fit_validation'], np.array([.8, .2]), 20261023)
    inner_map = inner.set_index('row_index').role
    chosen['upstream_role'] = chosen.row_index.map(inner_map).fillna('external')
    chosen.loc[chosen.role.eq('strategy_fit') & chosen.upstream_role.eq('external'), 'upstream_role'] = 'fit_excluded_mmsi'
    chosen.to_csv(HERE / 'table.csv', index=False)
    excluded.to_csv(HERE / 'role_exclusions.csv', index=False)
    inner_excluded.to_csv(HERE / 'fit_internal_exclusions.csv', index=False)
    pd.DataFrame([dict(pass_group=g, role=r) for g, r in sorted(assignments.items())]).to_csv(HERE / 'pass_assignment.csv', index=False)
    counts(chosen, ROLES).to_csv(HERE / 'role_class_counts.csv')
    counts(inner, ['fit_train', 'fit_validation']).to_csv(HERE / 'fit_class_counts.csv')
    np.savez_compressed(HERE / 'indices.npz', indices=chosen.row_index.to_numpy(),
        **{r: np.flatnonzero(chosen.role.eq(r)) for r in ROLES},
        **{r: np.flatnonzero(chosen.upstream_role.eq(r)) for r in ['fit_train', 'fit_validation']})
    for j, role in enumerate(ROLES):
        x = chosen[chosen.role.eq(role)]
        for other in ROLES[j+1:]:
            y = chosen[chosen.role.eq(other)]
            assert not set(x.mmsi) & set(y.mmsi)
            assert not set(x.pass_group) & set(y.pass_group)
    status = not (counts(chosen, ROLES).to_numpy() == 0).any() and not (counts(inner, ['fit_train', 'fit_validation']).to_numpy() == 0).any()
    summary = dict(status='ready' if status else 'coverage_needs_decision', source_pool=4702, retained=len(chosen), excluded=len(excluded),
        seed=20261023, fraction=FRACTIONS.tolist(), changes=changes, internal_changes=inner_changes,
        counts=chosen.groupby('role').agg(n=('row_index','size'), mmsi=('mmsi','nunique'), passes=('pass_group','nunique')).to_dict('index'),
        inner_counts=inner.groupby('role').size().to_dict(), historical_test_overlap=0,
        scope='One source split; all upstream fit/selection confined to strategy_fit; not whole-stack OOF within strategy_fit.')
    (HERE / 'split_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(counts(chosen, ROLES).to_string(), flush=True)
    print(counts(inner, ['fit_train', 'fit_validation']).to_string(), flush=True)
    assert status, 'No training until class coverage is resolved using metadata only.'


if __name__ == '__main__':
    main()
