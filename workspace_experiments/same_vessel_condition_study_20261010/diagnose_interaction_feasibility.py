"""Bounded diagnostics of saved response fits; no tuning or new training."""

import json
import numpy as np
import pandas as pd
from probe_structure_motion_interaction import BASE, OUT, design, context_features
from incidence_transport import pair_weights


def main():
    root = OUT/'response_probe'
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(OUT/'data.npz') as f:
        state = f['values'][:, :6]
        pairs = {r: f[r+'_pairs'] for r in ('fit', 'held')}
        held_ids = f['held_obs_ids']
        fit_ids = f['fit_obs_ids']
    evidence = np.full((len(meta), 6), np.nan)
    with np.load(BASE/'condition_evidence_20261011/evidence_features.npz') as f:
        evidence[f['obs_ids']] = f['features']
    output = {'new_fits': 0, 'seeds': {}}
    contributions = []
    for seed in (31415, 27182):
        scores = np.full((len(meta), 8), np.nan)
        with np.load(BASE/'geometry_train_teacher_20261011'/f'teachers_{seed}.npz') as f:
            np.testing.assert_array_equal(f['fit_obs_ids'], fit_ids)
            scores[fit_ids] = f['original_oof']
        with np.load(BASE/'crossfit_evidence_20261011'/f'full_{seed}_outputs.npz') as f:
            scores[held_ids] = f['prior'][held_ids]
        context = context_features(evidence, scores)
        row = {}
        for arm in ('anchor', 'additive', 'pairwise', 'triple', 'capacity'):
            with np.load(root/f'{arm}_{seed}.npz') as f:
                model = {k: f[k] for k in f.files}
            xx = {}
            for role in ('fit', 'held'):
                p = pairs[role]
                x = design(context, (state-model['center'])/model['scale'],
                           np.r_[p, p[:, ::-1]], arm)
                xx[role] = (x-model['xcenter'])/model['xscale']
            lo, hi = xx['fit'].min(0), xx['fit'].max(0)
            row[arm] = dict(held_coordinate_outside_fit_range=float(
                ((xx['held'] < lo-1e-8) | (xx['held'] > hi+1e-8)).mean()),
                fit_max_abs_z=float(np.abs(xx['fit']).max()),
                held_max_abs_z=float(np.abs(xx['held']).max()))
        a = pd.read_csv(root/f'held_anchor_{seed}.csv')
        t = pd.read_csv(root/f'held_triple_{seed}.csv')
        np.testing.assert_array_equal(a[['a', 'b']], t[['a', 'b']])
        w = np.tile(pair_weights(meta, pairs['held'])/2, 2)
        delta = t.class_margin_change_squared_error-a.class_margin_change_squared_error
        d = pd.DataFrame({'mmsi': a.mmsi, 'class_id': a.class_id,
                          'weighted_excess': w*delta, 'weight': w})
        g = d.groupby(['mmsi', 'class_id']).sum().reset_index()
        g['seed'] = seed
        contributions.append(g)
        row['margin_error'] = dict(ships_worse=int((g.weighted_excess > 0).sum()),
            ships_better=int((g.weighted_excess < 0).sum()),
            total_excess=float(g.weighted_excess.sum()),
            top3_positive_excess=float(g.weighted_excess.nlargest(3).sum()))
        with np.load(root/f'teacher_anchor_{seed}.npz') as f:
            ids = f['obs_ids']
            row['teacher_cohort'] = dict(n=len(ids), ships=int(meta.iloc[ids].mmsi.nunique()),
                classes=meta.iloc[ids].class_name.value_counts().to_dict())
        output['seeds'][str(seed)] = row
    pd.concat(contributions).to_csv(root/'held_ship_error_contributions.csv', index=False)
    (root/'ROOT_DIAGNOSTICS.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
