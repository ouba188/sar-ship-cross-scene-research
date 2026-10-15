"""Fixed AIS supervision mask and source-vessel-held-out attribute probes."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scene_training import sample_weights

BASE = Path(__file__).resolve().parent
OUT = BASE/'knowledge_train_only_20261011'
AUDIT = BASE/'knowledge_train_only_design_20261011'
RIDGE = .1


def clean_attributes(table):
    values = table[['raw_exact_length', 'raw_exact_beam']].to_numpy(float, copy=True)
    identity = table[['raw_identity_length', 'raw_identity_beam']].to_numpy(float)
    unique = (table.raw_identity_single_pair.to_numpy(bool)
              & table.raw_exact_single_pair.to_numpy(bool))
    agrees = np.isclose(values, identity, rtol=0, atol=1e-8).all(1)
    positive = np.isfinite(values).all(1) & (values > 0).all(1)
    length, beam = values.T
    ratio = np.divide(length, beam, out=np.full(len(table), np.nan), where=beam > 0)
    flagged = (length <= beam) | (length >= 500) | (ratio > 30)
    valid = unique & agrees & positive & ~flagged
    reason = np.full(len(table), 'usable_reported_pair', dtype=object)
    reason[flagged] = 'pending_dimension_review'
    reason[~positive] = 'missing_or_nonpositive'
    reason[~agrees & unique & positive] = 'product_identity_disagreement'
    reason[~unique] = 'missing_or_nonunique_pair'
    values[~valid] = np.nan
    return values, valid, reason


def fit_ridge(x, y, weights):
    """One fixed relative ridge, with training-only centering/scaling."""
    weights = np.asarray(weights, float)
    weights = weights/weights.sum()
    xm = weights@x
    xs = np.maximum(np.sqrt(weights@np.square(x-xm)), 1e-5)
    ym = weights@y
    a = (x-xm)/xs
    gram = a.T@(weights[:, None]*a)
    penalty = RIDGE*np.trace(gram)/x.shape[1]
    coef = np.linalg.solve(gram+penalty*np.eye(x.shape[1]),
                           a.T@(weights[:, None]*(y-ym)))
    return dict(center=xm, scale=xs, intercept=ym, coef=coef,
                penalty=np.array(penalty))


def predict_ridge(model, x):
    return ((x-model['center'])/model['scale'])@model['coef']+model['intercept']


def class_means(x, labels, weights):
    return np.stack([np.average(x[labels == c], axis=0, weights=weights[labels == c])
                     for c in range(8)])


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    from relative_content_condition import load_data
    OUT.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    del norm, fixed, h
    table = pd.read_csv(AUDIT/'source_dimension_join.csv').set_index('obs_id')
    archive, coverage, audit_rows = {}, [], []
    for role, key in [('fit', 'factor_fit'), ('held', 'test_source')]:
        ids = split[key]
        joined = table.loc[ids].reset_index()
        mm = meta.iloc[ids].reset_index(drop=True)
        np.testing.assert_array_equal(joined.mmsi.to_numpy(), mm.mmsi.to_numpy())
        np.testing.assert_array_equal(joined['split'].to_numpy(), np.full(len(ids), role))
        values, valid, reason = clean_attributes(joined)
        archive.update({role+'_obs_ids': ids, role+'_values': values, role+'_valid': valid})
        joined['attribute_valid'] = valid
        joined['attribute_reason'] = reason
        audit_rows.append(joined)
        for c in sorted(mm.class_id.unique()):
            mask = mm.class_id.to_numpy() == c
            coverage.append(dict(split=role, class_id=int(c), class_name=mm.loc[mask, 'class_name'].iloc[0],
                observations=int(mask.sum()), valid_observations=int((mask & valid).sum()),
                ships=int(mm.loc[mask, 'mmsi'].nunique()),
                valid_ships=int(mm.loc[mask & valid, 'mmsi'].nunique())))
    np.savez_compressed(OUT/'attributes.npz', **archive)
    pd.concat(audit_rows).to_csv(OUT/'attribute_mask.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT/'attribute_coverage.csv', index=False)
    with np.load(BASE/'crossfit_evidence_20261011/source_folds.npz') as f:
        np.testing.assert_array_equal(f['fit_obs_ids'], archive['fit_obs_ids'])
        assignment = f['assignment']
    good = archive['fit_valid']
    ids = archive['fit_obs_ids'][good]
    assignment = assignment[good]
    mm = meta.iloc[ids].reset_index(drop=True)
    x = raw[ids].astype(np.float64)
    y = np.log(archive['fit_values'][good])
    labels = mm.class_id.to_numpy()
    predictions = {name: np.zeros_like(y) for name in ('global', 'class_mean', 'sar', 'within_class')}
    scales = np.zeros_like(y)
    fold_info = []
    for fold in range(3):
        train, test = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
        assert not set(mm.iloc[train].mmsi) & set(mm.iloc[test].mmsi)
        w = sample_weights(mm.iloc[train]); w /= w.sum()
        mean = w@y[train]
        scale = np.maximum(np.sqrt(w@np.square(y[train]-mean)), 1e-5)
        cm = class_means(y[train], labels[train], w)
        xm = class_means(x[train], labels[train], w)
        direct = fit_ridge(x[train], y[train], w)
        conditional = fit_ridge(x[train]-xm[labels[train]], y[train]-cm[labels[train]], w)
        predictions['global'][test] = mean
        predictions['class_mean'][test] = cm[labels[test]]
        predictions['sar'][test] = predict_ridge(direct, x[test])
        predictions['within_class'][test] = cm[labels[test]]+predict_ridge(conditional, x[test]-xm[labels[test]])
        scales[test] = scale
        np.savez_compressed(OUT/f'probe_fold{fold}.npz',
            train_obs_ids=ids[train], test_obs_ids=ids[test], class_y=cm, class_x=xm,
            yscale=scale, **{'sar_'+k: v for k, v in direct.items()},
            **{'within_class_'+k: v for k, v in conditional.items()})
        fold_info.append(dict(fold=fold, train_ships=int(mm.iloc[train].mmsi.nunique()),
                              test_ships=int(mm.iloc[test].mmsi.nunique())))
    rows = []
    for fold in [-1, 0, 1, 2]:
        select = np.ones(len(mm), bool) if fold == -1 else assignment == fold
        for cls in [-1]+list(range(8)):
            selected = select & (np.ones(len(mm), bool) if cls == -1 else labels == cls)
            if not selected.any():
                continue
            w = sample_weights(mm.loc[selected]); w /= w.sum()
            for dim, attribute in enumerate(('length', 'beam')):
                errors = {name: float(w@np.square((pred[selected, dim]-y[selected, dim])/scales[selected, dim]))
                          for name, pred in predictions.items()}
                for name, pred in predictions.items():
                    rows.append(dict(fold=fold, class_id=cls, attribute=attribute, model=name,
                        observations=int(selected.sum()), ships=int(mm.loc[selected, 'mmsi'].nunique()),
                        standardized_MSE=errors[name],
                        log_MAE=float(w@np.abs(pred[selected, dim]-y[selected, dim])),
                        skill_vs_global=100*(1-errors[name]/errors['global']),
                        skill_vs_class=100*(1-errors[name]/errors['class_mean'])))
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT/'probe_results.csv', index=False)
    np.savez_compressed(OUT/'probe_oof_predictions.npz', obs_ids=ids, assignment=assignment,
                        truth=y, scales=scales, **predictions)
    # Reconstruct exported predictions: catches row order and model export mismatches.
    max_error = 0.
    for fold in range(3):
        test = np.flatnonzero(assignment == fold)
        with np.load(OUT/f'probe_fold{fold}.npz') as f:
            direct = {k: f['sar_'+k] for k in ('center', 'scale', 'intercept', 'coef')}
            cond = {k: f['within_class_'+k] for k in ('center', 'scale', 'intercept', 'coef')}
            replay = predict_ridge(direct, raw[f['test_obs_ids']].astype(float))
            replay_cond = f['class_y'][labels[test]]+predict_ridge(cond, x[test]-f['class_x'][labels[test]])
            max_error = max(max_error, float(np.max(np.abs(replay-predictions['sar'][test]))),
                            float(np.max(np.abs(replay_cond-predictions['within_class'][test]))))
    assert max_error < 1e-12
    pooled = scores[(scores.fold == -1) & (scores.class_id == -1)]
    gate = bool((pooled.loc[pooled.model == 'within_class', 'skill_vs_class'] > 0).any())
    write_json(OUT/'PROBE_PROTOCOL.json', dict(
        scope='Source-only fixed original vessel3fold; no held or target prediction/selection.',
        attributes='AIS reported L/W, NOT independently certified physical dimensions.',
        mask='Single joint positive pair both identity and product; exact agreement; exclude L<=W,L>=500,L/W>30; missing rows retained in later classification.',
        ridge=RIDGE, regularizer='ridge * training standardized weighted Gram trace/dimension; no sweep',
        weights='Equal classes then ships then observations, separately computed on each evaluated cohort.',
        probes='SAR direct is deployable; class_mean and within_class use true class ONLY for diagnostic decomposition, never deployment.',
        conditional='Train class-centered X to class-centered logdims; proves conditional predictability only, not causal identification.',
        gate='Proceed to fixed four-arm study if at least one source-OOF conditional log dimension beats class-mean error; report both and every fold; no held metrics consulted.',
        replay_max_error=max_error, folds=fold_info,
        continue_four_arms=gate))
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)
    print(pooled.to_string(index=False), flush=True)
    print('PROCEED_FOUR_ARMS', gate, 'REPLAY_ERROR', max_error, flush=True)


if __name__ == '__main__':
    main()
