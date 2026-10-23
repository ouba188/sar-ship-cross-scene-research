"""Frozen source-reference dependence inference; no fitting or selection."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent
DEP = FROZEN.parent
ROOT = DEP.parent
POOL = ROOT / 'query_evidence_pooling'
EXT = ROOT / 'external_confirmation'
sys.path.insert(0, str(POOL))
import run_pooling as pool
from match_readout import unit

# Load the parent by a unique name: pooling already imports support/run.py.
spec = importlib.util.spec_from_file_location('source_dependence', DEP / 'run.py')
dep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dep)
ARMS = ('joint', 'nonlinear', 'shuffle101', 'shuffle211', 'shuffle307')
SEEDS = (17, 29, 43)


def frozen_features(k, a, pi, valid, bank):
    """Parent formula, but all divisors and permutations are loaded, never fit."""
    present = pi > 0
    expected = (a.sum(-1) > 0)[:, :, None] & present.any(-1)[None]
    np.testing.assert_array_equal(expected, valid)
    loga = np.full(a.shape, -np.inf, dtype=np.float64)
    np.log(a, out=loga, where=a > 0)
    u = logsumexp(k.astype(np.float64) + loga[:, :, :, None, None], axis=2)
    u = np.where(valid[..., None] & present[None], u, np.nan)
    n = len(k)
    marginal = np.full((n, 6, 8), np.nan)
    for g in range(6):
        for c in range(8):
            rows = valid[:, g, c]
            refs = np.flatnonzero(present[g, c])
            if rows.any():
                marginal[rows, g, c] = dep.logmean(u[rows, g, c][:, refs])

    permutations = pd.read_csv(DEP / 'reference_permutations.csv')
    pair_valid = np.zeros((n, 7, 8), bool)
    raw = {key: np.full((n, 7, 8), np.nan)
           for key in ('joint', 'shuffle101', 'shuffle211', 'shuffle307')}
    for p, (g, h) in enumerate(dep.PAIRS):
        for c in range(8):
            refs = np.flatnonzero(present[g, c] & present[h, c])
            rows = valid[:, g, c] & valid[:, h, c] & (len(refs) > 0)
            pair_valid[:, p, c] = rows
            if not rows.any():
                continue
            ug, uh = u[rows, g, c][:, refs], u[rows, h, c][:, refs]
            raw['joint'][rows, p, c] = dep.dependence(ug, uh)
            for seed in dep.SHUFFLES:
                mapping = permutations.query('seed == @seed and pair == @p and class_id == @c')
                mapping = mapping.sort_values('first_reference')
                np.testing.assert_array_equal(mapping.first_reference.to_numpy(), refs)
                order = mapping.second_reference.to_numpy(dtype=int)
                np.testing.assert_array_equal(np.sort(order), refs)
                np.testing.assert_array_equal(mapping.first_table_index.to_numpy(), bank[c, refs])
                np.testing.assert_array_equal(mapping.second_table_index.to_numpy(), bank[c, order])
                raw[f'shuffle{seed}'][rows, p, c] = dep.dependence(ug, u[rows, h, c][:, order])

    scales = pd.read_csv(DEP / 'scales.csv')
    divisors = {family: part.sort_values('feature').divisor.to_numpy()
                for family, part in scales.groupby('family')}
    center = np.where(valid, marginal, 0.).sum(-1, keepdims=True) / np.maximum(valid.sum(-1, keepdims=True), 1)
    single = np.where(valid, marginal - center, np.nan).transpose(0, 2, 1)
    sv = valid.transpose(0, 2, 1)
    single_z = dep.standardize(single, sv, divisors['single'])
    pv = pair_valid.transpose(0, 2, 1)
    raw = {key: value.transpose(0, 2, 1) for key, value in raw.items()}
    nonlinear = np.stack([single_z[:, :, g] * single_z[:, :, h] for g, h in dep.PAIRS], -1)
    nonlinear = np.where(pv, nonlinear, np.nan)
    features = {key: dep.standardize(value, pv, divisors['joint_shared']) for key, value in raw.items()}
    features['nonlinear'] = dep.standardize(nonlinear, pv, divisors['nonlinear'])
    for value in features.values():
        assert value.shape == (n, 8, 7) and np.isfinite(value).all()
    return dict(u=u, L=marginal, marginal=single_z, single_valid=sv, pair_valid=pv,
                nonlinear_raw=nonlinear, **{f'D_{key}': value for key, value in raw.items()}, **features)


def predict(base, local, features, weights, seed, gain, scale):
    anchor = base.astype(np.float32).copy()
    anchor += gain * local / scale
    out = dict(full=base.astype(np.float32), anchor=anchor)
    for arm in ARMS:
        w = weights[f'{arm}_{seed}']
        assert w.shape == (7,)
        with torch.inference_mode():
            delta = dep.delta_from(torch.from_numpy(features[arm]), torch.from_numpy(w)).numpy()
        out[arm] = anchor + delta
        assert np.isfinite(out[arm]).all()
    return out


def replay(table, x, mass, bank, gains, scale, weights):
    saved = np.load(DEP / 'features.npz')
    idx = saved['idx']
    np.testing.assert_array_equal(bank, saved['bank'])
    missing = mass[idx, 1].sum(-1) == 0
    eval_rows = table.iloc[idx].role.to_numpy() == 'evaluation'
    selected = np.unique(np.r_[np.flatnonzero(~missing)[:4], np.flatnonzero(missing)[:4],
                               np.flatnonzero(eval_rows & missing)[:2], np.flatnonzero(eval_rows & ~missing)[:2]])
    assert missing[selected].any()
    source_ids = idx[selected]
    k, a, pi, valid = pool.extract(x, mass, bank, source_ids)
    scores, regions, _, _ = pool.aggregate(k, a, pi, valid)
    features = frozen_features(k, a, pi, valid, bank)
    checks = {}

    def check(name, actual, expected, tolerance=5e-6):
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0, equal_nan=True, err_msg=name)
        finite = np.isfinite(actual) & np.isfinite(expected)
        checks[name] = float(np.max(np.abs(actual[finite].astype(float) - expected[finite].astype(float)))) if finite.any() else 0.

    with np.load(POOL / 'details.npz') as old:
        for name, value in [('k', k), ('a', a), ('valid', valid)]:
            check(name, value, old[name][selected])
        check('pi', pi, old['pi'])
    with np.load(POOL / 'local_scores.npz') as old:
        check('local_joint', scores['joint'], old['joint'][selected])
    for name, value in features.items():
        expected = saved[name][selected]
        if name in ARMS:
            expected = expected[..., 6:]
        check(name, value, expected)
    check('marginal_vs_aggregate', features['L'][valid], regions['joint'][valid])
    for seed in SEEDS:
        base = np.load(pool.UP / f'strategies_seed{seed}.npz')['logits'][source_ids, 2]
        output = predict(base, scores['joint'], features, weights, seed, gains[seed], scale)
        with np.load(FROZEN / f'predictions_{seed}.npz') as old:
            for arm, value in output.items():
                check(f'{arm}_{seed}', value, old[arm][source_ids])
    result = dict(source_table_indices=source_ids.tolist(), source_feature_indices=selected.tolist(),
                  obs_ids=table.iloc[source_ids].obs_id.tolist(), n=len(selected),
                  missing_vh=int(missing[selected].sum()), errors=checks, max_error=max(checks.values()))
    (HERE / 'source_replay.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('source replay', result['n'], 'missing VH', result['missing_vh'], 'max error', result['max_error'], flush=True)


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    table, x, mass, bank = pool.load()
    gain_table = pd.read_csv(POOL / 'full_metrics.csv').query("arm == 'joint'").set_index('seed')
    gains = {seed: float(gain_table.loc[seed, 'gain']) for seed in SEEDS}
    scale = json.loads((POOL / 'receipt.json').read_text())['scale']
    weights = np.load(FROZEN / 'weights.npz')
    replay(table, x, mass, bank, gains, scale, weights)

    external = pd.read_csv(EXT / 'table.csv', dtype={'instance_id': str, 'mmsi': str})
    tokens = np.load(EXT / 'tokens.npz')
    assert len(external) == 3075
    np.testing.assert_array_equal(tokens['instance_ids'].astype(str), external.instance_id.to_numpy())
    projection = np.load(ROOT / 'projection_layer1.npz')
    ex = unit((tokens['layer1'] - projection['mean']) @ projection['components'].T).astype(np.float32)
    emass = tokens['mass'].astype(np.float32)
    ex[emass <= 0] = 0
    source_n = len(table)
    ids = source_n + np.arange(len(external))
    k, a, pi, valid = pool.extract(np.concatenate([x, ex]), np.concatenate([mass, emass]), bank, ids)
    scores, regions, _, _ = pool.aggregate(k, a, pi, valid)
    features = frozen_features(k, a, pi, valid, bank)
    np.savez_compressed(HERE / 'localcache.npz', idx=np.arange(len(external)), combined_ids=ids,
                        instance_ids=external.instance_id.to_numpy(dtype=str), bank=bank,
                        k=k, a=a, pi=pi, valid=valid, joint=scores['joint'], joint_region=regions['joint'])
    np.savez_compressed(HERE / 'features.npz', idx=np.arange(len(external)), bank=bank, **features)
    print('external features complete', len(external), flush=True)
    y = external.class_id.to_numpy(dtype=int)
    results, classes, pairs, grouped = [], [], [], []
    for seed in SEEDS:
        strategy = np.load(EXT / f'strategies_seed{seed}.npz')
        np.testing.assert_array_equal(strategy['instance_ids'].astype(str), external.instance_id.to_numpy())
        output = predict(strategy['logits'][:, 2], scores['joint'], features, weights, seed, gains[seed], scale)
        np.savez_compressed(HERE / f'predictions_seed{seed}.npz', idx=np.arange(len(external)),
                            instance_ids=external.instance_id.to_numpy(dtype=str), labels=y, bank=bank, **output)
        for arm, value in output.items():
            results.append(dict(seed=seed, arm=arm, **dep.metrics(y, value)))
            correct = value.argmax(-1) == y
            for reference in ('full', 'anchor'):
                old = output[reference].argmax(-1) == y
                pairs.append(dict(seed=seed, arm=arm, reference=reference,
                                  rescued=int((correct & ~old).sum()), harmed=int((~correct & old).sum()),
                                  net=int(correct.sum() - old.sum())))
            for c in range(8):
                select = y == c
                row = dict(seed=seed, arm=arm, class_id=c, class_name=external.loc[select, 'class_name'].iloc[0],
                           correct=int(correct[select].sum()), **dep.metrics(y[select], value[select]))
                for reference in ('full', 'anchor'):
                    old = output[reference].argmax(-1) == y
                    row.update({f'{reference}_rescued': int((select & correct & ~old).sum()),
                                f'{reference}_harmed': int((select & ~correct & old).sum()),
                                f'{reference}_net': int(correct[select].sum() - old[select].sum())})
                classes.append(row)
            for field in ('original_test_fold', 'port', 'pass_group'):
                for group, indices in external.groupby(field).groups.items():
                    selected = np.asarray(indices)
                    old = output['anchor'][selected].argmax(-1) == y[selected]
                    current = correct[selected]
                    grouped.append(dict(seed=seed, arm=arm, field=field, group=str(group),
                                        rescued=int((current & ~old).sum()), harmed=int((~current & old).sum()),
                                        net=int(current.sum()-old.sum()), **dep.metrics(y[selected], value[selected])))
    for name, rows in [('metrics', results), ('class_metrics', classes), ('paired_results', pairs), ('group_metrics', grouped)]:
        pd.DataFrame(rows).to_csv(HERE / f'{name}.csv', index=False)
    summary = pd.DataFrame(results).groupby('arm')[['OA', 'BA', 'NLL', 'Brier']].mean()
    summary.to_csv(HERE / 'summary.csv')
    receipt = dict(status='complete', n=len(external), ships=external.mmsi.nunique(),
                   passes=external.pass_group.nunique(), source_n=source_n, reference_count=bank.size,
                   new_fits=0, new_selections=0, parameters_per_arm=7, seeds=list(SEEDS), arms=list(ARMS),
                   source_scale=float(scale), source_gains=gains, source_scales_file=str(DEP / 'scales.csv'),
                   source_weights_file=str(FROZEN / 'weights.npz'), permutation_file=str(DEP / 'reference_permutations.csv'),
                   external_missing_vh_mass=int((emass[:, 1].sum(-1) == 0).sum()),
                   historical_not_blind=True, inherited_identity_exclusions=True,
                   no_target_scale_fit=True, no_git=True, raw_chip_review='main agent responsibility',
                   seconds=time.perf_counter()-started)
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(summary.to_string(), flush=True)
    print(pd.DataFrame(pairs).query("arm == 'joint' and reference == 'anchor'").to_string(index=False), flush=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
