"""Bounded reference-dependence probe with exactly preserved marginal controls."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POOL = ROOT / 'query_evidence_pooling'
UP = ROOT.parent / 'regional_utility_pilot/upstream'
sys.path.insert(0, str(UP))
from pipeline import exact_j, metrics, joint_loss

PAIRS = ((0, 1), (0, 2), (3, 4), (3, 5), (0, 3), (1, 4), (2, 5))
GROUPS = ('VV_body', 'VV_side', 'VV_end', 'VH_body', 'VH_side', 'VH_end')
SHUFFLES = (101, 211, 307)
SEEDS = (17, 29, 43)
ARMS = ('marginal', 'nonlinear', 'joint', 'shuffle101', 'shuffle211', 'shuffle307')
CHECKPOINTS = (0, 20, 50, 100, 200)


def logmean(x):
    return logsumexp(x, axis=-1) - np.log(x.shape[-1])


def dependence(g, h):
    return logmean(g + h) - logmean(g) - logmean(h)


def rms_valid(x, valid, train):
    values, mask = x[train], valid[train]
    count = mask.sum((0, 1))
    squared = np.where(mask, values, 0.) ** 2
    rms = np.sqrt(squared.sum((0, 1)) / np.maximum(count, 1))
    return np.where(rms > 1e-8, rms, 1.), rms, count


def standardize(x, valid, scale):
    # Missing contributions are explicitly masked; they are not zero observations.
    return np.where(valid, x / scale, 0.).astype(np.float32)


def delta_from(features, coefficient):
    delta = (features * coefficient).sum(-1)
    return delta - delta.mean(-1, keepdim=True)


def prepare(table, idx, bank, train):
    data = np.load(POOL / 'details.npz')
    k, a, pi, valid = (data[key] for key in ('k', 'a', 'pi', 'valid'))
    n = len(idx)
    assert k.shape == (n, 6, 8, 8, 8)
    assert a.shape == (n, 6, 8) and pi.shape == (6, 8, 8)
    assert valid.shape == (n, 6, 8) and valid.dtype == bool
    present = pi > 0
    expected = (a.sum(-1) > 0)[:, :, None] & present.any(-1)[None]
    np.testing.assert_array_equal(expected, valid)
    loga = np.full(a.shape, -np.inf, dtype=np.float64)
    np.log(a, out=loga, where=a > 0)
    u = logsumexp(k.astype(np.float64) + loga[:, :, :, None, None], axis=2)
    u = np.where(valid[..., None] & present[None], u, np.nan)
    marginal = np.full((n, 6, 8), np.nan)
    for g in range(6):
        for c in range(8):
            rows = valid[:, g, c]
            refs = np.flatnonzero(present[g, c])
            if rows.any():
                marginal[rows, g, c] = logmean(u[rows, g, c][:, refs])
    replay = float(np.max(np.abs(marginal[valid] - data['joint_region'][valid])))
    assert replay < 5e-6, replay

    pair_valid = np.zeros((n, 7, 8), bool)
    raw = {key: np.full((n, 7, 8), np.nan) for key in ('joint',) + tuple(f'shuffle{s}' for s in SHUFFLES)}
    common_counts, permutations, verification = [], [], []
    maps = {}
    for seed in SHUFFLES:
        rng = np.random.default_rng(seed)
        for p, (g, h) in enumerate(PAIRS):
            for c in range(8):
                refs = np.flatnonzero(present[g, c] & present[h, c])
                maps[seed, p, c] = rng.permutation(refs)
                for original, shuffled in zip(refs, maps[seed, p, c]):
                    permutations.append(dict(seed=seed, pair=p, class_id=c,
                                             first_reference=int(original), second_reference=int(shuffled),
                                             first_table_index=int(bank[c, original]),
                                             second_table_index=int(bank[c, shuffled])))
    errors = {seed: np.zeros((n, 2)) for seed in SHUFFLES}
    multisets = {seed: np.zeros(n) for seed in SHUFFLES}
    for p, (g, h) in enumerate(PAIRS):
        for c in range(8):
            refs = np.flatnonzero(present[g, c] & present[h, c])
            common_counts.append(dict(pair=p, first=GROUPS[g], second=GROUPS[h], class_id=c,
                                      common_references=len(refs)))
            rows = valid[:, g, c] & valid[:, h, c] & (len(refs) > 0)
            pair_valid[:, p, c] = rows
            if not rows.any():
                continue
            ug, uh = u[rows, g, c][:, refs], u[rows, h, c][:, refs]
            raw['joint'][rows, p, c] = dependence(ug, uh)
            for seed in SHUFFLES:
                order = maps[seed, p, c]
                np.testing.assert_array_equal(np.sort(order), refs)
                shuffled = u[rows, h, c][:, order]
                # Check each query/class/pair, not just the population mean.
                e1 = np.abs(logmean(ug) - logmean(ug.copy()))
                e2 = np.abs(logmean(uh) - logmean(shuffled))
                em = np.max(np.abs(np.sort(uh, axis=1) - np.sort(shuffled, axis=1)), axis=1)
                assert max(e1.max(), e2.max(), em.max()) < 1e-12
                errors[seed][rows, 0] = np.maximum(errors[seed][rows, 0], e1)
                errors[seed][rows, 1] = np.maximum(errors[seed][rows, 1], e2)
                multisets[seed][rows] = np.maximum(multisets[seed][rows], em)
                raw[f'shuffle{seed}'][rows, p, c] = dependence(ug, shuffled)
    for seed in SHUFFLES:
        for j in range(n):
            verification.append(dict(seed=seed, local_index=j, table_index=int(idx[j]),
                                     obs_id=int(table.iloc[idx[j]].obs_id),
                                     valid_pair_classes=int(pair_valid[j].sum()),
                                     first_marginal_error=errors[seed][j, 0],
                                     second_marginal_error=errors[seed][j, 1],
                                     second_multiset_error=multisets[seed][j]))

    center = np.where(valid, marginal, 0.).sum(-1, keepdims=True) / np.maximum(valid.sum(-1, keepdims=True), 1)
    single = np.where(valid, marginal - center, np.nan).transpose(0, 2, 1)
    single_valid = valid.transpose(0, 2, 1)
    single_scale, single_rms, single_count = rms_valid(single, single_valid, train)
    single_z = standardize(single, single_valid, single_scale)
    pv = pair_valid.transpose(0, 2, 1)
    raw = {name: value.transpose(0, 2, 1) for name, value in raw.items()}
    nonlinear = np.stack([single_z[:, :, g] * single_z[:, :, h] for g, h in PAIRS], -1)
    nonlinear = np.where(pv, nonlinear, np.nan)
    joint_scale, joint_rms, joint_count = rms_valid(raw['joint'], pv, train)
    nonlinear_scale, nonlinear_rms, nonlinear_count = rms_valid(nonlinear, pv, train)
    extra = {name: standardize(value, pv, joint_scale) for name, value in raw.items()}
    extra['nonlinear'] = standardize(nonlinear, pv, nonlinear_scale)
    features = {'marginal': single_z}
    for name, value in extra.items():
        features[name] = np.concatenate([single_z, value], axis=-1)
        np.testing.assert_array_equal(features[name][..., :6], single_z)
        assert np.isfinite(features[name]).all()
    assert np.isfinite(single_z).all()
    for value in raw.values():
        np.testing.assert_array_equal(np.isfinite(value), pv)

    synthetic_g = np.log(np.array([[2., 8.]]))
    positive = float(dependence(synthetic_g, synthetic_g)[0])
    negative = float(dependence(synthetic_g, synthetic_g[:, ::-1])[0])
    constant = float(dependence(synthetic_g, np.zeros_like(synthetic_g))[0])
    singleton = float(dependence(np.array([[2.]]), np.array([[3.]]))[0])
    assert positive > 0 and negative < 0 and abs(constant) < 1e-12 and abs(singleton) < 1e-12

    pd.DataFrame(common_counts).to_csv(HERE / 'common_reference_counts.csv', index=False)
    pd.DataFrame(permutations).to_csv(HERE / 'reference_permutations.csv', index=False)
    pd.DataFrame(verification).to_csv(HERE / 'marginal_verification.csv', index=False)
    availability = []
    roles = table.iloc[idx].role.to_numpy()
    for role in ('utility_train', 'selection', 'evaluation'):
        rows = roles == role
        for p, (g, h) in enumerate(PAIRS):
            for c in range(8):
                availability.append(dict(role=role, pair=p, first=GROUPS[g], second=GROUPS[h],
                                         class_id=c, queries=int(rows.sum()), valid=int(pair_valid[rows, p, c].sum()),
                                         valid_rate=float(pair_valid[rows, p, c].mean())))
    pd.DataFrame(availability).to_csv(HERE / 'availability.csv', index=False)
    scales = []
    for family, scale, rms, count in [('single', single_scale, single_rms, single_count),
                                      ('joint_shared', joint_scale, joint_rms, joint_count),
                                      ('nonlinear', nonlinear_scale, nonlinear_rms, nonlinear_count)]:
        for j in range(len(scale)):
            scales.append(dict(family=family, feature=j, divisor=scale[j], utility_rms=rms[j],
                               utility_valid_count=int(count[j])))
    pd.DataFrame(scales).to_csv(HERE / 'scales.csv', index=False)
    np.savez_compressed(HERE / 'features.npz', idx=idx, bank=bank, u=u, L=marginal,
                        single_valid=single_valid, pair_valid=pv, nonlinear_raw=nonlinear,
                        **{f'D_{name}': value for name, value in raw.items()}, **features)
    return features, dict(single_region_replay_error=replay,
                         max_first_marginal_error=max(float(v[:, 0].max()) for v in errors.values()),
                         max_second_marginal_error=max(float(v[:, 1].max()) for v in errors.values()),
                         max_multiset_error=max(float(v.max()) for v in multisets.values()),
                         synthetic_positive=positive, synthetic_negative=negative,
                         synthetic_constant=constant, synthetic_singleton=singleton)


def fit(features, base, labels, train, selection):
    x = torch.tensor(features)
    xt = x[train]
    bt = torch.tensor(base[train])
    yt = torch.tensor(labels[train], dtype=torch.long)
    coefficient = torch.nn.Parameter(torch.zeros(features.shape[-1]))
    optimizer = torch.optim.Adam([coefficient], lr=.01)
    initial_delta = delta_from(x, coefficient).detach().numpy()
    assert np.count_nonzero(initial_delta) == 0
    best = exact_j(labels[selection], base[selection].argmax(-1))
    selected, chosen = 0, coefficient.detach().clone()
    history, states = [], []
    for step in range(201):
        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            ce = joint_loss(bt + delta_from(xt, coefficient), yt)
            objective = ce + .01 * coefficient.square().sum()
            assert torch.isfinite(objective)
            objective.backward()
            assert torch.isfinite(coefficient.grad).all()
            optimizer.step()
        if step not in CHECKPOINTS:
            continue
        with torch.no_grad():
            sv = base[selection] + delta_from(x[selection], coefficient).numpy()
            train_ce = joint_loss(bt + delta_from(xt, coefficient), yt)
            l2 = .01 * coefficient.square().sum()
        j = exact_j(labels[selection], sv.argmax(-1))
        history.append(dict(step=step, train_CE=float(train_ce), L2=float(l2),
                            objective=float(train_ce + l2), exact_j=str(j),
                            **metrics(labels[selection], sv)))
        states.append(coefficient.detach().numpy().copy())
        if j > best:
            best, selected, chosen = j, step, coefficient.detach().clone()
    with torch.no_grad():
        delta = delta_from(x, chosen).numpy()
    score = base + delta
    assert np.isfinite(score).all()
    if selected == 0:
        np.testing.assert_array_equal(score, base)
    # Independent NumPy matrix contraction checks saved coefficients and class axis.
    direct = np.einsum('ncf,f->nc', features.astype(np.float64), chosen.numpy().astype(np.float64))
    direct -= direct.mean(-1, keepdims=True)
    replay = float(np.max(np.abs(direct - delta)))
    assert replay < 5e-6, replay
    return score, delta, chosen.numpy(), np.stack(states), history, selected, replay


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    table = pd.read_csv(UP / 'table.csv', dtype={'instance_id': str, 'mmsi': str})
    local = np.load(POOL / 'local_scores.npz')
    idx, bank = local['idx'], local['bank']
    expected = np.flatnonzero(table.role.isin(['utility_train', 'selection', 'evaluation']))
    np.testing.assert_array_equal(idx, expected)
    assert bank.shape == (8, 8) and len(np.unique(bank)) == 64
    np.testing.assert_array_equal(table.iloc[bank.ravel()].class_id.to_numpy().reshape(8, 8),
                                  np.broadcast_to(np.arange(8)[:, None], (8, 8)))
    role = table.iloc[idx].role.to_numpy()
    ids = {name: np.flatnonzero(role == name) for name in ('utility_train', 'selection', 'evaluation')}
    assert [len(v) for v in ids.values()] == [644, 612, 485]
    train, selection, evaluation = ids.values()
    labels = table.iloc[idx].class_id.to_numpy()
    features, checks = prepare(table, idx, bank, train)
    feature_seconds = time.perf_counter() - started
    results, curves, receipts, paired, classes, coefficients = [], [], [], [], [], []
    saved_weights = {}
    replay_max = 0.
    for seed in SEEDS:
        base_full = np.load(UP / f'strategies_seed{seed}.npz')['logits'][:, 2].astype(np.float32)
        base = base_full[idx]
        predictions = {'full': base_full.copy()}
        local_predictions = {'full': base}
        for arm in ARMS:
            tick = time.perf_counter()
            score, delta, weight, states, history, step, replay = fit(features[arm], base, labels, train, selection)
            replay_max = max(replay_max, replay)
            local_predictions[arm] = score
            predictions[arm] = base_full.copy()
            predictions[arm][idx] = score
            saved_weights[f'{arm}_{seed}'] = weight
            saved_weights[f'{arm}_{seed}_checkpoints'] = states
            receipts.append(dict(seed=seed, arm=arm, selected_step=step,
                                 parameters=len(weight), seconds=time.perf_counter() - tick,
                                 prediction_replay_error=replay))
            curves.extend(dict(seed=seed, arm=arm, **row) for row in history)
            for f, w in enumerate(weight):
                coefficients.append(dict(seed=seed, arm=arm, feature=f, coefficient=float(w), selected_step=step))
            print(seed, arm, 'selected', step, 'evaluation', metrics(labels[evaluation], score[evaluation]), flush=True)
        for arm, score in local_predictions.items():
            for name, rows in ids.items():
                results.append(dict(seed=seed, arm=arm, role=name, **metrics(labels[rows], score[rows])))
            prediction = score[evaluation].argmax(-1)
            truth = labels[evaluation]
            for c in range(8):
                mask = truth == c
                classes.append(dict(seed=seed, arm=arm, class_id=c, n=int(mask.sum()),
                                    correct=int((prediction[mask] == c).sum())))
            correct = prediction == truth
            for reference in ('full', 'marginal', 'joint'):
                old = local_predictions[reference][evaluation].argmax(-1) == truth
                paired.append(dict(seed=seed, arm=arm, reference=reference,
                                   rescued=int((correct & ~old).sum()), harmed=int((~correct & old).sum()),
                                   net=int(correct.sum() - old.sum())))
        np.savez_compressed(HERE / f'predictions_{seed}.npz', idx=idx, bank=bank, **predictions)
    np.savez_compressed(HERE / 'weights.npz', checkpoints=np.array(CHECKPOINTS), **saved_weights)
    for name, rows in [('metrics', results), ('selection_curve', curves), ('training_receipts', receipts),
                       ('paired_results', paired), ('class_results', classes), ('coefficients', coefficients)]:
        pd.DataFrame(rows).to_csv(HERE / f'{name}.csv', index=False)
    summary = pd.DataFrame(results).query("role == 'evaluation'").groupby('arm')[['OA', 'BA', 'J', 'NLL', 'balanced_NLL', 'Brier']].agg(['mean', 'std'])
    summary.to_csv(HERE / 'summary.csv')
    assert len(receipts) == 18 and len(curves) == 90
    receipt = dict(status='complete', utc=datetime.now(timezone.utc).isoformat(), fits=len(receipts),
                   seconds=time.perf_counter() - started, feature_seconds=feature_seconds,
                   seeds=list(SEEDS), shuffle_seeds=list(SHUFFLES), pairs=[list(p) for p in PAIRS],
                   roles={key: len(value) for key, value in ids.items()},
                   loss='0.5 CE + 0.5 balanced CE + 0.01 sum(theta^2)',
                   optimizer='Adam lr=0.01, 200 full-batch steps, zero initialization',
                   selection='source selection full exact_j; strict improvement; earliest tie; step0 fallback',
                   scale='utility only; joint and shuffle share actual-joint RMS',
                   scope='frozen source reference-dependence availability probe, not strong innovation',
                   no_backbone_training=True, no_hyperparameter_search=True, no_git=True, no_figures=True,
                   checks=checks, max_prediction_replay_error=replay_max,
                   raw_chip_review='deferred to main agent as requested')
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(summary.to_string(), flush=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
