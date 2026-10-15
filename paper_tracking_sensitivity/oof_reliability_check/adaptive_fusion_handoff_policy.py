"""Source-only selection of base versus equal fusion, not base versus expert."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from compact_fusion_policy import balanced_weights, utility_objective

FEATURES = ('base_margin', 'fusion_margin', 'expert_margin', 'base_entropy',
            'fusion_entropy', 'expert_entropy', 'base_preference', 'expert_preference')
THRESHOLDS = np.linspace(0., 1., 21)
PENALTY = .01


def features(pb, pe):
    fused = (pb + pe) * .5
    values = []
    for p in (pb, fused, pe):
        top = np.sort(p, axis=1)[:, -2:]
        values.append(top[:, 1] - top[:, 0])
    for p in (pb, fused, pe):
        values.append(-np.sum(p * np.log(np.maximum(p, 1e-15)), axis=1) / np.log(p.shape[1]))
    rows = np.arange(len(pb))
    base, fusion = pb.argmax(1), fused.argmax(1)
    values.extend((pb[rows, base] - pb[rows, fusion], pe[rows, fusion] - pe[rows, base]))
    return np.column_stack(values)


def fit(x, base, fusion, y, groups):
    weight = balanced_weights(y, groups)
    center = weight @ x
    scale = np.sqrt(weight @ ((x - center)**2))
    scale[scale < 1e-6] = 1.
    informative = (base == y) != (fusion == y)
    initial = np.zeros(x.shape[1] + 1)
    if not informative.any():
        return dict(theta=initial.tolist(), center=center.tolist(), scale=scale.tolist(),
                    events=0, convergence='No relative correctness events; all-fusion tie rule')
    w = weight[informative]
    w /= w.sum()
    target = (fusion[informative] == y[informative]).astype(float)
    p = np.clip(w @ target, 1e-3, 1. - 1e-3)
    initial[0] = np.log(p / (1. - p))
    design = np.column_stack((np.ones(len(x)), (x - center) / scale))[informative]
    args = (design, target, w, PENALTY)
    result = minimize(utility_objective, initial, args=args, jac=True, method='L-BFGS-B',
                      bounds=[(-8., 8.)] * len(initial),
                      options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success:
        raise RuntimeError(str(result.message))
    return dict(theta=result.x.tolist(), center=center.tolist(), scale=scale.tolist(),
                events=int(informative.sum()), fusion_rescue_events=int(target.sum()),
                constant_fusion_preference=float(w @ target), iterations=int(result.nit),
                initial_objective=utility_objective(initial, *args)[0],
                final_objective=float(result.fun), convergence=str(result.message))


def probability(model, x):
    theta = np.asarray(model['theta'])
    standardized = (x - np.asarray(model['center'])) / np.asarray(model['scale'])
    return expit(theta[0] + standardized @ theta[1:])


def select(base, fusion, p, threshold):
    accepted = (base != fusion) & (p >= threshold)
    return np.where(accepted, fusion, base), accepted


def choose_threshold(base, fusion, y, p, groups):
    weight = balanced_weights(y, groups)
    scores = np.array([weight @ (select(base, fusion, p, t)[0] == y) for t in THRESHOLDS])
    # Ties retain full fusion, then prefer the smaller rejection region.
    index = int(np.flatnonzero(scores >= scores.max() - 1e-12)[0])
    return float(THRESHOLDS[index]), scores.tolist()


def grouped_oof(x, base, fusion, y, groups):
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError('At least three source ports required')
    folds = np.searchsorted(unique, groups) % 3
    out = np.empty(len(y))
    for k in range(3):
        train, val = folds != k, folds == k
        model = fit(x[train], base[train], fusion[train], y[train], groups[train])
        out[val] = probability(model, x[val])
    return out, folds


def train_policy(x, base, fusion, y, groups):
    oof, folds = grouped_oof(x, base, fusion, y, groups)
    threshold, scores = choose_threshold(base, fusion, y, oof, groups)
    nested = np.empty(len(y), dtype=int)
    constant = np.empty(len(y), dtype=int)
    nested_threshold = np.empty(len(y))
    for k in range(3):
        train, val = folds != k, folds == k
        inner, _ = grouped_oof(x[train], base[train], fusion[train], y[train], groups[train])
        t, _ = choose_threshold(base[train], fusion[train], y[train], inner, groups[train])
        nested[val] = select(base[val], fusion[val], oof[val], t)[0]
        nested_threshold[val] = t
        w = balanced_weights(y[train], groups[train])
        use_fusion = w @ (fusion[train] == y[train]) >= w @ (base[train] == y[train])
        constant[val] = fusion[val] if use_fusion else base[val]
    model = fit(x, base, fusion, y, groups)
    w = balanced_weights(y, groups)
    model.update(threshold=threshold, threshold_grid=THRESHOLDS.tolist(),
                 source_oof_threshold_ba=(100 * np.array(scores)).tolist(),
                 constant_use_fusion=bool(w @ (fusion == y) >= w @ (base == y)),
                 features=FEATURES, penalty=PENALTY)
    return model, dict(oof_probability=oof, nested_prediction=nested,
                      nested_constant=constant, nested_threshold=nested_threshold, folds=folds)


def accounting(y, base, fusion, selected):
    bc, fc, sc = base == y, fusion == y, selected == y
    rescue, harm = ~bc & fc, bc & ~fc
    events = dict(fusion_rescues=rescue, fusion_harms=harm,
                  retained_rescues=rescue & sc, lost_rescues=rescue & ~sc,
                  avoided_harms=harm & sc, retained_harms=harm & ~sc)
    w = balanced_weights(y, np.zeros(len(y), int))
    result = {name + '_n': int(mask.sum()) for name, mask in events.items()}
    result.update({name + '_ba_pp': float(100 * w @ mask) for name, mask in events.items()})
    result['delta_oa'] = float(100 * np.mean(sc.astype(float) - fc))
    result['delta_ba'] = float(100 * w @ (sc.astype(float) - fc))
    np.testing.assert_allclose(result['delta_oa'],
        100 * (result['avoided_harms_n'] - result['lost_rescues_n']) / len(y), atol=1e-10)
    np.testing.assert_allclose(result['delta_ba'],
        result['avoided_harms_ba_pp'] - result['lost_rescues_ba_pp'], atol=1e-10)
    return result
