"""Source-only shallow applicability gate for center-response magnitude."""

from pathlib import Path
import numpy as np
import pandas as pd
from incidence_transport import matched_pairs, pair_weights
from condition_evidence_response import fit_model, predict_model, response_target

BASE = Path(__file__).resolve().parent
FEATURE_NAMES = np.asarray(['center_border_log_ratio_a', 'total_logenergy_a',
                            'abs_incidence_delta_deg', 'mean_incidence_deg'])
ARMS = ('anchor', 'capacity', 'condition', 'sham')


def load_features():
    """Return metadata, original fit/held global IDs, and a full N x 6 array."""
    meta = pd.read_csv(BASE/'observations.csv')
    np.testing.assert_array_equal(meta.obs_id.to_numpy(), np.arange(len(meta)))
    with np.load(BASE/'incidence_transport_20261011/source_cohorts.npz') as z:
        fit, held = z['fit_obs_ids'], z['held_obs_ids']
    with np.load(BASE/'condition_evidence_20261011/evidence_features.npz') as z:
        assert str(z['feature_names'][5]) == 'center_border_log_ratio'
        features = np.full((len(meta), 6), np.nan)
        features[z['obs_ids']] = z['features']
    assert np.isfinite(features[np.r_[fit, held]]).all()
    assert not set(meta.iloc[fit].mmsi) & set(meta.iloc[held].mmsi)
    return meta, fit, held, features


def gate_inputs(anchor_a, total_logenergy_a, angle_a, angle_b):
    """Raw A evidence, total log energy, absolute angle gap, mean angle."""
    a, b = np.asarray(angle_a, float), np.asarray(angle_b, float)
    return np.column_stack([anchor_a, total_logenergy_a, np.abs(b-a), (a+b)/2.])


def predict_gate(inputs, model):
    """Return condition blend in [0,1]; missing inputs fall back to anchor."""
    inputs = np.asarray(inputs, float)
    feature = int(model['feature_index'])
    leaf = np.zeros(len(inputs), dtype=int)
    if feature >= 0:
        leaf = (inputs[:, feature] > float(model['threshold'])).astype(int)
    result = np.asarray(model['leaf_mix'], float)[leaf].copy()
    result[~np.isfinite(inputs).all(axis=1)] = 0.
    return np.clip(result, 0., 1.)


def fit_gate(inputs, truth, anchor_prediction, condition_prediction, weights, ships):
    """One gain-regression stump, training quartiles only, 12 ships per leaf."""
    x, w, ships = np.asarray(inputs), np.asarray(weights, float), np.asarray(ships)
    if not np.isfinite(x).all():
        raise ValueError('Gate training requires finite observed conditions')
    w = w/w.sum()
    ea = np.square(truth-anchor_prediction)
    ec = np.square(truth-condition_prediction)
    gain = ea-ec

    def stats(mask):
        weight = w[mask]
        mass = float(weight.sum())
        mean = float(weight@gain[mask]/mass)
        error_a, error_c = float(weight@ea[mask]/mass), float(weight@ec[mask]/mass)
        return dict(ships=len(np.unique(ships[mask])), mass=mass, mean_gain=mean,
                    mse_anchor=error_a, mse_condition=error_c,
                    mix=float(np.clip((error_a-error_c)/(error_a+error_c+1e-12), 0., 1.)),
                    sse=float(weight@np.square(gain[mask]-mean)))

    root = stats(np.ones(len(x), bool))
    best_feature, best_threshold, best_sse = -1, np.nan, root['sse']
    best_leaves = [root, root]
    candidates = [dict(feature_index=-1, quantile=np.nan, threshold=np.nan,
                       left_ships=root['ships'], right_ships=0,
                       eligible=True, gain_sse=root['sse'])]
    for feature in range(4):
        for quantile in (.25, .5, .75):
            threshold = float(np.quantile(x[:, feature], quantile))
            left = x[:, feature] <= threshold
            nl, nr = len(np.unique(ships[left])), len(np.unique(ships[~left]))
            eligible = min(nl, nr) >= 12
            leaves = [stats(left), stats(~left)] if eligible else None
            sse = sum(v['sse'] for v in leaves) if eligible else np.nan
            candidates.append(dict(feature_index=feature, quantile=quantile,
                threshold=threshold, left_ships=nl, right_ships=nr,
                eligible=eligible, gain_sse=sse))
            if eligible and sse < best_sse:
                best_feature, best_threshold, best_sse = feature, threshold, sse
                best_leaves = leaves
    model = dict(feature_index=best_feature, threshold=best_threshold,
                 leaf_mix=np.asarray([v['mix'] for v in best_leaves]),
                 feature_names=FEATURE_NAMES, root_sse=root['sse'], gain_sse=best_sse,
                 leaf_ships=np.asarray([v['ships'] for v in best_leaves]),
                 leaf_mass=np.asarray([v['mass'] for v in best_leaves]),
                 leaf_mse_anchor=np.asarray([v['mse_anchor'] for v in best_leaves]),
                 leaf_mse_condition=np.asarray([v['mse_condition'] for v in best_leaves]),
                 training_ships=root['ships'])
    return model, candidates


def evidence_arrays(features):
    total = np.full(len(features), np.nan)
    valid = np.isfinite(features[:, :3]).all(axis=1)
    total[valid] = np.logaddexp.reduce(features[valid, :3], axis=1)
    return features[:, 5], total


def pair_data(meta, pairs, features):
    directed = np.r_[pairs, pairs[:, ::-1]]
    a, b = directed.T
    anchor, total = evidence_arrays(features)
    angles = meta.incidence_deg.to_numpy(float)
    return dict(directed=directed,
                inputs=gate_inputs(anchor[a], total[a], angles[a], angles[b]),
                truth=response_target(anchor, directed, 'magnitude'),
                weights=np.tile(pair_weights(meta, pairs)/2., 2),
                ships=meta.iloc[a].mmsi.to_numpy())


def fit_responses(meta, fitting, features, arms=ARMS):
    anchor, total = evidence_arrays(features)
    pairs = matched_pairs(meta, fitting)
    return {arm: fit_model(meta, fitting, pairs, anchor, total, anchor, 'magnitude', arm)
            for arm in arms}


def predict_responses(meta, directed, features, models):
    a, b = directed.T
    anchor, total = evidence_arrays(features)
    angle = meta.incidence_deg.to_numpy(float)
    return {arm: predict_model(anchor[a], total[a], angle[a], angle[b], model)
            for arm, model in models.items()}


def crossfit_responses(meta, fitting, features, assignment, arms=('anchor', 'condition')):
    """Response OOF within fitting only; assignment is indexed by global row ID."""
    fitting = np.asarray(fitting)
    pairs = matched_pairs(meta, fitting)
    if not np.array_equal(assignment[pairs[:, 0]], assignment[pairs[:, 1]]):
        raise ValueError('One vessel pair spans folds')
    data = pair_data(meta, pairs, features)
    directed = data['directed']
    fold_ids = np.unique(assignment[fitting])
    if len(fold_ids) < 2 or (fold_ids < 0).any():
        raise ValueError('Cross-fitting needs at least two source vessel folds')
    predictions = {arm: np.full(len(directed), np.nan) for arm in arms}
    audit = []
    for fold in fold_ids:
        train = fitting[assignment[fitting] != fold]
        evaluation = fitting[assignment[fitting] == fold]
        if set(meta.iloc[train].mmsi) & set(meta.iloc[evaluation].mmsi):
            raise ValueError('Response crossfit vessel leakage')
        mask = assignment[directed[:, 0]] == fold
        models = fit_responses(meta, train, features, arms)
        pred = predict_responses(meta, directed[mask], features, models)
        for arm in arms:
            predictions[arm][mask] = pred[arm]
        audit.append(dict(response_excluded_fold=int(fold),
                          train_obs=len(train), eval_obs=len(evaluation),
                          train_pairs=len(models[arms[0]]['pairs']),
                          eval_pairs=int(mask.sum()//2),
                          train_ships=meta.iloc[train].mmsi.nunique(),
                          eval_ships=meta.iloc[evaluation].mmsi.nunique()))
    assert all(np.isfinite(v).all() for v in predictions.values())
    return pairs, data, predictions, audit


def learn_gate(meta, fitting, features, assignment):
    pairs, data, predictions, audit = crossfit_responses(meta, fitting, features, assignment)
    gate, candidates = fit_gate(data['inputs'], data['truth'], predictions['anchor'],
        predictions['condition'], data['weights'], data['ships'])
    return gate, candidates, pairs, data, predictions, audit
