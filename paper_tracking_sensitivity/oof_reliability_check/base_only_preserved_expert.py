"""Base-only stable/protected activation with the historical expert unchanged."""

import numpy as np

from evidence_aware_handoff import fit_gate, predict_gate, route_scores
from receiver_consistent_handoff import confidence_parts
from residual_expert import fit_correction
from run_protected_drift_activation import make_transforms


PORTS = ('Antwerp-Bruges', 'Jebel Ali', 'Port Klang', 'Santos', 'Singapore', 'Tanger Med')
FACTORS = dict(matched_baseline=(False, False), stable_only=(False, True), joint=(True, True))
ARMS = ('original', *FACTORS)


def base_transforms(geometry):
    transforms = make_transforms(geometry)
    return {arm + '_base': transforms[arm + '_base'] for arm in FACTORS}


def base_views(bank, geometry):
    rank = geometry['stable_rank']
    stable = bank[:, 192:192 + rank] / np.sqrt(rank)
    drift = bank[:, 192 + rank:]
    answer = {}
    for arm, (suppress, activate) in FACTORS.items():
        core = (bank[:, :128] - .5 * (drift @ geometry['drift_to_xs'])
                if suppress else np.asarray(bank[:, :128], dtype=np.float32))
        base = (core - geometry['base_center']) / geometry['base_scale'] / np.sqrt(128)
        body = np.column_stack((base, stable)) if activate else base
        answer[arm] = np.column_stack((body, np.ones(len(bank))))
    return answer


def predict_bases(bank, views, heads, support, support_labels):
    result = {}
    for arm, phi in views.items():
        head = heads[arm + '_base']
        prior = bank @ head['coef'] + head['intercept']
        coefficient = fit_correction(phi[support], support_labels, prior[support])
        result[arm] = prior + phi @ coefficient
    return result


def gate_features(base, expert):
    bm, bh = confidence_parts(base)
    em, eh = confidence_parts(expert)
    return np.column_stack((em, bm, eh, bh))


def fit_episode_gate(episodes):
    joined = {key: np.concatenate([e[key] for e in episodes])
              for key in ('base', 'expert', 'labels', 'features', 'eligible', 'weights')}
    return fit_gate(joined['features'], joined['base'], joined['expert'],
                    joined['labels'], joined['eligible'], joined['weights'])


def route_episode(gate, episode):
    score = predict_gate(gate, episode['features'])
    return route_scores(episode['base'], episode['expert'], score,
                        episode['support_labels'], np.ones(len(score), dtype=bool), episode['rows'])


def verify_target_alignment(old, cached, rows):
    for key in ('query_rows', 'source_rows', 'support_rows', 'eval_mask'):
        np.testing.assert_array_equal(old[key], cached[key])
    np.testing.assert_array_equal(rows, old['query_rows'])
    if len(old['eval_mask']) != 5 or old['support_rows'].shape != (5, 64):
        raise ValueError('Expected five original64 support draws')
    for support_rows, evaluation, cap in zip(old['support_rows'], old['eval_mask'], old['cap']):
        support = np.searchsorted(rows, support_rows)
        np.testing.assert_array_equal(rows[support], support_rows)
        if len(np.unique(support)) != 64 or evaluation[support].any():
            raise ValueError('Support/query exclusion changed')
        if cap != int(np.floor(.05 * evaluation.sum())):
            raise ValueError('Original5% budget changed')
