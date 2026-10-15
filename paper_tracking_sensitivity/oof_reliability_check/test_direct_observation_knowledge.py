import numpy as np
import pandas as pd

from direct_incidence_conditioning import extend_transforms, predict_conditioned
from run_direct_observation_knowledge import metadata_features, columns


def test_metadata_has_no_label_or_port_input_and_missing_is_neutral():
    frame = pd.DataFrame({'product': ['S1A_a', 'S1C_b', 'S1A_c'],
                          'incidence_deg': [38., 43., np.nan]})
    u, known, platforms = metadata_features(frame, {'S1A_a': {'pass': 'Ascending'},
                                                  'S1C_b': {'pass': 'Descending'}})
    np.testing.assert_allclose(u, [[0, 0, 1, -1], [1, 1, -1, 1], [0, 0, 0, -1]])
    np.testing.assert_array_equal(known, [True, True, False])
    assert platforms.all()
    assert columns('incidence_platform') == [0, 1, 3]
    assert columns('incidence_pass_platform') == [0, 1, 2, 3]


def test_arbitrary_width_preserves_old_coordinates():
    original = np.arange(6).reshape(3, 2)
    expanded = extend_transforms({'x': original}, 4)['x']
    value = np.arange(7)
    np.testing.assert_allclose(value @ expanded, np.r_[value[:3] @ original, value[3:]])


def test_block_scaled_prediction_against_explicit_primal_solve():
    rng = np.random.default_rng(42)
    n = 13
    bank, u = rng.normal(size=(n, 5)), rng.normal(size=(n, 4))
    scales = np.array([1/np.sqrt(2), 1/np.sqrt(2), 1., 1.])
    retained, comp, stable = (rng.normal(size=(n, d)) for d in [3, 4, 2])
    phi = np.c_[rng.normal(size=(n, 6)), np.ones(n)]
    views = {'phis': {(True, True): phi}, 'common': {True: {'retained': retained, 'complement': comp}}, 'stable': stable}
    heads = {f'joint_{h}': {'coef': rng.normal(size=(9, 8)), 'intercept': rng.normal(size=8)} for h in ['base', 'expert']}
    support, labels = np.array([9, 2, 6, 1, 11]), np.array([1, 3, 3, 0, 1])
    actual = predict_conditioned(bank, views, heads, support, labels, u, scales)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = len(labels)/(len(counts)*counts[inverse])
    features = [np.c_[phi[:, :-1], u*scales, phi[:, -1]],
                np.c_[retained/np.sqrt(2), comp/np.sqrt(2), np.ones(n), stable*.5, u*scales*.5]]
    for h, feat, value in zip(['base', 'expert'], features, actual):
        prior = np.c_[bank, u] @ heads['joint_'+h]['coef'] + heads['joint_'+h]['intercept']
        xs = feat[support]
        residual = 2*np.eye(8)[labels]-1-prior[support]
        beta = np.linalg.solve(xs.T @ (weights[:, None]*xs)+8*np.eye(xs.shape[1]), xs.T @ (weights[:, None]*residual))
        np.testing.assert_allclose(value, prior+feat @ beta, atol=1e-11)
