import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import evidence_aware_handoff as method
from receiver_consistent_handoff import confidence_parts


def logits(predictions):
    scores = np.zeros((len(predictions), 8))
    scores[np.arange(len(predictions)), predictions] = 1.
    return scores


def episode(base, component, labels, query=None):
    return dict(base_scores=base, base_prior=component.copy(),
                stable_source=component.copy(), labels=np.asarray(labels),
                query=np.ones(len(base), dtype=bool) if query is None else query)


def test_zero_identity_and_first_tie():
    base = logits([0, 1, 2])
    ep = episode(base, np.zeros_like(base), [0, 1, 2])
    saved = base.copy()
    result = method.select_recalibration([ep])
    assert method.ALPHAS == (0., -.25, .25, -.5, .5)
    assert result['stable_alpha'] == result['scale_alpha'] == 0.
    assert all(point['ba'] == 1. for point in result['stable_curve'])
    np.testing.assert_array_equal(base, saved)


def test_nonzero_ties_prefer_smallest_magnitude_and_order():
    base = logits([0, 0])
    component = np.zeros_like(base)
    component[:, 1] = [8., -8.]
    result = method.select_recalibration([episode(base, component, [1, 1])])
    assert result['stable_alpha'] == -.25
    assert result['stable_curve'][0]['ba'] == 0.
    assert all(p['ba'] == .5 for p in result['stable_curve'][1:])


def test_query_only_labels_and_features_select_alpha():
    base = logits([0, 0, 1])
    component = np.zeros_like(base)
    component[0, 1] = 8.
    component[1:, 0] = 20.
    ep = episode(base, component, [1, 0, 0], np.array([True, False, False]))
    first = method.select_recalibration([ep])
    assert first['stable_alpha'] == .25
    ep['labels'][1:] = -999
    ep['stable_source'][1:] = np.nan
    ep['base_prior'][1:] = np.nan
    ep['base_scores'][1:] = np.nan
    assert method.select_recalibration([ep]) == first


def test_balanced_accuracy_and_equal_episode_weighting():
    first = episode(logits([0, 0, 0, 0]), np.zeros((4, 8)), [0, 0, 0, 1])
    second = episode(logits([1]), np.zeros((1, 8)), [1])
    result = method.select_recalibration([first, second])
    assert result['stable_curve'][0]['ba'] == .75


def test_stable_and_global_scale_controls_choose_independently():
    base = logits([0])
    component = np.zeros_like(base)
    component[0, 1] = 8.
    ep = episode(base, component, [1])
    ep['base_prior'] *= -1
    result = method.select_recalibration([ep])
    assert result['stable_alpha'] == .25
    assert result['scale_alpha'] == -.25


def test_recalibration_requires_source_query_rows():
    with pytest.raises(ValueError, match='episode'):
        method.select_recalibration([])
    with pytest.raises(ValueError, match='query'):
        method.select_recalibration([episode(logits([0]), np.zeros((1, 8)), [0], [False])])


@pytest.mark.parametrize('field', ['base_scores', 'base_prior', 'stable_source', 'labels'])
@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_recalibration_rejects_nonfinite_query_values(field, value):
    ep = episode(logits([0]), np.zeros((1, 8)), [0.])
    ep[field].flat[0] = value
    with pytest.raises(ValueError, match='nonfinite'):
        method.select_recalibration([ep])


def test_feature_order_and_pairwise_decomposition():
    rng = np.random.default_rng(42)
    base, expert, cb, ce = rng.normal(size=(4, 12, 8))
    features = method.gate_features(base, expert, cb, ce)
    assert set(features) == {'base_features', 'evidence_features'}
    basic, evidence = features['base_features'], features['evidence_features']
    bm, bh = confidence_parts(base)
    em, eh = confidence_parts(expert)
    np.testing.assert_allclose(basic[:, :4], np.column_stack((em, bm, eh, bh)))
    assert basic.shape == (12, 5)
    assert evidence.shape == (12, 6)
    np.testing.assert_array_equal(evidence[:, :5], basic)
    for i in range(len(base)):
        b, e = base[i].argmax(), expert[i].argmax()
        delta = expert[i] - base[i]
        stable = ce[i] - cb[i]
        residual = (expert[i] - ce[i]) - (base[i] - cb[i])
        assert basic[i, 4] == pytest.approx(delta[e] - delta[b])
        assert evidence[i, 5] == pytest.approx(stable[e] - stable[b])
        assert basic[i, 4] - evidence[i, 5] == pytest.approx(residual[e] - residual[b])


def test_features_recompute_classes_after_recalibration():
    base, expert = logits([0]), logits([1])
    cb, ce = np.zeros((1, 8)), np.zeros((1, 8))
    cb[0, 2], ce[0, 1] = 8., 3.
    original = method.gate_features(base, expert, cb, ce)['evidence_features']
    changed = method.gate_features(base + .25 * cb, expert, cb, ce)['evidence_features']
    assert original[0, 5] == 3.
    assert changed[0, 5] == 11.
    assert changed[0, 4] == 3.


def test_features_invariant_to_per_row_common_logit_shifts():
    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(4, 20, 8))
    shifted = inputs + rng.normal(size=(4, 20, 1))
    before, after = method.gate_features(*inputs), method.gate_features(*shifted)
    for key in before:
        np.testing.assert_allclose(before[key], after[key], atol=1e-14)


def gate_data():
    features = np.array([[-2., 7.], [-1., 7.], [1., 7.], [2., 7.],
                         [999., 7.], [-999., 7.]])
    base = logits([0, 0, 0, 0, 0, 0])
    expert = logits([1, 1, 1, 1, 1, 1])
    labels = np.array([0, 0, 1, 1, 1, 2])
    eligible = np.array([True, True, True, True, False, True])
    weights = np.array([1., 3., 2., 4., 1e6, 1e6])
    return features, base, expert, labels, eligible, weights


def test_lr_filtering_weighted_statistics_and_normalized_regularization():
    features, base, expert, labels, eligible, weights = gate_data()
    model = method.fit_gate(features, base, expert, labels, eligible, weights)
    assert model['counts'] == dict(eligible=5, training=4, positive=2, negative=2)
    w = np.array([.4, 1.2, .8, 1.6])
    center = np.average(features[:4], axis=0, weights=w)
    scale = np.sqrt(np.average((features[:4]-center)**2, axis=0, weights=w))
    scale[1] = 1e-6
    np.testing.assert_allclose(model['center'], center)
    np.testing.assert_allclose(model['scale'], scale)
    reference = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, tol=1e-7)
    reference.fit((features[:4]-center)/scale, [0, 0, 1, 1], sample_weight=w)
    np.testing.assert_allclose(method.predict_gate(model, features[:4]),
                               reference.decision_function((features[:4]-center)/scale))
    assert model['n_iter'][0] > 0
    assert model['coef'][0] > 0
    assert method.predict_gate(model, features[:4])[0] < 0
    assert method.predict_gate(model, features[:4])[-1] > 0


def test_weights_rescaling_and_excluded_rows_do_not_change_fit():
    data = list(gate_data())
    original = method.fit_gate(*data)
    data[-1] *= 19.
    data[0][4:] = np.nan
    data[3][4] = 0
    changed = method.fit_gate(*data)
    for key in ('center', 'scale', 'coef', 'intercept'):
        np.testing.assert_allclose(original[key], changed[key], atol=1e-12)


def test_no_informative_rows_fallback_is_zero():
    features = np.ones((3, 6))
    model = method.fit_gate(features, logits([0]*3), logits([0]*3),
                            [0]*3, [True]*3, [1.]*3)
    assert model['counts']['training'] == 0
    np.testing.assert_array_equal(method.predict_gate(model, features), np.zeros(3))


def test_one_utility_sign_has_constant_gate():
    features = np.arange(12.).reshape(3, 4)
    model = method.fit_gate(features, logits([0]*3), logits([1]*3),
                            [1]*3, [True]*3, [1.]*3)
    np.testing.assert_allclose(method.predict_gate(model, features), np.log(7.))
    assert model['n_iter'][0] == 0


def test_routing_excludes_support_unobserved_agreement_and_nonpositive():
    n = 43
    base, expert = logits([0]*n), logits([1]*n)
    expert[2] = logits([2])[0]
    expert[3] = base[3]
    evaluation = np.ones(n, dtype=bool)
    evaluation[:2] = False
    score = np.full(n, -1.)
    score[:8] = [999., 999., 999., 999., 0., -2., 5., 5.]
    score[8] = 5.
    rows = np.arange(n)
    rows[6:9] = [60, 20, 40]
    result = method.route_scores(base, expert, score, [0, 1], evaluation, rows)
    assert result['cap'] == 2  # floor(.05 * 41), not round or support count.
    np.testing.assert_array_equal(np.flatnonzero(result['selected']), [7, 8])
    assert not result['selected'][~evaluation].any()
    np.testing.assert_array_equal(result['predictions'][~result['selected']],
                                  result['base_predictions'][~result['selected']])
    np.testing.assert_array_equal(result['predictions'][result['selected']], [1, 1])


def test_routing_does_not_fill_cap_with_ineligible_rows():
    base, expert = logits([0]*20), logits([1]*20)
    result = method.route_scores(base, expert, np.zeros(20), [1],
                                 np.ones(20, bool), np.arange(20))
    assert result['cap'] == 1
    assert not result['selected'].any()
    result = method.route_scores(base, expert, np.ones(20), [1],
                                 np.ones(20, bool), np.arange(20), budget=0.)
    assert result['cap'] == 0
    assert not result['selected'].any()
