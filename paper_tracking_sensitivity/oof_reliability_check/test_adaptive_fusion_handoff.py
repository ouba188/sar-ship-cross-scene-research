import numpy as np

from adaptive_fusion_handoff_policy import (accounting, choose_threshold, features,
    fit, probability, select, train_policy)


def test_gain_accounting_with_unequal_class_sizes():
    y = np.array([0, 0, 0, 1])
    base = np.array([0, 1, 1, 1])
    fusion = np.array([1, 0, 0, 0])
    chosen = np.array([0, 1, 0, 1])
    r = accounting(y, base, fusion, chosen)
    assert r['avoided_harms_n'] == 2 and r['lost_rescues_n'] == 1
    np.testing.assert_allclose(r['delta_oa'], 25.)
    np.testing.assert_allclose(r['delta_ba'], 50.)


def test_threshold_endpoints_and_unchanged_predictions():
    base, fusion = np.array([0, 0, 1]), np.array([1, 0, 0])
    p = np.array([.2, .8, .6])
    np.testing.assert_array_equal(select(base, fusion, p, 0)[0], fusion)
    np.testing.assert_array_equal(select(base, fusion, p, 1)[0], base)
    assert not select(base, fusion, p, 0)[1][1]
    t, _ = choose_threshold(base, base, base, p, np.arange(3))
    assert t == 0.


def test_train_uses_fusion_correctness_not_expert_correctness():
    # Fusion changes every decision; confidence provides separable rescue/harm evidence.
    pb = np.tile([[.65, .35], [.51, .49]], (120, 1))
    pe = np.tile([[.1, .9], [.01, .99]], (120, 1))
    x = features(pb, pe)
    base, fusion = pb.argmax(1), ((pb + pe) / 2).argmax(1)
    y = np.tile([0, 1], 120)
    groups = np.repeat(np.arange(12), 20)
    model, result = train_policy(x, base, fusion, y, groups)
    assert np.mean(result['nested_prediction'] == y) == 1.
    assert np.mean(select(base, fusion, probability(model, x), model['threshold'])[0] == y) == 1.
    np.testing.assert_allclose(features(pb, pe), features(pb[:, ::-1], pe[:, ::-1]))


def test_no_unique_correctness_retains_fusion_on_tie():
    x = np.zeros((18, 8))
    y = np.zeros(18, int)
    base = np.ones(18, int)
    fusion = np.full(18, 2)
    groups = np.repeat(np.arange(9), 2)
    model, result = train_policy(x, base, fusion, y, groups)
    assert model['threshold'] == 0.
    np.testing.assert_array_equal(result['nested_prediction'], fusion)


def test_neutral_samples_are_not_false_harm_labels():
    x = np.arange(32).reshape(4, 8) / 10.
    y, b, f = np.array([0, 1, 2, 0]), np.array([0, 0, 0, 0]), np.array([1, 1, 1, 0])
    model = fit(x, b, f, y, np.zeros(4, int))
    assert model['events'] == 2
    assert model['fusion_rescue_events'] == 1
