"""Synthetic contract tests; no model checkpoints or experiment data are read."""

import numpy as np

from frequency_candidate_core import (
    candidate_difference,
    candidate_scores,
    class_evidence,
    fit_ranker,
    rank_margin,
    training_pairs,
)


def _local_example():
    # Flatten (channel, y, x): position p contains (p+1) times channel basis p.
    tile = np.zeros((16, 4), dtype=np.float32)
    tile[np.arange(4), np.arange(4)] = [1, 2, 3, 4]
    query = np.tile(tile.reshape(64), 4)[None]
    support = np.stack([
        tile,
        tile[:, [3, 1, 2, 0]],
        tile[:, [1, 2, 3, 0]],
        np.outer(np.eye(16)[0], [1, 2, 3, 4]),
    ]).reshape(1, 256)
    # The last band has directed nearest-position means 1/4 and 1.
    # Swapping endpoints leaves only channels 1 and 2 aligned: (4+9)/30.
    expected = np.array([1, 13 / 30, 0, 1 / 30,
                         1, .5, 0, .25,
                         1, 1, 1, .625])
    return query, support, expected


def test_positive_band_gate_weights_cancel_independently():
    rng = np.random.default_rng(19)
    query, support = rng.normal(size=(3, 256)), rng.normal(size=(6, 256))
    qc, sc = rng.normal(size=(3, 2)), rng.normal(size=(6, 2))
    labels = np.array([0, 1, 0, 2, 1, 2])
    before = [x.copy() for x in (query, support, qc, sc, labels)]
    expected, counts = class_evidence(query, support, qc, sc, labels)
    qgate = np.sqrt(rng.uniform(.001, 4, size=(3, 4, 1)))
    sgate = np.sqrt(rng.uniform(.001, 4, size=(6, 4, 1)))
    actual, actual_counts = class_evidence(
        (query.reshape(3, 4, 64) * qgate).reshape(3, 256),
        (support.reshape(6, 4, 64) * sgate).reshape(6, 256), qc, sc, labels)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(actual_counts, counts)
    for original, snapshot in zip((query, support, qc, sc, labels), before):
        np.testing.assert_array_equal(original, snapshot)


def test_channel_position_reshape_and_known_local_permutations():
    query, support, expected = _local_example()
    evidence, counts = class_evidence(
        query, support, np.zeros((1, 2)), np.zeros((1, 2)), np.array([3]))
    assert evidence.shape == (1, 8, 12)
    np.testing.assert_array_equal(counts, [0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(evidence[0, 3], expected, atol=2e-6)
    np.testing.assert_array_equal(evidence[0, counts == 0], 0)


def test_global_and_condition_attention_is_shared_by_all_three_evidences():
    query, other, pair_evidence = _local_example()
    support = np.concatenate([query, other])
    # Both condition coordinates contribute to squared distance.
    for conditions, distance in [
        (np.array([[.5, 0], [-.5, 1]]), [.25, 1.25]),
        (np.array([[2., 0], [0, 0]]), [4., 0]),
    ]:
        evidence, _ = class_evidence(
            query, support, np.zeros((1, 2)), conditions, np.array([0, 0]))
        logit_gap = (1 - pair_evidence[:4]) / .2 - (distance[0] - distance[1]) / .5
        other_weight = 1 / (1 + np.exp(logit_gap))
        expected = 1 + np.tile(other_weight, 3) * (pair_evidence - 1)
        np.testing.assert_allclose(evidence[0, 0], expected, atol=2e-6)


def test_joint_support_row_permutation_preserves_class_evidence():
    rng = np.random.default_rng(23)
    query, support = rng.normal(size=(4, 256)), rng.normal(size=(9, 256))
    qc, sc = rng.normal(size=(4, 2)), rng.normal(size=(9, 2))
    labels = np.array([0, 2, 0, 5, 2, 5, 0, 1, 2])
    expected, counts = class_evidence(query, support, qc, sc, labels)
    order = np.array([8, 2, 5, 0, 7, 1, 6, 4, 3])
    actual, actual_counts = class_evidence(
        query, support[order], qc, sc[order], labels[order])
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(actual_counts, counts)


def test_missing_or_single_support_disables_fallback_and_scores_route_whole_rows():
    rng = np.random.default_rng(29)
    evidence, counts = class_evidence(
        rng.normal(size=(6, 256)), rng.normal(size=(5, 256)),
        np.zeros((6, 2)), np.zeros((5, 2)), np.array([0, 0, 1, 1, 2]))
    np.testing.assert_array_equal(evidence[:, 3:], 0)
    old = np.tile(np.arange(8, dtype=float) / 10, (6, 1))
    old[:, 0] = 3
    new = np.tile(-np.arange(8, dtype=float) / 10, (6, 1))
    new[np.arange(6), [1, 2, 7, 0, 1, 1]] = 4
    old_before, new_before = old.copy(), new.copy()
    _, eligible = candidate_difference(evidence, counts, old.argmax(1), new.argmax(1))
    np.testing.assert_array_equal(eligible, [True, False, False, False, True, True])
    actual = candidate_scores(old, new, np.array([-1., -1., -1., -1., 0., 1.]), eligible)
    expected = new.copy()
    expected[0] = old[0]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(old, old_before)
    np.testing.assert_array_equal(new, new_before)


def test_candidate_reversal_and_weighted_rms_ridge_have_no_intercept():
    rng = np.random.default_rng(31)
    evidence = rng.normal(size=(3, 8, 12))
    counts = np.full(8, 2)
    old, new = np.array([0, 2, 4]), np.array([1, 3, 5])
    forward, eligible = candidate_difference(evidence, counts, old, new)
    reverse, reverse_eligible = candidate_difference(evidence, counts, new, old)
    np.testing.assert_allclose(forward, evidence[np.arange(3), new] - evidence[np.arange(3), old])
    np.testing.assert_array_equal(reverse, -forward)
    np.testing.assert_array_equal(eligible, np.ones(3, dtype=bool))
    np.testing.assert_array_equal(reverse_eligible, eligible)

    # Orthogonal rows give a closed-form Ridge(10) solution, independent of sklearn.
    amplitudes = np.arange(1, 13, dtype=float)
    features = np.diag(amplitudes)
    weights = amplitudes / amplitudes.mean()
    model = fit_ranker(features, weights)
    assert model is not None
    expected_scale = amplitudes * np.sqrt(weights / 12)
    expected_coef = np.sqrt(12 * weights) / 22
    np.testing.assert_allclose(model['scale'], expected_scale, rtol=2e-6)
    np.testing.assert_allclose(model['coef'], expected_coef, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(rank_margin(model, forward),
                               (forward / expected_scale) @ expected_coef, atol=2e-6)
    np.testing.assert_allclose(rank_margin(model, reverse), -rank_margin(model, forward), atol=1e-7)
    np.testing.assert_array_equal(rank_margin(model, np.zeros((2, 12))), 0)
    reversed_model = fit_ranker(-features, weights)
    np.testing.assert_allclose(reversed_model['coef'], -model['coef'], atol=1e-7)


def test_training_hardnegative_excludes_truth_and_insufficient_support():
    evidence = np.arange(4 * 8 * 12, dtype=float).reshape(4, 8, 12)
    counts = np.array([2, 2, 2, 1, 0, 0, 0, 0])
    old = np.array([[100, 3, 2, 500, 900, 0, 0, 0],
                    [7, 9, 100, 500, 900, 0, 0, 0],
                    [7, 3, 2, 500, 900, 0, 0, 0],
                    [7, 3, 2, 500, 900, 0, 0, 0]], dtype=float)
    new = np.array([[101, 4, 9, 600, 800, 0, 0, 0],
                    [10, 2, 101, 600, 800, 0, 0, 0],
                    [10, 2, 9, 600, 800, 0, 0, 0],
                    [10, 2, 9, 600, 800, 0, 0, 0]], dtype=float)
    labels = np.array([0, 2, 3, 4])
    before = [x.copy() for x in (evidence, counts, old, new, labels)]
    difference, valid, negative = training_pairs(evidence, counts, old, new, labels)
    np.testing.assert_array_equal(valid, [True, True, False, False])
    np.testing.assert_array_equal(negative[:2], [2, 0])
    np.testing.assert_array_equal(difference[:2], [evidence[0, 0] - evidence[0, 2],
                                                 evidence[1, 2] - evidence[1, 0]])
    assert np.all(negative[valid] != labels[valid])
    assert np.all(counts[negative[valid]] >= 2)
    _, isolated_valid, _ = training_pairs(
        evidence, np.array([2, 0, 0, 0, 0, 0, 0, 0]), old, new, np.zeros(4, dtype=int))
    assert not isolated_valid.any()
    for original, snapshot in zip((evidence, counts, old, new, labels), before):
        np.testing.assert_array_equal(original, snapshot)


def test_identical_evidence_and_untrained_ranker_preserve_new_model_output():
    query, _, _ = _local_example()
    evidence, counts = class_evidence(
        query, np.repeat(query, 4, axis=0), np.zeros((1, 2)), np.zeros((4, 2)),
        np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(evidence[0, :2], 1, atol=2e-6)
    difference, eligible = candidate_difference(evidence, counts, np.array([0]), np.array([1]))
    np.testing.assert_array_equal(difference, 0)
    np.testing.assert_array_equal(eligible, [True])
    old = np.array([[4., 1., 0., 0., 0., 0., 0., 0.]])
    new = np.array([[1., 4., 0., 0., 0., 0., 0., 0.]])
    empty = fit_ranker(np.empty((0, 12)), np.empty(0))
    assert empty is None
    np.testing.assert_array_equal(rank_margin(empty, np.ones((1, 12))), 0)
    model = fit_ranker(np.zeros((3, 12)), np.ones(3))
    for fitted in (empty, model):
        margin = rank_margin(fitted, difference)
        np.testing.assert_array_equal(margin, 0)
        np.testing.assert_array_equal(candidate_scores(old, new, margin, eligible), new)
    np.testing.assert_array_equal(candidate_scores(new, new, np.array([-1.]), eligible), new)
