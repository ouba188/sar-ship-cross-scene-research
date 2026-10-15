"""Independent contract checks for decision thresholds, actions and evidence."""

import numpy as np

from frequency_decision_core import (
    crossing_threshold,
    decision_features,
    endpoint_scores,
)


def _feature_inputs():
    rng = np.random.default_rng(20261009)
    old = rng.normal(size=(13, 8))
    new = rng.normal(size=(13, 8))
    new[:3] = old[:3] * 1.5
    similarity = np.stack([
        rng.permutation(np.linspace(0.1, 0.95, 64)) for _ in range(13)
    ])
    labels = np.tile(np.arange(8), 8)
    loo_old = rng.uniform(-0.3, 0.3, size=(64, 8))
    loo_new = rng.uniform(-0.3, 0.3, size=(64, 8))
    loo_old[np.arange(64), labels] += 2.0
    loo_new[np.arange(64), (labels + 1) % 8] += 2.0
    return old, new, similarity, labels, loo_old, loo_new


def _features(inputs):
    features, names, width = decision_features(*inputs)
    assert isinstance(features, np.ndarray)
    assert isinstance(names, list)
    assert isinstance(width, (int, np.integer))
    assert features.ndim == 2 and features.shape[0] == len(inputs[0])
    assert len(names) == features.shape[1]
    assert 0 < width < features.shape[1]
    assert np.isfinite(features).all()
    return features, names, width


def test_crossing_threshold_examples_include_non_old_top_limiter():
    old = np.full((4, 8), -10.0)
    new = old.copy()
    old[0, :3], new[0, :3] = [4, 3, 0], [0, 3, 4]
    old[1, :3], new[1, :3] = [4, 0, -1], [0, 1, -1]
    old[2, :3], new[2, :3] = [3, 1, 0], [4, 2, 1]
    old[3, :3], new[3, :3] = [0, 3, 1], [0, 3, 1]

    actual = crossing_threshold(old, new)
    assert isinstance(actual, np.ndarray) and actual.shape == (4,)
    # Row 0 crosses class 0 at 1/2, but class 1 still wins until 3/4.
    np.testing.assert_allclose(actual, [0.75, 0.8, 0.0, 0.0], atol=1e-7, rtol=0)
    for row in (0, 1):
        target = new[row].argmax()
        before = (1 - actual[row] + 1e-5) * old[row] + (actual[row] - 1e-5) * new[row]
        after = (1 - actual[row] - 1e-5) * old[row] + (actual[row] + 1e-5) * new[row]
        assert before.argmax() != target
        assert after.argmax() == target


def test_crossing_threshold_matches_independent_segment_sampling():
    rng = np.random.default_rng(781)
    old = rng.normal(size=(40, 8))
    new = rng.normal(size=(40, 8))
    new[:5] = old[:5] * 2.0
    actual = crossing_threshold(old, new)
    assert actual.shape == (40,)
    assert np.isfinite(actual).all()
    assert np.all((actual >= 0) & (actual <= 1))
    grid = np.linspace(0.0, 1.0, 10001)
    for row, threshold in enumerate(actual):
        if old[row].argmax() == new[row].argmax():
            assert threshold == 0.0
            continue
        scores = (1 - grid[:, None]) * old[row] + grid[:, None] * new[row]
        winner = new[row].argmax()
        first = np.flatnonzero(scores.argmax(axis=1) == winner)[0]
        assert first > 0
        assert grid[first - 1] - 1e-7 <= threshold <= grid[first] + 1e-7


def test_endpoint_selection_preserves_full_rows_and_never_interpolates():
    old = np.full((6, 8), -10.0)
    new = old.copy()
    old[:3, :3], new[:3, :3] = [4, 3, 0], [0, 3, 4]
    old[3:, :3], new[3:, :3] = [4, 1, 0], [6, 2, -1]
    benefit = np.array([-0.25, 0.0, 0.25, -0.25, 0.0, 0.25])
    expected = new.copy()
    expected[0] = old[0]

    selected = endpoint_scores(old, new, benefit)
    np.testing.assert_array_equal(selected, expected)
    # A continuous half-strength action would invent class 1 on these rows.
    assert np.all(((old[:3] + new[:3]) / 2).argmax(axis=1) == 1)
    np.testing.assert_array_equal(selected.argmax(axis=1), [0, 2, 2, 0, 0, 0])
    np.testing.assert_array_equal(selected[3:], new[3:])


def test_features_ignore_independent_common_logit_shifts_per_row():
    inputs = _feature_inputs()
    expected, names, width = _features(inputs)
    rng = np.random.default_rng(79)
    shifts = {
        index: rng.uniform(-4.0, 4.0, size=(len(inputs[index]), 1))
        for index in (0, 1, 4, 5)
    }
    for indices in ((0,), (1,), (4,), (5,), (0, 1, 4, 5)):
        shifted = list(inputs)
        for index in indices:
            shifted[index] = inputs[index] + shifts[index]
        actual, shifted_names, shifted_width = _features(shifted)
        assert shifted_names == names and shifted_width == width
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_features_ignore_joint_support_permutation_without_similarity_ties():
    old, new, similarity, labels, loo_old, loo_new = inputs = _feature_inputs()
    expected, names, width = _features(inputs)
    order = np.random.default_rng(309).permutation(64)
    actual, reordered_names, reordered_width = _features((
        old, new, similarity[:, order], labels[order], loo_old[order], loo_new[order]
    ))
    assert reordered_names == names and reordered_width == width
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_support_labels_change_evidence_only():
    inputs = _feature_inputs()
    expected, names, width = _features(inputs)
    changed = list(inputs)
    # Flip every LOO outcome from old-correct/new-wrong to new-correct/old-wrong.
    changed[3] = inputs[5].argmax(axis=1)
    actual, changed_names, changed_width = _features(changed)
    assert changed_names == names and changed_width == width
    np.testing.assert_array_equal(actual[:, :width], expected[:, :width])
    assert not np.allclose(actual[:, width:], expected[:, width:], rtol=2e-5, atol=2e-6)
