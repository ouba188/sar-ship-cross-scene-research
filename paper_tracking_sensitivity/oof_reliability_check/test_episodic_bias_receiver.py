from pathlib import Path

import numpy as np
import pytest

import episodic_bias_receiver as method
import residual_expert


def sample():
    rng = np.random.default_rng(17)
    prior = rng.normal(size=(113, 8))
    support = rng.choice(len(prior), 64, replace=False)
    labels = np.repeat([0, 3, 7], [49, 11, 4])
    return prior, support, labels


def test_matches_original_ones_design_correction():
    prior, support, labels = sample()
    expected = residual_expert.fit_correction(
        np.ones((64, 1)), labels, prior[support])[0]
    adapted, bias = method.adapt_bias(prior, support, labels)
    np.testing.assert_allclose(bias, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(adapted, prior + expected, atol=1e-12, rtol=1e-12)


def test_class_balancing_matches_equal_class_residual_means():
    prior, support, labels = sample()
    class_residuals = []
    for c in [0, 3, 7]:
        target = np.full(8, -1.)
        target[c] = 1.
        class_residuals.append(target - prior[support[labels == c]].mean(axis=0))
    expected = np.mean(class_residuals, axis=0) * (64 / 72)
    _, bias = method.adapt_bias(prior, support, labels)
    np.testing.assert_allclose(bias, expected, atol=1e-12)
    unweighted = (2 * np.eye(8)[labels] - 1 - prior[support]).sum(axis=0) / 72
    assert not np.allclose(bias, unweighted)


def test_support_order_and_episode_label_indexing():
    prior, support, support_labels = sample()
    labels = np.full(len(prior), 6)
    labels[support] = support_labels
    original_prior, original_labels = prior.copy(), labels.copy()
    expected = method.adapt_bias(prior, support, support_labels)
    adapted, bias, ev, observed = method.make_episode(prior, prior, labels, support)
    np.testing.assert_array_equal(adapted, expected[0])
    np.testing.assert_array_equal(bias, expected[1])
    np.testing.assert_array_equal(ev, ~np.isin(np.arange(len(prior)), support))
    assert ev.sum() == len(prior) - 64
    np.testing.assert_array_equal(observed, [0, 3, 7])
    permutation = np.random.default_rng(2).permutation(64)
    reordered = method.adapt_bias(prior, support[permutation], support_labels[permutation])
    np.testing.assert_allclose(reordered[0], adapted, atol=1e-12)
    np.testing.assert_allclose(reordered[1], bias, atol=1e-12)
    # Reordering labels alone must change the class-balanced prior residual.
    wrong = method.adapt_bias(prior, support, support_labels[permutation])
    assert not np.allclose(wrong[1], bias)
    np.testing.assert_array_equal(prior, original_prior)
    np.testing.assert_array_equal(labels, original_labels)


def test_heldout_source_labels_do_not_affect_episode():
    prior, support, support_labels = sample()
    labels = np.zeros(len(prior), dtype=int)
    labels[support] = support_labels
    first = method.make_episode(prior, prior, labels, support)
    labels[first[2]] = 7
    second = method.make_episode(prior, prior, labels, support)
    for actual, expected in zip(second, first):
        np.testing.assert_array_equal(actual, expected)


def test_episode_only_indexes_support_labels():
    prior, support, support_labels = sample()

    class SupportOnlyLabels:
        def __getitem__(self, rows):
            np.testing.assert_array_equal(rows, support)
            return support_labels

        def __array__(self, *args, **kwargs):
            raise AssertionError('full labels must not be materialized')

    episode = method.make_episode(prior, prior, SupportOnlyLabels(), support)
    np.testing.assert_array_equal(episode[0], method.adapt_bias(prior, support, support_labels)[0])


@pytest.mark.parametrize('support', [
    np.arange(63), np.arange(65), np.zeros(64, dtype=int),
    np.arange(64).reshape(8, 8), np.arange(64, dtype=float),
    np.arange(-1, 63), np.arange(50, 114),
])
def test_requires_exactly_64_distinct_valid_local_ids(support):
    prior, _, labels = sample()
    with pytest.raises(ValueError, match='support_local'):
        method.adapt_bias(prior, support, labels)


@pytest.mark.parametrize('labels', [
    np.zeros(63, dtype=int), np.zeros(113, dtype=int),
    np.full(64, -1), np.full(64, 8), np.zeros(64, dtype=float),
])
def test_requires_ordered_support_labels_only(labels):
    prior, support, _ = sample()
    with pytest.raises(ValueError, match='support_labels'):
        method.adapt_bias(prior, support, labels)


def test_reproduces_existing_target_bias_without_query_labels():
    root = Path(__file__).resolve().parent.parent
    port = 'Antwerp-Bruges'
    path = root / 'oof_residual_expert_results_20261007_v2' / port / 'policies.npz'
    with np.load(path, allow_pickle=False) as parent:
        query_rows = parent['query_rows']
        support_rows = parent['support_rows'][0]
        score_index = list(parent['score_names']).index('anchor_bias64')
        expected_scores = parent['scores'][0, score_index]
        expected_bias = parent['anchor_bias64'][0, 0]
    with np.load(root / 'pred_cache' / (port + '.npz'), allow_pickle=False) as cache:
        np.testing.assert_array_equal(query_rows, cache['qpos'])
        prior = cache['expert_logits'][query_rows]
        support_labels = cache['y_ids'][support_rows]
    support_local = np.searchsorted(query_rows, support_rows)
    np.testing.assert_array_equal(query_rows[support_local], support_rows)
    adapted, bias = method.adapt_bias(prior, support_local, support_labels)
    np.testing.assert_allclose(bias, expected_bias, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(adapted, expected_scores, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(adapted.argmax(axis=1), expected_scores.argmax(axis=1))
