import numpy as np

from task_stable_eightclass import predicted_pairs, predicted_delta, intervene
from run_generic_transfer import metrics


def test_predicted_pair_uses_score_columns_not_truth_pair():
    scores = np.array([[2., 1., 3., 0., 0., 0., 0., 0.]])
    np.testing.assert_array_equal(predicted_pairs(scores), [[0, 2]])
    # A true 0-vs-1 comparison would be correct, but the eight-class decision is not.
    assert metrics(np.array([0]), scores.argmax(1), 8)['acc'] == 0.


def test_no_geometry_for_predicted_pair_leaves_sample_unchanged():
    z = np.array([[3., 4.]])
    scores = np.array([[1., 2., 3., 0., 0., 0., 0., 0.]])
    geometry = {(0, 1): dict(direction=np.array([1., 0.]), midpoint=np.zeros(2))}
    np.testing.assert_array_equal(predicted_delta(z, scores, geometry), np.zeros_like(z))


def test_query_only_intervention_preserves_supports_and_zero_strength():
    scores = np.arange(24.).reshape(3, 8)
    ev = np.array([True, False, True])
    delta = np.ones((3, 2))
    gradient = np.ones((2, 8))
    base, expert = intervene(scores, scores, delta, gradient, gradient, ev, .25)
    np.testing.assert_array_equal(base[~ev], scores[~ev])
    np.testing.assert_allclose(expert[ev], scores[ev] - .5)
    base, expert = intervene(scores, scores, delta, gradient, gradient, ev, 0.)
    np.testing.assert_array_equal(base, scores)
    np.testing.assert_array_equal(expert, scores)


def test_absent_classes_do_not_enter_ba_as_artificial_zero_recall():
    value = metrics(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 0]), 8)
    assert value['acc'] == .75
    assert value['ba'] == .5
    assert value['class_recall'][2:] == [None] * 6
