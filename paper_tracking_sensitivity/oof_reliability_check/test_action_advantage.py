import numpy as np
from scipy.special import softmax

from run_action_advantage import action_advantage
from budgeted_residual_handoff import ranked_mask


def test_compares_actual_base_action_not_runner_up():
    logits = np.array([[4., 3.9, -4.], [4., 3., 1.]])
    base = np.array([2, 1])
    score = action_advantage(logits, base)
    assert score[0] > score[1]
    p = softmax(logits, axis=1)
    np.testing.assert_allclose(score, [p[0, 0]-p[0, 2], p[1, 0]-p[1, 1]])
    assert (p[0, 0]-p[0, 1]) < (p[1, 0]-p[1, 1])


def test_shared_head_agreement_and_row_shift():
    logits = np.array([[4., 2., 0.], [-2., 3., -1.]])
    base = np.array([0, 2])
    score = action_advantage(logits, base)
    assert score[0] == 0 and score[1] > 0
    np.testing.assert_allclose(score, action_advantage(logits+np.array([[1000.], [-1000.]]), base))


def test_cap_exclusion_and_global_row_tie_order():
    score = np.array([.9, .5, .5, .1])
    selected = ranked_mask(score, np.array([False, True, True, True]), np.array([1, 4, 3, 2]), 1)
    np.testing.assert_array_equal(selected, [False, False, True, False])
