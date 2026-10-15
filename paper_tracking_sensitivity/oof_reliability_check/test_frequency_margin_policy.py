import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from frequency_margin_policy import margin_evidence, support_utilities
from frequency_specialist_policy import local_choice, support_utility


def test_hard_utility_matches_original_and_ignores_negative_neighbors():
    rng = np.random.default_rng(10)
    labels = np.arange(64) % 8
    main, band = rng.normal(size=(2, 64, 8))
    sim = rng.normal(size=(20, 64))
    got = support_utilities(sim, main, band, labels)
    assert_allclose(got[:, 0], support_utility(sim, main, band, labels), atol=0, rtol=0)
    assert_array_equal(support_utilities(-np.ones((2, 64)), main, band, labels), 0)


def test_smooth_reveals_both_wrong_improvement_without_claiming_rescue():
    labels = np.zeros(8, dtype=int)
    main = np.tile([0., .2], (8, 1))
    band = np.tile([.1, .2], (8, 1))
    got = support_utilities(np.ones((1, 8)), main, band, labels)
    assert got[0, 0] == 0
    assert_allclose(got[0, 1], 1/(1+np.exp(1))-1/(1+np.exp(2)))
    assert_array_equal(band.argmax(1), 1)


def test_margin_offset_invariance_and_identical_candidate_zero():
    rng = np.random.default_rng(1)
    score, labels = rng.normal(size=(64, 8)), np.arange(64) % 8
    assert_allclose(margin_evidence(score, labels), margin_evidence(score+7, labels))
    assert_array_equal(support_utilities(np.ones((3, 64)), score, score, labels), 0)


def test_threshold_and_same_prediction_keep_reference():
    ref = np.array([[1., 0.], [1., 0.], [1., 0.]])
    bands = np.array([[[0., 1.]], [[0., 1.]], [[2., 0.]]])
    value = np.array([[.02], [.03], [.9]])
    score, choice = local_choice(ref, bands, value-.02)
    assert_array_equal(choice, [0, 1, 0])
    assert_array_equal(score, [[1., 0.], [0., 1.], [1., 0.]])
