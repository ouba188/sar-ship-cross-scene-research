import numpy as np
from numpy.testing import assert_allclose
from frequency_calibration_policy import confidence_features, fit_correctness, predict_correctness, fit_metadata


def test_offset_invariance():
    rng = np.random.default_rng(5)
    scores = rng.normal(size=(64, 5, 8))
    assert_allclose(confidence_features(scores), confidence_features(scores+7), atol=1e-14)


def test_high_margin_correctness_fit_and_parameter_count():
    labels = np.zeros(64, int)
    scores = np.zeros((64, 5, 8))
    scores[:32, :, 0] = 2.
    scores[32:, :, 1] = .1
    fitted = fit_correctness(scores, labels)
    probability = predict_correctness(fitted, scores)
    assert probability[:32].min() > probability[32:].max()
    assert len(fit_metadata(fitted)['coefficients']) == 6
    # Five repeated heads are not five independent label budgets.
    assert_allclose(probability, np.repeat(probability[:, :1], 5, axis=1), atol=1e-6)


def test_degenerate_correctness_keeps_equal_probabilities():
    scores = np.zeros((64, 5, 8))
    scores[:, :, 0] = 1
    for labels, expected in ((np.zeros(64, int), 1.), (np.ones(64, int), 0.)):
        fitted = fit_correctness(scores, labels)
        assert_allclose(predict_correctness(fitted, scores[:3]), expected)


def test_query_prediction_does_not_refit_scaler():
    rng = np.random.default_rng(8)
    fitted = fit_correctness(rng.normal(size=(64, 5, 8)), np.arange(64) % 8)
    before = fit_metadata(fitted)
    predict_correctness(fitted, rng.normal(size=(13, 5, 8))*5)
    assert before == fit_metadata(fitted)
