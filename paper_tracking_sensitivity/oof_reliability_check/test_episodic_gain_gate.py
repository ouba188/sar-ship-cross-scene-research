import numpy as np

import episodic_gain_gate as method


def test_neutral_probability_changes_gain_ranking_not_conditional_odds():
    # Row0 wins conditionally with90% probability but is almost always neutral.
    probability = np.array([[.01, .9, .09], [.2, .2, .6]])
    features = np.log(probability)
    model = dict(center=np.zeros(3), scale=np.ones(3), models={
        'gain_3way': dict(coef=np.eye(3), intercept=np.zeros(3))})
    scores, actual = method.predict_scores(model, features)
    np.testing.assert_allclose(actual['gain_3way'], probability)
    np.testing.assert_allclose(scores['gain_3way'], [.08, .4])
    assert probability[0, 2]/probability[0, 0] > probability[1, 2]/probability[1, 0]
    assert scores['gain_3way'][0] < scores['gain_3way'][1]


def test_model_class_order_and_query_prediction_does_not_refit():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(120, 4)); y = np.tile([-1, 0, 1], 40)
    model = method.fit_models(x, y, np.ones(120, bool), np.full(120, 1/120))
    np.testing.assert_array_equal(model['models']['gain_3way']['classes'], [-1, 0, 1])
    np.testing.assert_array_equal(model['models']['matched_binary']['classes'], [-1, 1])
    center = model['center'].copy(); scale = model['scale'].copy()
    first, _ = method.predict_scores(model, x[:4])
    second, _ = method.predict_scores(model, np.concatenate((x[:4], x[4:]+100)))
    for name in first:
        np.testing.assert_allclose(first[name], second[name][:4])
    np.testing.assert_array_equal(model['center'], center)
    np.testing.assert_array_equal(model['scale'], scale)
