import numpy as np

import context_reliability as method


def test_known_zero_is_not_missing_and_unseen_source_feature_disabled():
    values = np.array([[0., 999.], [2., -999.]])
    known = np.array([[True, False], [True, False]])
    scaler = method.fit_context_scaler(values, known, np.array([.5, .5]))
    x, mask, covered = method.context_design(np.array([[0., 7.], [0., 7.]]),
                                            np.array([[True, True], [False, True]]), scaler)
    np.testing.assert_array_equal(x, [[-1., 0.], [0., 0.]])
    np.testing.assert_array_equal(mask, [[1., 0.], [0., 0.]])
    np.testing.assert_array_equal(covered, [True, False])


def test_missing_storage_values_do_not_affect_scaling():
    known = np.array([[True], [False], [True]])
    a = method.fit_context_scaler(np.array([[0.], [1000.], [4.]]), known, np.array([.25, .25, .5]))
    b = method.fit_context_scaler(np.array([[0.], [-10.], [4.]]), known, np.array([.25, .25, .5]))
    np.testing.assert_allclose(a['center'], [8/3])
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_missing_context_exact_fallback_and_support_guard():
    rng = np.random.default_rng(12)
    base, expert = rng.normal(size=(80, 8)), rng.normal(size=(80, 8))
    labels = np.where(np.arange(80) % 2, base.argmax(1), expert.argmax(1))
    values, known = rng.normal(size=(80, 3)), np.ones((80, 3), dtype=bool)
    model = method.fit_source(base, expert, labels, np.repeat(['a', 'b'], 40), values, known)
    known[:10] = False
    scores, covered = method.predict_scores(model, base, expert, values, known)
    pred = method.route(scores, base.argmax(1), expert.argmax(1), np.zeros(64, dtype=int))
    for name in method.ARMS:
        np.testing.assert_array_equal(scores[name][~covered], scores['oof_lr0'][~covered])
        np.testing.assert_array_equal(pred[name][~covered], pred['oof_lr0'][~covered])
        use = expert.argmax(1) != 0
        np.testing.assert_array_equal(pred[name][use], base.argmax(1)[use])


def test_no_supported_source_returns_original():
    rng = np.random.default_rng(5)
    base, expert = rng.normal(size=(100, 8)), rng.normal(size=(100, 8))
    labels = np.where(np.arange(100) % 2, base.argmax(1), expert.argmax(1))
    model = method.fit_source(base, expert, labels, np.repeat(['a', 'b'], 50),
                              np.zeros((100, 2)), np.zeros((100, 2), dtype=bool))
    scores, covered = method.predict_scores(model, base, expert, np.ones((100, 2)), np.ones((100, 2), dtype=bool))
    assert model['fallback'] and not covered.any()
    for score in scores.values():
        np.testing.assert_array_equal(score, scores['oof_lr0'])
