import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

import enriched_reliability as m
import run_oof_reliability as old


def fixture():
    rng = np.random.default_rng(123)
    b, e = rng.normal(size=(256, 8)), rng.normal(size=(256, 8))
    y = np.arange(256) % 8
    ports = np.repeat(np.arange(4), 64)
    return b, e, y, ports


def test_richer_features_preserve_class_identity_and_logit_shift_invariance():
    b, e, _, _ = fixture()
    order = np.roll(np.arange(8), 1)
    assert np.allclose(m.features(b, e, False), m.features(b[:, order], e[:, order], False))
    assert not np.allclose(m.features(b, e, True), m.features(b[:, order], e[:, order], True))
    x = m.features(b, e, True)
    assert x.shape == (256, 82)
    assert np.array_equal(x[:, 18:].sum(1), np.ones(256))
    assert np.allclose(x, m.features(b + 17., e - 31., True), atol=1e-12)


def test_source_ridge_matches_independent_weighted_sklearn_solve():
    b, e, y, ports = fixture()
    with threadpool_limits(limits=1):
        model = m.fit_source(b, e, y, ports, True)
    X = old.design(m.features(b, e, True), model['center'], model['scale'])
    use = b.argmax(1) != e.argmax(1)
    w = old.source_weights(y, ports)[use]
    w /= w.sum()
    oracle = Ridge(alpha=.01, fit_intercept=False, solver='svd').fit(
        X[use], old.utility(y, b.argmax(1), e.argmax(1))[use], sample_weight=w)
    assert np.allclose(model['ridge'], oracle.coef_, atol=1e-11)


def test_confidence_residual_update_reproduces_archived_algorithm():
    b, e, y, ports = fixture()
    with threadpool_limits(limits=1):
        previous = old.fit_reliability(b, e, y, ports)
        model = m.fit_source(b, e, y, ports, False)
        adapted = old.adapt_reliability(previous, old.confidence_features(b[:64], e[:64]),
                                         y[:64], b[:64].argmax(1), e[:64].argmax(1))
        scores, pred, _ = m.predict(model, b, e, np.arange(64), y[:64])
    X = old.design(old.confidence_features(b, e), previous['center'], previous['scale'])
    assert np.allclose(scores['conf_ridge0'], X @ previous['ridge'], atol=1e-12)
    assert np.allclose(scores['conf_ridge64'], X @ adapted['ridge'], atol=1e-12)
    assert all(p.shape == y.shape for p in pred.values())


def test_no_support_disagreement_leaves_source_score_unchanged():
    b, e, y, ports = fixture()
    with threadpool_limits(limits=1):
        model = m.fit_source(b, e, y, ports, True)
        e[:64] = b[:64]
        scores, _, residuals = m.predict(model, b, e, np.arange(64), y[:64])
    assert all(np.array_equal(v, np.zeros_like(v)) for v in residuals.values())
    assert np.array_equal(scores['rich_ridge0'], scores['rich_ridge64'])
    assert np.array_equal(scores['rich_hgb0'], scores['rich_hgb64'])


def test_unobserved_expert_class_cannot_override_base():
    b, e, y, ports = fixture()
    with threadpool_limits(limits=1):
        model = m.fit_source(b, e, y, ports, False)
        _, pred, _ = m.predict(model, b, e, np.arange(64), np.zeros(64, dtype=int))
    rejected = e.argmax(1) != 0
    for output in pred.values():
        assert np.array_equal(output[rejected], b.argmax(1)[rejected])


def test_both_wrong_is_zero_reward():
    assert np.array_equal(old.utility(np.array([0, 0, 0]), np.array([1, 0, 1]),
                                     np.array([2, 1, 0])), [0., -1., 1.])
