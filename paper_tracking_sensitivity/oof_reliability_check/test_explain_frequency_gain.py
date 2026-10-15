import numpy as np
from explain_frequency_gain import dual_coefficients, parts


def test_weighted_dual_matches_independent_primal():
    r = np.random.default_rng(14)
    x, prior = r.normal(size=(13, 6)), r.normal(size=(13, 3))
    labels = np.array([0]*7+[1]*4+[2]*2)
    w = 13/(3*np.bincount(labels)[labels])
    expected = np.linalg.solve(x.T@(w[:, None]*x)+8*np.eye(6),
                              x.T@(w[:, None]*(2*np.eye(3)[labels]-1-prior)))
    alpha = dual_coefficients(x@x.T, labels, prior)
    np.testing.assert_allclose(x.T@alpha, expected, atol=1e-12)


def test_constant_frequency_is_condition_only_and_removal_is_identity():
    r = np.random.default_rng(5)
    n = 80
    data = dict(phi=r.normal(size=(n, 9)), prior=r.normal(size=(n, 8)),
                mean=r.normal(size=(n, 12)), conditions=r.random((n, 2)))
    data['mean'] /= np.linalg.norm(data['mean'], axis=1, keepdims=True)
    embedding = np.ones((n, 3))/np.sqrt(3)
    variants, values = parts(data, embedding, np.arange(64), np.arange(64)%8, .25)
    np.testing.assert_allclose(variants['learned_frequency'], variants['condition_only'], atol=1e-12)
    np.testing.assert_allclose(variants['learned_frequency'], variants['query_frequency_removed'], atol=1e-12)
