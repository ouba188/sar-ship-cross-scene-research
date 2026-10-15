import inspect
import json

import numpy as np
import pytest
from scipy.special import logsumexp

import direction_score_fusion as method


def _sample(n=96, seed=41):
    rng = np.random.default_rng(seed)
    original, c4 = rng.normal(size=(2, n, 8))
    labels = rng.integers(0, 8, size=n)
    weights = rng.uniform(.1, 2., size=n)
    context = rng.normal(size=(n, 6))
    context[:, 2:4] = rng.integers(0, 2, size=(n, 2))
    return original, c4, labels, weights, context


@pytest.fixture(scope='module')
def fitted():
    data = _sample()
    return data, method.fit_fusions(*data)


def _ce(scores, labels, weights):
    return np.average(logsumexp(scores, axis=1) - scores[np.arange(len(labels)), labels],
                      weights=weights)


def test_quality_definition_and_class_offset_invariance():
    original = np.array([[7., 2., 0., 0., 0., 0., 0., 0.],
                         [1., 1., 1., 1., 1., 1., 1., 1.]])
    c4 = np.array([[0., 6., 2., 0., 0., 0., 0., 0.],
                   [3., 3., 3., 3., 3., 3., 3., 3.]])
    quality = method.quality_features(original, c4)
    np.testing.assert_allclose(quality[:, :2], [[5., 4.], [0., 0.]])
    np.testing.assert_allclose(quality[:, 2], [np.sqrt(69. / 8. - .125 ** 2), 0.])
    np.testing.assert_array_equal(quality[:, 3], [1., 0.])
    np.testing.assert_allclose(method.quality_features(original + 100., c4 - 10.), quality)


@pytest.mark.parametrize('kind', ['class', 'bias', 'shared', 'quality', 'geometry'])
def test_analytic_gradients_match_central_differences(kind):
    original, c4, labels, weights, _ = _sample(13)
    weights /= weights.sum()
    rng = np.random.default_rng(18)
    if kind == 'class':
        parameters = rng.uniform(.2, .8, 8)
        objective = lambda p: method._class_objective(p, c4, original-c4, labels, weights, .37)
    elif kind == 'bias':
        parameters = rng.uniform(-.3, .3, 8)
        objective = lambda p: method._bias_objective(p, c4, labels, weights)
    else:
        shared = kind == 'shared'
        groups, features = (1 if shared else 8), (10 if kind == 'geometry' else 4)
        x = rng.normal(size=(len(labels), features))
        parameters = rng.normal(0., .3, groups * (features + 1))
        objective = lambda p: method._sigmoid_objective(
            p, x, c4, original-c4, labels, weights, -.7, shared)
    _, gradient = objective(parameters)
    numerical = np.empty_like(parameters)
    for i in range(len(parameters)):
        step = np.zeros_like(parameters)
        step[i] = 1e-6
        numerical[i] = (objective(parameters + step)[0] - objective(parameters - step)[0]) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-5)


def test_prediction_contract_and_serializable_diagnostics(fitted):
    data, models = fitted
    original, c4, _, _, context = data
    assert tuple(inspect.signature(method.predict_fusion).parameters) == (
        'model', 'original', 'c4', 'context')
    assert set(models) == {'global_mix', 'class_mix', 'quality_mix',
                           'class_quality_mix', 'class_geometry_mix', 'class_bias'}
    json.dumps(models, allow_nan=False)
    for kind, model in models.items():
        scores, alpha = method.predict_fusion(model, original, c4, context)
        assert scores.shape == original.shape and scores.dtype == np.float64
        assert np.isfinite(scores).all()
        assert isinstance(model['success'], bool)
        assert isinstance(model['status'], int)
        assert isinstance(model['message'], str)
        assert 0 <= model['nit'] <= 150
        assert np.isfinite(model['training_objective'])
        assert model['training_objective'] >= model['training_logloss'] - 1e-12
        if kind == 'class_bias':
            assert alpha.shape == (len(c4), 0)
            assert np.max(np.abs(model['bias'])) <= .5
            np.testing.assert_allclose(scores, c4 + model['bias'])
        else:
            assert alpha.shape == original.shape
            assert np.all((alpha >= 0.) & (alpha <= 1.))
            np.testing.assert_allclose(scores, c4 + alpha * (original-c4))


def test_prediction_rows_and_batches_are_equivariant(fitted):
    data, models = fitted
    original, c4, _, _, context = data
    permutation = np.random.default_rng(15).permutation(len(c4))
    before = json.dumps(models)
    for model in models.values():
        scores, alpha = method.predict_fusion(model, original, c4, context)
        permuted = method.predict_fusion(model, original[permutation], c4[permutation],
                                        context[permutation])
        np.testing.assert_allclose(permuted[0], scores[permutation], atol=1e-12)
        np.testing.assert_allclose(permuted[1], alpha[permutation], atol=1e-12)
        single = method.predict_fusion(model, original[:1], c4[:1], context[:1])
        np.testing.assert_allclose(single[0], scores[:1], atol=1e-12)
        np.testing.assert_allclose(single[1], alpha[:1], atol=1e-12)
    assert json.dumps(models) == before


def test_synthetic_complementary_class_evidence_beats_uniform():
    labels = np.tile(np.arange(8), 12)
    original = np.zeros((len(labels), 8))
    c4 = np.zeros_like(original)
    first = labels < 4
    original[np.flatnonzero(first), labels[first]] = 4.
    c4[np.flatnonzero(~first), labels[~first]] = 4.
    weights = np.ones(len(labels))
    models = method.fit_fusions(original, c4, labels, weights, None)
    scores, alpha = method.predict_fusion(models['class_mix'], original, c4, None)
    uniform = _ce(.5 * (original+c4), labels, weights)
    assert _ce(scores, labels, weights) < uniform - .3
    np.testing.assert_allclose(alpha[:, :4], 1., atol=1e-6)
    np.testing.assert_allclose(alpha[:, 4:], 0., atol=1e-6)
    assert abs(models['global_mix']['alpha'] - .5) < 1e-5


def test_constant_and_missing_features_are_finite_and_source_fixed():
    original = np.zeros((24, 8), dtype=np.float32)
    c4 = np.zeros_like(original)
    labels = np.tile(np.arange(8), 3)
    context = np.zeros((24, 6))
    context[::3, 0] = np.nan
    context[1::3, 4] = np.inf
    context[2::3, 5] = -np.inf
    models = method.fit_fusions(original, c4, labels, np.ones(24), context)
    for model in models.values():
        scores, alpha = method.predict_fusion(model, original, c4, context)
        assert np.isfinite(scores).all() and np.isfinite(alpha).all()
        assert not any(model['feature_active'])
        expected = method.predict_fusion(model, original, c4, None)
        np.testing.assert_allclose(scores, expected[0])
        np.testing.assert_allclose(alpha, expected[1])
    geometry = models['class_geometry_mix']
    # An unobserved source feature cannot activate merely because query varies.
    geometry['slopes'] = np.ones((8, 10)).tolist()
    a = method.predict_fusion(geometry, original, c4, None)[1]
    b = method.predict_fusion(geometry, original, c4, np.full((24, 6), 100.))[1]
    np.testing.assert_array_equal(a, b)


def test_weighted_loss_and_penalties_match_independent_calculation(fitted):
    data, models = fitted
    original, c4, labels, weights, context = data
    for kind, model in models.items():
        scores, _ = method.predict_fusion(model, original, c4, context)
        ce = _ce(scores, labels, weights)
        penalty = 0.
        if kind == 'class_mix':
            penalty = .01 * np.mean((np.asarray(model['alpha']) - model['global_alpha']) ** 2)
        elif kind == 'class_bias':
            penalty = .01 * np.mean(np.asarray(model['bias']) ** 2)
        elif 'slopes' in model:
            penalty = (.01 * np.mean(np.asarray(model['slopes']) ** 2)
                       + .001 * np.mean((np.asarray(model['intercepts']) - model['logit_global']) ** 2))
        np.testing.assert_allclose(model['training_logloss'], ce, atol=1e-12)
        np.testing.assert_allclose(model['training_objective'], ce + penalty, atol=1e-12)


def test_fit_equivariance_and_weight_normalization(fitted):
    data, models = fitted
    original, c4, labels, weights, context = data
    reverse = np.arange(len(labels))[::-1]
    permuted = method.fit_fusions(original[reverse], c4[reverse], labels[reverse],
                                  weights[reverse] * 13., context[reverse])
    for kind in models:
        a = method.predict_fusion(models[kind], original, c4, context)
        b = method.predict_fusion(permuted[kind], original, c4, context)
        np.testing.assert_allclose(a[0], b[0], atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(a[1], b[1], atol=2e-6, rtol=2e-6)


def test_logloss_is_stable_for_large_common_offset():
    scores = np.array([[10001., 9999., 10000., 9998., 9997., 9996., 9995., 9994.],
                       [-10001., -9999., -10000., -9998., -9997., -9996., -9995., -9994.]])
    labels, weights = np.array([0, 7]), np.array([.2, .8])
    loss, gradient = method._loss_gradient(scores, labels, weights)
    centered = scores - scores.max(1, keepdims=True)
    np.testing.assert_allclose(loss, _ce(centered, labels, weights), atol=1e-12)
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient.sum(1), 0., atol=1e-15)
