"""CPU checks for the fixed 64-label kernel expert and 5% handoff."""

import inspect

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

import projection_residual_expert as method
from receiver_consistent_handoff import gate_scores
from residual_expert import fit_correction


@pytest.mark.parametrize('dimensions,strength', [(13, 8.), (83, .75)])
def test_kernel_dual_matches_existing_correction_and_weighted_primal(dimensions, strength):
    rng = np.random.default_rng(713)
    features = rng.normal(size=(91, dimensions))*np.linspace(.2, 2., dimensions)
    features += np.linspace(-2., 3., dimensions)
    phi = np.column_stack((features, np.ones(len(features))))
    labels = rng.permutation(np.repeat([0, 2, 5, 7], [35, 17, 9, 3]))
    prior = rng.normal(size=(64, 8))+np.linspace(-1., 1., 8)
    weights = 64/(4*np.bincount(labels, minlength=8)[labels])
    reference = Ridge(alpha=strength, fit_intercept=False).fit(
        phi[:64], 2*np.eye(8)[labels]-1-prior, sample_weight=weights)
    beta = method.fit_kernel_correction(phi[:64]@phi[:64].T, labels, prior,
                                        strength=strength)
    assert beta.shape == (64, 8)
    actual = (phi@phi[:64].T)@beta
    np.testing.assert_allclose(actual, reference.predict(phi), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        actual, phi@fit_correction(phi[:64], labels, prior, strength=strength),
        rtol=1e-9, atol=1e-10)


def test_discarded_coordinates_annihilate_retained_space_and_preserve_complement():
    rng = np.random.default_rng(714)
    source = rng.normal(size=(80, 12))@rng.normal(size=(12, 12))+np.arange(12)
    pca = PCA(n_components=4, svd_solver='full').fit(source)
    x = rng.normal(size=(23, 12))+pca.mean_
    actual = method.discarded_coordinates(x, pca)
    np.testing.assert_allclose(
        actual, x-pca.inverse_transform(pca.transform(x)), atol=1e-12)
    np.testing.assert_allclose(actual@pca.components_.T, 0., atol=1e-12)
    perturbation = rng.normal(size=(23, 4))@pca.components_
    np.testing.assert_allclose(
        method.discarded_coordinates(x+perturbation, pca), actual, atol=1e-12)
    np.testing.assert_allclose(
        method.discarded_coordinates(pca.mean_+perturbation, pca), 0., atol=1e-12)
    np.testing.assert_allclose(
        method.discarded_coordinates(pca.mean_+actual, pca), actual, atol=1e-12)


def test_geometry_statistics_use_only_source_rows():
    rng = np.random.default_rng(717)
    base = rng.normal(size=(16, 7))+np.arange(7)
    extra = rng.normal(size=(16, 5))*np.arange(1, 6)
    source = np.array([11, 2, 14, 0, 7, 5, 9, 4, 12])
    model = dict(pca=PCA(n_components=3).fit(base[source]),
                 pca_ssl=PCA(n_components=2).fit(extra[source]))
    stats = method.fit_geometry_statistics(base, extra, source, model, batch_size=4)
    raw = np.column_stack((base[source], extra[source]))
    discarded = np.column_stack((
        base[source]-model['pca'].inverse_transform(model['pca'].transform(base[source])),
        extra[source]-model['pca_ssl'].inverse_transform(
            model['pca_ssl'].transform(extra[source]))))
    for name, values in [('raw', raw), ('discarded', discarded)]:
        np.testing.assert_allclose(stats[name+'_center'], values.mean(0), atol=1e-12)
        np.testing.assert_allclose(stats[name+'_scale'], values.std(0)+1e-6, atol=1e-12)
    query = np.setdiff1d(np.arange(len(base)), source)
    base[query] += 1000.
    extra[query] *= 1000.
    changed = method.fit_geometry_statistics(base, extra, source, model, batch_size=4)
    for name in stats:
        np.testing.assert_array_equal(changed[name], stats[name])


@pytest.fixture
def draw_inputs():
    rng = np.random.default_rng(715)
    n = 267
    support = rng.choice(n, 64, replace=False)
    labels = rng.permutation(np.repeat([0, 2, 5], [39, 19, 6]))
    base = np.zeros((n, 8))
    base[:, 0] = 3.
    prior = np.full((n, 8), -3.)
    prior[:, 0] = 0.
    prior[:, 2] = 2.
    gate = dict(center=np.zeros(4), scale=np.ones(4),
                lr=np.array([1., 0., 0., 0., -1.]))
    return dict(base_scores=base, prior_expert=prior,
                kernel_to_support=np.zeros((n, 64)), support_rows=support,
                support_labels=labels, gate=gate, row_ids=rng.permutation(n)+1000)


def test_frozen_draw_matches_primal_with_unsorted_supports_and_fixed_inputs(draw_inputs):
    rng = np.random.default_rng(716)
    data = draw_inputs
    n = len(data['base_scores'])
    phi = np.column_stack((rng.normal(size=(n, 15))+1.5, np.ones(n)))
    support = data['support_rows']
    data['kernel_to_support'] = phi@phi[support].T
    data['base_scores'] = rng.normal(size=(n, 8))
    data['prior_expert'] = rng.normal(size=(n, 8))
    data['gate'] = dict(center=np.array([.2, -.4, .8, 1.]),
                        scale=np.array([.7, 1.3, .9, 2.]),
                        lr=np.array([.6, -.3, .2, -.8, 1.]))
    before = {name: value.copy() for name, value in data.items()
              if isinstance(value, np.ndarray)}
    correction = fit_correction(phi[support], data['support_labels'],
                                data['prior_expert'][support])
    expected = data['prior_expert']+phi@correction
    result = method.freeze_kernel_draw(**data)
    assert result['coefficients'].shape == (64, 8)
    np.testing.assert_allclose(result['expert_scores'], expected, atol=1e-10)
    np.testing.assert_allclose(
        result['expert_scores'],
        data['prior_expert']+data['kernel_to_support']@result['coefficients'],
        atol=1e-10)
    np.testing.assert_array_equal(result['expert_predictions'], expected.argmax(1))
    np.testing.assert_allclose(
        result['gate_scores'], gate_scores(data['base_scores'], expected, data['gate']),
        atol=1e-10)
    for name, value in before.items():
        np.testing.assert_array_equal(data[name], value)


def test_budget_excludes_supports_and_nonpositive_or_agreeing_rows(draw_inputs):
    data = draw_inputs
    support = data['support_rows']
    evaluation = np.ones(len(data['base_scores']), dtype=bool)
    evaluation[support] = False
    query = np.flatnonzero(evaluation)
    data['prior_expert'][support, 2] = 20.
    data['prior_expert'][query[:3], 2] = [1., .5, 2.]
    data['prior_expert'][query[2], 0] = 5.
    data['prior_expert'][query[3:6], 2] = [5., 4., 3.]
    result = method.freeze_kernel_draw(**data)
    np.testing.assert_array_equal(result['eval_mask'], evaluation)
    assert result['cap'] == 10  # floor(.05*203), excluding all 64 supports.
    assert result['selected'].sum() == result['cap']
    assert not result['selected'][support].any()
    assert not result['selected'][query[:3]].any()
    np.testing.assert_allclose(result['gate_scores'][query[:2]], [0., -.5])
    # Admit the three higher scores first, then break ties by original row ID.
    candidates = query[6:]
    expected = np.zeros(len(evaluation), dtype=bool)
    expected[query[3:6]] = True
    expected[candidates[np.argsort(data['row_ids'][candidates])[:7]]] = True
    np.testing.assert_array_equal(result['selected'], expected)
    base = data['base_scores'].argmax(1)
    np.testing.assert_array_equal(result['predictions'] != base, result['selected'])
    np.testing.assert_array_equal(
        result['predictions'], np.where(expected, result['expert_predictions'], base))


def test_sub_twenty_query_rows_have_zero_budget(draw_inputs):
    data = draw_inputs
    for name in ('base_scores', 'prior_expert', 'kernel_to_support', 'row_ids'):
        data[name] = data[name][:83]
    data['support_rows'] = np.arange(64)
    result = method.freeze_kernel_draw(**data)
    assert result['eval_mask'].sum() == 19
    assert result['cap'] == 0
    assert not result['selected'].any()
    np.testing.assert_array_equal(result['predictions'], data['base_scores'].argmax(1))


def test_observed_class_guard_rejects_unseen_winner(draw_inputs):
    data = draw_inputs
    data['prior_expert'][:, 7] = 9.
    result = method.freeze_kernel_draw(**data)
    assert (result['gate_scores'] > 0).all()
    np.testing.assert_array_equal(result['expert_predictions'], np.full(267, 7))
    assert result['cap'] == 10
    assert not result['selected'].any()
    np.testing.assert_array_equal(result['predictions'], data['base_scores'].argmax(1))


def test_zero_kernel_preserves_expert_prior_without_forcing_base_fallback(draw_inputs):
    result = method.freeze_kernel_draw(**draw_inputs)
    np.testing.assert_array_equal(result['expert_scores'], draw_inputs['prior_expert'])
    np.testing.assert_array_equal(
        result['expert_predictions'], draw_inputs['prior_expert'].argmax(1))
    assert result['selected'].sum() == 10
    base = draw_inputs['base_scores'].argmax(1)
    np.testing.assert_array_equal(result['predictions'] != base, result['selected'])


def test_frozen_draw_api_accepts_only_support_truth():
    parameters = inspect.signature(method.freeze_kernel_draw).parameters
    assert list(parameters) == [
        'base_scores', 'prior_expert', 'kernel_to_support', 'support_rows',
        'support_labels', 'gate', 'row_ids']
    assert all(parameter.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                      inspect.Parameter.VAR_KEYWORD)
               for parameter in parameters.values())


@pytest.fixture
def source_discriminant_inputs(monkeypatch):
    rng = np.random.default_rng(718)
    base = rng.normal(size=(83, 7))+np.arange(7)
    extra = rng.normal(size=(83, 5))*np.arange(1, 6)
    source = rng.choice(len(base), 19, replace=False)
    labels = rng.integers(0, 8, len(base))
    labels[source] = rng.permutation(np.repeat(np.arange(8), [5, 4, 3, 2, 1, 1, 1, 2]))
    model = dict(pca=PCA(n_components=3).fit(base[source]),
                 pca_ssl=PCA(n_components=2).fit(extra[source]))
    prior_coef = rng.normal(size=(12, 8))*.15
    prior_bias = rng.normal(size=8)

    def fixed_geometry(model, b, e):
        retained = np.column_stack((model['pca'].transform(b),
                                    model['pca_ssl'].transform(e)))/np.sqrt(5)
        prior = np.column_stack((b, e))@prior_coef+prior_bias
        return None, prior, None, np.column_stack((retained, np.ones(len(b))))

    monkeypatch.setattr(method, 'target_geometry', fixed_geometry)
    stats = method.fit_geometry_statistics(base, extra, source, model, batch_size=7)
    return dict(base=base, extra=extra, source_rows=source, source_labels=labels,
                model=model, stats=stats)


def test_source_discriminant_ignores_outside_data_but_uses_source_labels(
        source_discriminant_inputs):
    data = source_discriminant_inputs
    projection = method.fit_source_discriminant(**data, batch_size=7)
    outside = np.setdiff1d(np.arange(len(data['base'])), data['source_rows'])
    changed_labels = data['source_labels'].copy()
    changed_labels[outside] = (changed_labels[outside]+1) % 8
    changed_base, changed_extra = data['base'].copy(), data['extra'].copy()
    changed_base[outside] += 1000.
    changed_extra[outside] *= -100.
    for changes in (dict(source_labels=changed_labels),
                    dict(base=changed_base, extra=changed_extra)):
        refitted = method.fit_source_discriminant(**{**data, **changes}, batch_size=7)
        for name in ('coef', 'center', 'scale'):
            np.testing.assert_array_equal(refitted[name], projection[name])
    changed_labels = data['source_labels'].copy()
    source = data['source_rows']
    changed_labels[source] = np.roll(changed_labels[source], 1)
    refitted = method.fit_source_discriminant(
        **{**data, 'source_labels': changed_labels}, batch_size=7)
    assert not np.allclose(refitted['coef'], projection['coef'])
    discarded = method.feature_views(data['base'], data['extra'], data['model'],
                                     data['stats'])['discarded']
    assert not np.allclose(method.source_discriminant_view(discarded, refitted),
                           method.source_discriminant_view(discarded, projection))


def test_guided_kernel_matches_half_weight_features_with_one_bias(source_discriminant_inputs):
    data = source_discriminant_inputs
    base, extra, model, stats = (data[name] for name in ('base', 'extra', 'model', 'stats'))
    projection = method.fit_source_discriminant(**data, batch_size=7)
    support = np.random.default_rng(719).choice(len(base), 64, replace=False)
    retained = np.column_stack((model['pca'].transform(base),
                                model['pca_ssl'].transform(extra)))/np.sqrt(5)
    discarded = np.column_stack((
        base-model['pca'].inverse_transform(model['pca'].transform(base)),
        extra-model['pca_ssl'].inverse_transform(model['pca_ssl'].transform(extra))))
    discarded = (discarded-stats['discarded_center'])/stats['discarded_scale']/np.sqrt(12)
    raw_guided = np.column_stack((discarded, np.ones(len(base))))@projection['coef']
    guided = (raw_guided-projection['center'])/projection['scale']/np.sqrt(8)
    assert guided.shape == (83, 8)
    np.testing.assert_allclose(
        method.source_discriminant_view(discarded, projection), guided, atol=1e-12)
    phi = np.column_stack((retained/np.sqrt(2), guided/np.sqrt(2), np.ones(len(base))))
    actual = method.build_guided_kernel(base, extra, model, stats, projection,
                                        support, batch_size=7)
    assert actual.shape == (83, 64)
    np.testing.assert_allclose(actual, phi@phi[support].T, rtol=1e-12, atol=1e-12)
