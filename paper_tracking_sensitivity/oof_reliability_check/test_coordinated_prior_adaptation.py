import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.utils.class_weight import compute_sample_weight

from coordinated_prior_adaptation import fit_component_response, ridge_objective
from residual_expert import fit_correction


def problem(labels):
    rng = np.random.default_rng(704)
    phi = np.column_stack((rng.normal(size=(91, 74)), np.ones(91)))
    support = rng.choice(len(phi), 64, replace=False)
    prior = rng.normal(size=(len(phi), 8))
    component = rng.normal(size=prior.shape) + np.arange(8)
    return phi, support, np.asarray(labels), prior, component


LABELS = [
    np.tile(np.arange(8), 8),
    np.repeat(np.arange(8), [24, 16, 8, 6, 4, 3, 2, 1]),
    np.repeat([0, 3, 7], [48, 12, 4]),
    np.full(64, 5),
]


@pytest.mark.parametrize('labels', LABELS,
                         ids=['balanced', 'unbalanced', 'missing_classes', 'one_class'])
@pytest.mark.parametrize('alpha', [-.5, .25])
def test_update_matches_independent_updated_prior_fit(labels, alpha):
    phi, support, labels, prior, component = problem(labels)
    old_coef = fit_correction(phi[support], labels, prior[support])
    old_scores = prior + phi@old_coef
    result = fit_component_response(phi, support, labels, component)
    assert set(result) == {'coefficient', 'residual_component'}
    assert result['coefficient'].shape == (phi.shape[1], 8)
    assert result['residual_component'].shape == prior.shape

    new_prior = prior + alpha*component
    desired = np.full((64, 8), -1.)
    desired[np.arange(64), labels] = 1.
    direct = Ridge(alpha=8., fit_intercept=False, solver='svd').fit(
        phi[support], desired-new_prior[support],
        sample_weight=compute_sample_weight('balanced', labels))
    new_coef = old_coef-alpha*result['coefficient']
    new_scores = old_scores+alpha*result['residual_component']
    np.testing.assert_allclose(new_coef, direct.coef_.T, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(new_scores, new_prior+direct.predict(phi),
                               rtol=1e-10, atol=1e-11)


def test_zero_alpha_is_exact_and_inputs_are_unchanged():
    phi, support, labels, prior, component = problem(LABELS[2])
    originals = [x.copy() for x in (phi, support, labels, component)]
    coefficient = fit_correction(phi[support], labels, prior[support])
    scores = prior+phi@coefficient
    result = fit_component_response(phi, support, labels, component)
    np.testing.assert_array_equal(scores+0.*result['residual_component'], scores)
    np.testing.assert_array_equal(coefficient-0.*result['coefficient'], coefficient)
    for value, original in zip((phi, support, labels, component), originals):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize('strength', [1.5, 8.])
def test_coordinated_refit_minimizes_same_objective_vs_post_addition(strength):
    phi, support, labels, prior, component = problem(LABELS[2])
    old_coef = fit_correction(phi[support], labels, prior[support], strength)
    result = fit_component_response(phi, support, labels, component, strength)
    for alpha in (-.5, .25, .5):
        new_prior = prior+alpha*component
        post_scores = new_prior+phi@old_coef
        new_coef = old_coef-alpha*result['coefficient']
        new_scores = new_prior+phi@new_coef
        post = ridge_objective(post_scores[support], labels, old_coef, strength)
        coordinated = ridge_objective(new_scores[support], labels, new_coef, strength)
        assert coordinated < post
        desired = np.full((64, 8), -1.)
        desired[np.arange(64), labels] = 1.
        # Independent class-by-class sum verifies observed-class balancing.
        expected = sum(64./3*np.mean(np.sum(
            (new_scores[support][labels == c]-desired[labels == c])**2, axis=1))
            for c in (0, 3, 7)) + strength*np.sum(new_coef**2)
        assert coordinated == pytest.approx(expected, rel=1e-12)


def test_objective_counts_missing_class_columns_and_regularizes_bias():
    labels = np.array([0, 0, 0, 2])
    scores = np.zeros((4, 4))
    coefficient = np.array([[0., 0., 0., 0.], [1., 2., 3., 4.]])
    assert ridge_objective(scores, labels, coefficient) == 16.+8.*30.
    scores[:, 1] = 1.
    assert ridge_objective(scores, labels, coefficient) == 28.+8.*30.


def test_query_components_do_not_affect_fitted_coefficient():
    phi, support, labels, _, component = problem(LABELS[2])
    result = fit_component_response(phi, support, labels, component)
    query = np.ones(len(phi), dtype=bool)
    query[support] = False
    changed = component.copy()
    changed[query] += 100.
    updated = fit_component_response(phi, support, labels, changed)
    np.testing.assert_array_equal(updated['coefficient'], result['coefficient'])
    np.testing.assert_array_equal(updated['residual_component'][support],
                                  result['residual_component'][support])
    np.testing.assert_allclose(updated['residual_component'][query],
                               result['residual_component'][query]+100.)
