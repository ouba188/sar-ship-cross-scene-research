import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import RidgeClassifier

from protected_drift_activation import protected_drift_basis, fit_source_heads


def test_manual_equal_class_and_cell_weights_preserve_row_order():
    means = np.array([[0., 1., 0.], [10., 2., 0.], [3., 1., 0.],
                      [12., 4., 0.], [6., 1., 0.], [14., 6., 0.],
                      [16., 8., 0.]])
    labels = np.array([2, 7, 2, 7, 2, 7, 7])
    expected = np.array([[-3/np.sqrt(6), 0, 0],
                         [-3/np.sqrt(8), -3/np.sqrt(8), 0],
                         [0, 0, 0], [-1/np.sqrt(8), -1/np.sqrt(8), 0],
                         [3/np.sqrt(6), 0, 0],
                         [1/np.sqrt(8), 1/np.sqrt(8), 0],
                         [3/np.sqrt(8), 3/np.sqrt(8), 0]])
    result = protected_drift_basis(means, labels, np.eye(3)[:, :1])
    assert_allclose(result['u'], expected, atol=1e-15)
    assert_allclose(result['nuisance_energy_in_stable_fraction'], 5.5/8)
    assert_allclose(result['singular_values'], [np.sqrt(2.5), 0, 0], atol=1e-15)
    assert_allclose(result['selected_energy_fraction'], 1.)


def test_stable_coordinates_preserved_at_real_feature_dimension():
    rng = np.random.default_rng(8)
    qs = np.linalg.qr(rng.normal(size=(2048, 26)), mode='reduced')[0]
    # Small residuals amplify projection roundoff and exercise reorthogonalization.
    means = rng.normal(size=(24, 26)) @ qs.T + 1e-8*rng.normal(size=(24, 2048))
    result = protected_drift_basis(means, np.repeat(np.arange(8), 3), qs)
    qd = result['q_drift']
    assert qd.shape == (2048, 16)
    assert result['protected_overlap_max'] < 1e-8
    assert_allclose(qs.T @ qd, 0., atol=1e-8)
    assert_allclose(qd.T @ qd, np.eye(16), atol=1e-12)
    x = rng.normal(size=(5, 2048))
    suppressed = x - .5*(x @ qd) @ qd.T
    assert_allclose(suppressed @ qs, x @ qs, atol=1e-10, rtol=1e-10)


def test_suppression_quarters_selected_energy_and_reports_fraction():
    means = np.vstack([np.diag([4., 2., 1.]), -np.diag([4., 2., 1.])])
    result = protected_drift_basis(means, np.zeros(6, dtype=int),
                                   np.empty((3, 0)), rank=2)
    qd, u = result['q_drift'], result['u']
    suppressed = u - .5*(u @ qd) @ qd.T
    assert_allclose(np.sum((suppressed @ qd)**2), .25*np.sum((u @ qd)**2))
    assert_allclose(result['selected_energy_fraction'], 20/21)
    assert_allclose(suppressed - (suppressed @ qd) @ qd.T,
                    u - (u @ qd) @ qd.T, atol=1e-14)


@pytest.mark.parametrize('case', ['zero', 'contained', 'empty'])
def test_no_fake_directions_for_empty_or_contained_nuisance(case):
    rng = np.random.default_rng(12)
    qs = np.linalg.qr(rng.normal(size=(40, 7)), mode='reduced')[0]
    means = rng.normal(size=(12, 7)) @ qs.T
    labels = np.repeat([0, 1, 2], 4)
    if case == 'zero':
        means = np.zeros_like(means)
    elif case == 'empty':
        means, labels = means[:0], labels[:0]
    result = protected_drift_basis(means, labels, qs)
    assert result['q_drift'].shape == (40, 0)
    assert result['selected_energy_fraction'] == 0.
    assert result['protected_overlap_max'] == 0.
    assert_allclose(result['nuisance_energy_in_stable_fraction'],
                    1. if case == 'contained' else 0., atol=1e-14)


def test_rank_cap_and_relative_singular_value_threshold():
    means = np.vstack([np.eye(25), -np.eye(25)])
    labels = np.zeros(50, dtype=int)
    for rank, expected in [(0, 0), (5, 5), (16, 16), (100, 16)]:
        result = protected_drift_basis(means, labels, np.empty((25, 0)), rank=rank)
        assert result['q_drift'].shape == (25, expected)
        assert_allclose(result['selected_energy_fraction'], expected/25)
    diagonal = np.diag([1., 2e-6, 5e-7])
    result = protected_drift_basis(np.vstack([diagonal, -diagonal]),
                                   np.zeros(6, dtype=int), np.empty((3, 0)))
    assert result['q_drift'].shape == (3, 2)
    assert_allclose(result['q_drift'][2], 0., atol=1e-14)


def source_bank():
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(8), np.arange(3, 11))
    bank = (rng.normal(size=(len(labels), 9)) + np.arange(9)).astype(np.float32)
    return rng, bank, labels


def test_all_heads_match_independent_sklearn_fits_including_2x2():
    rng, bank, labels = source_bank()
    basis = np.linalg.qr(rng.normal(size=(9, 4)), mode='reduced')[0]
    qs, qd = basis[:, :2], basis[:, 2:]
    transforms = dict(random=rng.normal(size=(9, 9)),
                      rectangular=rng.normal(size=(9, 4)), zero=np.zeros((9, 3)))
    for drift_on in (False, True):
        transform = np.eye(9) - (.5 if drift_on else 0.) * qd @ qd.T
        for stable_on in (False, True):
            transforms[f'{drift_on}_{stable_on}'] = (
                np.column_stack([transform, qs]) if stable_on else transform)
    originals = {name: value.copy() for name, value in transforms.items()}
    bank_before = bank.copy()
    heads = fit_source_heads(bank, labels, transforms, strength=.7)
    query = rng.normal(size=(13, 9))
    for name, transform in transforms.items():
        reference = RidgeClassifier(alpha=.7, class_weight='balanced', solver='cholesky')
        reference.fit(bank.astype(np.float64) @ transform, labels)
        head = heads[name]
        assert head['coef'].shape == (9, 8)
        assert head['intercept'].shape == (8,)
        assert head['coef'].dtype == np.float64
        assert_allclose(head['coef'], transform @ reference.coef_.T, atol=2e-11)
        assert_allclose(head['intercept'], reference.intercept_, atol=2e-11)
        assert_allclose(query @ head['coef'] + head['intercept'],
                        reference.decision_function(query @ transform), atol=2e-11)
        assert_array_equal(transform, originals[name])
    assert_array_equal(bank, bank_before)


def test_zero_width_head_and_present_class_balancing():
    rng, bank, labels = source_bank()
    keep = np.isin(labels, [1, 4, 7])
    bank, labels = bank[keep], labels[keep]
    transform = rng.normal(size=(9, 5))
    heads = fit_source_heads(bank, labels,
                            dict(projected=transform, empty=np.empty((9, 0))))
    expected = -np.ones(8)
    expected[[1, 4, 7]] = -1/3
    assert_allclose(heads['empty']['intercept'], expected, atol=1e-15)
    assert_array_equal(heads['empty']['coef'], np.zeros((9, 8)))
    reference = RidgeClassifier(alpha=1., class_weight='balanced', solver='cholesky')
    reference.fit(bank.astype(np.float64) @ transform, labels)
    head = heads['projected']
    predictions = bank @ head['coef'] + head['intercept']
    assert_allclose(predictions[:, [1, 4, 7]],
                    reference.decision_function(bank.astype(np.float64) @ transform), atol=1e-11)
    assert_allclose(predictions[:, [0, 2, 3, 5, 6]], -1., atol=1e-14)


def test_affine_bank_translation_preserves_predictions_and_unpenalized_intercept():
    rng, bank, labels = source_bank()
    bank = bank.astype(np.float64)
    shift = 100*rng.normal(size=9)
    transforms = dict(projected=rng.normal(size=(9, 4)))
    before = fit_source_heads(bank, labels, transforms)['projected']
    after = fit_source_heads(bank + shift, labels, transforms)['projected']
    query = rng.normal(size=(10, 9))
    assert_allclose(after['coef'], before['coef'], atol=1e-12)
    assert_allclose((query + shift) @ after['coef'] + after['intercept'],
                    query @ before['coef'] + before['intercept'], atol=1e-11)


def test_float32_bank_across_accumulation_blocks_matches_sklearn():
    rng = np.random.default_rng(19)
    labels = np.repeat(np.arange(8), np.arange(130, 138))
    bank = rng.normal(size=(len(labels), 5)).astype(np.float32)
    head = fit_source_heads(bank, labels, {'identity': np.eye(5)})['identity']
    reference = RidgeClassifier(alpha=1., class_weight='balanced', solver='cholesky')
    reference.fit(bank.astype(np.float64), labels)
    assert_allclose(head['coef'], reference.coef_.T, atol=1e-13)
    assert_allclose(head['intercept'], reference.intercept_, atol=1e-13)


def test_public_apis_only_take_source_inputs():
    assert list(inspect.signature(protected_drift_basis).parameters) == [
        'cell_means', 'cell_labels', 'stable_q', 'rank']
    assert list(inspect.signature(fit_source_heads).parameters) == [
        'bank', 'labels', 'transforms', 'strength', 'classes']
