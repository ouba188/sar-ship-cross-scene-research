"""Small synthetic CPU checks for source provenance and full readout geometry."""

import pickle

import numpy as np
import pytest

from full_ssl_geometry import base_parts, fit_geometry, transform


@pytest.fixture
def data():
    rng = np.random.default_rng(104)
    raw = rng.normal(size=(96, 12)).astype(np.float32)
    c4 = (.4*raw+rng.normal(size=raw.shape)).astype(np.float32)
    extra = rng.normal(size=(96, 7)).astype(np.float32)
    rows = np.arange(8, 88)
    labels = np.zeros(96, dtype=int)
    labels[rows] = np.r_[np.zeros(24, dtype=int), np.tile(np.arange(1, 8), 8)]
    return raw, c4, extra, labels, rows


def fit(data, batch_size=13):
    return fit_geometry(*data, raw_rank=4, extra_rank=3, batch_size=batch_size)


def test_forbidden_rows_cannot_change_source_fit(data):
    raw, c4, extra, labels, rows = data
    baseline = fit(data)
    forbidden = np.setdiff1d(np.arange(len(raw)), rows)
    arrays = [x.copy() for x in (raw, c4, extra)]
    for values in arrays:
        values[forbidden] = np.nan
    other_labels = labels.copy()
    other_labels[forbidden] = 999
    changed = fit((*arrays, other_labels, rows))
    probe = (raw[:6], c4[:6], extra[:6])
    for name, expected in transform(*probe, baseline).items():
        np.testing.assert_array_equal(transform(*probe, changed)[name], expected)
    for key in ('pca', 'pca_ssl'):
        assert baseline[key].components_.dtype == np.float32
        np.testing.assert_array_equal(baseline[key].components_, changed[key].components_)
    restored = pickle.loads(pickle.dumps(changed))
    for name, expected in transform(*probe, changed).items():
        np.testing.assert_array_equal(transform(*probe, restored)[name], expected)


def test_direct_projections_and_kernel_weights(data):
    raw, c4, extra, _, rows = data
    state = fit(data)
    r, e = raw[:9].astype(float), extra[:9].astype(float)
    pc, pe = state['pca'], state['pca_ssl']
    xs = ((r-pc.mean_)@pc.components_.astype(float).T-state['center'])/state['scale']
    xp = (e-pe.mean_)@pe.components_.astype(float).T
    retained = (np.c_[xs, xp]-state['expert_center'])/state['expert_scale']
    residuals = []
    for values, source, pca in ((r, raw[rows].astype(float), pc),
                                (e, extra[rows].astype(float), pe)):
        components = pca.components_.astype(float)
        projection = np.eye(values.shape[1])-components.T@components
        source_residual = (source-pca.mean_)@projection
        mean, scale = source_residual.mean(0), source_residual.std(0)+1e-6
        residuals.append(((values-pca.mean_)@projection-mean)/scale)
    result = transform(raw[:9], c4[:9], extra[:9], state)
    expected_single = .5*retained[:, :4]@retained[:, :4].T/4
    expected_single += .5*residuals[0]@residuals[0].T/12+1
    discarded = np.column_stack(residuals)
    expected_dual = .5*retained@retained.T/7+.5*discarded@discarded.T/19+1
    np.testing.assert_allclose(result['single_expert']@result['single_expert'].T,
                               expected_single, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(result['dual_expert']@result['dual_expert'].T,
                               expected_dual, atol=1e-11, rtol=1e-11)
    assert result['single_expert'].shape == (9, 17)
    assert result['dual_expert'].shape == (9, 27)
    parts = base_parts(c4[:9], state)
    expected_xs = ((c4[:9].astype(float)-pc.mean_)@pc.components_.astype(float).T
                   -state['center'])/state['scale']
    normalized = (expected_xs-state['residual_center'])/state['residual_scale']
    np.testing.assert_allclose(parts['xs'], expected_xs, atol=1e-12)
    np.testing.assert_allclose(parts['phi'], np.c_[normalized/2, np.ones(9)], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(parts['mean'], axis=1), 1., atol=1e-12)
    for values in result.values():
        assert values.dtype == np.float64
    one_block = fit(data, batch_size=2048)
    np.testing.assert_allclose(state['discarded_scale'], one_block['discarded_scale'], atol=1e-12)


def test_single_outputs_do_not_depend_on_extra(data):
    raw, c4, extra, _, _ = data
    state = fit(data)
    before = transform(raw[:8], c4[:8], extra[:8], state)
    after = transform(raw[:8], c4[:8], extra[:8]*-5+17, state)
    for name in ('phi', 'mean', 'prior', 'raw_phi', 'raw_prior', 'single_expert', 'single_prior'):
        np.testing.assert_array_equal(before[name], after[name])
    assert not np.allclose(before['dual_expert'], after['dual_expert'])
    assert not np.allclose(before['dual_prior'], after['dual_prior'])
    base_state = {key: state[key] for key in
                  ('pca', 'center', 'scale', 'residual_center', 'residual_scale', 'base')}
    parts = base_parts(c4[:8], base_state)
    for name in ('phi', 'mean', 'prior'):
        np.testing.assert_array_equal(before[name], parts[name])


def test_all_four_priors_match_balanced_ridge_with_free_intercept(data):
    raw, c4, extra, labels, rows = data
    state = fit(data)
    result = transform(raw[rows], c4[rows], extra[rows], state)
    original_xs = base_parts(raw[rows], state)['xs']
    extra_xs = ((extra[rows].astype(float)-state['pca_ssl'].mean_)
                @state['pca_ssl'].components_.astype(float).T)
    joint = (np.c_[original_xs, extra_xs]-state['expert_center'])/state['expert_scale']
    weights = len(rows)/(8*np.bincount(labels[rows])[labels[rows]])
    targets = 2*np.eye(8)[labels[rows]]-1
    for head, key, features in (
            ('base', 'prior', base_parts(c4[rows], state)['xs']),
            ('raw_base', 'raw_prior', original_xs),
            ('single', 'single_prior', joint[:, :4]),
            ('dual', 'dual_prior', joint)):
        design = np.c_[features, np.ones(len(rows))]
        penalty = np.diag(np.r_[np.ones(features.shape[1]), 0.])
        coef = np.linalg.lstsq(np.r_[design*np.sqrt(weights[:, None]), penalty],
                              np.r_[targets*np.sqrt(weights[:, None]),
                                    np.zeros((design.shape[1], 8))], rcond=None)[0]
        np.testing.assert_allclose(result[key], design@coef, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(state[head].classes_, np.arange(8))
        assert result[key].shape == (80, 8)
