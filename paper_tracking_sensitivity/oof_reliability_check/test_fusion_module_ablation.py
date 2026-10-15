import numpy as np

from fusion_module_ablation import parts, kernel_for, uniform_kernel, ARMS
from frequency_band_core import old_kernel


def fixture():
    rng = np.random.default_rng(621)
    data = dict(phi=rng.normal(size=(13, 5)), mean=rng.normal(size=(13, 4)),
                conditions=rng.uniform(size=(13, 2)))
    return data, rng.normal(size=(13, 6)), np.array([2, 7, 0, 11])


def test_uniform_equals_explicit_shared_and_identity_features():
    data, _, s = fixture()
    x, rho = data['mean'], .37
    explicit = np.c_[np.sqrt(rho)*x, np.diag(np.sqrt((1-rho)*np.sum(x*x, axis=1)))]
    actual = uniform_kernel(x@x[s].T, s, rho)
    np.testing.assert_allclose(actual, explicit@explicit[s].T, atol=1e-12)
    assert np.linalg.eigvalsh(actual[s]).min() > -1e-10


def test_current_full_kernel_replays_existing_operator():
    data, z, s = fixture()
    expected, condition = old_kernel(data, s, .25)
    expected += .25*(z@z[s].T)*condition
    np.testing.assert_array_equal(kernel_for(parts(data, z, s), s, .25, 'full'), expected)


def test_condition_control_preserves_diagonal_not_full_normalization():
    data, z, s = fixture()
    p = parts(data, z, s)
    for prefix in ('mean', 'full'):
        a = kernel_for(p, s, .25, prefix+'_plain')
        b = kernel_for(p, s, .25, prefix+'_uniform')
        c = kernel_for(p, s, .25, 'full' if prefix == 'full' else 'mean_condition')
        np.testing.assert_allclose(np.diag(a[s]), np.diag(b[s]))
        np.testing.assert_allclose(np.diag(a[s]), np.diag(c[s]))
    q = {k: v.copy() for k, v in data.items()}
    q['conditions'][np.setdiff1d(np.arange(13), s)] += 9
    assert parts(q, z, s)['rho'] == p['rho']


def test_zero_mean_strength_collapses_only_expected_arms():
    data, z, s = fixture()
    p = parts(data, z, s)
    for arm in ARMS[:4]:
        np.testing.assert_array_equal(kernel_for(p, s, 0., arm), p['phi'])
    np.testing.assert_array_equal(kernel_for(p, s, 0., 'full'),
                                  kernel_for(p, s, 0., 'frequency_condition'))
