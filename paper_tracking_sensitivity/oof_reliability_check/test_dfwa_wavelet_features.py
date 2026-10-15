import numpy as np
import pywt

from dfwa_wavelet_features import features, LOW_MASK


def test_constant_image_has_no_high_frequency_evidence():
    x = np.full((2, 2, 128, 128), 100, dtype=np.uint8)
    f = features(x)
    assert f.shape == (2, 192)
    np.testing.assert_allclose(f[:, ~LOW_MASK], 0, atol=1e-7)
    assert np.all(f[:, LOW_MASK] > 0)


def test_checkerboard_keeps_high_frequency_energy():
    grid = ((np.indices((128, 128)).sum(0)%2)*255).astype(np.uint8)
    x = np.stack([grid, grid])[None]
    f = features(x)
    assert np.max(f[:, ~LOW_MASK]) > .1
    ll, detail = pywt.dwt2(x.astype(float), 'haar', mode='periodization', axes=(-2, -1))
    np.testing.assert_allclose(np.sum(x.astype(float)**2), np.sum(ll**2)+sum(np.sum(a**2) for a in detail))


def test_c4_invariance_and_polarization_separation():
    rng = np.random.default_rng(73)
    x = rng.integers(0, 256, (3, 2, 128, 128), dtype=np.uint8)
    f = features(x)
    np.testing.assert_allclose(features(np.rot90(x, axes=(-2, -1))), f, atol=2e-7)
    swapped = features(x[:, ::-1]).reshape(3, 2, -1)
    np.testing.assert_allclose(swapped[:, ::-1], f.reshape(3, 2, -1))


def test_added_blocks_keep_kernel_scale_and_source_only_normalization():
    from run_dfwa_wavelet_probe import branch_vectors
    rng = np.random.default_rng(20)
    source = rng.normal(size=(80, 192))
    target = rng.normal(size=(13, 192))
    mean = rng.normal(size=(13, 128))
    vectors, center, scale = branch_vectors(mean, target, source)
    for value in vectors.values():
        np.testing.assert_allclose(np.sum(value**2, axis=1), 1, atol=1e-12)
    _, other_center, other_scale = branch_vectors(mean, target*2, source)
    np.testing.assert_array_equal(center, other_center)
    np.testing.assert_array_equal(scale, other_scale)
