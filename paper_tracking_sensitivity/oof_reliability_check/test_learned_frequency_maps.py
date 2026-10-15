"""Haar mathematical checks and the frozen cache contract, without target labels."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import extract_learned_frequency_maps as feature_maps
from extract_learned_frequency_maps import haar_bands, inverse_haar


torch.set_num_threads(2)


def test_haar_forward_against_orthonormal_matrix():
    x = torch.arange(2*3*8*10, dtype=torch.float64).reshape(2, 3, 8, 10)/17
    matrix = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1],
                           [1, 1, -1, -1], [1, -1, -1, 1]], dtype=x.dtype)/2
    torch.testing.assert_close(matrix@matrix.T, torch.eye(4, dtype=x.dtype))
    patches = torch.nn.functional.unfold(x, kernel_size=2, stride=2).reshape(2, 3, 4, 4, 5)
    expected = torch.einsum('ij,bcjhw->bichw', matrix, patches)
    actual = haar_bands(x)
    assert actual.shape == (2, 4, 3, 4, 5)
    torch.testing.assert_close(actual, expected)


def test_haar_inverse_random_bands_and_inputs():
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(2, 3, 8, 10, generator=generator, dtype=torch.float64)
    bands = torch.randn(2, 4, 3, 4, 5, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(inverse_haar(haar_bands(x)), x)
    torch.testing.assert_close(haar_bands(inverse_haar(bands)), bands)


def test_haar_energy_independent_of_inverse():
    x = torch.randn(3, 5, 16, 12, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    bands = haar_bands(x)
    torch.testing.assert_close(bands.square().sum(dim=(1, 3, 4)), x.square().sum(dim=(2, 3)))


@pytest.mark.parametrize('band', range(4))
def test_constant_checkerboard_and_axial_patterns(band):
    patterns = torch.tensor([[[1, 1], [1, 1]], [[1, -1], [1, -1]],
                             [[1, 1], [-1, -1]], [[1, -1], [-1, 1]]], dtype=torch.float64)
    x = patterns[band].repeat(4, 5)[None, None]
    actual = haar_bands(x)
    expected = torch.zeros(1, 4, 1, 4, 5, dtype=x.dtype)
    expected[:, band] = 2
    torch.testing.assert_close(actual, expected)


def test_haar_forward_and_inverse_gradients():
    x = torch.randn(1, 2, 4, 6, dtype=torch.float64, requires_grad=True)
    bands = torch.randn(1, 4, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(haar_bands, (x,))
    assert torch.autograd.gradcheck(inverse_haar, (bands,))
    gains = torch.tensor([1., 2., 3., 4.], dtype=x.dtype).reshape(1, 4, 1, 1, 1).requires_grad_()
    inverse_haar(haar_bands(x)*gains).square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.isfinite(gains.grad).all() and (gains.grad.abs() > 0).all()


def test_haar_rejects_odd_size():
    with pytest.raises(ValueError, match='even'):
        haar_bands(torch.zeros(1, 2, 7, 8))


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
def test_parent_haar_outputs_and_gradients_match(dtype):
    from learned_frequency_model import haar

    x = torch.randn(2, 5, 16, 16, generator=torch.Generator().manual_seed(23),
                    dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual, expected = haar_bands(x), haar(reference_x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.arange(1, 5, dtype=dtype).reshape(1, 4, 1, 1, 1)
    (actual*weights).sum().backward()
    (expected*weights).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)


def test_encoder_matches_existing_paired_front_and_is_frozen():
    path = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/extract_full_six_spatial_features.py')
    spec = importlib.util.spec_from_file_location('original_paired_encoder', path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    device = torch.device('cpu')
    encoder = feature_maps.build_encoder(device)
    reference = original.build_encoder(device)[:6]
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(not module.training for module in encoder.modules())
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
    images = torch.randint(0, 256, (2, 2, 128, 128), generator=torch.Generator().manual_seed(11))
    x = (images.float()/255.-0.5)/0.25
    with torch.inference_mode():
        actual = encoder(x)
        assert actual.shape == (2, 512, 16, 16)
        torch.testing.assert_close(actual, reference(x), rtol=0, atol=0)


def test_limited_cache_is_separate_and_f0921_order_is_preserved(tmp_path, monkeypatch):
    frame = pd.DataFrame({'sample_id': ['first', 'second', 'third'], 'row': [9, 2, 17]})
    pixels = np.full((3, 2, 128, 128), 64, dtype=np.uint8)
    pixels[1] = 255
    np.save(tmp_path/'inputs128.npy', pixels)
    np.save(tmp_path/'feature_rows.npy', frame.row.to_numpy())
    monkeypatch.setattr(feature_maps, 'OUT', tmp_path/'maps')
    monkeypatch.setattr(feature_maps, 'load_manifest', lambda dataset: frame)
    monkeypatch.setitem(feature_maps.MANIFESTS, 'f0921', tmp_path/'target_manifest.csv')

    class StubEncoder(torch.nn.Module):
        def forward(self, x):
            return x[:, :1, ::8, ::8].expand(-1, 512, -1, -1)

    encoder = StubEncoder()
    feature_maps.extract('f0921', encoder, torch.device('cpu'), batch_size=2, limit=2)
    assert not list(tmp_path.rglob('COMPLETE.json'))
    assert not (tmp_path/'maps/f0921').exists()
    assert (tmp_path/'smoke/f0921/SMOKE.json').exists()
    info = feature_maps.extract('f0921', encoder, torch.device('cpu'), batch_size=2)
    maps = np.load(tmp_path/'maps/f0921/maps.npy', mmap_mode='r')
    assert maps.shape == (3, 512, 16, 16) and maps.dtype == np.float16
    np.testing.assert_array_equal(np.load(tmp_path/'maps/f0921/sample_ids.npy'), frame.sample_id)
    np.testing.assert_array_equal(maps[:, 0, 0, 0], ((pixels[:, 0, 0, 0]/255.-0.5)/0.25).astype(np.float16))
    assert info['full_population'] and not info['target_labels_read']
    # Even a limit exceeding the manifest length must not claim full completion.
    feature_maps.extract('f0921', encoder, torch.device('cpu'), batch_size=2, limit=10)
    assert not (tmp_path/'smoke/f0921/COMPLETE.json').exists()
    assert (tmp_path/'maps/f0921/COMPLETE.json').exists()
