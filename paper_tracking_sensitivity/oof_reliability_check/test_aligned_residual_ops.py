"""Check deployment parity and gradients through support adaptation."""

import numpy as np
import pytest
import torch

from aligned_residual_ops import FixedExpertGeometry, residual_scores
from residual_expert import fit_correction


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


def episode(labels):
    rng = np.random.default_rng(17)
    support_phi = np.column_stack((rng.normal(size=(64, 192))/np.sqrt(192), np.ones(64)))
    query_phi = np.column_stack((rng.normal(size=(9, 192))/np.sqrt(192), np.ones(9)))
    return (support_phi, rng.normal(size=(64, 8)), np.asarray(labels),
            query_phi, rng.normal(size=(9, 8)))


def numpy_scores(support_phi, support_prior, labels, query_phi, query_prior, strength=8.):
    return query_prior+query_phi@fit_correction(support_phi, labels, support_prior, strength)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('strength', [8., 2.5])
@pytest.mark.parametrize('labels', [
    np.repeat(np.arange(8), [29, 13, 8, 5, 4, 2, 2, 1]),
    np.repeat([0, 3, 7], [45, 14, 5]),
    np.full(64, 7),
], ids=['imbalanced-all-classes', 'absent-classes', 'single-present-class'])
def test_scores_match_numpy_float64(device, strength, labels):
    arrays = episode(labels)
    inputs = [torch.as_tensor(value, device=device) for value in arrays]
    actual = residual_scores(*inputs, strength=strength)
    assert actual.dtype == torch.float64
    np.testing.assert_allclose(actual.cpu().numpy(), numpy_scores(*arrays, strength),
                               rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('device', DEVICES)
def test_default_strength_preserves_float32(device):
    arrays = episode(np.repeat([0, 3, 7], [45, 14, 5]))
    arrays = tuple(value if index == 2 else value.astype(np.float32)
                   for index, value in enumerate(arrays))
    actual = residual_scores(*[torch.as_tensor(value, device=device) for value in arrays])
    assert actual.dtype == torch.float32
    np.testing.assert_allclose(actual.cpu().numpy(), numpy_scores(*arrays),
                               rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('input_index', [0, 1], ids=['support-phi', 'support-prior'])
def test_support_gradient_matches_numpy_finite_difference(device, input_index):
    arrays = episode(np.repeat([0, 3, 7], [45, 14, 5]))
    inputs = [torch.as_tensor(value, device=device) for value in arrays]
    inputs[input_index].requires_grad_()
    rng = np.random.default_rng(31)
    upstream = rng.normal(size=arrays[-1].shape)
    loss = (residual_scores(*inputs)*torch.as_tensor(upstream, device=device)).sum()
    gradient, = torch.autograd.grad(loss, inputs[input_index])
    assert torch.isfinite(gradient).all()
    direction = rng.normal(size=arrays[input_index].shape)
    if input_index == 0:
        direction[:, -1] = 0.  # The intercept is fixed; perturb the support features.
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5
    plus, minus = list(arrays), list(arrays)
    plus[input_index] = arrays[input_index]+epsilon*direction
    minus[input_index] = arrays[input_index]-epsilon*direction
    expected = ((numpy_scores(*plus)-numpy_scores(*minus))*upstream).sum()/(2*epsilon)
    actual = (gradient.cpu().numpy()*direction).sum()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)


def geometry_arrays():
    rng = np.random.default_rng(9)
    return dict(
        ssl_mean=rng.normal(size=512),
        ssl_components=rng.normal(size=(64, 512))/np.sqrt(512),
        expert_center=rng.normal(size=192),
        expert_scale=rng.uniform(.5, 2., size=192),
        expert_coef=rng.normal(size=(8, 192))/np.sqrt(192),
        expert_intercept=rng.normal(size=8))


def numpy_geometry(arrays, xs, extra):
    z = (extra-arrays['ssl_mean'])@arrays['ssl_components'].T
    joint = np.concatenate((xs, z), axis=1)
    xe = (joint-arrays['expert_center'])/arrays['expert_scale']
    prior = xe@arrays['expert_coef'].T+arrays['expert_intercept']
    phi = np.column_stack((xe/np.sqrt(xe.shape[1]), np.ones(len(xe))))
    return prior, phi


@pytest.mark.parametrize('device', DEVICES)
def test_fixed_geometry_matches_numpy_and_has_only_buffers(device):
    arrays = geometry_arrays()
    geometry = FixedExpertGeometry(arrays).to(device)
    assert not list(geometry.parameters())
    assert set(dict(geometry.named_buffers())) == set(arrays)
    assert all(not buffer.requires_grad for buffer in geometry.buffers())
    rng = np.random.default_rng(13)
    xs, extra = rng.normal(size=(7, 128)), rng.normal(size=(7, 512))
    actual = geometry(torch.as_tensor(xs, device=device), torch.as_tensor(extra, device=device))
    for result, expected in zip(actual, numpy_geometry(arrays, xs, extra)):
        assert result.dtype == torch.float64
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-11, atol=1e-11)
    arrays['ssl_mean'][:] = 0.
    assert torch.count_nonzero(geometry.ssl_mean) == 512


@pytest.mark.parametrize('device', DEVICES)
def test_extra_support_gradient_through_geometry_and_ridge(device):
    arrays = geometry_arrays()
    geometry = FixedExpertGeometry(arrays).to(device)
    rng = np.random.default_rng(23)
    xs, extra = rng.normal(size=(73, 128)), rng.normal(size=(73, 512))
    labels = np.repeat([0, 3, 7], [45, 14, 5])
    extra_tensor = torch.tensor(extra, device=device, requires_grad=True)
    prior, phi = geometry(torch.as_tensor(xs, device=device), extra_tensor)
    scores = residual_scores(phi[:64], prior[:64], torch.as_tensor(labels, device=device),
                             phi[64:], prior[64:])
    upstream = rng.normal(size=(9, 8))
    gradient, = torch.autograd.grad((scores*torch.as_tensor(upstream, device=device)).sum(),
                                    extra_tensor)
    assert torch.isfinite(gradient).all()
    direction = np.zeros_like(extra)
    direction[:64] = rng.normal(size=(64, 512))
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5
    losses = []
    for sign in (1., -1.):
        np_prior, np_phi = numpy_geometry(arrays, xs, extra+sign*epsilon*direction)
        adapted = numpy_scores(np_phi[:64], np_prior[:64], labels, np_phi[64:], np_prior[64:])
        losses.append((adapted*upstream).sum())
    expected = (losses[0]-losses[1])/(2*epsilon)
    actual = (gradient.cpu().numpy()*direction).sum()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)
