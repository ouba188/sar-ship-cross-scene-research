"""NumPy parity, kernel weighting, and training gradients for joint geometry."""

import numpy as np
import pytest
import torch

from aligned_residual_ops import residual_scores as aligned_residual_scores
from joint_residual_ops import ComplementExpertGeometry, residual_scores
from residual_expert import fit_correction


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


def episode(dimensions=(7, 5, 3, 2)):
    base_dim, extra_dim, base_rank, extra_rank = dimensions
    rng = np.random.default_rng(117)
    components, _ = np.linalg.qr(rng.normal(size=(extra_dim, extra_rank)))
    retained_dim = base_rank+extra_rank
    arrays = dict(
        ssl_mean=rng.normal(size=extra_dim),
        ssl_components=components.T.copy(),
        expert_center=rng.normal(size=retained_dim),
        expert_scale=rng.uniform(.5, 2., size=retained_dim),
        expert_coef=rng.normal(size=(8, retained_dim))/np.sqrt(retained_dim),
        expert_intercept=rng.normal(size=8),
        discarded_center=rng.normal(size=base_dim+extra_dim),
        discarded_scale=rng.uniform(.5, 2., size=base_dim+extra_dim))
    xs = rng.normal(size=(73, base_rank))
    extra = rng.normal(size=(73, extra_dim))
    base_discarded = rng.normal(size=(73, base_dim))
    labels = np.repeat([0, 3, 7], [45, 14, 5])
    return arrays, xs, extra, base_discarded, labels


def numpy_geometry(arrays, xs, extra, base_discarded):
    centered = extra-arrays['ssl_mean']
    components = arrays['ssl_components']
    retained = np.column_stack((xs, centered@components.T))
    retained = (retained-arrays['expert_center'])/arrays['expert_scale']
    prior = retained@arrays['expert_coef'].T+arrays['expert_intercept']
    projector = np.eye(extra.shape[1])-components.T@components
    offset = base_discarded.shape[1]
    discarded_extra = ((centered@projector-arrays['discarded_center'][offset:])/
                       arrays['discarded_scale'][offset:])
    discarded = np.column_stack((base_discarded, discarded_extra))
    phi = np.column_stack((retained/np.sqrt(2*retained.shape[1]),
                           discarded/np.sqrt(2*discarded.shape[1]),
                           np.ones(len(xs))))
    return prior, phi


def numpy_scores(arrays, xs, extra, base_discarded, labels, strength=8.):
    prior, phi = numpy_geometry(arrays, xs, extra, base_discarded)
    correction = fit_correction(phi[:64], labels, prior[:64], strength)
    return prior[64:]+phi[64:]@correction


def adapted_scores(geometry, xs, extra, base_discarded, labels, strength=8.):
    prior, phi = geometry(xs, extra, base_discarded)
    return residual_scores(phi[:64], prior[:64], labels, phi[64:], prior[64:], strength)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('dimensions', [(7, 5, 3, 2), (2048, 512, 128, 64)],
                         ids=['generic', 'r50-ssl'])
def test_geometry_and_correction_match_numpy(device, dtype, dimensions):
    arrays, xs, extra, base_discarded, labels = episode(dimensions)
    geometry = ComplementExpertGeometry(arrays).to(device)
    inputs = [torch.tensor(value, dtype=dtype, device=device)
              for value in (xs, extra, base_discarded)]
    prior, phi = geometry(*inputs)
    expected_prior, expected_phi = numpy_geometry(arrays, xs, extra, base_discarded)
    tolerance = dict(rtol=2e-5, atol=3e-6) if dtype == torch.float32 else dict(
        rtol=1e-11, atol=1e-11)
    assert prior.dtype == phi.dtype == dtype
    assert phi.shape == (73, dimensions[2]+dimensions[3]+sum(dimensions[:2])+1)
    np.testing.assert_allclose(prior.cpu().numpy(), expected_prior, **tolerance)
    np.testing.assert_allclose(phi.cpu().numpy(), expected_phi, **tolerance)
    score_tolerance = dict(rtol=2e-5, atol=1e-5) if dtype == torch.float32 else tolerance
    for strength in (8., 2.5):
        scores = adapted_scores(geometry, *inputs, torch.tensor(labels, device=device),
                                strength=strength)
        assert scores.dtype == dtype
        np.testing.assert_allclose(
            scores.cpu().numpy(),
            numpy_scores(arrays, xs, extra, base_discarded, labels, strength),
            **score_tolerance)


@pytest.mark.parametrize('device', DEVICES)
def test_one_intercept_and_half_weighted_kernels(device):
    arrays, xs, extra, base_discarded, _ = episode()
    geometry = ComplementExpertGeometry(arrays).to(device)
    inputs = [torch.tensor(value, device=device) for value in (xs, extra, base_discarded)]
    _, phi = geometry(*inputs)
    centered = extra-arrays['ssl_mean']
    components = arrays['ssl_components']
    xe = ((np.column_stack((xs, centered@components.T))-arrays['expert_center'])/
          arrays['expert_scale'])
    discarded_extra = centered@(
        np.eye(extra.shape[1])-components.T@components)
    offset = base_discarded.shape[1]
    discarded_extra = ((discarded_extra-arrays['discarded_center'][offset:])/
                       arrays['discarded_scale'][offset:])
    discarded = np.column_stack((base_discarded, discarded_extra))
    expected = (.5*(xe@xe.T)/xe.shape[1]+
                .5*(discarded@discarded.T)/discarded.shape[1]+1.)
    np.testing.assert_allclose((phi@phi.T).cpu().numpy(), expected, atol=1e-12)
    np.testing.assert_array_equal(phi[:, -1].cpu().numpy(), np.ones(len(xs)))
    constant_columns = torch.all(phi == phi[:1], dim=0)
    assert constant_columns.sum().item() == 1
    assert constant_columns[-1].item()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('rows', [slice(0, 64), slice(64, None)], ids=['support', 'query'])
@pytest.mark.parametrize('branch', ['retained', 'complement'])
def test_extra_gradient_matches_numpy_finite_difference(device, rows, branch):
    arrays, xs, extra, base_discarded, labels = episode()
    geometry = ComplementExpertGeometry(arrays).to(device)
    extra_tensor = torch.tensor(extra, device=device, requires_grad=True)
    scores = adapted_scores(geometry, torch.tensor(xs, device=device), extra_tensor,
                            torch.tensor(base_discarded, device=device),
                            torch.tensor(labels, device=device))
    rng = np.random.default_rng(119)
    upstream = rng.normal(size=(9, 8))
    gradient, = torch.autograd.grad(
        (scores*torch.tensor(upstream, device=device)).sum(), extra_tensor)
    assert torch.isfinite(gradient).all()
    components = arrays['ssl_components']
    projector = components.T@components
    if branch == 'complement':
        projector = np.eye(extra.shape[1])-projector
    direction = np.zeros_like(extra)
    direction[rows] = rng.normal(size=extra[rows].shape)@projector
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5
    plus = numpy_scores(arrays, xs, extra+epsilon*direction, base_discarded, labels)
    minus = numpy_scores(arrays, xs, extra-epsilon*direction, base_discarded, labels)
    expected = ((plus-minus)*upstream).sum()/(2*epsilon)
    actual = (gradient.cpu().numpy()*direction).sum()
    assert abs(expected) > 1e-7
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('rows', [slice(0, 64), slice(64, None)], ids=['support', 'query'])
def test_training_uses_complement_direction(device, rows):
    arrays, xs, extra, base_discarded, labels = episode()
    geometry = ComplementExpertGeometry(arrays).to(device)
    geometry.train()
    xs, extra, base_discarded = [torch.tensor(value, device=device)
                                 for value in (xs, extra, base_discarded)]
    labels = torch.tensor(labels, device=device)
    rng = np.random.default_rng(121)
    components = arrays['ssl_components']
    direction = np.zeros(extra.shape)
    direction[rows] = rng.normal(size=direction[rows].shape)@(
        np.eye(extra.shape[1])-components.T@components)
    direction /= np.linalg.norm(direction)
    direction = torch.tensor(direction, device=device)
    # One trainable feature direction isolates what the frozen geometry exposes.
    amplitude = torch.nn.Parameter(extra.new_zeros(()))
    optimizer = torch.optim.SGD([amplitude], lr=1e-3)
    before_prior, before_phi = geometry(xs, extra, base_discarded)
    before_scores = adapted_scores(geometry, xs, extra, base_discarded, labels)
    upstream = torch.tensor(rng.normal(size=(9, 8)), device=device)
    scores = adapted_scores(geometry, xs, extra+amplitude*direction,
                            base_discarded, labels)
    loss = (scores*upstream).sum()
    loss.backward()
    assert torch.isfinite(amplitude.grad)
    assert abs(amplitude.grad.item()) > 1e-7
    optimizer.step()
    with torch.no_grad():
        changed_extra = extra+amplitude*direction
        after_prior, after_phi = geometry(xs, changed_extra, base_discarded)
        after_scores = adapted_scores(geometry, xs, changed_extra, base_discarded, labels)
    retained_dim = len(arrays['expert_center'])
    torch.testing.assert_close(after_prior, before_prior, rtol=0., atol=1e-12)
    torch.testing.assert_close(after_phi[:, :retained_dim], before_phi[:, :retained_dim],
                               rtol=0., atol=1e-12)
    assert torch.linalg.vector_norm(after_phi-before_phi).item() > 1e-9
    assert torch.linalg.vector_norm(after_scores-before_scores).item() > 1e-9
    assert (after_scores*upstream).sum().item() < loss.item()


@pytest.mark.parametrize('device', DEVICES)
def test_geometry_has_copied_fixed_buffers_and_reuses_aligned_solver(device):
    arrays, _, _, _, _ = episode()
    geometry = ComplementExpertGeometry(arrays).to(device)
    assert residual_scores is aligned_residual_scores
    assert not list(geometry.parameters())
    expected_names = {name if name.startswith('discarded_') else 'retained.'+name
                      for name in arrays}
    assert set(dict(geometry.named_buffers())) == expected_names
    assert set(geometry.state_dict()) == expected_names
    for name, array in arrays.items():
        buffer_name = name if name.startswith('discarded_') else 'retained.'+name
        buffer = geometry.get_buffer(buffer_name)
        assert not buffer.requires_grad
        assert buffer.device.type == device
        expected = array.copy()
        array[...] = 99.
        np.testing.assert_array_equal(buffer.cpu().numpy(), expected)
