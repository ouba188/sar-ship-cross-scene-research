"""Synthetic checks for polar features and the unchanged 64-support ridge."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from aligned_residual_ops import FixedExpertGeometry, residual_scores
from polar_residual_expert import PolarResidualExpert, pair_directions


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def backbone_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        net = resnet18(weights=None)
    return {name: value for name, value in net.state_dict().items()
            if not name.startswith('fc.')}


def make_geometry(device='cpu', dtype=torch.float32):
    rng = torch.Generator().manual_seed(17)
    arrays = dict(
        ssl_mean=torch.randn(512, generator=rng)*.1,
        ssl_components=torch.randn(64, 512, generator=rng)/512**.5,
        expert_center=torch.randn(192, generator=rng)*.1,
        expert_scale=torch.rand(192, generator=rng)*1.5+.5,
        expert_coef=torch.randn(8, 192, generator=rng)/192**.5,
        expert_intercept=torch.randn(8, generator=rng)*.1)
    return FixedExpertGeometry(arrays).to(device=device, dtype=dtype)


def make_maps(rows=3, device='cpu', dtype=torch.float32):
    rng = torch.Generator(device=device).manual_seed(31)
    h = torch.randn(rows, 512, 4, 4, generator=rng, device=device, dtype=dtype)
    v = torch.randn(rows, 512, 4, 4, generator=rng, device=device, dtype=dtype)
    xs = torch.randn(rows, 128, generator=rng, device=device, dtype=dtype)
    return h, v, xs


def activate_projection(model):
    with torch.no_grad():
        model.out_proj.weight.copy_(.2*torch.eye(
            512, device=model.out_proj.weight.device, dtype=model.out_proj.weight.dtype))


@pytest.mark.parametrize('mode', ['contrast', 'pooled'])
def test_zero_init_is_exact_vh_mean(backbone_state, mode):
    model = PolarResidualExpert(backbone_state, mode)
    h, v, xs = make_maps()
    output, details = model.fuse(h, v, xs, make_geometry(), return_details=True)
    assert torch.equal(output, h.mean((2, 3)))
    assert torch.equal(details['base_features'], output)
    assert torch.count_nonzero(details['residual_norm']) == 0
    torch.testing.assert_close(details['global_delta'], (v-h).mean((2, 3)))
    assert output.shape == (3, 512)
    assert details['residual_norm'].shape == details['contrast_norm'].shape == (3,)
    assert model.out_proj.bias is None
    assert not hasattr(model, 'classifier') and not hasattr(model, 'utility')


def test_seed_matching_and_exact_backbone_loading(backbone_state):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        candidate = PolarResidualExpert(backbone_state, 'contrast')
        torch.manual_seed(123)
        pooled = PolarResidualExpert(backbone_state, 'pooled')
    for name, value in candidate.state_dict().items():
        assert torch.equal(value, pooled.state_dict()[name]), name
    for name, value in candidate.layer4.state_dict().items():
        assert torch.equal(value, backbone_state['layer4.'+name]), name
    assert set(dict(candidate.named_children())) == {
        'layer4', 'common_norm', 'delta_norm', 'query', 'key', 'content', 'out_proj'}


@pytest.mark.parametrize('invalid', ['missing-layer', 'unexpected-key', 'included-fc'])
def test_backbone_requires_exact_missing_fc(backbone_state, invalid):
    state = dict(backbone_state)
    if invalid == 'missing-layer':
        state.pop('layer4.0.conv1.weight')
    elif invalid == 'unexpected-key':
        state['other.weight'] = torch.zeros(1)
    else:
        state.update({'fc.weight': torch.zeros(1000, 512), 'fc.bias': torch.zeros(1000)})
    with pytest.raises(ValueError, match='without classification head'):
        PolarResidualExpert(state)


def test_invalid_mode(backbone_state):
    with pytest.raises(ValueError, match='mode'):
        PolarResidualExpert(backbone_state, mode='other')


def test_single_shared_encode_forward_and_frozen_bn(backbone_state):
    model = PolarResidualExpert(backbone_state)
    assert model.eval().train() is model
    batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    assert batch_norms and all(not bn.training for bn in batch_norms)
    assert all(bn.weight.requires_grad and bn.bias.requires_grad for bn in batch_norms)
    before = {name: value.clone() for name, value in model.layer4.named_buffers()}
    vh = torch.randn(2, 256, 8, 8)
    vv = torch.randn_like(vh)
    calls = []
    handle = model.layer4.register_forward_pre_hook(lambda module, args: calls.append(args[0]))
    with torch.no_grad():
        h, v = model.encode(vh, vv)
    handle.remove()
    assert len(calls) == 1
    assert torch.equal(calls[0], torch.cat((vh, vv)))
    assert h.shape == v.shape == (2, 512, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(h, model.layer4(vh))
        torch.testing.assert_close(v, model.layer4(vv))
        output = model(vh, vv, torch.randn(2, 128), make_geometry())
    assert torch.equal(output, h.mean((2, 3)))
    for name, value in model.layer4.named_buffers():
        assert torch.equal(value, before[name]), name


@pytest.mark.parametrize('mode', ['contrast', 'pooled'])
def test_identical_branches_have_exact_zero_residual(backbone_state, mode):
    model = PolarResidualExpert(backbone_state, mode)
    activate_projection(model)
    spatial = torch.randn(2, 256, 8, 8)
    with torch.no_grad():
        h, v = model.encode(spatial, spatial.clone())
        output, details = model.fuse(h, v, torch.randn(2, 128), make_geometry(),
                                     return_details=True)
    assert torch.equal(h, v)
    assert torch.equal(output, details['base_features'])
    assert torch.count_nonzero(details['residual_norm']) == 0
    assert torch.count_nonzero(details['contrast_norm']) == 0


@pytest.mark.parametrize('mode', ['contrast', 'pooled'])
def test_both_encoder_inputs_receive_gradients(backbone_state, mode):
    model = PolarResidualExpert(backbone_state, mode)
    activate_projection(model)
    vh = torch.randn(1, 256, 8, 8, requires_grad=True)
    vv = torch.randn(1, 256, 8, 8, requires_grad=True)
    model(vh, vv, torch.randn(1, 128), make_geometry()).square().mean().backward()
    for value in (vh.grad, vv.grad, model.layer4[0].conv1.weight.grad):
        assert value is not None and torch.isfinite(value).all()
        assert value.norm() > 1e-6


def test_pair_direction_matches_direct_head_gradient_and_finite_difference():
    geometry = make_geometry(dtype=torch.float64)
    h, _, xs = make_maps(dtype=torch.float64)
    extra = h.mean((2, 3)).requires_grad_()
    classes = geometry(xs, extra)[0].detach().topk(2, dim=1).indices

    def direct_margin(features):
        z = F.linear(features-geometry.ssl_mean, geometry.ssl_components)
        xe = (torch.cat((xs, z), dim=1)-geometry.expert_center)/geometry.expert_scale
        logits = F.linear(xe, geometry.expert_coef, geometry.expert_intercept)
        selected = logits.gather(1, classes)
        return selected[:, 1]-selected[:, 0]

    actual = pair_directions(geometry, classes)
    expected, = torch.autograd.grad(direct_margin(extra).sum(), extra)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert not actual.requires_grad
    probe = torch.randn_like(extra)
    probe /= probe.norm(dim=1, keepdim=True)
    epsilon = 1e-5
    difference = (direct_margin(extra+epsilon*probe)-direct_margin(extra-epsilon*probe))/(2*epsilon)
    torch.testing.assert_close((actual*probe).sum(1), difference, rtol=1e-7, atol=1e-9)


def test_contrast_and_ablation_keep_global_projection(backbone_state):
    model = PolarResidualExpert(backbone_state)
    activate_projection(model)
    h, v, xs = make_maps()
    geometry = make_geometry()
    queries = []
    handle = model.query.register_forward_pre_hook(lambda module, args: queries.append(args[0]))
    output, details = model.fuse(h, v, xs, geometry, return_details=True)
    handle.remove()
    expected_classes = geometry(xs, h.mean((2, 3)))[0].topk(2, dim=1).indices
    assert torch.equal(details['expert_classes'], expected_classes)
    assert not queries[0].requires_grad
    torch.testing.assert_close(queries[0], F.normalize(pair_directions(geometry, expected_classes), dim=1))
    for name in ('a_plus', 'a_minus'):
        weights = details[name]
        assert weights.shape == (3, 16)
        assert (weights >= 0).all()
        torch.testing.assert_close(weights.sum(1), torch.ones(3))
    contrast = ((details['a_plus']-details['a_minus']).unsqueeze(1)*(v-h).flatten(2)).sum(2)
    assert contrast.norm() > 1e-5
    torch.testing.assert_close(details['contrast_norm'], contrast.norm(dim=1))
    torch.testing.assert_close(output, details['base_features']+model.out_proj(
        details['global_delta']+contrast))
    unused_geometry = Mock(side_effect=AssertionError('ablation must skip prior/attention'))
    ablated, disabled = model.fuse(h, v, xs, unused_geometry, disable_contrast=True,
                                   return_details=True)
    torch.testing.assert_close(ablated, h.mean((2, 3))+model.out_proj((v-h).mean((2, 3))))
    assert not torch.allclose(ablated, output)
    assert not torch.allclose(ablated, disabled['base_features'])
    assert torch.count_nonzero(disabled['contrast_norm']) == 0
    assert 'a_plus' not in disabled and 'a_minus' not in disabled
    pooled = PolarResidualExpert(backbone_state, 'pooled')
    pooled.load_state_dict(model.state_dict())
    handle = pooled.key.register_forward_pre_hook(
        lambda *args: pytest.fail('pooled mode called attention'))
    assert torch.equal(pooled.fuse(h, v, xs, unused_geometry), ablated)
    handle.remove()


def test_spatially_constant_delta_cancels_contrast(backbone_state):
    model = PolarResidualExpert(backbone_state)
    activate_projection(model)
    # Binary fractions make v-h exactly spatially constant, despite varying h.
    h = torch.randint(-8, 9, (3, 512, 4, 4)).float()/4
    delta = torch.randint(-4, 5, (3, 512, 1, 1)).float()/4
    v = h+delta
    output, details = model.fuse(h, v, torch.randn(3, 128), make_geometry(), return_details=True)
    torch.testing.assert_close(details['contrast_norm'], torch.zeros(3), rtol=0, atol=3e-6)
    torch.testing.assert_close(output, h.mean((2, 3))+model.out_proj(delta.flatten(1)),
                               rtol=1e-6, atol=1e-6)


def test_zero_pair_direction_is_finite_and_cancels(backbone_state):
    model = PolarResidualExpert(backbone_state)
    activate_projection(model)
    geometry = make_geometry()
    geometry.expert_coef[:, 128:] = 0
    h, v, xs = make_maps()
    output, details = model.fuse(h, v, xs, geometry, return_details=True)
    assert torch.isfinite(output).all()
    assert torch.equal(details['a_plus'], details['a_minus'])
    assert torch.count_nonzero(details['contrast_norm']) == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cpu_dtype_and_geometry_conversion(backbone_state, dtype):
    model = PolarResidualExpert(backbone_state).to(dtype=dtype)
    activate_projection(model)
    h, v, xs = make_maps(dtype=dtype)
    geometry = make_geometry(dtype=torch.float64 if dtype == torch.float32 else torch.float32)
    output, details = model.fuse(h, v, xs, geometry, return_details=True)
    assert output.dtype == dtype and torch.isfinite(output).all()
    for value in details.values():
        assert torch.isfinite(value).all()
        if value.is_floating_point():
            assert value.dtype == dtype


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason='CUDA bfloat16 unavailable'))])
@pytest.mark.parametrize('mode', ['contrast', 'pooled'])
def test_bfloat16_autocast_and_cached_maps(backbone_state, device, mode):
    model = PolarResidualExpert(backbone_state, mode).to(device)
    activate_projection(model)
    geometry = make_geometry(device=device, dtype=torch.float64)
    vh = torch.randn(1, 256, 8, 8, device=device)
    vv = torch.randn_like(vh)
    xs = torch.randn(1, 128, device=device)
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        h, v = model.encode(vh, vv)
    assert h.dtype == v.dtype == torch.bfloat16
    h, v = h.detach().requires_grad_(), v.detach().requires_grad_()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        output, details = model.fuse(h, v, xs, geometry, return_details=True)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    for value in details.values():
        assert torch.isfinite(value).all()
    cached = model.fuse(h, v, xs, geometry)
    assert torch.equal(output, cached)
    output.square().mean().backward()
    for gradient in (h.grad, v.grad, model.out_proj.weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.float().norm() > 1e-6


@pytest.mark.parametrize('mode', ['contrast', 'pooled'])
def test_64_support_residual_scores_backpropagate_into_fusion(backbone_state, mode):
    model = PolarResidualExpert(backbone_state, mode)
    activate_projection(model)
    geometry = make_geometry()
    h, v, xs = make_maps(rows=69)
    h.requires_grad_()
    v.requires_grad_()
    features = model.fuse(h, v, xs, geometry)
    prior, phi = geometry(xs, features)
    labels = torch.arange(64) % 8
    scores = residual_scores(phi[:64], prior[:64], labels, phi[64:], prior[64:])
    assert scores.shape == (5, 8) and torch.isfinite(scores).all()
    F.cross_entropy(scores, torch.arange(5)).backward()
    for gradient in (h.grad[:64], h.grad[64:], v.grad[:64], v.grad[64:], model.out_proj.weight.grad):
        assert torch.isfinite(gradient).all() and gradient.norm() > 1e-7
    for layer in (model.query, model.key, model.content, model.common_norm, model.delta_norm):
        if mode == 'contrast':
            assert layer.weight.grad is not None
            assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.norm() > 1e-7
        else:
            assert layer.weight.grad is None
    assert all(buffer.grad is None for buffer in geometry.buffers())
