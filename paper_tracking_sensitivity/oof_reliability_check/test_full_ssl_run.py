import numpy as np
import torch
from torch import nn

from full_ssl_run import views, frequency_embedding
from learned_frequency_model import FrequencyResidual, support_scores
from unified_head_core import coefficients


def test_native_14_maps_haar_embeddings_and_gradients():
    torch.manual_seed(31)
    model = FrequencyResidual('selective')
    maps = torch.randn(4, 512, 14, 14)
    z, weights = model(maps, torch.rand(4, 2))
    assert z.shape == (4, 256)
    torch.testing.assert_close(z.square().sum(1), torch.ones(4))
    torch.testing.assert_close(weights.sum(1), torch.ones(4))
    (z[:, :64].sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_live_bf16_maps_match_cached_training_map_precision():
    torch.manual_seed(17)
    model = FrequencyResidual('selective').eval()
    live = torch.randn(4, 512, 14, 14).bfloat16()
    cached_training = live.half().float()
    conditions = torch.rand(4, 2)
    expected = model(cached_training, conditions)
    actual = frequency_embedding(model, live, conditions)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0., rtol=0.)


def test_frequency_training_solve_matches_float64_cpu_readout():
    rng = np.random.default_rng(9)
    phi, prior, mean, conditions, z = [rng.normal(size=(12, n)) for n in (5, 3, 4, 2, 6)]
    y = np.array([0, 1, 2, 0, 0, 1, 2, 1])
    relation = np.exp(-((conditions[:, None]-conditions[None, :8])**2).sum(2)/.5)
    kernel = phi @ phi[:8].T + (.25*mean @ mean[:8].T + .25*z @ z[:8].T)*relation
    expected = prior + kernel @ coefficients(kernel[:8], prior[:8], y, 8.)
    args = [torch.as_tensor(a, dtype=torch.float64) for a in (phi, prior, mean, conditions, z)]
    actual = support_scores(*args, torch.as_tensor(y), .25, support_n=8)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12, rtol=1e-12)


def test_c4_path_keeps_original_view_and_rotation_mean():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            for name in ('maxpool', 'relu', 'bn1', 'conv1', 'layer1', 'layer2', 'layer3', 'layer4'):
                setattr(self, name, nn.Identity())
            self.avgpool = nn.AdaptiveAvgPool2d(1)
    x = torch.rand(2, 2, 128, 128)
    net = Toy()
    zero, average, spatial = views(net, x)
    _, rotated, _ = views(net, torch.rot90(x, 1, (-2, -1)))
    assert spatial.shape[-2:] == (112, 112)
    torch.testing.assert_close(zero, average, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(average, rotated, atol=1e-6, rtol=1e-6)
