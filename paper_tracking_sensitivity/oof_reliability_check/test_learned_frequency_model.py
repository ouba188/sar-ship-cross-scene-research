import numpy as np
import torch
from learned_frequency_model import FrequencyResidual, haar, support_scores


def test_haar_energy_and_constant():
    torch.manual_seed(3)
    x = torch.randn(2, 512, 16, 16, requires_grad=True)
    bands = haar(x)
    torch.testing.assert_close(bands.square().sum(), x.square().sum())
    assert haar(torch.ones_like(x))[:, 1:].abs().max() == 0
    bands[:, 1:].square().mean().backward()
    assert x.grad.abs().sum() > 0


def test_unit_kernel_and_uniform_initial_fusion():
    torch.manual_seed(8)
    model = FrequencyResidual('selective')
    x, c = torch.randn(3, 512, 16, 16), torch.rand(3, 2)
    z, weights = model(x, c)
    torch.testing.assert_close(z.square().sum(1), torch.ones(3))
    torch.testing.assert_close(weights, torch.full((3, 4), .25))
    fixed = FrequencyResidual('fixed')
    fixed.blocks.load_state_dict(model.blocks.state_dict())
    torch.testing.assert_close(fixed(x, c)[0], z)


def test_zero_replays_explicit_augmented_ridge_and_support_only():
    rng = np.random.default_rng(8)
    n, s = 91, 64
    phi = rng.normal(size=(n, 11))
    mean = rng.normal(size=(n, 16))
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    prior = rng.normal(size=(n, 8))
    labels = np.arange(s)%8
    conditions = np.zeros((n, 2))
    values = [torch.tensor(a, dtype=torch.float64) for a in (phi, prior, mean, conditions)]
    emb = torch.randn(n, 13, dtype=torch.float64, requires_grad=True)
    actual = support_scores(*values, emb, torch.tensor(labels), eta=.25, beta=0).detach().numpy()
    # Independent primal reference, not the implementation's dual solve.
    aug = np.c_[phi, .5*mean]
    wanted = 2*np.eye(8)[labels]-1-prior[:s]
    coef = np.linalg.solve(aug[:s].T@aug[:s]+8*np.eye(aug.shape[1]), aug[:s].T@wanted)
    np.testing.assert_allclose(actual, prior+aug@coef, atol=1e-10)
    changed = torch.randn_like(emb)
    other = support_scores(*values, changed, torch.tensor(labels), eta=.25, beta=0)
    torch.testing.assert_close(other, torch.tensor(actual))


def test_selective_gate_and_band_processors_receive_query_gradients():
    torch.manual_seed(4)
    model = FrequencyResidual('selective')
    n = 70
    x, c = torch.randn(n, 512, 16, 16), torch.rand(n, 2)
    z, _ = model(x, c)
    score = support_scores(torch.randn(n, 9), torch.randn(n, 8),
                           torch.nn.functional.normalize(torch.randn(n, 10), dim=1),
                           c, z, torch.arange(64)%8, eta=.25)
    score[64:, 0].square().mean().backward()
    assert model.gate[-1].weight.grad.abs().sum() > 0
    assert all(block[0].weight.grad.abs().sum() > 0 for block in model.blocks)
