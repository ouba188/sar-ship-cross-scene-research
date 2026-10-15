import numpy as np
import torch

from episodic_spatial_pooling import EpisodicPool, adapted_scores, balanced_ce, balanced_subset
from projection_residual_expert import fit_kernel_correction


def data():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(30, 12))
    support = np.arange(8)
    kernel = x @ x[support].T / 12 + 1
    prior = rng.normal(size=(30, 4))
    labels = np.arange(8) % 4
    return rng, support, kernel, prior, labels


def test_zero_kernel_replays_original_numpy_solver():
    _, support, kernel, prior, labels = data()
    result = adapted_scores(torch.tensor(prior), torch.tensor(kernel),
        torch.zeros(30, 5), torch.tensor(support), torch.tensor(labels)).numpy()
    expected = prior + kernel @ fit_kernel_correction(kernel[support], labels, prior[support])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_augmented_kernel_matches_explicit_numpy_solver():
    rng, support, kernel, prior, labels = data()
    z = rng.normal(size=(30, 5))
    augmented = kernel + .25 * z @ z[support].T / 5
    expected = prior + augmented @ fit_kernel_correction(augmented[support], labels, prior[support])
    result = adapted_scores(torch.tensor(prior), torch.tensor(kernel), torch.tensor(z),
        torch.tensor(support), torch.tensor(labels)).numpy()
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_initial_attention_equals_fixed_pool_and_has_live_gradient():
    torch.manual_seed(4)
    model = EpisodicPool(12)
    tokens = torch.randn(30, 4, 12)
    tokens -= tokens.mean(1, keepdim=True)
    logits = torch.randn(30, 4).abs()
    features, alpha = model(tokens, logits, torch.ones(12), torch.ones(12))
    torch.testing.assert_close(alpha, logits.softmax(1))
    _, support, kernel, prior, labels = data()
    scores = adapted_scores(torch.tensor(prior), torch.tensor(kernel), features,
        torch.tensor(support), torch.tensor(labels))
    loss = balanced_ce(scores[8:], torch.arange(22) % 4)
    loss.backward()
    assert torch.isfinite(model.output.weight.grad).all()
    assert model.output.weight.grad.abs().max() > 1e-7


def test_gradient_through_support_solve_matches_finite_difference():
    rng, support, kernel, prior, labels = data()
    z = torch.tensor(rng.normal(size=(30, 5)), requires_grad=True)
    def loss(v):
        return adapted_scores(torch.tensor(prior), torch.tensor(kernel), v,
            torch.tensor(support), torch.tensor(labels))[8:].square().mean()
    loss(z).backward()
    epsilon = 1e-5
    plus, minus = z.detach().clone(), z.detach().clone()
    plus[2, 1] += epsilon
    minus[2, 1] -= epsilon
    finite = (loss(plus) - loss(minus)) / (2 * epsilon)
    torch.testing.assert_close(z.grad[2, 1], finite, atol=1e-8, rtol=1e-5)


def test_query_sampler_never_includes_support_and_balances_present_classes():
    labels = np.arange(80) % 4
    sampled = balanced_subset(labels, np.arange(8, 80), 5, np.random.default_rng(7))
    assert len(np.unique(sampled)) == 20
    assert sampled.min() >= 8
    np.testing.assert_array_equal(np.bincount(labels[sampled]), np.full(4, 5))
