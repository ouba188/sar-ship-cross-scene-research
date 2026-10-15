import numpy as np
import torch

from budget_conditioned_metric import channel_weights, metric_scores, gate_policy, training_loss
from evidence_aware_handoff import route_scores, predict_gate
from projection_residual_expert import fit_kernel_correction
from run_stable_guided_spatial_pooling import gate_features


def test_positive_unit_mass_metric_and_identity():
    torch.testing.assert_close(channel_weights(torch.zeros(12), 8), torch.ones(12))
    weights = channel_weights(torch.linspace(-9, 9, 12), 8)
    assert weights.min() > 0
    torch.testing.assert_close(weights[:8].mean(), torch.tensor(1.))
    torch.testing.assert_close(weights[8:].mean(), torch.tensor(1.))


def test_identity_metric_replays_original_ridge():
    rng = np.random.default_rng(17)
    retained, complement = rng.normal(size=(30, 6)), rng.normal(size=(30, 12))
    prior, support, labels = rng.normal(size=(30, 4)), np.arange(8), np.arange(8) % 4
    kernel = .5 * retained @ retained[support].T + .5 * complement @ complement[support].T + 1
    expected = prior + kernel @ fit_kernel_correction(kernel[support], labels, prior[support])
    actual = metric_scores(*[torch.tensor(a) for a in (prior, retained, complement, support, labels)],
        torch.zeros(12, dtype=torch.double), 8)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)


def test_support_adaptation_metric_gradient_matches_finite_difference():
    torch.manual_seed(3)
    retained, complement, prior = [torch.randn(*s, dtype=torch.double) for s in ((30, 6), (30, 12), (30, 4))]
    theta = torch.zeros(12, dtype=torch.double, requires_grad=True)
    def loss(t):
        return metric_scores(prior, retained, complement, torch.arange(8), torch.arange(8) % 4, t, 8)[8:].square().mean()
    loss(theta).backward()
    step = torch.zeros(12, dtype=torch.double)
    step[3] = 1e-5
    finite = (loss(theta.detach() + step) - loss(theta.detach() - step)) / 2e-5
    torch.testing.assert_close(theta.grad[3], finite, atol=1e-8, rtol=1e-5)
    assert theta.grad.abs().max() > 1e-6


def test_full_episode_gate_matches_existing_numpy_policy():
    rng = np.random.default_rng(16)
    base, expert = rng.normal(size=(100, 4)), rng.normal(size=(100, 4))
    gate = dict(center=np.zeros(4), scale=np.ones(4), coef=np.array([1., -.2, -.3, .1]), intercept=np.array([1.]))
    support_labels, query, rows = np.arange(8) % 4, np.arange(100) >= 8, rng.permutation(100)
    score = predict_gate(gate, gate_features(base, expert))
    expected = route_scores(base, expert, score, support_labels, query, rows)
    selected, actual_score = gate_policy(torch.tensor(base), torch.tensor(expert),
        {k: torch.tensor(v) for k, v in gate.items()}, torch.tensor(support_labels), torch.tensor(query), torch.tensor(rows))
    np.testing.assert_array_equal(selected.numpy(), expected['selected'])
    np.testing.assert_allclose(actual_score.numpy(), score, atol=1e-12)


def test_budget_loss_preserves_query_class_weights_and_excludes_support():
    torch.manual_seed(8)
    scores = torch.randn(10, 3, dtype=torch.double, requires_grad=True)
    y = torch.tensor([0, 1, 2, 0, 0, 0, 0, 1, 1, 2])
    query = torch.arange(10) >= 3
    selected = torch.zeros(10, dtype=torch.bool)
    selected[[4, 9]] = True
    losses = torch.nn.functional.cross_entropy(scores, y, reduction='none')
    full = (losses[3:7].mean() + losses[7:9].mean() + losses[9]) / 3
    tail = (losses[4] / 4 + losses[9]) / 1.25
    actual = training_loss(scores, y, query, selected, True)
    torch.testing.assert_close(actual, .5 * (full + tail))
    actual.backward()
    assert not scores.grad[:3].any()
    torch.testing.assert_close(training_loss(scores, y, query, selected & False, True), full)
