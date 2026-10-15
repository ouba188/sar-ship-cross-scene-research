import numpy as np

import receiver_consistent_handoff as method
import run_oof_reliability as old


def test_unchanged_fusion_reproduces_original_features():
    rng = np.random.default_rng(1)
    base, expert = rng.normal(size=(40, 8)), rng.normal(size=(40, 8))
    prior = dict(center=rng.normal(size=4), scale=np.arange(1, 5), lr=rng.normal(size=5))
    expected = old.design(old.confidence_features(base, expert), prior['center'], prior['scale'])@prior['lr']
    np.testing.assert_allclose(method.gate_scores(base, expert, prior), expected, atol=1e-14)


def test_restricted_confidence_ignores_excluded_high_score():
    allowed = np.array([True, False, True, False, False, False, False, False])
    scores = np.array([[0., 100., 0., -1., -1., -1., -1., -1.]])
    margin, entropy = method.confidence_parts(scores, allowed)
    np.testing.assert_array_equal(margin, [0.])
    np.testing.assert_allclose(entropy, [np.log(2.)])


def test_expert_change_updates_score_without_changing_parameters():
    base = np.zeros((2, 8))
    expert = base.copy(); expert[:, 1] = 1.
    prior = dict(center=np.zeros(4), scale=np.ones(4), lr=np.array([1., 0., 0., 0., 0.]))
    first = method.gate_scores(base, expert, prior)
    expert[:, 1] = 3.
    second = method.gate_scores(base, expert, prior)
    np.testing.assert_allclose(first, [1., 1.])
    np.testing.assert_allclose(second, [3., 3.])
