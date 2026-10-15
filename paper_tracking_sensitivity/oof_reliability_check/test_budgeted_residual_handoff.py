import numpy as np

import budgeted_residual_handoff as method


def inputs(n=100):
    logits = np.zeros((n, 8))
    logits[:, 0] = np.linspace(.01, 1, n)
    return logits, np.ones(n, dtype=int), np.linspace(1, 2, n), np.arange(n), np.ones(n, dtype=bool)


def test_caps_support_exclusion_and_nested_masks():
    logits, fusion, score, rows, ev = inputs()
    ev[:10] = False
    masks, caps = method.admission_masks(logits, fusion, score, np.array([1]), rows, ev)
    np.testing.assert_array_equal(caps, [4, 9, 18])
    assert not masks[..., :10].any()
    np.testing.assert_array_equal(masks.sum(-1), np.tile(caps, (2, 1)))
    assert np.all(masks[:, :-1] <= masks[:, 1:])
    np.testing.assert_array_equal(np.flatnonzero(masks[0, 0]), np.arange(96, 100))
    np.testing.assert_array_equal(np.flatnonzero(masks[1, 0]), np.arange(10, 14))


def test_guard_underuse_and_zero_budget():
    logits, fusion, score, rows, ev = inputs(10)
    masks, caps = method.admission_masks(logits, fusion, score, np.array([0]), rows, ev)
    np.testing.assert_array_equal(caps, [0, 1, 2])
    assert not masks[0].any() and not masks[:, 0].any()
    np.testing.assert_array_equal(masks[1].sum(-1), caps)


def test_ties_use_global_row_ids_and_handoff_keeps_equal_admissions():
    rows = np.array([8, 4, 2, 9])
    selected = method.ranked_mask(np.ones(4), np.ones(4, bool), rows, 2)
    np.testing.assert_array_equal(selected, [False, True, True, False])
    base = np.zeros(4, dtype=int)
    experts = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    masks = selected[None, None]
    predictions = method.handoff(base, experts, masks)
    assert masks.sum() == 2
    assert np.count_nonzero(predictions[0, 0, 0] != base) == 0
    assert np.count_nonzero(predictions[0, 0, 1] != base) == 2


def test_base_uncertainty_independent_of_fusion_and_guard():
    logits, fusion, score, rows, ev = inputs()
    first, _ = method.admission_masks(logits, fusion, score, np.array([1]), rows, ev)
    second, _ = method.admission_masks(logits, np.zeros(100, int), -score, np.array([7]), rows, ev)
    np.testing.assert_array_equal(first[1], second[1])


def test_recovery_harm_uses_full_class_denominator():
    truth = np.array([0, 0, 0, 1])
    base = np.array([0, 0, 1, 1])
    prediction = np.array([0, 0, 0, 0])
    result = method.score_policy(prediction, base, truth, np.ones(4, bool))
    assert result['ba'] == .5 and result['acc'] == .75
    np.testing.assert_allclose(result['recover_ba'], 1/6)
    np.testing.assert_allclose(result['harm_ba'], .5)
