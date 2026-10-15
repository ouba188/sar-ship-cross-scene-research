import numpy as np

from run_handoff_budget_sweep import freeze_budget_draw


def example():
    n = 100
    base = np.zeros(n, dtype=int)
    expert = np.ones(n, dtype=int)
    score = np.arange(n, dtype=float)
    ev = np.ones(n, dtype=bool)
    ev[:4] = False
    return base, expert, score, ev, np.array([0, 1]), np.arange(n), 3


def test_cap_uses_queries_not_support_and_ranks_highest_first():
    pred, selected, eligible = freeze_budget_draw(*example())
    assert np.flatnonzero(selected[1]).tolist() == [96, 97, 98, 99]
    assert selected[0].sum() == 0
    assert selected[2].sum() == 9
    assert not selected[:, :4].any()
    assert np.array_equal(selected[-2], eligible)
    assert np.array_equal(pred[0], np.zeros(100))


def test_same_scores_are_tied_by_original_row_id():
    args = list(example())
    args[2] = np.ones(100)
    args[5] = np.arange(100)[::-1]
    _, selected, _ = freeze_budget_draw(*args)
    assert np.flatnonzero(selected[1]).tolist() == [96, 97, 98, 99]


def test_uncapped_gate_is_not_all_expert():
    args = list(example())
    args[1][20] = 2
    args[2][30] = 0
    args[2][40] = -1
    args[1][50] = 0
    pred, selected, _ = freeze_budget_draw(*args)
    assert not selected[-2, [20, 30, 40, 50]].any()
    assert selected[-1, [20, 30, 40, 50]].all()
    assert pred[-1, 20] == 2
    assert pred[-2, 20] == 0


def test_budget_plateaus_when_candidates_run_out():
    args = list(example())
    args[2][:92] = -1
    _, selected, eligible = freeze_budget_draw(*args)
    assert eligible.sum() == 8
    assert selected[1].sum() == 4
    assert np.array_equal(selected[2], selected[-2])
    assert selected[-1].sum() == 96
