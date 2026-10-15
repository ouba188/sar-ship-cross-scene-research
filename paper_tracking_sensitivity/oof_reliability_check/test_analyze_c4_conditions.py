import numpy as np

from analyze_c4_conditions import aggregate_events


def test_event_mass_matches_direct_paired_oa_and_ba():
    y = np.array([0, 0, 0, 1, 1, 2])
    ev = np.array([[1, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1]], dtype=bool)
    rng = np.random.default_rng(7)
    p = rng.integers(0, 3, (2, 4, 6))
    rows = aggregate_events(p, y, ev)
    for i, phase in enumerate(['reference', 'query_c4', 'support_c4', 'matched_c4']):
        oa, ba = [], []
        for d, mask in enumerate(ev):
            correct = p[d, i] == y
            oa.append(correct[mask].mean())
            ba.append(np.mean([correct[mask & (y == c)].mean() for c in np.unique(y[mask])]))
        np.testing.assert_allclose(rows[phase+'_oa_pp'].sum(), 100*np.mean(oa))
        np.testing.assert_allclose(rows[phase+'_ba_pp'].sum(), 100*np.mean(ba))
    np.testing.assert_allclose(rows.support_total_delta_ba_pp.sum(),
        rows.support_c4_ba_pp.sum()-rows.reference_ba_pp.sum())


def test_lost_rescue_excludes_reference_correct_and_support_rows():
    y = np.array([0, 0, 0, 0])
    p = np.array([[[1, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0]]])
    ev = np.array([[True, True, False, True]])
    rows = aggregate_events(p, y, ev, tasks=2)
    np.testing.assert_array_equal(rows.c4_rescue_lost_to_refit_draw_events, [1, 0, 0, 0])
    np.testing.assert_array_equal(rows.source_refit_harm_draw_events, [1, 1, 0, 0])
    np.testing.assert_allclose(rows.oa_weight.sum(), .5)
    assert rows.loc[2, 'query_draws'] == 0
