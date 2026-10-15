import numpy as np

from report_conditional_direction import event_stats


def test_error_flow_distinguishes_c4_retention_and_new_repairs():
    truth = np.array([0, 0, 0, 1, 1, 1])
    original = np.array([1, 0, 1, 1, 0, 0])
    baseline = np.array([0, 1, 1, 1, 1, 1])
    candidate = np.array([1, 0, 0, 1, 1, 0])
    ev = np.array([True, True, True, True, True, False])
    result = event_stats(truth, candidate, baseline, original, ev)
    assert result['n'] == 5
    assert result['rescues'] == 2 and result['harms'] == 1
    assert result['c4_rescues'] == 2 and result['c4_rescues_destroyed'] == 1
    assert result['c4_harms'] == 1 and result['c4_harms_repaired'] == 1
    assert result['changes'] == 3
    assert result['oa'] == 80
    np.testing.assert_allclose(result['ba'], 100*(2/3+1)/2)
