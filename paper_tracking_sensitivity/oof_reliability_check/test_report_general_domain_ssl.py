import numpy as np

from report_general_domain_ssl import error_flow


def test_rescue_accounting_respects_class_imbalance():
    y = np.array([0, 0, 0, 0, 1])
    a = np.array([0, 0, 0, 0, 0])
    b = np.array([1, 1, 0, 0, 1])
    result = error_flow(y, a, b)
    assert result['rescued'] == 1 and result['harmed'] == 2
    assert result['delta_oa'] == -20.
    assert result['delta_ba'] == 25.
    assert result['rescue_ba'] == 50. and result['harm_ba'] == 25.


def test_no_change_has_no_rescue_or_harm():
    result = error_flow(np.array([0, 1]), np.array([0, 0]), np.array([0, 0]))
    assert all(v == 0 for v in result.values())
