"""Detect perfect ranking with a shifted conditional distribution."""

import numpy as np

from run_conditional_invariance_probe import QUANTILES, pair_readout, source_statistics


def test_shifted_perfect_ranking_is_not_distributional_invariance():
    noise = np.linspace(-.5, .5, 100)
    z = np.r_[2+noise, -2+noise]
    y = np.repeat([0, 1], 100)
    args = (y, 0, 1, 2., -2., noise.var(), noise.var(),
            np.quantile(z[:100], QUANTILES), np.quantile(z[100:], QUANTILES))
    unchanged = pair_readout(z, *args)
    shifted = pair_readout(z+5., *args)
    assert unchanged['auc'] == shifted['auc'] == 1.
    assert unchanged['conditional_w1'] == 0.
    assert shifted['conditional_w1'] > 10.
    assert unchanged['fixed_threshold_ba'] == 1.
    assert shifted['fixed_threshold_ba'] == .5


def test_class_statistics_balance_ports_not_sample_counts():
    z = np.r_[np.full(5, -1.), np.zeros(5), np.ones(100)][:, None]
    y = np.zeros(len(z), dtype=int)
    domains = np.repeat(['a', 'b', 'c'], [5, 5, 100])
    stats = source_statistics(z, y, domains)
    np.testing.assert_allclose(stats['class_mean'][0], 0., atol=1e-12)
    assert stats['class_domains'][0] == 3
