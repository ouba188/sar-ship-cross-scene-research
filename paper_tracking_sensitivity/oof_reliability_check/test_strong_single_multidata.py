import numpy as np
import pandas as pd
import torch

from strong_single_r50_task import grouped_split, support_scores


def test_grouped_split_purges_shared_identity_and_preserves_outer_exclusion():
    frame = pd.DataFrame({'port': np.repeat(list('ABCDEFG'), 160),
                          'mmsi': [str(i) for i in range(1120)],
                          'product': [str(i//8) for i in range(1120)]})
    frame.loc[::20, 'mmsi'] = 'shared_vessel'
    allowed = np.flatnonzero(frame.port.to_numpy() != 'G')
    train, val, _ = grouped_split(frame, allowed, validation_ports=2)
    assert set(frame.iloc[train].port).isdisjoint(frame.iloc[val].port)
    assert set(frame.iloc[train].mmsi).isdisjoint(frame.iloc[val].mmsi)
    assert set(frame.iloc[train]['product']).isdisjoint(frame.iloc[val]['product'])
    assert not np.intersect1d(np.r_[train, val], np.flatnonzero(frame.port == 'G')).size


def test_chunked_readout_matches_independent_primal_solution():
    torch.manual_seed(19)
    x = torch.randn(64, 13, dtype=torch.float64)
    q = torch.randn(117, 13, dtype=torch.float64)
    y = torch.arange(64) % 7  # One absent support class must retain its source evidence.
    ps = torch.randn(64, 8, dtype=torch.float64)
    pq = torch.randn(117, 8, dtype=torch.float64)
    counts = torch.bincount(y)
    weights = 64/(7*counts[y].double())
    desired = 2*torch.nn.functional.one_hot(y, 8).double()-1
    beta = torch.linalg.solve(x.T@(weights[:, None]*x)+8*torch.eye(13),
                              x.T@(weights[:, None]*(desired-ps)))
    expected = pq+q@beta
    actual = torch.cat([support_scores(x, y, q[i:i+23], support_prior=ps,
                                      query_prior=pq[i:i+23]) for i in range(0, len(q), 23)])
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(actual[:, 7]).all()
