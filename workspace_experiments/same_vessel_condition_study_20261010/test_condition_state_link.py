import unittest
import numpy as np
import pandas as pd
from condition_state_link import endpoint_blocks, pair_blocks, fit_predict, context_coordinates


class StateInteractionChecks(unittest.TestCase):
    def test_equal_capacity_and_pair_reversal(self):
        rng = np.random.default_rng(8)
        state, context = rng.normal(size=(90, 10)), rng.normal(size=(90, 32))
        blocks = endpoint_blocks(state, context)
        self.assertEqual([b.shape[1] for b in blocks], [65, 384])
        p = pd.DataFrame(dict(a=[0, 1, 2], b=[3, 4, 5]))
        r = p.rename(columns={'a': 'b', 'b': 'a'})
        x, xr = pair_blocks(blocks, p), pair_blocks(blocks, r)
        for a, b in zip(x, xr):
            np.testing.assert_allclose(a, -b)
        pt, pr, _ = fit_predict(x, xr, np.array([1., -.3, .5]), np.ones(3))
        np.testing.assert_allclose(pt, -pr, atol=1e-12)
        for a, b in zip(pair_blocks(blocks, p, True), pair_blocks(blocks, r, True)):
            np.testing.assert_allclose(a, b)

    def test_projection_does_not_fit_held_rows(self):
        values = np.random.default_rng(9).normal(size=(90, 40))
        train = np.arange(70)
        original = context_coordinates(values, train, 8)
        values[70:] += 1000.
        changed = context_coordinates(values, train, 8)
        np.testing.assert_allclose(original[train], changed[train])

    def test_cross_terms_really_depend_on_state(self):
        state = np.zeros((2, 10))
        context = np.ones((2, 32))
        state[1, 0] = 2.
        cross = endpoint_blocks(state, context)[1][:, 64:]
        np.testing.assert_array_equal(cross[1, :32]-cross[0, :32], np.full(32, 2.))


if __name__ == '__main__':
    unittest.main()
