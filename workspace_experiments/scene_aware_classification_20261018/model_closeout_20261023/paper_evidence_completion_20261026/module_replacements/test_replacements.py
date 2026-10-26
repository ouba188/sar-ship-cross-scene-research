"""Only checks replacement semantics and source-only scaling, not local baselines."""
import unittest
import numpy as np
import run_replacements as run


class ReplacementTest(unittest.TestCase):
    def test_valid_area_merge_and_quality(self):
        s = dict(summary1=np.zeros((1, 2, 3, 4, 3)), quality=np.zeros((1, 2, 3, 4)),
                 geometry=np.arange(8).reshape(1, 2, 4), valid=np.array([[True, False]]))
        s['summary1'][0, 0, :, :, :] = np.array([2., 8., 100.])[:, None, None]
        s['quality'][0, 0] = [[1, .5, 4, 1], [3, 1, 8, 1], [100, 1, 100, 0]]
        out = run.merged_summaries(s)
        np.testing.assert_allclose(out['summary1'][0, 0], 6.5)
        np.testing.assert_allclose(out['quality'][0, 0], np.tile([4, .875, 7, 1], (3, 1)))
        np.testing.assert_array_equal(out['summary1'][0, 1], 0)
        np.testing.assert_array_equal(out['quality'][0, 1], 0)
        np.testing.assert_array_equal(out['geometry'], s['geometry'])
        self.assertEqual(s['summary1'][0, 0, 2, 0, 0], 100)

    def test_source_constant_mask_ignores_held_activation(self):
        raw = np.array([[[1., 5.], [3., 5.]], [[100., 999.], [-100., -999.]]])
        sc = run.fit_scaler(raw, np.array([0]))
        np.testing.assert_array_equal(sc['active'], [[[True, False]]])
        x = run.transform(raw, sc)
        np.testing.assert_array_equal(x[..., 1], 0)
        np.testing.assert_allclose(x[0, :, 0], [-1, 1])

    def test_m1_preserves_architecture_mean_and_validity(self):
        rng = np.random.default_rng(17)
        base = rng.normal(size=(2, 4))
        sums = dict(mean3=rng.normal(size=(2, 2, 4)), hb3=rng.normal(size=(2, 2, 4, 2)),
                    valid=np.ones((2, 2), bool))
        both = run.raw_features('m1_both_refit', base, sums)
        mean = run.raw_features('m1_mean_only', base, sums)
        keep, zero = [0, 1, 2, 7, 8, 13, 14], [3, 4, 5, 6, 9, 10, 11, 12]
        np.testing.assert_array_equal(both[..., keep], mean[..., keep])
        np.testing.assert_array_equal(mean[..., zero], 0)
        self.assertEqual(sum(p.numel() for p in run.front.upstream.AttentionHead().parameters()), 266)
        self.assertEqual(sum(p.numel() for p in run.front.upstream.RegionHead().parameters()), 853)


if __name__ == '__main__':
    unittest.main()
