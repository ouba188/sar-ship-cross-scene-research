"""Synthetic operator checks only; no training, labels or target data are read."""
import time
import unittest

import numpy as np
import torch

from local_baselines import dn4_scores, regional_prototype_scores, balanced_emd_scores

METHODS = (dn4_scores, regional_prototype_scores, balanced_emd_scores)


def empty(n, d=2):
    return np.zeros((n, 2, 20, d), np.float64), np.zeros((n, 2, 20), bool)


class LocalBaselinesTest(unittest.TestCase):
    def test_hand_computed_dn4_class_pool(self):
        q, qv = empty(1)
        b, bv = empty(4)
        q[0, 0, :2] = [[1, 0], [0, 1]]
        qv[0, 0, :2] = True
        b[0, 0, :2], b[1, 0, :2] = [[1, 0], [1, 0]], [[0, 1], [-1, 0]]
        bv[:2, 0, :2] = True
        b[2, 0, :3] = [-1, 0]
        bv[2, 0, :3] = True
        # c0: top3 sums are 2 and 1; c1: -3 and 0. No clipping negatives.
        score = dn4_scores(q, qv, b, bv, np.array([0, 0, 1, 1]))
        np.testing.assert_allclose(score, [[1.5, -1.5]])

    def test_all_eight_references_participate(self):
        q, qv = empty(1)
        b, bv = empty(8)
        q[0, 0, 0], qv[0, 0, 0] = [1, 0], True
        b[:5, 0, 0], b[5:, 0, 0], bv[:, 0, 0] = [-1, 0], [1, 0], True
        np.testing.assert_allclose(dn4_scores(q, qv, b, bv, np.zeros(8, int)), [[3]])

    def test_mask_nan_zero_and_fewer_than_k(self):
        q, qv = empty(1)
        b, bv = empty(1)
        q[0, 0, 0], b[0, 0, 0] = [1, 0], [-1, 0]
        qv[0, 0, 0], bv[0, 0, 0] = True, True
        q[0, 0, 1], b[0, 0, 1] = np.nan, [1, 0]
        q[0, 0, 2], b[0, 0, 2] = np.inf, np.nan
        qv[0, 0, 2:4], bv[0, 0, 2:4] = True, True
        # Masked attractive support, nonfinite tokens and zero vectors cannot vote.
        for method in METHODS:
            with self.subTest(method=method.__name__):
                s, a = method(q, qv, b, bv, np.array([0]), return_available=True)
                np.testing.assert_allclose(s, [[-1]])
                self.assertTrue(a[0, 0])

    def test_equal_polarizations_not_equal_token_counts(self):
        q, qv = empty(1)
        b, bv = empty(1)
        q[0, 0, 0], qv[0, 0, 0] = [1, 0], True
        q[0, 1, :3], qv[0, 1, :3] = [1, 0], True
        b[0, 0, :3], b[0, 1, :3], bv[0, :, :3] = [1, 0], [-1, 0], True
        for method in METHODS:
            with self.subTest(method=method.__name__):
                np.testing.assert_allclose(method(q, qv, b, bv, np.array([0])), [[0]], atol=1e-12)
                no_vh = bv.copy()
                no_vh[:, 1] = False
                expected = 3 if method is dn4_scores else 1
                np.testing.assert_allclose(method(q, qv, b, no_vh, np.array([0])), [[expected]])

    def test_region_mean_and_merged_mean(self):
        q, qv = empty(1)
        b, bv = empty(1)
        q[:, 0], b[:, 0], qv[:, 0], bv[:, 0] = [1, 0], [1, 0], True, True
        b[:, 0, 16:] = [-1, 0]
        np.testing.assert_allclose(regional_prototype_scores(q, qv, b, bv, np.array([0])), [[1 / 3]])
        np.testing.assert_allclose(regional_prototype_scores(q, qv, b, bv, np.array([0]), merge_regions=True), [[1]])
        # Mean direction is normalized after pooling; this is not mean pair cosine.
        qv[:, 0, 1:], bv[:, 0, 2:] = False, False
        b[0, 0, :2] = [[1, 0], [0, 1]]
        np.testing.assert_allclose(regional_prototype_scores(q, qv, b, bv, np.array([0])), [[1 / np.sqrt(2)]])

    def test_uniform_balanced_transport_known_cost(self):
        q, qv = empty(1)
        b, bv = empty(1)
        q[0, 0, :2], qv[0, 0, :2] = [[1, 0], [0, 1]], True
        b[0, 0, :3], bv[0, 0, :3] = [[1, 0], [1, 0], [0, 1]], True
        # Excess e1 bank mass is 2/3 - 1/2 = 1/6, transported at cost 1.
        np.testing.assert_allclose(balanced_emd_scores(q, qv, b, bv, np.array([0])), [[5 / 6]])

    def test_same_ship_exclusion_and_missing_evidence(self):
        q, qv = empty(3)
        b, bv = empty(2)
        q[:2, 0, 0], qv[:2, 0, 0] = [1, 0], True
        b[:, 0, 0], bv[:, 0, 0] = [1, 0], True
        for method in METHODS:
            with self.subTest(method=method.__name__):
                s, a = method(q, qv, b, bv, np.array([0, 1]),
                              query_ids=['a', 'b', 'c'], bank_ids=['a', 'b'],
                              num_classes=3, return_available=True)
                np.testing.assert_array_equal(a, [[False, True, False], [True, False, False], [False, False, False]])
                np.testing.assert_array_equal(s[~a], 0)
                np.testing.assert_allclose(s[a], 1)
                z, az = method(q, np.zeros_like(qv), b, bv, np.array([0, 1]), return_available=True)
                self.assertFalse(az.any())
                np.testing.assert_array_equal(z, 0)

    def test_reference_order_batch_and_numpy_torch_agree(self):
        rng = np.random.default_rng(29)
        q = rng.normal(size=(3, 2, 20, 4))
        b = rng.normal(size=(6, 2, 20, 4))
        qv, bv = rng.random(q.shape[:-1]) > .2, rng.random(b.shape[:-1]) > .25
        y, ids = np.array([0, 1, 0, 1, 0, 1]), np.array(['a', 'b', 'c', 'd', 'e', 'f'])
        order = np.array([5, 0, 3, 2, 4, 1])
        for method in METHODS:
            with self.subTest(method=method.__name__):
                opts = dict(query_ids=['a', 'c', 'g'], return_available=True)
                s, a = method(q, qv, b, bv, y, bank_ids=ids, **opts)
                sp, ap = method(q, qv, b[order], bv[order], y[order], bank_ids=ids[order], batch_size=1, **opts)
                st, at = method(torch.tensor(q), torch.tensor(qv), torch.tensor(b), torch.tensor(bv),
                                torch.tensor(y), bank_ids=ids, batch_size=2, **opts)
                np.testing.assert_allclose(s, sp, atol=1e-12)
                np.testing.assert_array_equal(a, ap)
                np.testing.assert_allclose(s, st.numpy(), atol=1e-12)
                np.testing.assert_array_equal(a, at.numpy())
                self.assertFalse(st.requires_grad)

    def test_empty_query_empty_bank_and_zero_prototype(self):
        q, qv = empty(0)
        b, bv = empty(0)
        for method in METHODS:
            with self.subTest(method=method.__name__):
                self.assertEqual(method(q, qv, b, bv, np.array([], int), num_classes=4).shape, (0, 4))
                q1, qv1 = empty(1)
                s, a = method(q1, qv1, b, bv, np.array([], int), num_classes=4, return_available=True)
                np.testing.assert_array_equal(s, 0)
                self.assertFalse(a.any())
        q, qv = empty(1)
        b, bv = empty(1)
        q[0, 0, 0], qv[0, 0, 0] = [1, 0], True
        b[0, 0, :2], bv[0, 0, :2] = [[1, 0], [-1, 0]], True
        s, a = regional_prototype_scores(q, qv, b, bv, np.array([0]), return_available=True)
        self.assertFalse(a.any())
        np.testing.assert_array_equal(s, 0)


def benchmark():
    """Small synthetic timing, never a target-data benchmark."""
    rng = np.random.default_rng(17)
    q, b = rng.normal(size=(8, 2, 20, 32)).astype('float32'), rng.normal(size=(32, 2, 20, 32)).astype('float32')
    qv, bv, y = np.ones(q.shape[:-1], bool), np.ones(b.shape[:-1], bool), np.repeat(np.arange(4), 8)
    for method in METHODS:
        t = time.perf_counter()
        s = method(q, qv, b, bv, y)
        print(f'{method.__name__}: {time.perf_counter() - t:.4f}s, shape={s.shape}, finite={np.isfinite(s).all()}')


if __name__ == '__main__':
    import sys
    torch.set_num_threads(2)
    if '--benchmark' in sys.argv:
        benchmark()
    else:
        unittest.main()
