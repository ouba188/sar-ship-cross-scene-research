"""Checks for the specific area-merging and missing-region failure modes."""
import unittest
import numpy as np
from controls import merged_regions, bridge


class PoolingTests(unittest.TestCase):
    def test_merge_matches_direct_pixels(self):
        image = np.arange(64,dtype=float).reshape(8,8)**2
        target = np.zeros((8,8),bool)
        target[1:7,3:5] = True
        target[1:3,1:3] = True
        other = np.zeros_like(target)
        other[6:,6:] = True
        regions = [target,other,~(target|other)]
        tokens, masses = [], []
        for r,c in [(0,0),(0,4),(4,0),(4,4)]:
            cell = np.zeros_like(target)
            cell[r:r+4,c:c+4] = True
            for region in regions:
                mask = cell & region
                masses.append(mask.sum())
                tokens.append(image[mask].mean() if mask.any() else 0.)
        merged, valid = merged_regions(np.array(tokens)[None,None,:,None],np.array(masses)[None])
        np.testing.assert_allclose(merged[0,0,:,0],[image[target].mean(),image[~target].mean()])
        self.assertTrue(valid.all())
        self.assertNotAlmostEqual(float(np.mean([t for t,m in zip(tokens[::3],masses[::3]) if m])),float(image[target].mean()))

    def test_absent_target_stays_zero(self):
        tokens = np.ones((3,2,12,4))
        masses = np.ones((3,12));masses[:,::3] = 0
        pooled, valid = merged_regions(tokens,masses)
        z,_ = bridge.token_descriptor(pooled,valid,[0,1])
        np.testing.assert_array_equal(pooled[:,:,0],0)
        np.testing.assert_array_equal(z[:,:,0],0)
        self.assertFalse(valid[:,0].any())

    def test_query_cannot_change_support_centering(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(5,2,2,4));valid=np.ones((5,2),bool)
        before,_ = bridge.token_descriptor(x,valid,[0,1,2])
        x[3:] *= 100
        after,_ = bridge.token_descriptor(x,valid,[0,1,2])
        np.testing.assert_array_equal(before[:3],after[:3])


if __name__=='__main__':
    unittest.main()
