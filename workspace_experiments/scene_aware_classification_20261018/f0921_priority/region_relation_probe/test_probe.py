import unittest
import numpy as np
from evaluate import ap_at_r,choose_support,cosine,observed_region_cosine
from prepare import split_regions


class ProbeTests(unittest.TestCase):
    def test_regions_cover_once_and_prioritize_target(self):
        target=np.zeros((8,8),bool);target[2:4,2:4]=True
        other=np.zeros_like(target);other[3:5,3:5]=True
        wc=np.full((8,8),80,dtype=np.uint8);wc[:2]=0
        regions=split_regions(target,other,wc)
        np.testing.assert_array_equal(regions.sum(0),np.ones((8,8)))
        self.assertEqual(regions[0,3,3],1)
        self.assertEqual(regions[1,3,3],0)
        self.assertEqual(regions[3,0,0],1)

    def test_average_precision_examples(self):
        self.assertEqual(ap_at_r(np.array([4.,3.,2.,1.]),np.array([True,True,False,False])),1)
        self.assertEqual(ap_at_r(np.array([4.,3.,2.,1.]),np.array([False,True,False,True])),.25)

    def test_source_draws_do_not_use_queries(self):
        source=np.arange(3,33);labels=np.arange(36)%7
        s=choose_support(source,labels,0)
        self.assertEqual(len(s),16);self.assertEqual(len(set(s)),16)
        self.assertTrue(set(s).issubset(set(source)))
        self.assertEqual(set(labels[s]),set(range(7)))

    def test_query_values_cannot_change_support_kernel(self):
        x=np.random.default_rng(1).normal(size=(8,2,12))
        before=cosine(x,np.arange(4))
        x[4:]*=100
        after=cosine(x,np.arange(4))
        np.testing.assert_allclose(before[:4,:4],after[:4,:4])
        np.testing.assert_allclose(before,before.T,atol=1e-12)

    def test_missing_blocks_do_not_create_a_descriptor(self):
        x=np.ones((2,2,4));x[1,:,1:]=0
        area=np.ones((2,2,4));area[1,:,1:]=0
        original=cosine(x,[0])
        fixed=observed_region_cosine(x,area,[0])
        self.assertAlmostEqual(original[1,1],1)
        self.assertEqual(fixed[1,1],0)

    def test_diagnostic_preserves_fully_observed_features(self):
        x=np.random.default_rng(2).normal(size=(8,2,12))
        area=np.ones((8,2,4))
        np.testing.assert_allclose(observed_region_cosine(x,area,np.arange(4)),cosine(x,np.arange(4)),atol=1e-12)


if __name__=='__main__':unittest.main()
