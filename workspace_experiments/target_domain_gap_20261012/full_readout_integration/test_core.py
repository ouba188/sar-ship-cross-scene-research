import unittest
from core import *


class StatisticsTests(unittest.TestCase):
    def test_joint_and_source_fixed_match_direct_objectives(self):
        rng=np.random.default_rng(461)
        y=np.repeat(np.arange(8),np.arange(4,12))
        hs=np.c_[rng.normal(size=(len(y),9))+1.,np.ones(len(y))]
        yt=np.array([0,0,1,3,3,3,5,6])
        ht=np.c_[rng.normal(size=(len(yt),9)),np.ones(len(yt))]
        stats=source_statistics(hs,y)
        heads,info=fit_variants(stats,ht,yt)
        counts=np.bincount(np.r_[y,yt],minlength=8)
        w1=len(y)/(8.*counts)
        for arm,sw in [('joint',w1[y]),('source_fixed',len(y)/(8.*np.bincount(y)[y]))]:
            h=np.r_[hs,ht]
            w=np.r_[sw,w1[yt]]
            expected=solve(h.T@(w[:,None]*h),h.T@(w[:,None]*Y8[np.r_[y,yt]]))
            np.testing.assert_allclose(heads[arm],expected,atol=1e-12)
        self.assertEqual(info['missing_classes'],[2,4,7])

    def test_equal_class_means_remove_composition_interaction(self):
        y=np.tile(np.arange(8),3)
        hs=np.c_[np.eye(8)[y],np.ones(len(y))]
        stats=source_statistics(hs,y)
        yt=np.array([0,0,1,2,3,4,5,6,7])
        heads,_=fit_variants(stats,np.c_[np.eye(8)[yt],np.ones(len(yt))],yt)
        np.testing.assert_allclose(heads['joint'],heads['TS'],atol=1e-12)
        np.testing.assert_allclose(heads['SS'],heads['ST'],atol=1e-12)


if __name__=='__main__':
    unittest.main()
