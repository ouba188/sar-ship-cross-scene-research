import unittest
import numpy as np
from sklearn.linear_model import RidgeClassifier
from experiment import source_statistics, increment, integrate, coefficients


class IntegrationTests(unittest.TestCase):
    def test_fixed_source_update_matches_weighted_ridge(self):
        rng=np.random.default_rng(123)
        x=rng.normal(size=(160,7))
        y=np.tile(np.arange(8),20)
        xt=rng.normal(size=(64,7))
        yt=rng.integers(0,6,size=64)
        h=np.c_[x,np.ones(len(x))]
        ht=np.c_[xt,np.ones(len(xt))]
        stats=source_statistics(h,y)
        delta=increment(stats,ht,yt,ht)
        n=np.bincount(y,minlength=8)
        k=np.bincount(yt,minlength=8)
        weights=np.r_[len(y)/(8*n[y]), len(y)/(8*(n+k)[yt])]
        model=RidgeClassifier(alpha=1.).fit(np.r_[x,xt],np.r_[y,yt],sample_weight=weights)
        np.testing.assert_allclose(ht@stats['theta0']+delta,model.decision_function(xt),atol=1e-10)

    def test_cancellation_identity_and_support_permutation(self):
        rng=np.random.default_rng(42)
        features=rng.normal(size=(90,12))
        support=rng.choice(90,64,replace=False)
        k=features@features[support].T+1
        y=rng.integers(0,6,size=64)
        prior=rng.normal(size=(90,8))
        delta=rng.normal(size=(90,8))*.1
        scores=integrate(prior,k,support,y,delta)
        _,inv,counts=np.unique(y,return_inverse=True,return_counts=True)
        root=np.sqrt(64/(len(counts)*counts[inv]))
        removed=k@(root[:,None]*np.linalg.solve(root[:,None]*k[support]*root[None]+8*np.eye(64),root[:,None]*delta[support]))
        np.testing.assert_allclose(scores[1]-scores[0],delta-removed,atol=1e-12)
        np.testing.assert_allclose(scores[3]-scores[0],delta,atol=1e-12)
        perm=rng.permutation(64)
        shuffled=integrate(prior,k[:,perm],support[perm],y[perm],delta)
        np.testing.assert_allclose(scores,shuffled,atol=1e-12)
        zero=integrate(prior,k,support,y,np.zeros_like(delta))
        np.testing.assert_allclose(zero,np.broadcast_to(zero[0],zero.shape),atol=1e-12)


if __name__=='__main__':
    unittest.main()
