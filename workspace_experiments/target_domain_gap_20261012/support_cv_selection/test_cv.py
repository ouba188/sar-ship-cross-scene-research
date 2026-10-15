import unittest
import numpy as np
from run import folds,fold_predict,pick,old,ARMS


class CVTests(unittest.TestCase):
    def test_group_isolation_and_coverage(self):
        y=np.repeat(np.arange(8),8)
        groups=np.array([f'g{i//2}' for i in range(64)])
        count=np.zeros(64,int)
        for train,val in folds(y,groups,20261012):
            self.assertFalse(set(groups[train])&set(groups[val]))
            count[val]+=1
        np.testing.assert_array_equal(count,1)

    def test_fold_uses_only_training_labels_and_order(self):
        rng=np.random.default_rng(85)
        x=rng.normal(size=(160,5))
        stats=old.source_statistics(np.c_[x,np.ones(160)],np.tile(np.arange(8),20))
        h=np.c_[rng.normal(size=(64,5)),np.ones(64)]
        y=np.tile(np.arange(8),8)
        pe=rng.normal(size=(64,8))
        pb=rng.normal(size=(64,8))
        a=rng.normal(size=(64,11))
        b=rng.normal(size=(64,9))
        ke,kb=a@a.T+1,b@b.T+1
        train=np.arange(48)
        p=fold_predict(stats,h,pe,ke,pb,kb,train,y[train])
        shuffled=rng.permutation(train)
        np.testing.assert_allclose(p,fold_predict(stats,h,pe,ke,pb,kb,shuffled,y[shuffled]),atol=1e-12)
        changed=y.copy()
        changed[48:]=(changed[48:]+1)%8
        np.testing.assert_array_equal(p,fold_predict(stats,h,pe,ke,pb,kb,train,changed[train]))
        changed[0]=(changed[0]+1)%8
        self.assertGreater(np.abs(p-fold_predict(stats,h,pe,ke,pb,kb,train,changed[train])).max(),1e-6)

    def test_selection_uses_full_fusion_oa_and_ba_constraint(self):
        rows=[dict(arm=a,oa=50.,ba=40.,nll=1.,macro_nll=1.) for a in ARMS]
        rows[1].update(oa=55.,ba=39.)
        rows[2].update(oa=51.,ba=41.)
        rows[3].update(oa=50.,ba=40.,nll=.8)
        self.assertEqual(pick(rows),'retain_half')
        self.assertEqual(pick(rows,'nll'),'retain_full')


if __name__=='__main__':
    unittest.main()
