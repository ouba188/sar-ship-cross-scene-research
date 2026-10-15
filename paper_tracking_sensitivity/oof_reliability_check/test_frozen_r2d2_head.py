import unittest
import torch
import frozen_r2d2_head as module
from frozen_r2d2_head import meta_scores


class MetaHeadTests(unittest.TestCase):
    def test_official_normalized_formula(self):
        torch.manual_seed(7)
        x = torch.randn(12, 4, dtype=torch.float64)
        s = torch.arange(6)
        y = torch.tensor([0, 1, 1, 2, 4, 7])
        ll, ls = torch.tensor(-2.,dtype=torch.float64), torch.tensor(.3,dtype=torch.float64)
        module.NORMALIZED = True
        try:
            actual = meta_scores(x,s,y,ll,ls)
        finally:
            module.NORMALIZED = False
        aug = torch.cat([x, torch.ones((12,1), dtype=x.dtype)],1)
        a = aug[s]/(6**.5)
        b = torch.nn.functional.one_hot(y,8).double()/(6**.5)
        w = torch.linalg.solve(a.T@a+ll.exp()*torch.eye(5), a.T@b)
        torch.testing.assert_close(actual,ls.exp()*aug@w)

    def test_scores_and_gradients(self):
        torch.manual_seed(5)
        x = torch.randn(11, 4, dtype=torch.float64)
        y = torch.tensor([0, 1, 1, 3, 5])
        s = torch.arange(5)
        ll = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
        ls = torch.tensor(-.1, dtype=torch.float64, requires_grad=True)
        scores = meta_scores(x, s, y, ll, ls)
        targets = torch.nn.functional.one_hot(y, 8).double()
        w = torch.linalg.solve(x[s].T@x[s]+ll.exp()*torch.eye(4), x[s].T@targets)
        torch.testing.assert_close(scores, ls.exp()*x@w)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b: meta_scores(x,s,y,a,b), (ll,ls)))


if __name__ == '__main__':
    unittest.main()
