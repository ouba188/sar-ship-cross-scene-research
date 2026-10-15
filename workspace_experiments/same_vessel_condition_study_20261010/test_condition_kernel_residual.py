import unittest
import torch
from condition_kernel_residual import KernelResidual, score


class KernelResidualTests(unittest.TestCase):
    def test_identity_bound_and_intercept(self):
        model = KernelResidual()
        ph = torch.cat([torch.randn(70, 512), torch.ones(70, 1)], 1)
        context = torch.randn(70, 32)
        torch.testing.assert_close(model(ph, context), ph, atol=0, rtol=0)
        model(ph, context).square().mean().backward()
        self.assertGreater(float(model.map.weight.grad.norm()), 0)
        with torch.no_grad():
            model.map.weight.fill_(10.)
        self.assertLessEqual(float(model.gains(context).abs().max().detach()), .1+1e-7)
        torch.testing.assert_close(model(ph, context)[:, -1], ph[:, -1], atol=0, rtol=0)

    def test_fixed_prior_and_absent_class_scores(self):
        model = KernelResidual()
        ph = torch.cat([torch.randn(70, 512)/512**.5, torch.ones(70, 1)], 1)
        prior = torch.randn(70, 8)
        original = prior.clone()
        labels = torch.arange(64).remainder(3).numpy()
        before = score(ph, prior, labels)
        with torch.no_grad():
            model.map.bias.fill_(.7)
        after = score(model(ph, torch.randn(70, 32)), prior, labels)
        torch.testing.assert_close(prior, original, atol=0, rtol=0)
        torch.testing.assert_close(after[:, 3:], prior[64:, 3:], atol=0, rtol=0)
        self.assertGreater(float((before[:, :3]-after[:, :3]).abs().max().detach()), 1e-5)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
