import unittest
import torch
from causal_split import (PolarChannelSplit, donor_permutation, combine,
    swap_consistency, bias_prior_loss, effective_prior)


class SplitTests(unittest.TestCase):
    def test_complement_capacity_and_gradient(self):
        layer = PolarChannelSplit(12, .5)
        x = torch.randn(3, 2, 12, 5, 5, requires_grad=True)
        c, b = layer(x)
        self.assertTrue(torch.equal(c+b, x))
        self.assertEqual(float((c*b).abs().max()), 0.)
        self.assertTrue(torch.equal(layer.mask().sum(-1), torch.tensor([6., 6.])))
        c.square().sum().backward()
        self.assertGreater(float(layer.scores.grad.norm()), 0.)
        layer.eval()
        self.assertTrue(torch.equal(c, layer(x)[0]))

    def test_shared_coordinate_mask_and_no_polar_mixing(self):
        layer = PolarChannelSplit(6)
        x = torch.ones(4, 2, 6, 3, 3)
        c, b = layer(x)
        self.assertTrue(torch.equal(c[0], c[-1]))
        self.assertTrue(torch.equal(c[..., 0, 0], c[..., 2, 2]))
        self.assertEqual(float((c[0]*b[1]).abs().sum()), 0.)

    def test_donors_eligibility_singletons(self):
        av = torch.tensor([[1,1],[1,1],[1,0],[1,0],[0,1],[0,0]]).bool()
        donor, ok = donor_permutation(av, torch.Generator().manual_seed(7))
        self.assertEqual(ok.tolist(), [True,True,True,True,False,False])
        self.assertTrue(torch.equal(av[ok], av[donor[ok]]))
        self.assertTrue(bool((donor[ok] != torch.arange(6)[ok]).all()))

    def test_swap_does_not_modify_causal_coordinates_or_padding(self):
        c = torch.zeros(2, 2, 2, 4, 4)
        c[:,:,0] = 3
        b = torch.zeros_like(c)
        b[0,:,1], b[1,:,1] = 5, 7
        valid = torch.ones(2,2,1,4,4)
        valid[0,:,:,:2] = 0
        result = combine(c,b,torch.tensor([1,0]),valid,torch.ones(2,2))
        self.assertTrue(torch.equal(result[:,:,0], c[:,:,0]))
        self.assertEqual(float(result[0,0,1,3,0]), 7.)
        self.assertEqual(float(result[1,0,1,3,0]), 5.)
        self.assertEqual(float(result[:,:,:,0].sub(c[:,:,:,0]).abs().max()), 0.)

    def test_loss_targets_detached_and_empty_swap_safe(self):
        original = torch.randn(3,4,requires_grad=True)
        changed = torch.randn(3,4,requires_grad=True)
        zero = swap_consistency(changed,original,torch.zeros(3,dtype=torch.bool))
        self.assertEqual(float(zero), 0.)
        loss = swap_consistency(changed,original,torch.tensor([True,False,True]))
        loss.backward()
        self.assertIsNone(original.grad)
        self.assertEqual(float(changed.grad[1].abs().sum()), 0.)
        self.assertGreater(float(changed.grad[0].norm()), 0.)

    def test_source_prior_not_batch_one_hot(self):
        prior = effective_prior(torch.tensor([0,0,0,1]),torch.tensor([1.,1.,1.,3.]),2)
        self.assertTrue(torch.equal(prior,torch.tensor([.5,.5])))
        z = torch.tensor([[2.,0.]],requires_grad=True)
        bias_prior_loss(z,prior,torch.ones(1,2,dtype=torch.bool)).backward()
        self.assertGreater(float(z.grad[0,0]),0.)

    def test_full_capacity_replays(self):
        layer = PolarChannelSplit(6,1.)
        x = torch.randn(2,2,6,4,4)
        c,b = layer(x)
        self.assertTrue(torch.equal(x,c))
        self.assertEqual(float(b.abs().sum()),0.)

    def test_checkpoint_preserves_capacity(self):
        first,second = PolarChannelSplit(8,.75),PolarChannelSplit(8,.5)
        second.load_state_dict(first.state_dict(),strict=True)
        self.assertEqual(second.keep,6)
        self.assertTrue(torch.equal(first.mask(),second.mask()))


if __name__ == '__main__':
    unittest.main()
