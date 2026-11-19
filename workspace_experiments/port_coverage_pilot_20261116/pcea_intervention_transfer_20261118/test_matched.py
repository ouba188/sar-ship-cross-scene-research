import unittest
import torch
from objective import matched_plan, matched_auxiliary, fulfillment


class MatchedTests(unittest.TestCase):
    def data(self):
        y = torch.tensor([0,0,0,1,2,2])
        clean = torch.zeros(6,3)
        clean[torch.arange(6),y] = -.2
        gain = torch.tensor([[0,.02,.08],[0,.07,.04],[0,.05,.01],
                             [.03,0,.09],[.04,.06,0],[.08,.02,0]])
        weights = torch.ones(6,3)/2
        weights[torch.arange(6),y] = 0
        t = dict(use=torch.tensor([1,1,1,1,1,0],dtype=torch.bool),
                 clean=clean, relation_gain=gain, competitor_weights=weights)
        active = torch.ones(6,dtype=torch.bool)
        t.update(matched_plan(t,y,active,17))
        return y,t,active

    def test_global_class_donors_and_singletons(self):
        y,t,active = self.data()
        self.assertEqual(t['matched'].tolist(),[True,True,True,False,False,False])
        keep = t['matched']
        self.assertTrue(torch.equal(y[keep],y[t['donor'][keep]]))
        self.assertTrue(bool((t['donor'][keep] != torch.where(keep)[0]).all()))
        self.assertTrue(torch.equal(t['measured'][keep].sum(0),t['shuffled'][keep].sum(0)))
        for key in ('measured','shuffled','fixed'):
            self.assertEqual(float(t[key][~keep].abs().sum()),0.)
            self.assertEqual(float(t[key].gather(1,y[:,None]).abs().sum()),0.)
        other = matched_plan(t,y,active,17)
        self.assertTrue(torch.equal(other['donor'],t['donor']))

    def test_analytic_loss_and_detachment(self):
        y = torch.tensor([0])
        teacher = torch.tensor([[-.2,0.,-.4]],requires_grad=True)
        off = teacher.detach().clone().requires_grad_()
        on = teacher.detach().clone().requires_grad_()
        t = dict(clean=teacher,use=torch.tensor([True]),matched=torch.tensor([True]),
                 measured=torch.tensor([[0.,.08,.03]],requires_grad=True),
                 competitor_weights=torch.tensor([[0.,.75,.25]],requires_grad=True))
        loss,_ = matched_auxiliary(on,off,y,torch.ones(1),t,torch.tensor([True]),'measured')
        self.assertAlmostEqual(float(loss.detach()),.0135,places=6)
        loss.backward()
        self.assertIsNone(off.grad)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(t['measured'].grad)
        self.assertIsNone(t['competitor_weights'].grad)
        self.assertLess(float(on.grad[0,0]),0)
        self.assertGreater(float(on.grad[0,1]),0)

    def test_eligible_count_is_not_active_count(self):
        y,t,active = self.data()
        _,stats = matched_auxiliary(t['clean'],t['clean'],y,torch.ones(6),t,active,'measured')
        self.assertEqual(stats['active'],6)
        self.assertEqual(stats['eligible'],5)
        self.assertEqual(stats['selected'],3)

    def test_no_gain_still_protected_and_inactive_is_zero(self):
        y = torch.tensor([0])
        t = dict(clean=torch.tensor([[.1,0.,-.4]]),use=torch.tensor([False]),
            matched=torch.tensor([False]),measured=torch.zeros(1,3),
            competitor_weights=torch.tensor([[0.,.5,.5]]))
        z = torch.tensor([[0.,.1,-.4]],requires_grad=True)
        for mode in ('protect','measured'):
            loss,stats = matched_auxiliary(z,z,y,torch.ones(1),t,torch.tensor([True]),mode)
            self.assertAlmostEqual(float(loss.detach()),.04,places=6)
            self.assertEqual(stats['selected'],0)
        for mode in ('none','protect','measured'):
            loss,_ = matched_auxiliary(z,z,y,torch.ones(1),t,torch.tensor([False]),mode)
            self.assertEqual(float(loss),0.)
        loss,_ = matched_auxiliary(z,z,y,torch.ones(1),t,torch.tensor([True]),'none')
        self.assertEqual(float(loss),0.)

    def test_not_confusing_off_damage_with_on_gain(self):
        y,t,active = self.data()
        on = t['clean'].clone()
        off = on.clone()
        off[torch.arange(len(y)),y] -= .3
        f = fulfillment(on,off,y,t,active)
        self.assertTrue(torch.allclose(f['on_minus_off'],torch.full((6,),.3)))
        self.assertEqual(float(f['on_minus_teacher'].abs().max()),0.)

    def test_probe_and_actual_supervision_are_separate(self):
        y,t,active = self.data()
        for mode in ('none','protect'):
            f = fulfillment(t['clean'],t['clean'],y,t,active,mode)
            self.assertEqual(int(f['requested_relations'].sum()),0)
            self.assertGreater(int(f['measured_probe_relations'].sum()),0)
        f = fulfillment(t['clean'],t['clean'],y,t,active,'measured')
        self.assertEqual(int(f['requested_relations'].sum()),6)
        self.assertEqual(int(f['requested_relations'][~t['matched']].sum()),0)


if __name__ == '__main__':
    unittest.main()
