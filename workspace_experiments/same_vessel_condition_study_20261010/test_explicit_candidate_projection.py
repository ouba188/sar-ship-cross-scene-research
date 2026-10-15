import unittest
import numpy as np
import pandas as pd
import torch
from explicit_candidate_projection import CandidateProjector, identity_episode
from target_response_adapter import residual_head


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)
        self.x = torch.randn(96,40)
        self.y = torch.arange(96)%8
        self.model = CandidateProjector(self.x,self.y,np.ones(96),dimensions=12)

    def test_orthogonal_decomposition_and_nontrivial_variance(self):
        q,scale = self.model.geometry()
        torch.testing.assert_close(q.T@q,torch.eye(12),atol=1e-6,rtol=1e-5)
        z = self.model.encode(self.x)
        torch.testing.assert_close(z.var(0,unbiased=False),torch.full((12,),1/12),atol=1e-6,rtol=1e-4)
        component = (z*scale*np.sqrt(12))@q.T
        residual = self.x-self.model.center-component
        torch.testing.assert_close(residual@q,torch.zeros(96,12),atol=3e-6,rtol=0)
        torch.testing.assert_close(component+residual,self.x-self.model.center)

    def test_joint_readout_gradient_reaches_projection(self):
        z,p = self.model(self.x)
        phi = torch.cat([z,torch.ones(len(z),1)],1)
        beta,_,_ = residual_head(phi[:64],p[:64],self.y[:64],balanced=True)
        query = p[64:]+phi[64:]@beta
        loss = torch.nn.functional.cross_entropy(query,self.y[64:])
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.frame.grad).all())
        self.assertGreater(float(self.model.frame.grad.norm()),0)
        self.assertGreater(float(self.model.head.weight.grad.norm()),0)

    def test_exported_geometry_reproduces_vectors_and_checkpoint(self):
        q,scale = self.model.geometry()
        expected = (self.x-self.model.center)@q/scale/np.sqrt(12)
        torch.testing.assert_close(self.model.encode(self.x),expected)
        replacement = CandidateProjector(self.x,self.y,np.ones(96),dimensions=12)
        replacement.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(replacement.encode(self.x),expected)

    def test_episode_excludes_support_identity_not_just_chip(self):
        meta = pd.DataFrame(dict(mmsi=np.repeat(np.arange(90),3)))
        s,q = identity_episode(meta,np.random.default_rng(7),64,96)
        self.assertEqual(len(np.unique(meta.iloc[s].mmsi)),64)
        self.assertFalse(set(meta.iloc[s].mmsi)&set(meta.iloc[q].mmsi))
        s2,q2 = identity_episode(meta,np.random.default_rng(7),64,96)
        np.testing.assert_array_equal(s,s2); np.testing.assert_array_equal(q,q2)


if __name__ == '__main__':
    unittest.main()
