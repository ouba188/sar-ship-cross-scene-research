"""Focused regressions for the coupled geometry and support-kernel integration."""
import unittest
import numpy as np
import torch
from local_reader import regions,LocalReader
from train import predict
from common import public


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.d=dict(native_h=85,native_w=85,cx=42.,cy=42.,length=22.,width=6.,ship_axis=.31)

    def test_unsigned_axes(self):
        for arm in ('ship','ship_radar_encoding','joint'):
            a=regions(self.d,.1,.6,arm=arm)
            b=regions({**self.d,'ship_axis':self.d['ship_axis']+np.pi},.1+np.pi,.6,arm=arm)
            np.testing.assert_allclose(a[1],b[1],atol=1e-6)
            np.testing.assert_allclose(a[2],b[2],atol=1e-6)

    def test_conv1_center_alignment(self):
        # A delta kernel at its center locates input pixel (2j,2i) exactly.
        im=torch.arange(112*112,dtype=torch.float32).reshape(1,1,112,112)
        kernel=torch.zeros(1,1,7,7);kernel[0,0,3,3]=1
        result=torch.nn.functional.conv2d(im,kernel,stride=2,padding=3)
        np.testing.assert_array_equal(result.numpy()[0,0],im.numpy()[0,0,::2,::2])
        d={**self.d,'native_w':112,'native_h':112,'cx':20.,'cy':22.,'length':4.,'width':3.,'ship_axis':0.}
        body,*_=regions(d,0,.6)
        self.assertTrue(body[11,10])
        self.assertFalse(body[10,10])

    def test_encoding_does_not_change_content(self):
        a=regions(self.d,0,.6,arm='ship')
        b=regions(self.d,0,.6,arm='ship_radar_encoding')
        np.testing.assert_array_equal(a[1],b[1])
        self.assertGreater(np.abs(a[2]-b[2]).max(),.1)

    def test_joint_radar_changes_sampling(self):
        a=regions(self.d,0,.6,arm='joint')
        b=regions(self.d,.8,.6,arm='joint')
        self.assertGreater(np.abs(a[1]-b[1]).sum(),1.)
        for mask in (a[1],b[1]):
            np.testing.assert_allclose(mask.sum((1,2)),1.,atol=1e-6)

    def test_reader_is_token_permutation_invariant(self):
        torch.manual_seed(3);net=LocalReader()
        body=torch.randn(2,128);tokens=torch.randn(2,12,64);pos=torch.randn(2,12,16)
        pos[:,:,3]=1
        order=torch.randperm(12)
        a,_=net(body,tokens,pos);b,_=net(body,tokens[:,order],pos[:,order])
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)

    def test_full_readout_matches_independent_solver(self):
        rng=np.random.default_rng(4);ns,nq=16,7
        z=rng.normal(size=(ns+nq,12));raw=torch.tensor(rng.normal(size=(ns+nq,32)),dtype=torch.float32)
        y=np.arange(ns)%8;prior=rng.normal(size=(ns+nq,8));base=torch.tensor(np.full((nq,8),1/8))
        root=np.sqrt(ns/(8*np.bincount(y,minlength=8)[y]))
        task={key:torch.tensor(value,dtype=torch.float64) for key,value in
              dict(ss=z[:ns]@z[:ns].T,qs=z[ns:]@z[:ns].T,prior_s=prior[:ns],prior_q=prior[ns:],
                   desired=2*np.eye(8)[y]-1,root=root,old=np.full((nq,8),1/8)).items()}
        task.update(ns=ns,base=base)
        p,h=predict(raw,torch.ones(ns+nq,dtype=torch.bool),task,4.)
        hn=h.numpy();ss=task['ss'].numpy()+4*hn[:ns]@hn[:ns].T
        qs=task['qs'].numpy()+4*hn[ns:]@hn[:ns].T
        logits=prior[ns:]+qs@public.coefficients(ss,prior[:ns],y,8.)
        from scipy.special import softmax
        np.testing.assert_allclose(p.numpy(),.5*(base.numpy()+softmax(logits,axis=1)),atol=1e-12)
        p0,_=predict(raw,torch.zeros(ns+nq,dtype=torch.bool),task,4.)
        np.testing.assert_array_equal(p0.numpy(),task['old'].numpy())


if __name__=='__main__':
    unittest.main()
