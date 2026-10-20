import unittest
import numpy as np
import torch
from run import LENGTH, StripResidual, sample_strips, shuffle, strip_layout, symmetric_permutation


class SequenceTests(unittest.TestCase):
    def test_fixed_spacing_and_rotated_geometry(self):
        view={'priors':{'ship_axis':{'center':[32.,31.],'angle':.63,'length_px':28.,'width_px':9.},
                        'radar_kind':'orbit_los','radar_axis':1.2}}
        place=dict(target_present=True,pad_left=48,pad_top=48,crop_x=0,crop_y=0)
        co,v,_=strip_layout(view,place)
        self.assertEqual(co.shape,(2,3,LENGTH,2))
        np.testing.assert_allclose(np.linalg.norm(np.diff(co,axis=2),axis=-1),4,atol=2e-5)
        rotated,v1,_=strip_layout(view,place,1)
        expected=np.stack((co[...,1],159-co[...,0]),-1)
        np.testing.assert_allclose(rotated,expected,atol=2e-5)
        np.testing.assert_array_equal(v,v1)
        view['priors']['ship_axis']['angle']+=np.pi
        reverse,_,_=strip_layout(view,place)
        np.testing.assert_allclose(reverse[0,0],co[0,0,::-1],atol=2e-5)
        np.testing.assert_allclose(reverse[0,1],co[0,2,::-1],atol=2e-5)

    def test_sampling_coordinate_not_resize(self):
        y,x=torch.meshgrid(torch.arange(40.),torch.arange(40.),indexing='ij')
        feature=(4*x+1+2*(4*y+1))[None,None]
        co=torch.zeros(1,2,3,LENGTH,2)
        co[...,0]=42.; co[...,1]=51.
        masks=torch.ones(1,2,160,160)/2
        values,mass=sample_strips(feature,masks,co,torch.ones(1,2,LENGTH))
        torch.testing.assert_close(values,torch.full_like(values,144.),atol=2e-5,rtol=0)
        torch.testing.assert_close(mass,torch.full_like(mass,.5))

    def test_shuffle_is_content_preserving_and_reverse_equivariant(self):
        order=symmetric_permutation(8,71)
        np.testing.assert_array_equal(np.sort(order),np.arange(LENGTH))
        np.testing.assert_array_equal(order[::-1],LENGTH-1-order)
        x=torch.randn(2,2,4,2,3,LENGTH,64); m=torch.rand(2,2,4,2,3,LENGTH,2)
        p=torch.tensor(order)[None,None,None,None].expand(2,2,4,2,-1)
        xx,mm=shuffle(x,m,p)
        torch.testing.assert_close((x*m.sum(-1,keepdim=True)).sum(-2),(xx*mm.sum(-1,keepdim=True)).sum(-2))
        inverse=torch.argsort(p,-1)
        xr,mr=shuffle(xx,mm,inverse)
        torch.testing.assert_close(xr,x); torch.testing.assert_close(mr,m)

    def test_reader_has_no_endpoint_or_side_preference(self):
        torch.manual_seed(91)
        x=torch.randn(2,2,4,2,3,LENGTH,64); m=torch.rand(2,2,4,2,3,LENGTH,2)
        for spatial in (False,True):
            model=StripResidual(spatial)
            self.assertTrue(torch.equal(model(x,m),torch.zeros(2,8)))
            torch.nn.init.normal_(model.output.weight,std=.1)
            expected=model(x,m)
            torch.testing.assert_close(model(x.flip(-2),m.flip(-2)),expected,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(model(x[:,:,:,:,[0,2,1]],m[:,:,:,:,[0,2,1]]),expected,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(model(x,m*0),torch.zeros_like(expected),atol=0,rtol=0)
            model(x,m).square().mean().backward()
            self.assertGreater(model.content.weight.grad.abs().sum().item(),0)


if __name__=='__main__': unittest.main()
