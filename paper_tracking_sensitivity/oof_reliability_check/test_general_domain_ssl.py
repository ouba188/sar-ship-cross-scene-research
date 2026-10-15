import numpy as np
import torch

from general_domain_ssl import augment, nt_xent, rgb


def test_polarizations_and_pair_mean_preserved():
    pair = torch.tensor([[[[.2, .3]], [[.7, .9]]]])
    result = rgb(pair)
    torch.testing.assert_close(result[:, :2], pair)
    torch.testing.assert_close(result[:, 2], pair.mean(1))
    assert not torch.equal(result[:, 0], result[:, 1])


def test_nt_xent_positive_order_and_gradients():
    positives = torch.eye(4)
    good = torch.cat((positives, positives)).requires_grad_()
    bad = torch.cat((positives, positives.roll(1, 0)))
    assert nt_xent(good) < nt_xent(bad)
    nt_xent(good).backward()
    assert torch.isfinite(good.grad).all()
    np.testing.assert_allclose(float(nt_xent(torch.ones(8, 4))), np.log(7), atol=1e-6)


def test_augmentation_aligns_polarization_geometry():
    # Identical VV/VH stay identical before the fixed channel normalization.
    a = torch.rand(4, 1, 128, 128)
    z = augment(a.repeat(1, 2, 1, 1))
    image = z*torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)+torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    torch.testing.assert_close(image[:, 0], image[:, 1], atol=1e-6, rtol=0)
    torch.testing.assert_close(image[:, 0], image[:, 2], atol=1e-6, rtol=0)
    assert image.shape == (4, 3, 112, 112)
