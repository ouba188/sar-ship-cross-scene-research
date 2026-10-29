"""Focused loss checks: shared-bias invariance and useful rank gradients."""
import torch
from experiment import paired_loss

torch.manual_seed(17)
z = torch.randn(8,4,dtype=torch.float64,requires_grad=True)
y = torch.tensor([0,1,2,3,0,3,1,2])
bias = torch.tensor([[3.,-4.,1.,6.]],dtype=torch.float64)
assert torch.allclose(paired_loss(z,y),paired_loss(z+bias,y),atol=1e-12)
g, = torch.autograd.grad(paired_loss(z,y),z)
assert torch.allclose(g.sum(0),torch.zeros(4,dtype=torch.float64),atol=1e-12)
better=z.detach().clone()
better[torch.arange(8),y] += .1
assert paired_loss(better,y) < paired_loss(z,y)
swapped=torch.arange(8).reshape(-1,2).flip(1).flatten()
assert torch.allclose(paired_loss(z[swapped],y[swapped]),paired_loss(z,y),atol=1e-12)
print('PASS: shared-bias invariance, cancelling common gradient, correct rank direction, pair reversal')
