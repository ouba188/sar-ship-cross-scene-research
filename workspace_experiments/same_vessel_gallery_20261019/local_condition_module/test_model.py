import torch
from model import LocalReader

torch.set_num_threads(2)
torch.manual_seed(7)
maps=torch.randn(2,2,64,8,8)
masks=torch.zeros(2,2,3,8,8); masks[:,:,0,:,:4]=1; masks[:,:,2,:,4:]=1
condition=torch.randn(2,2,6)
counts=[]
for arm in ('plain','post','pre','wrong'):
    net=LocalReader(arm); counts.append(sum(p.numel() for p in net.parameters()))
    assert torch.equal(net(maps,masks,condition),torch.zeros(2,8))
    d=net.descriptor(maps,masks,condition)
    assert torch.equal(d[:,:,1],torch.zeros_like(d[:,:,1]))
    altered=maps.clone(); altered[:,:,:,:,4:]+=50
    torch.testing.assert_close(net.descriptor(altered,masks,condition)[:,:,0],d[:,:,0])
    if arm=='pre':
        condition.requires_grad_(True)
        net.descriptor(maps,masks,condition).square().sum().backward()
        assert condition.grad.abs().sum()>0
assert len(set(counts))==1
print('PASS: exact zero residual, empty role, role isolation, condition gradients, equal parameters',counts[0])
