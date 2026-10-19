"""Role-preserving conditional filter mixtures and content-weighted readout."""
import torch
from torch import nn
from torch.nn import functional as F


class LocalReader(nn.Module):
    def __init__(self,arm,channels=12):
        super().__init__()
        assert arm in ('plain','post','pre','wrong')
        self.arm=arm; self.channels=channels
        self.project=nn.Conv2d(64,channels,1)
        self.filters=nn.Conv2d(channels,channels*3,3,padding=1,groups=channels,bias=False)
        self.gate=nn.Sequential(nn.Linear(channels+6,24),nn.GELU(),nn.Linear(24,channels*3))
        self.attention=nn.Conv2d(channels,1,1)
        self.head=nn.Linear(2*3*channels,8)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def pool(self,z,mass):
        logits=self.attention(z).flatten(1).clamp(-20,20)
        weights=torch.exp(logits-logits.max(1,keepdim=True).values)*mass.flatten(1)
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
        return (z.flatten(2)*weights[:,None]).sum(2)

    def descriptor(self,maps,masks,condition):
        n,p,c,h,w=maps.shape
        projected=self.project(maps.reshape(n*p,c,h,w))
        z=projected[:,None].expand(-1,3,-1,-1,-1).reshape(n*p*3,self.channels,h,w)
        m=masks.reshape(n*p*3,1,h,w)
        context=(z*m).sum((2,3))/m.sum((2,3)).clamp_min(1e-8)
        cond=condition[:,:,None].expand(-1,-1,3,-1).reshape(n*p*3,6)
        if self.arm=='plain': cond=torch.zeros_like(cond)
        gate=self.gate(torch.cat((context,cond),1)).reshape(-1,self.channels,3).softmax(-1)
        # Mask before the new convolution; do not claim to undo frozen stem mixing.
        mass=F.avg_pool2d(m,3,stride=1,padding=1)*9
        values=self.filters(z*m).reshape(-1,self.channels,3,h,w)/mass[:,None].clamp_min(1e-8)
        values=values*9
        if self.arm=='post':
            pooled=torch.stack([self.pool(F.gelu(values[:,:,j]),m) for j in range(3)],-1)
            output=(pooled*gate).sum(-1)
        else:
            local=F.gelu((values*gate[:,:,:,None,None]).sum(2))
            output=self.pool(local,m)
        present=(m.sum((1,2,3))>0).to(output.dtype)
        return (output*present[:,None]).reshape(n,2,3,self.channels)

    def forward(self,maps,masks,condition,drop_role=None):
        features=self.descriptor(maps,masks,condition)
        if drop_role is not None:
            features=features.clone(); features[:,:,drop_role]=0
        return self.head(features.flatten(1))
