"""Distinguish within-cell nonlinear content from neighboring spatial patterns."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'learned_geometry_crosspass_20261019'))
from model import Reader


def region_pool(maps,masks):
    n,p,c,h,w=maps.shape
    mass=F.avg_pool2d(masks,4)
    summed=F.avg_pool2d((maps[:,:,None]*masks[:,None,:,None]).reshape(-1,c,h,w),4)
    value=summed.reshape(n,p,2,c,8,8)/mass[:,None,:,None].clamp_min(1e-8)
    return value.permute(0,1,2,4,5,3).reshape(n,p,128,c),mass.flatten(1)


def scramble_cells(maps,masks,permutation):
    """Jointly permute content and ownership inside each 4x4 feature cell."""
    def reorder(x):
        shape=x.shape
        block=x.reshape(*shape[:-2],8,4,8,4).transpose(-3,-2)
        block=block.flatten(-2)[...,permutation]
        return block.reshape(*shape[:-2],8,8,4,4).transpose(-3,-2).reshape(shape)
    return reorder(maps),reorder(masks)


class ConvReader(Reader):
    def __init__(self,mode):
        super().__init__('A')
        self.mode=mode
        self.pre_norm=nn.LayerNorm(32)
        if mode=='conv_first':
            self.spatial=nn.Conv2d(32,32,3,padding=1,groups=32,bias=False)
            nn.init.zeros_(self.spatial.weight)

    def forward(self,maps,masks,position,geometry,available,drop=None):
        n=len(maps)
        if self.mode=='pool_first':
            tokens,mass=region_pool(maps,masks)
            local=F.gelu(self.pre_norm(self.content(tokens)))
        else:
            projected=self.content(maps.permute(0,1,3,4,2))
            if self.mode=='conv_first':
                spatial=projected.permute(0,1,4,2,3).reshape(-1,32,32,32)
                projected=projected+self.spatial(spatial).reshape(n,2,32,32,32).permute(0,1,3,4,2)
            local=F.gelu(self.pre_norm(projected)).permute(0,1,4,2,3)
            local,mass=region_pool(local,masks)
        h=F.gelu(self.norm(local+self.position(position)[:,None])).reshape(n,2,2,64,32)
        mass=mass.reshape(n,1,2,64).expand(-1,2,-1,-1)
        summary=(h*mass[...,None]).sum((2,3))/mass.sum((2,3))[...,None].clamp_min(1e-8)
        h=self.interact(h,self.context(summary)[:,:,None,None])
        output,weights=self.pool(h,mass)
        if drop is not None:
            kept=weights*(~drop)
            kept=kept/kept.sum(-1,keepdim=True).clamp_min(1e-8)
            output=(kept[...,None]*self.value(h)).sum(-2)
        present=mass.sum(-1)>1e-8
        output=output*present[...,None]*available[:,None,None,None]
        return output.flatten(2),weights
