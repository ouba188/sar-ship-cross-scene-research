"""Matched region-conditioned messages on fixed or ship-axis offset pairs."""
import math
import torch
from torch import nn


class RolePairConv(nn.Module):
    def __init__(self, sample_offset, axis=True, early_mean=False, channels=8):
        super().__init__()
        self.sample_offset = sample_offset
        self.axis = axis
        self.early_mean = early_mean
        self.embedding = nn.Embedding(6, 4)
        self.gate = nn.Sequential(nn.Linear(2*channels+23, 12), nn.GELU(), nn.Linear(12, 4))
        self.basis = nn.Parameter(torch.randn(6, 4, 2, channels)*.1)
        self.output = nn.Parameter(torch.zeros(6, channels, channels))
        self.register_buffer('radii', torch.tensor([1., 2., 4., 1., 2., 4.]))

    def message(self, center, neighbor, receiver, sender, geometry, embedding, type_id):
        relation = (receiver[:, :, None]*sender[:, None]).flatten(1, 2).movedim(1, -1)
        inputs = torch.cat((center.movedim(1, -1), neighbor.movedim(1, -1),
                            relation, geometry, embedding), -1)
        weight = self.gate(inputs).softmax(-1)
        filt = torch.einsum('nhwk,nkvc->nvchw', weight, self.basis[type_id])
        return filt[:, 0]*neighbor + filt[:, 1]*(neighbor-center)

    def forward(self, x, regions, valid, theta, type_id):
        role = torch.cat((regions, (1-regions.sum(1, keepdim=True)).clamp_min(0)), 1)
        role = role/role.sum(1, keepdim=True).clamp_min(1e-12)
        n, _, h, w = x.shape
        emb = self.embedding(type_id)[:, None, None].expand(-1, h, w, -1)
        update = torch.zeros_like(x)
        for angle in (0., math.pi/2, math.pi/4, -math.pi/4):
            direction = (theta if self.axis else torch.zeros_like(theta))+angle
            dx, dy = self.radii[type_id]*direction.cos(), self.radii[type_id]*direction.sin()
            vp, vm = self.sample_offset(valid, dx, dy), self.sample_offset(valid, -dx, -dy)
            plus = self.sample_offset(x*valid, dx, dy)/vp.clamp_min(1e-12)
            minus = self.sample_offset(x*valid, -dx, -dy)/vm.clamp_min(1e-12)
            rp = self.sample_offset(role, dx, dy)
            rm = self.sample_offset(role, -dx, -dy)
            relative = direction-theta
            a, b = relative.cos(), relative.sin()
            geo = torch.stack((a.square(), b.square(), a*b), -1)[:, None, None].expand(-1, h, w, -1)
            if self.early_mean:
                pair = self.message(x, (plus+minus)*.5, role, (rp+rm)*.5, geo, emb, type_id)
            else:
                pair = (self.message(x, plus, role, rp, geo, emb, type_id)
                        +self.message(x, minus, role, rm, geo, emb, type_id))*.5
            coverage = valid*(vp > .999).to(x.dtype)*(vm > .999).to(x.dtype)
            update = update+pair*coverage
        delta = torch.einsum('noc,nchw->nohw', self.output[type_id], update/4)
        return x+delta*valid
