"""Capacity-matched content readouts and differentiable full residual adaptation."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Reader(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.content = nn.Linear(256,32)
        self.position = nn.Linear(8,32,bias=False)
        self.norm = nn.LayerNorm(32)
        self.context = nn.Linear(32,4)
        self.gate = nn.Linear(4,32)
        self.mix = nn.Linear(32,32)
        self.attention = nn.Linear(32,1)
        self.value = nn.Linear(32,32)

    def interact(self, h, condition):
        return F.gelu(self.mix(h*(1+torch.tanh(self.gate(condition)))))

    def pool(self, h, mass):
        logits = self.attention(h).squeeze(-1)+torch.log(mass.clamp_min(1e-12))
        w = torch.softmax(logits.masked_fill(mass<=1e-8,-1e4),dim=-1)
        w = w*(mass>1e-8)
        return (w[...,None]*self.value(h)).sum(-2),w

    def forward(self, content, position, mass, geometry, available):
        n = len(content)
        h = F.gelu(self.norm(self.content(content)+self.position(position)[:,None]))
        h = h.reshape(n,2,2,64,32)
        mass = mass.reshape(n,1,2,64).expand(-1,2,-1,-1)
        summary = (h*mass[...,None]).sum((2,3))/mass.sum((2,3))[...,None].clamp_min(1e-8)
        condition = self.context(summary)
        if self.arm != 'A':
            condition = condition+geometry[:,None]
        condition = condition[:,:,None,None]
        if self.arm == 'B':
            pooled, weights = self.pool(h,mass)
            output = self.interact(pooled,condition.squeeze(3))
        else:
            h = self.interact(h,condition)
            output,weights = self.pool(h,mass)
        present = mass.sum(-1)>1e-8
        output = output*present[...,None]*available[:,None,None,None]
        return output.flatten(2),weights


def centered(raw, available, support_count):
    ok = available[:support_count]
    mean = (raw[:support_count]*ok[:,None,None]).sum(0)/ok.sum().clamp_min(1)
    z = F.normalize(raw-mean,dim=-1)*available[:,None,None]
    return z.flatten(1)/(raw.shape[1]**.5)


def residual_probability(kernel, prior, expert, support_labels, ns):
    support_labels = support_labels.long()
    counts = torch.bincount(support_labels,minlength=7)
    weights = ns/((counts>0).sum()*counts[support_labels].to(kernel.dtype))
    root = weights.sqrt()
    target = 2*F.one_hot(support_labels,7).to(kernel.dtype)-1
    ss = root[:,None]*kernel[:ns,:ns]*root[None,:]+8*torch.eye(ns,device=kernel.device,dtype=kernel.dtype)
    alpha = root[:,None]*torch.linalg.solve(ss,root[:,None]*(target-prior[:ns]))
    score = prior+kernel[:,:ns]@alpha
    prob = .5*score.softmax(-1)+.5*expert.softmax(-1)
    return prob[ns:],alpha,score


def substitute_geometry(geometry, available, donors, rng):
    """Complete relative-angle/incidence tuples; never transplant a map azimuth."""
    g = geometry.copy()
    ids = np.flatnonzero(available)
    choice = rng.choice(donors,size=len(ids),replace=True)
    # Avoid self-donations where possible, without consulting classes.
    for j,i in enumerate(ids):
        if choice[j] == i and len(donors)>1:
            choice[j] = rng.choice(donors[donors!=i])
    g[ids] = geometry[choice]
    return g,ids,choice
