"""Explicit source-normalized linear candidate space and identity-split episodes."""

import numpy as np
import torch
from torch import nn


class CandidateProjector(nn.Module):
    def __init__(self, source, labels, weights, dimensions=32):
        super().__init__()
        mean = source.mean(0)
        centered = source-mean
        covariance = centered.T@centered/len(source)
        _, vectors = torch.linalg.eigh(covariance)
        self.register_buffer('center', mean)
        self.register_buffer('covariance', covariance)
        self.frame = nn.Parameter(vectors[:, -dimensions:].flip(1).clone())
        self.head = nn.Linear(dimensions, 8)
        with torch.no_grad():
            z = self.encode(source)
            coef = ridge_probe(z, labels, weights)
            self.head.weight.copy_(coef[:-1].T)
            self.head.bias.copy_(coef[-1])

    def geometry(self):
        q, r = torch.linalg.qr(self.frame, mode='reduced')
        sign = torch.where(r.diagonal() < 0, -1., 1.)
        q = q*sign
        scale = ((self.covariance@q)*q).sum(0).clamp_min(1e-6).sqrt()
        return q, scale

    def encode(self, x):
        q, scale = self.geometry()
        return ((x-self.center)@q)/scale/np.sqrt(q.shape[1])

    def forward(self, x):
        z = self.encode(x)
        return z, self.head(z)


def ridge_probe(z, labels, weights, ridge=1.):
    phi = torch.cat([z, torch.ones((len(z),1),dtype=z.dtype)],1)
    w = torch.as_tensor(weights,dtype=z.dtype)
    w = w/w.mean()
    target = 2*torch.eye(8,dtype=z.dtype)[labels]-1
    penalty = torch.eye(phi.shape[1],dtype=z.dtype)*ridge
    penalty[-1,-1] = 0
    return torch.linalg.solve(phi.T@(w[:,None]*phi)+penalty,phi.T@(w[:,None]*target))


def identity_episode(meta, rng, support_size=64, query_size=96):
    groups = {v:g.index.to_numpy() for v,g in meta.reset_index(drop=True).groupby('mmsi',sort=True)}
    ships = np.array(list(groups))
    chosen = rng.choice(ships,support_size,replace=False)
    support = np.array([rng.choice(groups[v]) for v in chosen])
    remaining = ships[~np.isin(ships,chosen)]
    if query_size is None:
        query = np.concatenate([groups[v] for v in remaining])
    else:
        sampled = rng.choice(remaining,query_size,replace=len(remaining)<query_size)
        query = np.array([rng.choice(groups[v]) for v in sampled])
    return support,query


def balanced_mse(scores, labels):
    target = 2*torch.eye(8,dtype=scores.dtype)[labels]-1
    counts = torch.bincount(labels,minlength=8)
    w = len(labels)/((counts>0).sum()*counts[labels].to(scores.dtype))
    return (w*(scores-target).square().mean(1)).mean()
