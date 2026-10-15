"""Trainable spatial/subband residual embeddings for support-conditioned ridge."""

import torch
from torch import nn
from torch.nn import functional as F


def haar(x):
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return torch.stack(((a+b+c+d)/2, (a-b+c-d)/2,
                        (a+b-c-d)/2, (a-b-c+d)/2), dim=1)


def processor(channels):
    return nn.Sequential(nn.Conv2d(512, channels, 1, bias=False),
                         nn.GroupNorm(4, channels), nn.GELU(),
                         nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False),
                         nn.GroupNorm(4, channels), nn.GELU())


class FrequencyResidual(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == 'unsplit':
            self.blocks = nn.ModuleList([processor(64)])
        else:
            self.blocks = nn.ModuleList([processor(16) for _ in range(4)])
        if mode == 'selective':
            self.gate = nn.Sequential(nn.Linear(66, 32), nn.GELU(), nn.Linear(32, 4))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x, conditions):
        if self.mode == 'unsplit':
            z = self.blocks[0](x)
            out = F.normalize(F.adaptive_avg_pool2d(z, 2).flatten(1).float(), dim=1)
            return out, torch.full((len(x), 4), .25, device=x.device)
        bands = haar(x)
        z = [block(bands[:, i]) for i, block in enumerate(self.blocks)]
        vectors = torch.stack([F.normalize(F.adaptive_avg_pool2d(a, 2).flatten(1).float(), dim=1)
                               for a in z], dim=1)
        if self.mode == 'selective':
            context = torch.cat([a.mean((-1, -2)) for a in z]+[conditions], dim=1)
            weights = self.gate(context).float().softmax(1)
        else:
            weights = torch.full((len(x), 4), .25, device=x.device)
        # Concatenated weighted blocks preserve band identity and unit diagonal.
        return (weights.sqrt().unsqueeze(-1)*vectors).flatten(1), weights


def support_scores(phi, prior, mean, conditions, embedding, labels, eta,
                   beta=.25, strength=8., support_n=64):
    labels = labels.long()
    s = slice(0, support_n)
    dist = (conditions[:, None]-conditions[None, :support_n]).square().sum(-1)
    condition = torch.exp(-dist/.5)
    km = (mean @ mean[s].T)*condition
    kernel = phi @ phi[s].T + eta*km
    if beta:
        kernel = kernel+beta*(embedding @ embedding[s].T)*condition
    _, inverse, count = torch.unique(labels, return_inverse=True, return_counts=True)
    root = (len(labels)/(len(count)*count[inverse].to(phi.dtype))).sqrt()
    desired = 2*F.one_hot(labels, prior.shape[1]).to(prior.dtype)-1
    dual = torch.linalg.solve(root[:, None]*kernel[s]*root[None, :]
                              +strength*torch.eye(support_n, device=phi.device, dtype=phi.dtype),
                              root[:, None]*(desired-prior[s]))
    return prior+kernel@(root[:, None]*dual)
