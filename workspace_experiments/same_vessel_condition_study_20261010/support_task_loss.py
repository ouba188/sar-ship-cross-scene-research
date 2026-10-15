"""Differentiable support correction with the deployed ridge and probability fusion."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def torch_support(raw_s, delta, cols, dose):
    if dose == 0:
        return raw_s.clone()
    block = raw_s[:, cols]+dose*delta
    rc, tc = raw_s[:, cols].mean(0), block.mean(0)
    re = (raw_s[:, cols]-rc).square().sum(1).mean()
    te = (block-tc).square().sum(1).mean()
    result = raw_s.clone()
    result[:, cols] = (block-tc)*torch.sqrt(re/te.clamp_min(1e-20))+rc
    return result


def class_weights(labels):
    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    return len(labels)/(len(counts)*counts[inverse].to(torch.float64))


def torch_scores(phi_s, phi_q, prior_s, prior_q, labels_s, delta, cols, dose, strength):
    support = torch_support(phi_s, delta, cols, dose)
    root = class_weights(labels_s).sqrt()
    target = 2*F.one_hot(labels_s, prior_s.shape[1]).to(prior_s.dtype)-1
    matrix = root[:, None]*(support@support.T)*root[None, :]
    matrix = matrix+strength*torch.eye(len(support), dtype=matrix.dtype, device=matrix.device)
    alpha = root[:, None]*torch.linalg.solve(matrix, root[:, None]*(target-prior_s))
    return prior_q+(phi_q@support.T)@alpha


def fusion_ce(base_logits, expert_logits, labels_q):
    log_p = torch.logaddexp(F.log_softmax(base_logits, dim=1),
                            F.log_softmax(expert_logits, dim=1))-math.log(2.)
    loss = -log_p[torch.arange(len(labels_q), device=labels_q.device), labels_q]
    weights = class_weights(labels_q)
    return (loss*weights).sum()/weights.sum()


class ResidualCorrector(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.register_buffer('center', torch.as_tensor(initial['image_mean_x'], dtype=torch.float64).clone())
        self.coef = nn.Parameter(torch.as_tensor(initial['image_coef'], dtype=torch.float64).clone())
        self.bias = nn.Parameter(torch.as_tensor(initial['image_mean_d'], dtype=torch.float64).clone())

    def forward(self, x):
        return self.bias+(x-self.center)@self.coef

    def numpy_state(self):
        return dict(image_mean_x=self.center.detach().cpu().numpy(),
                    image_coef=self.coef.detach().cpu().numpy(),
                    image_mean_d=self.bias.detach().cpu().numpy())


def common_class_targets(x, meta, weights):
    """Training-only same-class means, excluding the entire focal vessel."""
    target = np.empty_like(x)
    y, ships = meta.class_id.to_numpy(), meta.mmsi.to_numpy()
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        for ship in np.unique(ships[members]):
            focal = members[ships[members] == ship]
            other = members[ships[members] != ship]
            if not len(other):
                raise ValueError('Each fitting class needs at least two distinct vessels')
            target[focal] = np.average(x[other], axis=0, weights=weights[other])
    return target-x
