"""Standalone band supervision with no shared classifier or visual kernel."""

import torch
from torch.nn import functional as F


def standalone_scores(embedding, conditions, support_labels, classes=8):
    n = len(support_labels)
    bands = F.normalize(embedding.reshape(-1, 4, 64), dim=-1)
    condition = torch.exp(-(conditions[:, None]-conditions[None, :n]).square().sum(-1)/.5)
    kernel = .25*torch.einsum('nbd,sbd->bns', bands, bands[:n])*condition[None]
    _, inverse, count = torch.unique(support_labels, return_inverse=True, return_counts=True)
    root = (n/(len(count)*count[inverse].to(embedding.dtype))).sqrt()
    desired = 2*F.one_hot(support_labels, classes).to(embedding.dtype)-1
    matrix = root[None, :, None]*kernel[:, :n]*root[None, None, :]
    matrix = matrix+8*torch.eye(n, device=embedding.device, dtype=embedding.dtype)
    dual = torch.linalg.solve(matrix, (root[:, None]*desired).expand(4, -1, -1))
    return (kernel@(root[None, :, None]*dual)).permute(1, 0, 2)


def auxiliary_objective(embedding, conditions, support_labels, reference, query_labels, arm):
    # The deployed band kernel is unchanged; temperature only scales supervision.
    alone = standalone_scores(embedding, conditions, support_labels)[len(support_labels):]/.1
    available = torch.bincount(support_labels, minlength=8) > 0
    eligible = available[query_labels]
    if available.sum() < 2 or not eligible.any():
        return alone.sum()*0
    labels = query_labels[eligible]
    scores = alone[eligible]
    weights = 1/torch.bincount(labels, minlength=8).clamp_min(1)[labels].float()
    if arm == 'standalone':
        scores = scores.masked_fill(~available[None, None], -torch.inf)
        target = labels[:, None].expand(-1, 4)
        losses = F.cross_entropy(scores.reshape(-1, 8), target.reshape(-1), reduction='none').reshape(-1, 4)
    elif arm == 'pair':
        rival_scores = reference.detach()[eligible].clone()
        rival_scores[:, ~available] = -torch.inf
        rival_scores.scatter_(1, labels[:, None], -torch.inf)
        rivals = rival_scores.argmax(1)
        true = scores.gather(2, labels[:, None, None].expand(-1, 4, 1)).squeeze(-1)
        other = scores.gather(2, rivals[:, None, None].expand(-1, 4, 1)).squeeze(-1)
        losses = F.softplus(other-true)
    else:
        raise ValueError(arm)
    return (losses*weights[:, None]).sum()/(4*weights.sum())
