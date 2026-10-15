"""Differentiable, gate-preserving local support comparison supervision."""

import torch
from torch.nn import functional as F


def visual_scores(embedding, conditions, support_labels, support_n=64):
    bands = embedding.float().reshape(-1, 4, 64)
    amplitude = torch.linalg.vector_norm(bands, dim=2).clamp_min(1e-12)
    units = bands/amplitude.unsqueeze(-1)
    tokens = F.normalize(units.reshape(-1, 4, 16, 4).transpose(2, 3), dim=3)
    q, s = units[support_n:], units[:support_n]
    qt, st = tokens[support_n:], tokens[:support_n]
    counts = torch.bincount(support_labels.long(), minlength=8)
    condition = (conditions[support_n:, None]-conditions[None, :support_n]).square().sum(-1)/.5
    class_scores = [embedding[support_n:].sum(1)*0 for _ in range(8)]
    for band in range(4):
        global_sim = q[:, band]@s[:, band].T
        local = torch.einsum('qpc,src->qspr', qt[:, band], st[:, band])
        aligned = local.diagonal(dim1=2, dim2=3).mean(2)
        matched = .5*(local.max(2).values.mean(2)+local.max(3).values.mean(2))
        importance = amplitude[support_n:, band, None]*amplitude[None, :support_n, band]
        evidence = importance*(global_sim+aligned+matched)/3
        for cl in torch.nonzero(counts, as_tuple=False).flatten().tolist():
            mask = support_labels == cl
            attention = (global_sim[:, mask]/.2-condition[:, mask]).softmax(1)
            class_scores[cl] = class_scores[cl]+(attention*evidence[:, mask]).sum(1)
    return torch.stack(class_scores, dim=1), counts


def comparison_loss(visual, counts, classification_logits, labels):
    rr = torch.arange(len(labels), device=labels.device)
    candidate = classification_logits.detach().clone()
    candidate[:, counts < 2] = -torch.inf
    candidate[rr, labels] = -torch.inf
    negative = candidate.argmax(1)
    valid = (counts[labels] >= 2) & torch.isfinite(candidate[rr, negative])
    if not valid.any():
        return visual.sum()*0, valid, negative
    margin = (visual[rr, labels]-visual[rr, negative])[valid]
    per_class = torch.bincount(labels[valid], minlength=8).clamp_min(1)
    weight = 1./per_class[labels[valid]].float()
    loss = (F.softplus(-margin/.1)*weight).sum()/weight.sum()
    return loss, valid, negative
