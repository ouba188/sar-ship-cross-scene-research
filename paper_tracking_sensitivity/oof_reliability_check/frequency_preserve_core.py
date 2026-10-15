"""Preserve demonstrated single-band rescues without suppressing other bands."""

import torch
from torch.nn import functional as F
from frequency_specialist_core import class_margin


def unique_anchors(teacher_scores, main_scores, labels):
    correct = teacher_scores.detach().argmax(-1) == labels[:, None]
    unique = correct & (correct.sum(1) == 1)[:, None]
    unique = unique & (main_scores.detach().argmax(-1) != labels)[:, None]
    y = labels[:, None].expand(-1, 4)
    margin = class_margin(teacher_scores.detach(), y).clamp(0, .1)
    return unique.detach(), margin.detach()


def retention_loss(scores, labels, mask, target_margin):
    if not mask.any():
        return scores.sum()*0
    y = labels[:, None].expand(-1, 4)
    active_labels = labels[mask.any(1)]
    count = torch.bincount(active_labels, minlength=scores.shape[-1]).clamp_min(1)
    weight = mask.detach()/count[labels, None].to(scores.dtype)
    hinge = F.relu(target_margin.detach()-class_margin(scores, y))
    return (weight*hinge).sum()/weight.sum()
