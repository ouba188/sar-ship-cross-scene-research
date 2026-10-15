"""Source-only branch competence and anchored rescue specialization."""

import torch
from torch.nn import functional as F
from learned_frequency_model import support_scores


def class_margin(scores, labels):
    true = scores.gather(-1, labels[..., None]).squeeze(-1)
    other = scores.masked_fill(F.one_hot(labels, scores.shape[-1]).bool(), -torch.inf).max(-1).values
    return true-other


def rescue_responsibility(initial_band_scores, reference, labels, temperature=.1):
    """Frozen source-label supervision; no pressure to make representations orthogonal."""
    y = labels[:, None].expand(-1, 4)
    margins = class_margin(initial_band_scores, y)
    weights = .5+2*torch.softmax(margins/temperature, dim=1)
    reference_wrong = reference.argmax(1) != labels
    return torch.where(reference_wrong[:, None], weights, torch.ones_like(weights)).detach()


def per_band_scores(args, embedding, labels, eta, support_n=64):
    bands = embedding.reshape(-1, 4, 64)*2
    return torch.stack([support_scores(*args, bands[:, b], labels, eta, support_n=support_n)
                        for b in range(4)], dim=1)


def balanced_ce(scores, labels):
    weight = 1./torch.bincount(labels, minlength=scores.shape[-1]).clamp_min(1)[labels].float()
    return (F.cross_entropy(scores, labels, reduction='none')*weight).sum()/weight.sum()


def branch_objective(scores, reference, labels, responsibility=None):
    n, bands, classes = scores.shape
    y = labels[:, None].expand(-1, bands)
    weight = 1./torch.bincount(labels, minlength=classes).clamp_min(1)[labels].float()
    ce = F.cross_entropy(scores.reshape(-1, classes), y.reshape(-1), reduction='none').reshape(n, bands)
    allocation = torch.ones_like(ce) if responsibility is None else responsibility.detach()
    classification = (ce*allocation*weight[:, None]).sum()/(bands*weight.sum())
    target_margin = class_margin(reference.detach(), labels).clamp(0, .1)
    protected = reference.argmax(1) == labels
    hinge = F.relu(target_margin[:, None]-class_margin(scores, y))
    protection = (hinge*protected[:, None]*weight[:, None]).sum()/(bands*weight.sum())
    return classification+.25*protection, classification, protection
