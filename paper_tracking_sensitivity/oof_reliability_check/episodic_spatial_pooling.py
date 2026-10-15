"""Learn token pooling through the existing balanced support ridge solve."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class EpisodicPool(nn.Module):
    def __init__(self, dimension=128, hidden=16):
        super().__init__()
        self.hidden = nn.Linear(dimension, hidden)
        self.output = nn.Linear(hidden, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, tokens, fixed_logits, token_scale, residual_scale):
        adjustment = self.output(torch.tanh(self.hidden(tokens / token_scale))).squeeze(-1)
        weights = torch.softmax(fixed_logits + adjustment, dim=1)
        residual = torch.einsum('nt,ntd->nd', weights - .25, tokens) / residual_scale
        return residual, weights


def adapted_scores(prior, kernel, features, support, labels, strength=.25):
    """Keep the prior exactly unchanged and differentiate through target adaptation."""
    prior, kernel, features = prior.double(), kernel.double(), features.double()
    augmented = kernel + strength * (features @ features[support].T) / features.shape[1]
    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    root = (len(labels) / (len(counts) * counts[inverse].double())).sqrt()
    desired = 2 * F.one_hot(labels, prior.shape[1]).double() - 1
    gram = root[:, None] * augmented[support] * root[None, :]
    beta = root[:, None] * torch.linalg.solve(
        gram + 8 * torch.eye(len(labels), dtype=gram.dtype, device=gram.device),
        root[:, None] * (desired - prior[support]))
    return prior + augmented @ beta


def balanced_ce(scores, labels):
    losses = F.cross_entropy(scores, labels, reduction='none')
    return torch.stack([losses[labels == c].mean() for c in labels.unique()]).mean()


def balanced_subset(labels, query, per_class, rng):
    query = np.asarray(query, dtype=int)
    return np.concatenate([rng.choice(query[labels[query] == c],
        size=min(per_class, np.sum(labels[query] == c)), replace=False)
        for c in np.unique(labels[query])])
