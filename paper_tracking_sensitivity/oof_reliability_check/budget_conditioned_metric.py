"""Budget-conditioned learning of the deployed expert's complement metric."""

import torch
from torch.nn import functional as F


def channel_weights(theta, split=2048):
    positive = 1 + .5 * torch.tanh(theta)
    return torch.cat([part / part.mean() for part in (positive[:split], positive[split:])])


def metric_scores(prior, retained, complement, support, labels, theta, split=2048):
    weights = channel_weights(theta, split)
    kernel = (.5 * (retained @ retained[support].T)
              + .5 * ((complement * weights) @ complement[support].T) + 1).double()
    prior = prior.double()
    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    root = (len(labels) / (len(counts) * counts[inverse].double())).sqrt()
    desired = 2 * F.one_hot(labels, prior.shape[1]).double() - 1
    gram = root[:, None] * kernel[support] * root[None, :]
    beta = root[:, None] * torch.linalg.solve(
        gram + 8 * torch.eye(len(labels), dtype=gram.dtype, device=gram.device),
        root[:, None] * (desired - prior[support]))
    return prior + kernel @ beta


def gate_policy(base, expert, gate, support_labels, query, rows):
    """Exact deployed gate features and full-episode floor-five-percent ranking."""
    with torch.no_grad():
        def confidence(x):
            ordered = x.sort(1).values
            logp = x.log_softmax(1)
            return ordered[:, -1] - ordered[:, -2], -(logp.exp() * logp).sum(1)
        bm, bh = confidence(base.double())
        em, eh = confidence(expert.double())
        features = torch.stack((em, bm, eh, bh), 1)
        score = ((features - gate['center']) / gate['scale']) @ gate['coef'] + gate['intercept']
        b, e = base.argmax(1), expert.argmax(1)
        eligible = query & (b != e) & torch.isin(e, support_labels) & (score > 0)
        index = torch.where(eligible)[0]
        index = index[torch.argsort(rows[index], stable=True)]
        index = index[torch.argsort(score[index], descending=True, stable=True)]
        selected = torch.zeros_like(query)
        selected[index[:int(.05 * int(query.sum()))]] = True
    return selected, score


def training_loss(scores, labels, query, selected, budgeted):
    losses = F.cross_entropy(scores[query], labels[query], reduction='none')
    _, inverse, counts = torch.unique(labels[query], return_inverse=True, return_counts=True)
    weight = 1 / counts[inverse].double()
    full = (weight * losses).sum() / weight.sum()
    chosen = selected[query]
    if not budgeted or not chosen.any():
        return full
    tail = (weight[chosen] * losses[chosen]).sum() / weight[chosen].sum()
    return .5 * (full + tail)
