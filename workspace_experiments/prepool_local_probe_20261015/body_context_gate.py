"""Differentiable body/context maps and the frozen full residual readout."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class Gate(nn.Module):
    """Map normalized body [N,64] and context [N,4] to local [N,132].

    Inputs share the module's floating dtype/device. ``gate_b`` replaces only
    the conditional gate input; the explicit context branch always uses b.
    Zero initialization has the same kernel as [sqrt(.5)*s, sqrt(.5)*b].
    Static mode has only c (64 parameters); conditional mode adds W (64x4).
    """

    def __init__(self, mode="conditional"):
        super().__init__()
        if mode not in ("static", "conditional"):
            raise ValueError("mode must be 'static' or 'conditional'")
        self.mode = mode
        self.c = nn.Parameter(torch.zeros(64))
        if mode == "conditional":
            self.W = nn.Parameter(torch.zeros(64, 4))

    def forward(self, s, b, gate_b=None):
        logits = self.c
        if self.mode == "conditional":
            context = b if gate_b is None else gate_b
            logits = F.linear(context, self.W, self.c)
        u = F.normalize(s * (1. + .25 * torch.tanh(logits)), dim=-1)
        return torch.cat((.5 * s, math.sqrt(.5) * b, .5 * u), dim=-1)


def _coefficients(kernel, prior, labels):
    """Class-balanced support residual ridge, matching source_context_pilot."""
    _, inverse, counts = torch.unique(labels, return_inverse=True,
                                      return_counts=True)
    sample_weight = labels.numel() / (counts.numel() * counts[inverse].to(kernel.dtype))
    root = sample_weight.sqrt()
    target = 2. * F.one_hot(labels, num_classes=prior.shape[1]).to(prior.dtype) - 1.
    system = root[:, None] * kernel * root[None, :]
    system = system + 8. * torch.eye(len(labels), dtype=kernel.dtype,
                                   device=kernel.device)
    return root[:, None] * torch.linalg.solve(
        system, root[:, None] * (target - prior))


def readout(base, expert, prior_b, prior_e, labels, q, s,
            relation=None, weights=None, local=None):
    """Return [len(q), C] probabilities with full differentiable support solves.

    All row-aligned inputs may be compact per-task tensors. Floating tensors
    share dtype/device; labels and q/s are torch.long. Shapes are base [N,Db],
    expert [N,257], priors [N,C], labels [N], relation [N,R], weights [R], and
    local [N,L]. Inputs already contain the original design scaling/bias.
    ``weights`` are relation multipliers, not support sample weights. Only
    support labels are read. ``local=None`` preserves the original expert;
    otherwise 25% of its D18 kernel is replaced in both SS and QS blocks.

    Relation blending is .75/.25, ridge is 8, and separately softmaxed branch
    probabilities are averaged. No dtype conversion or gradient detachment is
    performed on floating inputs. Only SxS and QxS kernel blocks are formed.
    """
    sb, qb = base[s], base[q]
    sz, qz = expert[s], expert[q]
    support_labels = labels[s]
    base_score = prior_b[q] + (qb @ sb.T) @ _coefficients(
        sb @ sb.T, prior_b[s], support_labels)
    ss, qs = sz @ sz.T, qz @ sz.T
    if local is not None:
        if expert.shape[1] != 257:
            raise ValueError("local replacement requires the 257-column expert design")
        ls, lq = local[s], local[q]
        ds, dq = sz[:, 128:256], qz[:, 128:256]
        ss = ss + .25 * (16. * (ls @ ls.T) - ds @ ds.T)
        qs = qs + .25 * (16. * (lq @ ls.T) - dq @ ds.T)
    if relation is not None:
        sr, qr = relation[s], relation[q]
        multiplier = 1. if weights is None else weights
        ss = .75 * ss + .25 * ((sr * multiplier) @ sr.T)
        qs = .75 * qs + .25 * ((qr * multiplier) @ sr.T)
    expert_score = prior_e[q] + qs @ _coefficients(ss, prior_e[s], support_labels)
    return .5 * (torch.softmax(base_score, dim=1) +
                 torch.softmax(expert_score, dim=1))
