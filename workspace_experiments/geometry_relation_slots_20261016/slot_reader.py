"""Region-preserving local relations and task-centered additive Gram kernels."""

import torch
from torch import nn


class SlotReader(nn.Module):
    def __init__(self, mode):
        super().__init__()
        assert mode in ("spatial", "conditional", "geometry")
        self.mode = mode
        self.norm = nn.LayerNorm(64)
        self.relation = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 16, bias=False))
        self.condition = nn.Linear(8, 16, bias=False)
        nn.init.zeros_(self.condition.weight)
        self.register_buffer("scale", torch.ones(()))

    @staticmethod
    def angular(geometry):
        a = geometry[:, :3]
        s, r, inc = a.unbind(-1)
        return torch.stack((torch.sin(2*s), torch.cos(2*s), torch.sin(2*r), torch.cos(2*r),
                            torch.sin(2*(s-r)), torch.cos(2*(s-r)), torch.sin(inc), torch.cos(inc)), -1)

    def forward(self, tokens, areas, geometry):
        h = self.norm(tokens)
        body = h[:, :1].expand(-1, 3, -1, -1)
        pairs = torch.cat((body, h[:, 1:]), -1)
        z = self.relation(pairs)
        angles = self.angular(geometry)
        if self.mode == "spatial":
            angles = torch.zeros_like(angles)
        z = z * (1. + .25 * torch.tanh(self.condition(angles))[:, None, None])
        occupied = (areas[:, :1] > 0) & (areas[:, 1:] > 0)
        z = z * occupied[..., None]
        return z.flatten(1) / self.scale


def centered_features(raw, valid, ns):
    """Use only valid supports to center both roles; <=1 supports gives no kernel."""
    active = valid[:ns]
    if int(active.sum()) < 2:
        return raw * 0.
    mean = raw[:ns][active].mean(0)
    return (raw - mean) * valid[:, None]


def predict_increment(ss, qs, prior_s, prior_q, support_labels, base_prob, old_prob,
                      raw, valid, coefficient_fn, strength=.75):
    ns = len(support_labels)
    psi = centered_features(raw, valid, ns).double()
    ds, dq = psi[:ns], psi[ns:]
    new_ss = ss + strength * (ds @ ds.T)
    new_qs = qs + strength * (dq @ ds.T)
    coeff = coefficient_fn(new_ss, prior_s, support_labels)
    score = prior_q + new_qs @ coeff
    prob = .5 * (base_prob + score.softmax(-1))
    enabled = valid[ns:] & (valid[:ns].sum() >= 2)
    prob = torch.where(enabled[:, None], prob, old_prob)
    return prob, new_qs, psi
