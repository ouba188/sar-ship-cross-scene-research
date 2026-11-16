"""Candidate regional polarization interaction; not a trained formal-model replacement."""
import torch
from torch import nn


class PolarizationConditionedEvidenceArbitration(nn.Module):
    """Tokens [B,2,R,K+1,D], with slot0=mean and slots1..K=all candidate classes.

    The upstream CCREL reader defines class semantics. No query labels are accepted.
    Returned tokens retain polarization and region identity for separate spatial writeback.
    """

    def __init__(self, dim=32, rank=16, classes=4, regions=3):
        super().__init__()
        self.project = nn.ModuleList(nn.Linear(dim, rank, bias=False) for _ in range(2))
        self.region_embedding = nn.Parameter(torch.zeros(regions, rank))
        self.class_embedding = nn.Parameter(torch.zeros(classes + 1, rank))
        self.gate = nn.Sequential(nn.Linear(4 * rank, rank), nn.SiLU(), nn.Linear(rank, 3))
        self.output = nn.ModuleList(nn.Linear(rank, dim, bias=False) for _ in range(2))
        for layer in self.output:
            nn.init.zeros_(layer.weight)

    def forward(self, tokens, available, region_valid, *, product=True, constant_gate=False):
        u, v = (self.project[p](tokens[:, p]).tanh() for p in range(2))
        b, r, k, h = u.shape
        er = self.region_embedding[None, :, None].expand(b, r, k, h)
        ek = self.class_embedding[None, None].expand(b, r, k, h)
        logits = self.gate(torch.cat((u, v, er, ek), -1))
        alpha = logits[..., :2].softmax(-1)
        beta = logits[..., 2:].sigmoid()
        if constant_gate:
            alpha, beta = torch.full_like(alpha, .5), torch.full_like(beta, .5)
        joint = u * v
        fused = alpha[..., :1] * u + alpha[..., 1:] * v
        if product:
            fused = fused + beta * joint
        # No cross-polar update when either physical observation/region is unavailable.
        active = available.bool().all(1)[:, None] & region_valid.bool().all(1)
        deltas = []
        for p in range(2):
            scale = tokens[:, p].square().mean(-1, keepdim=True).sqrt().detach()
            delta = .1 * scale * self.output[p](fused).tanh()
            deltas.append(delta * active[:, :, None, None])
        delta = torch.stack(deltas, 1)
        return tokens + delta, dict(alpha=alpha, beta=beta, product=joint,
                                   active=active, delta=delta)


def regional_writeback(delta, masks, projection):
    """Concatenate all candidate slots, project, then write to that polarization's own mask.

    delta: [B,2,R,K+1,D]; masks: [B,2,R,H,W]. Output: [B,2,C,H,W].
    Overlap totals are capped at one; fractional boundary coverage is retained.
    Pixels outside all masks receive zero.
    """
    masks = masks / masks.sum(2, keepdim=True).clamp_min(1)
    return torch.stack([torch.einsum('brc,brhw->bchw',
        projection[p](delta[:, p].flatten(2)), masks[:, p]) for p in range(2)], 1)
