"""Task-learned local evidence weighting with a shared PSD feature map."""
import torch
from torch import nn
from torch.nn import functional as F


class EvidenceReader(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.body = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 32), nn.GELU())
        self.context = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 32), nn.GELU())
        self.position = nn.Sequential(nn.Linear(16, 8), nn.GELU())
        self.quality = nn.Sequential(nn.Linear(10, 8), nn.GELU())
        self.relation = nn.Sequential(nn.Linear(112, 32), nn.GELU())
        self.content = nn.Linear(32, 16)
        self.preference = nn.Parameter(torch.zeros(13), requires_grad=arm != 'uniform')
        self.gate = nn.Linear(32, 1)
        self.null_gate = nn.Linear(32, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.null_gate.weight)
        nn.init.zeros_(self.null_gate.bias)
        if arm != 'dynamic':
            self.gate.requires_grad_(False)
            self.null_gate.requires_grad_(False)

    def forward(self, body, tokens, positions, quality, gate_override=None):
        b = self.body(body)[:, None].expand(-1, tokens.shape[1], -1)
        c = self.context(tokens)
        p, q = self.position(positions), self.quality(quality)
        relation = self.relation(torch.cat([b, c, b*c, p, q], dim=-1))
        content = F.normalize(self.content(relation), dim=-1, eps=1e-6)
        correction = (torch.cat([self.gate(relation).squeeze(-1), self.null_gate(relation.mean(1))], dim=1)
                      if self.arm == 'dynamic' else positions.new_zeros(len(body), 13))
        if gate_override is not None:
            correction = gate_override.expand_as(correction)
        logits = self.preference[None]+correction
        coverage = positions[:, :, 2].clamp(0, 1)
        active = (positions[:, :, 3] > 0) & (coverage > 0)
        regional = (logits[:, :12]+coverage.clamp_min(1e-12).log()).masked_fill(~active, -1e4)
        weights = torch.cat([regional, logits[:, 12:]], dim=1).softmax(-1)
        raw = (content*weights[:, :12].clamp_min(1e-12).sqrt()[:, :, None]*active[:, :, None]).flatten(1)
        return raw, weights, correction


def retrieval_loss(h, labels, ns, valid, temperature=.2):
    sy, qy = labels[:ns][valid[:ns]], labels[ns:][valid[ns:]]
    sh, qh = h[:ns][valid[:ns]], h[ns:][valid[ns:]]
    if len(sy) == 0 or len(qy) == 0:
        return h.sum()*0
    present = sy.unique(sorted=True)
    similarity = qh@sh.T/temperature
    scores = torch.stack([torch.logsumexp(similarity[:, sy == c], dim=1)-torch.log((sy == c).sum().to(h.dtype))
                          for c in present], dim=1)
    eligible = (qy[:, None] == present[None]).any(1)
    if not eligible.any():
        return h.sum()*0
    target = (qy[eligible, None] == present[None]).long().argmax(1)
    loss = F.cross_entropy(scores[eligible], target, reduction='none')
    return torch.stack([loss[qy[eligible] == c].mean() for c in qy[eligible].unique()]).mean()
