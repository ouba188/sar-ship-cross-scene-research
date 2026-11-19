"""GDOS-inspired continuous gates and classification consistency (not causal ID)."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def occupancy(valid, available, size):
    if valid.ndim == 4:
        valid = valid[:, :, None]
    mask = F.adaptive_avg_pool2d(valid.flatten(0, 1).float(), size)
    return mask.reshape(len(valid), 2, 1, *size)*available[:, :, None, None, None]


class PolarGatedSplit(nn.Module):
    """Sample-conditioned full-width complementary features, separately per polar."""
    def __init__(self, channels=256, hidden=16, initial_gate=.95):
        super().__init__()
        if not 0 < initial_gate < 1:
            raise ValueError('initial_gate must be strictly between zero and one')
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels))
            for _ in range(2)])
        self.fusions = nn.ModuleList([nn.Conv2d(2*channels, channels, 1, bias=False)
                                     for _ in range(2)])
        for gate, fusion in zip(self.gates, self.fusions):
            # Near-identity continuation, not random deletion of pretrained channels.
            nn.init.normal_(gate[-1].weight, std=1e-3)
            nn.init.constant_(gate[-1].bias, math.log(initial_gate/(1-initial_gate)))
            with torch.no_grad():
                fusion.weight.zero_()
                eye = torch.eye(channels)
                fusion.weight[:, :channels, 0, 0].copy_(eye)
                fusion.weight[:, channels:, 0, 0].copy_(eye)

    def forward(self, features, valid, available):
        mask = occupancy(valid, available, features.shape[-2:])
        descriptor = (features*mask).sum((-1, -2))/mask.sum((-1, -2)).clamp_min(1)
        gate = torch.stack([net(descriptor[:, p]).sigmoid()
                            for p, net in enumerate(self.gates)], 1)[:, :, :, None, None]
        return features*gate, features*(1-gate), gate

    def recombine(self, content, remainder, donor, valid, available):
        donor = donor.to(content.device)
        mask = occupancy(valid, available, content.shape[-2:])
        overlap = torch.minimum(mask, mask[donor])
        changed = torch.stack([net(torch.cat((content[:, p], remainder[donor, p]), 1))
                               for p, net in enumerate(self.fusions)], 1)
        # A padded chip is not a fixed-size RoI: keep the original content outside
        # real-image overlap, including singleton availability groups (no donor).
        eligible = (donor != torch.arange(len(donor), device=donor.device))
        weight = overlap*eligible[:, None, None, None, None]
        return content + weight*(changed-content)


def weighted_consistency(changed, clean, eligible):
    """GDOS Eq.25; omit clutter probability because no clutter class is trained."""
    logp = clean.detach().log_softmax(-1)
    p = logp.exp()
    confidence = (1+(p*logp).sum(-1)/math.log(p.shape[-1])).clamp(0, 1)
    kl = F.kl_div(changed.log_softmax(-1), p, reduction='none').sum(-1)
    loss = (kl*confidence*eligible.to(kl)).mean()
    return loss, dict(kl=float(kl.detach().mean()),
                      reliability=float(confidence.mean()),
                      probability_mse=float((changed.detach().softmax(-1)-p).square().mean()))
