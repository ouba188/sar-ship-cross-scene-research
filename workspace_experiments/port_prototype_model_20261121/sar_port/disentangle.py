"""Optional full-width gated decomposition and source-donor consistency.

This preserves the prior GDOS-inspired gated_split.py mechanism; the two
branches are candidate content and remainder, not identified causal factors.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _occupancy(valid, available, size):
    if valid.ndim == 4:
        valid = valid[:, :, None]
    area = F.adaptive_avg_pool2d(valid.flatten(0, 1).float(), size)
    return area.reshape(len(valid), 2, 1, *size) * available[:, :, None, None, None]


class PolarDisentangler(nn.Module):
    """Per-polar validGAP -> C/hidden/C sigmoid, with complementary C channels.

    Normal classification uses ``content``. ``swap`` combines own content with
    a compatible donor remainder only inside the valid-image overlap. ``full``
    is the all-open control. Missing views are zero in all three modes.
    """

    def __init__(self, channels=256, hidden=16, initial_gate=.95):
        super().__init__()
        if not 0 < initial_gate < 1:
            raise ValueError("initial_gate must be strictly between zero and one")
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels))
            for _ in range(2)
        )
        self.fusions = nn.ModuleList(nn.Conv2d(2 * channels, channels, 1, bias=False) for _ in range(2))
        for gate, fusion in zip(self.gates, self.fusions):
            nn.init.normal_(gate[-1].weight, std=1e-3)
            nn.init.constant_(gate[-1].bias, math.log(initial_gate / (1 - initial_gate)))
            with torch.no_grad():
                fusion.weight.zero_()
                eye = torch.eye(channels)
                fusion.weight[:, :channels, 0, 0].copy_(eye)
                fusion.weight[:, channels:, 0, 0].copy_(eye)

    def split(self, features, valid, available):
        area = _occupancy(valid, available, features.shape[-2:]).to(features)
        descriptor = (features * area).sum((-1, -2)) / area.sum((-1, -2)).clamp_min(1)
        present = (area.sum((-1, -2)) > 0)[:, :, :, None, None]
        gate = torch.stack([net(descriptor[:, p]).sigmoid()
                            for p, net in enumerate(self.gates)], 1)[:, :, :, None, None]
        gate = gate * present
        visible = features * present
        return visible * gate, visible * (1 - gate), gate

    def forward(self, features, valid, available, mode="content", donor=None):
        content, remainder, _ = self.split(features, valid, available)
        if mode == "content":
            return content
        if mode == "full":
            return content + remainder
        if mode != "swap":
            raise ValueError("mode must be content, swap or full")
        if donor is None:
            raise ValueError("swap requires donor indices")
        donor = donor.to(device=features.device, dtype=torch.long)
        area = _occupancy(valid, available, features.shape[-2:]).to(features)
        overlap = torch.minimum(area, area[donor])
        compatible = (available.bool() == available[donor].bool()).all(-1)
        compatible &= available.bool().any(-1)
        compatible &= donor != torch.arange(len(donor), device=donor.device)
        mixed = torch.stack([net(torch.cat((content[:, p], remainder[donor, p]), 1))
                             for p, net in enumerate(self.fusions)], 1)
        weight = overlap * compatible[:, None, None, None, None]
        return content + weight * (mixed - content)


def donor_permutation(rows, available, seed=None):
    """Choose same-availability donors with different vessel and group identities.

    Choices are independently sampled from each recipient's legal donor list,
    so donors may be reused (many-to-one), rather than claiming a bijection.
    A row with no legal donor maps to itself and has eligible=False. Metadata
    fields are vessel_id/group_id, matching the package data protocol.
    """
    patterns = available.detach().cpu().bool().numpy()
    if len(rows) != len(patterns):
        raise ValueError("rows and availability batch dimensions must agree")
    rng = np.random.default_rng(seed)
    donor = np.arange(len(rows), dtype=np.int64)
    eligible = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        candidates = [j for j, other in enumerate(rows)
                      if j != i and patterns[i].any()
                      and np.array_equal(patterns[i], patterns[j])
                      and row["vessel_id"] != other["vessel_id"]
                      and row["group_id"] != other["group_id"]]
        if candidates:
            donor[i] = rng.choice(candidates)
            eligible[i] = True
    return (torch.as_tensor(donor, device=available.device),
            torch.as_tensor(eligible, device=available.device))


def weighted_consistency(changed_logits, clean_logits, eligible, temperature=1.):
    """Entropy-weighted KL(stopgrad(clean) || changed), averaged over the batch.

    Temperature scales both distributions; T squared preserves the conventional
    distillation gradient scale. Confidence is computed at the same temperature.
    No pure-environment, clutter-class or class-prior target is introduced.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    classes = clean_logits.shape[-1]
    if classes < 2:
        raise ValueError("Consistency requires at least two classes")
    logp = (clean_logits.detach().float() / temperature).log_softmax(-1)
    probability = logp.exp()
    confidence = (1 + (probability * logp).sum(-1) / math.log(classes)).clamp(0, 1)
    log_changed = (changed_logits.float() / temperature).log_softmax(-1)
    kl = F.kl_div(log_changed, probability, reduction="none").sum(-1)
    return (kl * confidence * eligible.to(kl)).mean() * temperature ** 2
