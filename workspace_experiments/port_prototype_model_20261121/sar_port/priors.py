"""Class-level regional priors and signed operation responses, without query labels."""

import torch
from torch import nn


REGIONS = ("body", "ends", "side")
OPERATIONS = ("fine", "middle")


def attention_from_response(response, valid, available=None):
    """Normalize class logits over valid area, returning zero for absent views.

    response: [B,2,C,H,W]; valid: [B,2,1,H,W], fractional coverage allowed.
    This is a weak model-response distribution, not a physical annotation.
    """
    area = valid.to(response.dtype)
    if available is not None:
        area = area * available[:, :, None, None, None].to(area.dtype)
    logits = response.float().flatten(-2)
    area = area.float().flatten(-2)
    logits = logits.masked_fill(area <= 0, -1e4)
    weights = logits.softmax(-1) * area
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
    return weights.reshape_as(response).to(response.dtype)


class PriorEffectStudent(nn.Module):
    """Produce attention and either parent utility or regional frequency effects.

    Utility uses eight integrated parent masks and fine-minus-coarse margins;
    frequency uses body/ends/side and original-minus-intervened margins.
    Cross-polar context does not assume aligned VV/VH pixel coordinates.
    """

    def __init__(self, channels=512, classes=8, hidden=64, operations=2, effect_target="frequency"):
        super().__init__()
        if effect_target not in ("frequency", "utility"):
            raise ValueError("effect_target must be frequency or utility")
        self.effect_target = effect_target
        self.classes = classes
        self.operations = operations if effect_target == "frequency" else 1
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU())
            for _ in range(2)
        )
        self.context = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.GELU())
            for _ in range(2)
        )
        self.attention_heads = nn.ModuleList(nn.Conv2d(hidden, classes, 1) for _ in range(2))
        self.effect_heads = nn.ModuleList(nn.Linear(hidden, classes * self.operations) for _ in range(2))

    def forward(self, features, valid, regions, available):
        if regions.shape[2] != (8 if self.effect_target == "utility" else 3):
            raise ValueError("regions shape must match eight utility parents or three frequency regions")
        support = valid.to(features.dtype) * available[:, :, None, None, None].to(features.dtype)
        local = torch.stack([self.project[p](features[:, p]) for p in range(2)], 1)
        pooled = (local * support).sum((-1, -2)) / support.sum((-1, -2)).clamp_min(1e-12)
        # A nominally available view with no valid pixels contributes no context.
        present = support.sum((-1, -2, -3)) > 0
        logits, effects = [], []
        for p in range(2):
            condition = torch.cat((pooled[:, p], pooled[:, 1 - p],
                                   present[:, [p, 1 - p]].to(local.dtype)), -1)
            hidden = local[:, p] + self.context[p](condition)[:, :, None, None]
            logits.append(self.attention_heads[p](hidden))
            region_area = regions[:, p].to(hidden.dtype)
            region_area = (region_area * available[:, p, None, None, None] if self.effect_target == "utility"
                           else region_area * support[:, p])
            mass = region_area.sum((-1, -2))
            summary = torch.einsum("brhw,bdhw->brd", region_area, hidden)
            summary = summary / mass[..., None].clamp_min(1e-12)
            effect = self.effect_heads[p](summary)
            effect = effect.reshape(len(features), regions.shape[2], self.classes, self.operations).permute(0, 2, 1, 3)
            effects.append(effect * (mass > 0)[:, None, :, None])
        attention = attention_from_response(torch.stack(logits, 1), valid, available)
        values = torch.stack(effects, 1)
        key = "utility" if self.effect_target == "utility" else "effects"
        return {"attention": attention, key: values.squeeze(-1) if key == "utility" else values}
