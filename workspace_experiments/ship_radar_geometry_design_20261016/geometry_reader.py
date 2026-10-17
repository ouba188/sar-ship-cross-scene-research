"""Matched local readers; coordinates and axes must share the image xy frame."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GeometryReader(nn.Module):
    """Return a unit 64-vector from body/touch/near/far spatial tokens.

    Inputs: tokens (N,4,16,64), positions (N,4,16,2), areas
    (N,4,16), geometry (N,7). Geometry columns are ship/radar/incidence
    radians, their three validity masks, and ship confidence in [0,1].
    Ship and radar axes are BOTH unsigned modulo pi. Radar coordinates use
    a length-preserving doubled-angle embedding, not signed range/azimuth.
    Areas are nonnegative actual cell coverage, not region-normalized scores.
    A ignores geometry; B conditions the pooled content; C conditions relations.
    An entirely empty sample returns the fixed first coordinate unit vector.
    """

    def __init__(self, mode="A"):
        super().__init__()
        if mode not in ("A", "B", "C"):
            raise ValueError("mode must be A, B or C")
        self.mode = mode
        self.content = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 64))
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.output = nn.Linear(64, 64, bias=False)
        self.image_relation = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 64))
        if mode == "A":
            self.adapter = nn.Sequential(nn.Linear(64, 40), nn.GELU(), nn.Linear(40, 64))
        elif mode == "B":
            self.adapter = nn.Sequential(nn.Linear(74, 36), nn.GELU(), nn.Linear(36, 64))
        else:
            self.geometry_relation = nn.Sequential(nn.Linear(12, 64), nn.GELU(), nn.Linear(64, 64))

    @staticmethod
    def _geometry(geometry):
        valid = geometry[:, 3:6] > 0
        # Mask before trigonometry, including NaN sentinels for absent fields.
        angles = torch.where(valid, geometry[:, :3], torch.zeros_like(geometry[:, :3]))
        confidence = torch.where(valid[:, 0], geometry[:, 6], torch.zeros_like(geometry[:, 6])).clamp(0, 1)
        masks = valid.to(geometry.dtype)
        strength = torch.stack((masks[:, 0] * confidence, masks[:, 1], masks[:, 2]), dim=1)
        angular = torch.stack((torch.sin(2 * angles[:, 0]), torch.cos(2 * angles[:, 0]),
                               torch.sin(2 * angles[:, 1]), torch.cos(2 * angles[:, 1]),
                               torch.sin(angles[:, 2]), torch.cos(angles[:, 2])), dim=1)
        angular = angular * strength.repeat_interleave(2, dim=1)
        encoded = torch.cat((angular, masks, confidence[:, None]), dim=1)
        return angles, strength, encoded

    @staticmethod
    def _rotate(delta, angle):
        c = angle.cos()[:, None, None, None]
        s = angle.sin()[:, None, None, None]
        x, y = delta.unbind(-1)
        return torch.stack((c * x + s * y, -s * x + c * y), dim=-1)

    def _physical_relation(self, delta, angles, strength, reverse):
        ship = self._rotate(delta, angles[:, 0])
        if reverse:
            ship = -ship
        radar = self._rotate(delta, angles[:, 1])
        # (r*cos(2*phi), r*sin(2*phi)) is invariant to reversing the radar
        # axis. Clamp before sqrt so coincident positions have finite gradients.
        u, v = radar.unbind(-1)
        length = (u.square() + v.square()).clamp_min(1e-12).sqrt()
        radar = torch.stack((u.square() - v.square(), 2 * u * v), dim=-1) / length[..., None]
        shape = delta.shape[:-1]
        weights = strength[:, None, None, None, :].expand(*shape, 3)
        inc = torch.stack((angles[:, 2].sin(), angles[:, 2].cos()), dim=-1)
        inc = (inc * strength[:, 2:3])[:, None, None, None, :].expand(*shape, 2)
        # Geometry affects relative positions, not a constant angle-only logit.
        features = torch.cat((delta, ship * weights[..., :1], radar * weights[..., 1:2],
                              inc, weights, delta.square().sum(-1, keepdim=True)), dim=-1)
        present = strength.amax(dim=1)[:, None, None, None, None]
        return self.geometry_relation(features) * present

    def _attend(self, q, k, v, relation, areas):
        n = q.shape[0]
        q = q.reshape(n, 16, 4, 16).permute(0, 2, 1, 3)
        k = k.reshape(n, 3, 16, 4, 16).permute(0, 1, 3, 2, 4)
        v = v.reshape(n, 3, 16, 4, 16).permute(0, 1, 3, 2, 4)
        r = relation.reshape(n, 3, 16, 16, 4, 16).permute(0, 1, 4, 2, 3, 5)
        score = (torch.einsum("nhid,nrhjd->nrhij", q, k)
                 + torch.einsum("nhid,nrhijd->nrhij", q, r)) / math.sqrt(16)
        key_area = areas[:, 1:]
        valid = key_area > 0
        region_valid = valid.any(-1)
        log_area = key_area.clamp_min(torch.finfo(score.dtype).tiny).log()
        score = score + log_area[:, :, None, None, :]
        score = score.masked_fill(~valid[:, :, None, None, :], -torch.inf)
        # Empty regions must not pass all -inf logits into softmax/backward.
        score = torch.where(region_valid[:, :, None, None, None], score, torch.zeros_like(score))
        attention = score.softmax(-1) * valid[:, :, None, None, :]
        out = torch.einsum("nrhij,nrhjd->nrhid", attention, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(n, 3, 16, 64)
        body_area = areas[:, 0]
        body_weight = body_area / body_area.sum(-1, keepdim=True).clamp_min(torch.finfo(out.dtype).tiny)
        per_region = (out * body_weight[:, None, :, None]).sum(2)
        return (per_region * region_valid[..., None]).sum(1) / region_valid.sum(1, keepdim=True).clamp_min(1)

    def forward(self, tokens, positions, areas, geometry):
        valid = areas > 0
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        positions = torch.where(valid[..., None], positions, torch.zeros_like(positions))
        h = self.content(tokens)
        h = torch.where(valid[..., None], h, torch.zeros_like(h))
        q, k, v = self.query(h[:, 0]), self.key(h[:, 1:]), self.value(h[:, 1:])
        delta = positions[:, 1:, None, :, :] - positions[:, None, 0, :, None, :]
        relation = self.image_relation(delta)
        if self.mode == "C":
            angles, strength, _ = self._geometry(geometry)
            first = self._attend(q, k, v, relation + self._physical_relation(delta, angles, strength, False), areas)
            second = self._attend(q, k, v, relation + self._physical_relation(delta, angles, strength, True), areas)
            pooled = (first + second) * 0.5
        else:
            pooled = self._attend(q, k, v, relation, areas)
        tiny = torch.finfo(h.dtype).tiny
        regional = (h * areas[..., None]).sum(2) / areas.sum(2, keepdim=True).clamp_min(tiny)
        body_present = valid[:, 0].any(1)
        context_present = valid[:, 1:].any(2)
        context_fallback = (regional[:, 1:] * context_present[..., None]).sum(1) / context_present.sum(1, keepdim=True).clamp_min(1)
        pooled = torch.where(body_present[:, None], pooled, context_fallback)
        pooled = torch.where((~context_present.any(1) & body_present)[:, None], regional[:, 0], pooled)
        pooled = self.output(pooled)
        if self.mode == "A":
            pooled = pooled + self.adapter(pooled)
        elif self.mode == "B":
            _, _, encoded = self._geometry(geometry)
            pooled = pooled + self.adapter(torch.cat((pooled, encoded), dim=-1))
        norm = pooled.norm(dim=-1, keepdim=True)
        usable = valid.any(dim=(1, 2))[:, None] & (norm > 1e-12)
        fallback = torch.zeros_like(pooled)
        fallback[:, 0] = 1
        return torch.where(usable, F.normalize(pooled, dim=-1, eps=1e-12), fallback)
