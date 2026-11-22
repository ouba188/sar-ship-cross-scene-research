"""Frozen-input H: geometry-only transport and bounded query evidence weights."""
from dataclasses import dataclass
from collections import defaultdict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import EncodedShip, weighted_pool


@dataclass
class HConfig:
    classes: int
    descriptor_dim: int = 128
    alpha: float = .25
    prototype_floor: float = .2
    prototype_temperature: float = 1.
    evidence_temperature: float = .2
    gamma: float = 1.
    match_loss_weight: float = 1.
    ot_temperature: float = .1
    sinkhorn_iterations: int = 60
    position_weight: float = .2
    size_weight: float = .2
    size_tolerance: float = 2.
    adjacent_scales: bool = True
    descriptor_mode: str = "frozen_g"
    learn_edge_weights: bool = True
    prototype_weighting: str = "floor"
    batched_matching: bool = True
    pair_batch_size: int = 64

    def __post_init__(self):
        if self.classes < 2 or self.descriptor_dim < 1:
            raise ValueError("classes >= 2 and descriptor_dim >= 1 are required")
        if not 0 <= self.alpha <= 1 or not 0 <= self.prototype_floor <= 1:
            raise ValueError("alpha and prototype_floor must lie in [0, 1]")
        if min(self.prototype_temperature, self.evidence_temperature,
               self.ot_temperature) <= 0 or self.sinkhorn_iterations < 1:
            raise ValueError("temperatures and Sinkhorn iterations must be positive")
        if min(self.gamma, self.match_loss_weight, self.position_weight,
               self.size_weight) < 0 or self.size_tolerance < 1:
            raise ValueError("weights must be nonnegative and size_tolerance >= 1")
        if self.descriptor_mode not in {"frozen_g", "raw_fc"}:
            raise ValueError("descriptor_mode must be frozen_g or raw_fc")
        if self.prototype_weighting not in {"floor", "uniform", "attention"}:
            raise ValueError("prototype_weighting must be floor, uniform or attention")
        if self.pair_batch_size < 1:
            raise ValueError("pair_batch_size must be positive")


def _resize(x, shape):
    return F.interpolate(x, size=shape, mode="bilinear", align_corners=False)


def _union_slots(ship):
    supplied = getattr(ship.geometry, "union_slots", None)
    return supplied if supplied is not None else ship.geometry.levels[0].slots.amax(0)


def _union_regions(ship):
    supplied = getattr(ship.geometry, "union_regions", None)
    return supplied if supplied is not None else ship.geometry.levels[0].regions.amax(1)


def _region_union_coverage(ship):
    """Exact union fraction using physical union area and per-pol geometry.

    Each per-pol area equals its full-region area times its coverage fraction.
    Summing these two equalities recovers the denominator without adding the
    polarizations' overlapping areas to the numerator.
    """
    level = ship.geometry.levels[0]
    stride = ship.geometry.canvas_size / level.regions.shape[-1]
    polar_area = level.regions.sum((-1, -2)) * stride ** 2
    fraction = ship.geometry.coverage[:, :, 0]
    region_area = polar_area.sum(1) / fraction.sum(1).clamp_min(1e-12)
    return (ship.geometry.area / region_area.clamp_min(1e-12)).clamp(0, 1)


class SpatialBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depth = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.point = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x + self.point(F.gelu(self.depth(x)))


class SpatialClassifier(nn.Module):
    """Three-scale spatial teacher; only its ordinary forward remains in H."""
    def __init__(self, classes):
        super().__init__()
        self.register_buffer("rms", torch.ones(2, 3))
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 32, 1),
                          nn.Conv2d(32, 32, 3, padding=1, groups=32),
                          nn.GELU(), nn.Conv2d(32, 32, 1)) for _ in range(6)])
        self.polarizations = nn.ModuleList([nn.Conv2d(96, 64, 1) for _ in range(2)])
        self.fusion = nn.Conv2d(130, 64, 1)
        self.blocks = nn.ModuleList([SpatialBlock(64), SpatialBlock(64)])
        self.head = nn.Sequential(nn.Linear(197, 64), nn.GELU(), nn.Linear(64, classes))

    def forward(self, ship):
        size = ship.content[0].shape[-2:]
        polar = []
        masks = []
        for p in range(2):
            branch = []
            for l in range(3):
                valid = (ship.geometry.levels[l].valid[p:p + 1] > 0).to(ship.content[l])
                x = ship.content[l][p:p + 1] / self.rms[p, l].clamp_min(1e-8)
                x = x * valid
                for layer in self.branches[3 * p + l]:
                    x = layer(x) * valid
                branch.append(_resize(x, size))
            mask = (ship.geometry.levels[0].valid[p:p + 1] > 0).to(branch[0])
            polar.append(self.polarizations[p](torch.cat(branch, 1)) * mask)
            masks.append(mask)
        mask = torch.maximum(*masks)
        x = self.fusion(torch.cat(polar + masks, 1)) * mask
        for block in self.blocks:
            x = block(x) * mask
        slots = _union_slots(ship)
        flags = ship.geometry.levels[0].slot_valid
        root_valid = flags[:, 0].any()
        halves_valid = flags[:, 1:].all(1).any()
        active = torch.stack((root_valid, halves_valid, halves_valid))
        pooled = weighted_pool(x.expand(3, -1, -1, -1), slots)
        pooled = pooled * active[:, None]
        availability = flags[:, 0].to(x)
        original = torch.cat((pooled.flatten(), availability, active.to(x)))
        order = torch.tensor([0, 2, 1], device=x.device)
        swapped = torch.cat((pooled[order].flatten(), availability, active[order].to(x)))
        return .5 * (self.head(original) + self.head(swapped))


class RegionWeights(nn.Module):
    """Scores actual cut regions. Never changes geometry or matching."""
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 16, 1), nn.GELU())
                                       for _ in range(6)])
        self.fusion = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.GELU())
        self.score = nn.Sequential(nn.Linear(72, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, ship):
        size = ship.content[0].shape[-2:]
        values = []
        for p in range(2):
            for l in range(3):
                valid = (ship.geometry.levels[l].valid[p:p + 1] > 0).to(ship.content[l])
                x = self.branches[3 * p + l](ship.content[l][p:p + 1] * valid) * valid
                values.append(_resize(x, size))
        mask = (ship.geometry.levels[0].valid.amax(0, keepdim=True) > 0).to(values[0])
        x = self.fusion(torch.cat(values, 1)) * mask
        local = weighted_pool(x.expand(len(ship.geometry.area), -1, -1, -1),
                              _union_regions(ship))
        root = weighted_pool(x, _union_slots(ship)[0:1]).expand_as(local)
        geometry = torch.cat((ship.geometry.coverage.flatten(1),
                              ship.geometry.intervals[:, 1:2],
                              ship.geometry.intervals[:, 0:1].abs()), 1)
        b = self.score(torch.cat((local, root, geometry), 1)).squeeze(-1)
        return torch.softmax(b + ship.geometry.mass.log(), 0)


def partial_transport(scores, valid, query_mass, reference_mass, dustbin,
                      temperature=.1, iterations=60):
    """Log-Sinkhorn with unit real mass on each side and a unit dustbin.

    The final real block is scaled down only if finite-iteration roundoff
    exceeds a capacity; residual rows/columns then go to the dustbins.
    """
    n, m = scores.shape
    top = torch.cat((scores.masked_fill(~valid, -torch.inf),
                     dustbin.expand(n, 1)), 1)
    augmented = torch.cat((top, dustbin.expand(1, m + 1)), 0)
    log_kernel = augmented / temperature
    row = torch.cat((query_mass, query_mass.new_ones(1)))
    col = torch.cat((reference_mass, reference_mass.new_ones(1)))
    log_row, log_col = row.log(), col.log()
    u, v = torch.zeros_like(row), torch.zeros_like(col)
    for _ in range(iterations):
        u = log_row - torch.logsumexp(log_kernel + v[None, :], 1)
        v = log_col - torch.logsumexp(log_kernel + u[:, None], 0)
    real = torch.exp(log_kernel + u[:, None] + v[None, :])[:n, :m]
    real = real.masked_fill(~valid, 0)
    real = real * (query_mass / real.sum(1).clamp_min(1e-12)).clamp(max=1)[:, None]
    real = real * (reference_mass / real.sum(0).clamp_min(1e-12)).clamp(max=1)[None, :]
    query_rejected = (query_mass - real.sum(1)).clamp_min(0)
    reference_rejected = (reference_mass - real.sum(0)).clamp_min(0)
    return torch.cat((torch.cat((real, query_rejected[:, None]), 1),
                      torch.cat((reference_rejected, real.sum()[None]))[None, :]), 0)


def geometric_cost(query, reference, direction, config):
    qi, ri = query.geometry.intervals, reference.geometry.intervals
    n, m = len(qi), len(ri)
    if not (query.geometry.axis_valid and reference.geometry.axis_valid):
        zero = qi.new_zeros(n, m)
        return zero, zero
    position = ((qi[:, 0, None] - direction * ri[None, :, 0]).abs()
                - .5 * (qi[:, 1, None] + ri[None, :, 1])).clamp_min(0)
    size = ((qi[:, 1, None] / ri[None, :, 1]).log().abs()
            - math.log(config.size_tolerance)).clamp_min(0)
    return position, size


def partial_transport_batched(scores, valid, query_mass, reference_mass, dustbin,
                              temperature=.1, iterations=60):
    """The identical area/dustbin Sinkhorn update for equal-shaped pair batches."""
    batch, n, m = scores.shape
    top = torch.cat((scores.masked_fill(~valid, -torch.inf), dustbin.expand(batch, n, 1)), -1)
    augmented = torch.cat((top, dustbin.expand(batch, 1, m+1)), -2)
    log_kernel = augmented / temperature
    row = torch.cat((query_mass, query_mass.new_ones(batch, 1)), -1)
    col = torch.cat((reference_mass, reference_mass.new_ones(batch, 1)), -1)
    log_row, log_col = row.log(), col.log()
    u, v = torch.zeros_like(row), torch.zeros_like(col)
    for _ in range(iterations):
        u = log_row - torch.logsumexp(log_kernel + v[:, None, :], -1)
        v = log_col - torch.logsumexp(log_kernel + u[:, :, None], -2)
    real = torch.exp(log_kernel + u[:, :, None] + v[:, None, :])[:, :n, :m]
    real = real.masked_fill(~valid, 0)
    real = real * (query_mass / real.sum(-1).clamp_min(1e-12)).clamp(max=1)[:, :, None]
    real = real * (reference_mass / real.sum(-2).clamp_min(1e-12)).clamp(max=1)[:, None, :]
    query_rejected = (query_mass - real.sum(-1)).clamp_min(0)
    reference_rejected = (reference_mass - real.sum(-2)).clamp_min(0)
    return torch.cat((torch.cat((real, query_rejected[:, :, None]), -1),
                      torch.cat((reference_rejected, real.sum((-2, -1))[:, None]), -1)[:, None, :]), -2)


class HModel(nn.Module):
    def __init__(self, config: HConfig):
        super().__init__()
        self.config = config
        d = config.descriptor_dim
        self.h0 = SpatialClassifier(config.classes)
        self.attention = RegionWeights() if config.alpha > 0 else None
        input_channels = 64 if config.descriptor_mode == "frozen_g" else 256
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(input_channels, d), nn.LayerNorm(d))
                                          for _ in range(6)])
        self.edge_score = (nn.Sequential(nn.Linear(2 * d + 13, 32), nn.GELU(), nn.Linear(32, 1))
                           if config.learn_edge_weights else None)
        self.prototype_score = (nn.Sequential(nn.Linear(9, 16), nn.GELU(), nn.Linear(16, 1))
                                if config.prototype_weighting != "uniform" else None)
        for scorer in (self.edge_score, self.prototype_score):
            if scorer is not None:
                nn.init.zeros_(scorer[-1].weight)
                nn.init.zeros_(scorer[-1].bias)
        self.dustbin = nn.Parameter(torch.tensor(0.))
        self.stage = "base"
        self.set_stage("base")

    def set_stage(self, stage):
        if stage not in {"base", "match"}:
            raise ValueError("stage must be base or match")
        self.stage = stage
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("h0.") == (stage == "base"))
            if not parameter.requires_grad:
                parameter.grad = None
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.stage == "match":
            self.h0.eval()
        return self

    @torch.no_grad()
    def estimate_rms(self, source):
        squares = torch.zeros_like(self.h0.rms)
        counts = torch.zeros_like(squares)
        for ship in source:
            for p in range(2):
                for l in range(3):
                    weights = ship.geometry.levels[l].valid[p]
                    squares[p, l] += (ship.content[l][p].square() * weights).sum()
                    counts[p, l] += weights.sum() * ship.content[l].shape[1]
        result = (squares / counts.clamp_min(1)).sqrt().clamp_min(1e-6)
        result = torch.where(counts > 0, result, torch.ones_like(result))
        self.h0.rms.copy_(result)
        return result.clone()

    def forward_base(self, query):
        return torch.stack([self.h0(ship) for ship in query])

    def region_weights(self, ship):
        mu = ship.geometry.mass
        if self.attention is None or len(mu) == 1:
            return mu
        return (1 - self.config.alpha) * mu + self.config.alpha * self.attention(ship)

    def _descriptors(self, ship):
        n = len(ship.geometry.area)
        source = ship.spatial if self.config.descriptor_mode == "frozen_g" else ship.content
        descriptors, masks = [], []
        for p in range(2):
            values, valid = [], []
            for l in range(3):
                level = ship.geometry.levels[l]
                pooled = weighted_pool(source[l][p:p + 1].expand(n, -1, -1, -1),
                                       level.regions[:, p])
                z = self.projections[3 * p + l](pooled)
                if not bool(torch.isfinite(pooled).all() and torch.isfinite(z).all()):
                    raise ValueError("nonfinite regional descriptor")
                # Observation eligibility cannot be changed by learned phi or
                # by the sample's feature norm; a zero vector stays a legal edge.
                active = level.match_valid[:, p]
                values.append(F.normalize(z, dim=1) * active[:, None])
                valid.append(active)
            descriptors.append(torch.stack(values, 1))
            masks.append(torch.stack(valid, 1))
        return torch.stack(descriptors, 1), torch.stack(masks, 1)

    def _edges(self, query, reference, qdesc, rdesc):
        q, qvalid = qdesc
        r, rvalid = rdesc
        n, m = len(q), len(r)
        cosines, logits, masks, polar = [], [], [], []
        qavail = query.geometry.levels[0].slot_valid[:, 0].to(q)
        ravail = reference.geometry.levels[0].slot_valid[:, 0].to(q)
        availability = torch.cat((qavail, ravail)).expand(n, m, 4)
        for p in range(2):
            for l in range(3):
                for t in range(3):
                    if abs(l - t) > (1 if self.config.adjacent_scales else 0):
                        continue
                    qv = q[:, p, l, None, :].expand(n, m, -1)
                    rv = r[None, :, p, t, :].expand(n, m, -1)
                    cosine = (qv * rv).sum(-1).clamp(-1, 1)
                    mask = qvalid[:, p, l, None] & rvalid[None, :, p, t]
                    if self.edge_score is None:
                        logits.append(torch.zeros_like(cosine))
                    else:
                        qc = query.geometry.coverage[:, p, l, None].expand(n, m)
                        rc = reference.geometry.coverage[None, :, p, t].expand(n, m)
                        layer_id = torch.cat((F.one_hot(torch.tensor(l, device=q.device), 3),
                                              F.one_hot(torch.tensor(t, device=q.device), 3))).to(q)
                        # 13 metadata values: cosine 1, coverage 2, layer IDs 6,
                        # query/reference polarization availability 4. No A/omega.
                        metadata = torch.cat((cosine[..., None], qc[..., None], rc[..., None],
                                              layer_id.expand(n, m, 6), availability), -1)
                        logits.append(self.edge_score(torch.cat((qv, rv, metadata), -1)).squeeze(-1))
                    cosines.append(cosine)
                    masks.append(mask)
                    polar.append(p)
        valid = torch.stack(masks, -1)
        scores = torch.stack(logits, -1).masked_fill(~valid, -1e4)
        beta = torch.softmax(scores, -1) * valid
        beta = beta / beta.sum(-1, keepdim=True).clamp_min(1e-12)
        appearance = (beta * torch.stack(cosines, -1)).sum(-1)
        common = torch.stack([valid[..., [i for i, item in enumerate(polar) if item == p]].any(-1)
                              for p in range(2)], -1)
        return appearance, valid.any(-1), beta, valid, common

    def match_pair(self, query, reference, qdesc=None, rdesc=None):
        qdesc = self._descriptors(query) if qdesc is None else qdesc
        rdesc = self._descriptors(reference) if rdesc is None else rdesc
        appearance, valid, beta, branch_valid, common = self._edges(query, reference, qdesc, rdesc)
        mu, nu = query.geometry.mass, reference.geometry.mass
        pos, size = geometric_cost(query, reference, 1, self.config)
        reverse_pos, _ = geometric_cost(query, reference, -1, self.config)
        plus = appearance - self.config.position_weight * pos - self.config.size_weight * size
        minus = appearance - self.config.position_weight * reverse_pos - self.config.size_weight * size
        identical = bool(torch.allclose(plus.detach(), minus.detach(), atol=1e-7, rtol=0))
        candidates = []
        for direction, scores in [(1, plus)] + ([] if identical else [(-1, minus)]):
            transport = partial_transport(scores, valid, mu, nu, self.dustbin,
                                          self.config.ot_temperature, self.config.sinkhorn_iterations)
            real = transport[:-1, :-1]
            mass = real.sum(1)
            rate = mass / mu
            support = (real * ((appearance + 1) / 2)).sum(1) / mass.clamp_min(1e-12)
            evidence = rate * support
            candidates.append((direction, transport, rate, support, evidence,
                               (mu * evidence).sum()))
        best = max(range(len(candidates)), key=lambda i: float(candidates[i][-1].detach()))
        direction, transport, rate, support, evidence, quality = candidates[best]
        real, row_mass = transport[:-1, :-1], transport[:-1, :-1].sum(1)
        count = valid.sum(1)
        probability = real / row_mass[:, None].clamp_min(1e-12)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(1)
        entropy_valid = (count > 1) & (row_mass > 0)
        entropy = entropy / count.clamp_min(2).to(appearance).log()
        entropy = entropy * entropy_valid
        qcoverage = _region_union_coverage(query)
        rcoverage = _region_union_coverage(reference)
        rcoverage = (valid * rcoverage[None, :]).sum(1) / count.clamp_min(1)
        pair_fraction = (branch_valid.to(appearance).sum(-1) / 14 * valid).sum(1) / count.clamp_min(1)
        stats = torch.stack((rate, support, entropy, entropy_valid.to(appearance),
                             qcoverage, rcoverage, common[:, :, 0].any(1).to(appearance),
                             common[:, :, 1].any(1).to(appearance), pair_fraction), 1)
        return dict(evidence=evidence, stats=stats, valid=valid.any(1), transport=transport,
                    beta=beta, appearance=appearance, edge_valid=valid,
                    geometry_cost=self.config.position_weight * (pos if direction == 1 else reverse_pos)
                    + self.config.size_weight * size, direction=direction,
                    direction_identifiable=not identical, direction_solves=len(candidates),
                    rate=rate, mean_support=support, quality=quality)

    def _edges_batched(self, query, refs, qdesc, rdesc):
        q = torch.stack([item[0] for item in qdesc])
        qvalid = torch.stack([item[1] for item in qdesc])
        r = torch.stack([item[0] for item in rdesc])
        rvalid = torch.stack([item[1] for item in rdesc])
        batch, n, _, _, _ = q.shape
        m = r.shape[1]
        qcoverage = torch.stack([ship.geometry.coverage for ship in query])
        rcoverage = torch.stack([ship.geometry.coverage for ship in refs])
        availability = torch.stack([torch.cat((a.geometry.levels[0].slot_valid[:, 0],
                                               b.geometry.levels[0].slot_valid[:, 0]))
                                    for a, b in zip(query, refs)]).to(q)
        availability = availability[:, None, None, :].expand(batch, n, m, 4)
        cosines, logits, masks, polar = [], [], [], []
        # Keep the legacy branch order exactly: polarization, query scale, reference scale.
        for p in range(2):
            for l in range(3):
                for t in range(3):
                    if abs(l-t) > (1 if self.config.adjacent_scales else 0):
                        continue
                    qv = q[:, :, p, l, None, :].expand(batch, n, m, -1)
                    rv = r[:, None, :, p, t, :].expand(batch, n, m, -1)
                    cosine = (qv*rv).sum(-1).clamp(-1, 1)
                    mask = qvalid[:, :, p, l, None] & rvalid[:, None, :, p, t]
                    if self.edge_score is None:
                        logits.append(torch.zeros_like(cosine))
                    else:
                        qc = qcoverage[:, :, p, l, None].expand(batch, n, m)
                        rc = rcoverage[:, None, :, p, t].expand(batch, n, m)
                        layer_id = torch.cat((F.one_hot(torch.tensor(l, device=q.device), 3),
                                              F.one_hot(torch.tensor(t, device=q.device), 3))).to(q)
                        metadata = torch.cat((cosine[..., None], qc[..., None], rc[..., None],
                                              layer_id.expand(batch, n, m, 6), availability), -1)
                        logits.append(self.edge_score(torch.cat((qv, rv, metadata), -1)).squeeze(-1))
                    cosines.append(cosine); masks.append(mask); polar.append(p)
        valid = torch.stack(masks, -1)
        scores = torch.stack(logits, -1).masked_fill(~valid, -1e4)
        beta = torch.softmax(scores, -1)*valid
        beta = beta/beta.sum(-1, keepdim=True).clamp_min(1e-12)
        appearance = (beta*torch.stack(cosines, -1)).sum(-1)
        common = torch.stack([valid[..., [i for i, item in enumerate(polar) if item == p]].any(-1)
                              for p in range(2)], -1)
        return appearance, valid.any(-1), beta, valid, common

    def _match_group(self, query, refs, qdesc, rdesc, qcoverage, rcoverage):
        appearance, valid, beta, branch_valid, common = self._edges_batched(query, refs, qdesc, rdesc)
        mu = torch.stack([ship.geometry.mass for ship in query])
        nu = torch.stack([ship.geometry.mass for ship in refs])
        qi = torch.stack([ship.geometry.intervals for ship in query])
        ri = torch.stack([ship.geometry.intervals for ship in refs])
        axes = torch.tensor([a.geometry.axis_valid and b.geometry.axis_valid for a, b in zip(query, refs)],
                            dtype=torch.bool, device=appearance.device)[:, None, None]
        pos = ((qi[:, :, 0, None]-ri[:, None, :, 0]).abs()
               -.5*(qi[:, :, 1, None]+ri[:, None, :, 1])).clamp_min(0)
        reverse_pos = ((qi[:, :, 0, None]+ri[:, None, :, 0]).abs()
                       -.5*(qi[:, :, 1, None]+ri[:, None, :, 1])).clamp_min(0)
        size = ((qi[:, :, 1, None]/ri[:, None, :, 1]).log().abs()
                -math.log(self.config.size_tolerance)).clamp_min(0)
        pos, reverse_pos, size = [torch.where(axes, value, 0.) for value in (pos, reverse_pos, size)]
        plus = appearance-self.config.position_weight*pos-self.config.size_weight*size
        minus = appearance-self.config.position_weight*reverse_pos-self.config.size_weight*size
        identical = ((plus.detach()-minus.detach()).abs() <= 1e-7).flatten(1).all(1)
        transport = partial_transport_batched(plus, valid, mu, nu, self.dustbin,
                                              self.config.ot_temperature, self.config.sinkhorn_iterations)
        # Only identifiable pairs require a second solve, as in match_pair.
        reverse_ids = torch.where(~identical)[0]
        choose_reverse = torch.zeros(len(query), dtype=torch.bool, device=appearance.device)
        if len(reverse_ids):
            reverse = partial_transport_batched(minus[reverse_ids], valid[reverse_ids], mu[reverse_ids],
                                                  nu[reverse_ids], self.dustbin,
                                                  self.config.ot_temperature, self.config.sinkhorn_iterations)
            def quality(plan, app, mass):
                real = plan[:, :-1, :-1]
                row_mass = real.sum(-1)
                rate = row_mass/mass
                support = (real*((app+1)/2)).sum(-1)/row_mass.clamp_min(1e-12)
                return (mass*(rate*support)).sum(-1)
            forward_quality = quality(transport[reverse_ids], appearance[reverse_ids], mu[reverse_ids])
            reverse_quality = quality(reverse, appearance[reverse_ids], mu[reverse_ids])
            # Strict greater-than preserves legacy max's first-candidate tie rule.
            choose = reverse_quality.detach() > forward_quality.detach()
            choose_reverse[reverse_ids] = choose
            selected = torch.where(choose[:, None, None], reverse, transport[reverse_ids])
            transport = transport.index_copy(0, reverse_ids, selected)
        real = transport[:, :-1, :-1]
        row_mass = real.sum(-1)
        rate = row_mass/mu
        support = (real*((appearance+1)/2)).sum(-1)/row_mass.clamp_min(1e-12)
        evidence = rate*support
        quality = (mu*evidence).sum(-1)
        count = valid.sum(-1)
        probability = real/row_mass[:, :, None].clamp_min(1e-12)
        entropy = -(probability*probability.clamp_min(1e-12).log()).sum(-1)
        entropy_valid = (count > 1) & (row_mass > 0)
        entropy = entropy/count.clamp_min(2).to(appearance).log()*entropy_valid
        reference_coverage = (valid*torch.stack(rcoverage)[:, None, :]).sum(-1)/count.clamp_min(1)
        pair_fraction = (branch_valid.to(appearance).sum(-1)/14*valid).sum(-1)/count.clamp_min(1)
        stats = torch.stack((rate, support, entropy, entropy_valid.to(appearance), torch.stack(qcoverage),
                             reference_coverage, common[..., 0].any(-1).to(appearance),
                             common[..., 1].any(-1).to(appearance), pair_fraction), -1)
        geometry_cost = (self.config.position_weight*torch.where(choose_reverse[:, None, None], reverse_pos, pos)
                         +self.config.size_weight*size)
        directions = choose_reverse.detach().cpu().tolist()
        identities = identical.detach().cpu().tolist()
        return [dict(evidence=evidence[i], stats=stats[i], valid=valid[i].any(-1), transport=transport[i],
                     beta=beta[i], appearance=appearance[i], edge_valid=valid[i], geometry_cost=geometry_cost[i],
                     direction=-1 if directions[i] else 1, direction_identifiable=not identities[i],
                     direction_solves=1 if identities[i] else 2, rate=rate[i], mean_support=support[i],
                     quality=quality[i]) for i in range(len(query))]

    def _match_all_batched(self, query, refs, qtokens, rtokens, legal):
        groups = defaultdict(list)
        # Copy eligibility once, avoiding a CUDA synchronization per candidate pair.
        for qi, row in enumerate(legal.detach().cpu().tolist()):
            for ri, allowed in enumerate(row):
                if allowed:
                    groups[len(query[qi].geometry.area), len(refs[ri].geometry.area)].append((qi, ri))
        result = [{} for _ in query]
        qcoverage = [_region_union_coverage(ship) for ship in query]
        rcoverage = [_region_union_coverage(ship) for ship in refs]
        for indices in groups.values():
            for start in range(0, len(indices), self.config.pair_batch_size):
                pairs = indices[start:start+self.config.pair_batch_size]
                matches = self._match_group([query[q] for q, _ in pairs], [refs[r] for _, r in pairs],
                    [qtokens[q] for q, _ in pairs], [rtokens[r] for _, r in pairs],
                    [qcoverage[q] for q, _ in pairs], [rcoverage[r] for _, r in pairs])
                for (qi, ri), match in zip(pairs, matches):
                    result[qi][ri] = match
        # Preserve the public reference-index insertion order in details.
        return [dict(sorted(pairs.items())) for pairs in result]

    def forward(self, query, refs, ref_labels, ref_ports, legal=None):
        if len(refs) != len(ref_ports) or len(refs) != len(ref_labels):
            raise ValueError("reference metadata lengths differ")
        if not query:
            raise ValueError("query batch is empty")
        if legal is None:
            legal = torch.ones(len(query), len(refs), dtype=torch.bool,
                               device=query[0].content[0].device)
        if tuple(legal.shape) != (len(query), len(refs)):
            raise ValueError("legal must have shape [queries, references]")
        labels = [int(x) for x in ref_labels]
        if any(c < 0 or c >= self.config.classes for c in labels):
            raise ValueError("reference class outside configured range")
        groups = defaultdict(list)
        for index, (label, port) in enumerate(zip(labels, ref_ports)):
            groups[label, port].append(index)
        z0 = self.forward_base(query)
        qtokens = [self._descriptors(ship) for ship in query]
        rtokens = [self._descriptors(ship) for ship in refs]
        batched_pairs = (self._match_all_batched(query, refs, qtokens, rtokens, legal)
                         if self.config.batched_matching else None)
        evidence_rows, support_rows, weights, details = [], [], [], []
        for qi, ship in enumerate(query):
            omega = self.region_weights(ship)
            pairs = (batched_pairs[qi] if batched_pairs is not None else
                     {ri: self.match_pair(ship, ref, qtokens[qi], rtokens[ri])
                      for ri, ref in enumerate(refs) if bool(legal[qi, ri])})
            class_evidence, class_support = [], []
            mixtures = {}
            for c in range(self.config.classes):
                ports, port_valid = [], []
                for (label, port), indices in groups.items():
                    if label != c:
                        continue
                    selected = [ri for ri in indices if ri in pairs]
                    if not selected:
                        continue
                    values = torch.stack([pairs[ri]["evidence"] for ri in selected])
                    mask = torch.stack([pairs[ri]["valid"] for ri in selected])
                    stats = torch.stack([pairs[ri]["stats"] for ri in selected])
                    uniform = mask / mask.sum(0, keepdim=True).clamp_min(1)
                    if self.prototype_score is None:
                        pi = uniform
                    else:
                        logits = self.prototype_score(stats).squeeze(-1) / self.config.prototype_temperature
                        learned = torch.softmax(logits.masked_fill(~mask, -1e4), dim=0) * mask
                        learned = learned / learned.sum(0, keepdim=True).clamp_min(1e-12)
                        floor = self.config.prototype_floor if self.config.prototype_weighting == "floor" else 0.
                        pi = (1 - floor) * learned + floor * uniform
                    ports.append((pi * values).sum(0))
                    port_valid.append(mask.any(0))
                    mixtures[c, port] = dict(indices=selected, weights=pi)
                if ports:
                    active = torch.stack(port_valid)
                    per_region = torch.stack(ports).sum(0) / active.sum(0).clamp_min(1)
                    class_evidence.append((omega * per_region).sum())
                    class_support.append(active.any())
                else:
                    class_evidence.append(self.dustbin * 0)
                    class_support.append(torch.tensor(False, device=z0.device))
            evidence_rows.append(torch.stack(class_evidence))
            support_rows.append(torch.stack(class_support))
            weights.append(omega)
            details.append(dict(pairs=pairs, mixtures=mixtures))
        evidence = torch.stack(evidence_rows)
        logits = z0 + self.config.gamma * evidence / self.config.evidence_temperature
        return dict(logits=logits, evidence=evidence, z0=z0,
                    support=torch.stack(support_rows), weights=weights, details=details)

    def loss(self, output, labels):
        final = F.cross_entropy(output["logits"], labels)
        rows = torch.arange(len(labels), device=labels.device)
        supported = output["support"][rows, labels]
        if bool(supported.any()):
            match = F.cross_entropy(output["evidence"][supported] / self.config.evidence_temperature,
                                    labels[supported])
        else:
            match = output["evidence"].sum() * 0
        return dict(total=final + self.config.match_loss_weight * match,
                    final=final, match=match, match_samples=supported.sum())
