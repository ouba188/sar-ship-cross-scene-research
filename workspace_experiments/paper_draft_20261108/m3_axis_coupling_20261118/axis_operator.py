"""Ship-relative geometry and a shared axis hypothesis in the actual value path."""
from pathlib import Path
import math
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'m3_relation_context_20261118'))
from relation_operator import RelationContextMatch
from operator_block import weighted_cosine
from sar_m123.tokens import REGIONS


def slot_positions():
    body = [(u, v) for u in (-.375, -.125, .125, .375) for v in (-.25, .25)]
    side = [(u, v) for u in (-.375, -.125, .125, .375) for v in (-.75, .75)]
    ends = [(u, v) for u in (-.625, .625) for v in (-.25, .25)]
    return torch.tensor(body+side+ends)


REVERSE = torch.tensor(list(range(7, -1, -1))+list(range(15, 7, -1))+list(range(19, 15, -1)))


class AxisCoupledMatch(RelationContextMatch):
    regions = REGIONS

    def __init__(self, seed=17, classes=8, geometry='axis', coupling='paired', strength=.25):
        super().__init__(seed=seed, classes=classes)
        if geometry not in ('none', 'signed', 'axis') or coupling not in ('paired', 'aggregate'):
            raise ValueError((geometry, coupling))
        self.geometry, self.coupling = geometry, coupling
        self.position_strength_raw = torch.nn.Parameter(torch.full((2,), math.log(math.expm1(strength))))
        self.register_buffer('positions', slot_positions(), persistent=False)

    def region_candidates(self, cost, qa, ra, allowed, ix, pol):
        if self.geometry == 'none':
            return self.transport(cost, qa, ra, allowed)[..., None, :, :], cost[..., None, :, :]
        pos = self.positions[ix].to(cost)
        signs = (1., -1.) if self.geometry == 'axis' else (1.,)
        costs, plans = [], []
        for sign in signs:
            distance = (pos[:, None]-sign*pos[None]).square().sum(-1)
            adjusted = cost-F.softplus(self.position_strength_raw[pol])*distance
            costs.append(adjusted)
            plans.append(self.transport(adjusted, qa, ra, allowed))
        return torch.stack(plans, -3), torch.stack(costs, -3)

    def orientation_weights(self, scores):
        return (scores/.1).softmax(-1)

    def comparison_distance(self, query, reference, plans, orientation, class_beta,
                            metric, ref_labels, aligned, prototype):
        distance = ((query[:, None]-prototype).square()*metric[:, :, None]).sum(-1)
        return distance, {}

    def forward(self, q, r, qm, rm, write, legal, ref_labels, arm='aggregate', details=False):
        if len(r) == 0:
            return super().forward(q, r, qm, rm, write, legal, ref_labels, arm=arm, details=details)
        if arm not in ('competitive', 'shared', 'aggregate'):
            raise ValueError(arm)
        self.context_traces = []
        labels = F.one_hot(ref_labels.long(), self.classes).to(q)
        updates, sums, counts, traces = [], [], [], []
        for p in range(2):
            kq, vq = self.encode(q[:, p], p)
            kr, vr = self.encode(r[:, p], p)
            records, hypothesis_scores, hypothesis_valid = [], [], []
            for region in self.regions:
                ix = region.tolist()
                qa, ra = qm[:, p, ix], rm[:, p, ix]
                live_ref = legal[:, :, p] & (ra.sum(-1) > 0)[None] & (qa.sum(-1) > 0)[:, None]
                mass = live_ref[..., None]*ra[None]
                total = torch.einsum('brj,rc->bc', mass, labels)
                available = total > 0
                mu = torch.einsum('brj,rc,rjd->bcd', mass, labels, kr[:, ix])/total[..., None].clamp_min(1e-20)
                n = available.sum(-1, keepdim=True)
                all_mu = mu.sum(1, keepdim=True)/n[..., None].clamp_min(1)
                other = self.other_class_context(mu, available)
                w = 1+.5*self.metric(self.metric_condition(mu, other, all_mu, arm)).tanh()
                sim = weighted_cosine(kq[:, ix], kr[:, ix], w[:, ref_labels])
                qa = qa/qa.sum(-1, keepdim=True).clamp_min(1e-20)
                ra = ra/ra.sum(-1, keepdim=True).clamp_min(1e-20)
                allowed = live_ref[..., None, None] & (qa > 0)[:, None, :, None] & (ra > 0)[None, :, None, :]
                best = torch.stack([sim.masked_fill(~(allowed & (ref_labels == c)[None, :, None, None]),
                                                   -1e4).amax((1, 3)) for c in range(self.classes)], 1)
                active = available[..., None] & (qa > 0)[:, None]
                best = best*active
                others = (best.sum(1, keepdim=True)-best)/(n-available.long()).clamp_min(1)[..., None]
                competition = self.cost_competition((best-others)*active*(n > 1)[..., None], arm)
                cost = sim+competition[:, ref_labels, :, None]
                plans, costs = self.region_candidates(cost, qa, ra, allowed, ix, p)
                mass_h = plans.sum((-1, -2))
                score_h = (plans*costs).sum((-1, -2))/mass_h.clamp_min(1e-20)
                hypothesis_scores.append(score_h)
                hypothesis_valid.append(mass_h > 0)
                records.append((ix, qa, available, active, w, sim, competition, plans))
            # One direction distribution per query/reference/polarization, shared by all regions.
            hs, hv = torch.stack(hypothesis_scores), torch.stack(hypothesis_valid)
            hs = (hs*hv).sum(0)/hv.sum(0).clamp_min(1)
            orientation = self.orientation_weights(hs)
            fields, regions = [], []
            ds, dn = q.new_zeros((len(q), self.classes)), q.new_zeros((len(q), self.classes))
            for ix, qa, available, active, w, sim, competition, plans in records:
                original = (plans*orientation[..., None, None]).sum(-3)
                row, col = original.sum(-1), original.sum(-2)
                score = (original*sim).sum(-1)/row.clamp_min(1e-20)
                live = row > 0
                beta = (score/.1).masked_fill(~live, -1e4).softmax(1)*live
                beta = beta/beta.sum(1, keepdim=True).clamp_min(1e-20)
                transport = original
                if self.coupling == 'aggregate':
                    transport = row[..., None]*col[..., None, :]/row.sum(-1)[..., None, None].clamp_min(1e-20)
                aligned = torch.einsum('brij,rjd->brid', transport, vr[:, ix])/row[..., None].clamp_min(1e-20)
                class_beta = beta[:, :, None]*labels[None, :, :, None]
                denom = class_beta.sum(1)
                class_beta = class_beta/denom[:, None].clamp_min(1e-20)
                # Average references of the same class, NOT query locations.
                proto = torch.einsum('brci,brid->bcid', class_beta, aligned)
                active = (denom > 0) & active
                distance, comparison_trace = self.comparison_distance(
                    vq[:, ix], vr[:, ix], plans, orientation, class_beta,
                    w, ref_labels, aligned, proto)
                evidence = (-distance/.1)*active
                centered = (evidence-evidence.sum(1, keepdim=True)/active.sum(1, keepdim=True).clamp_min(1))*active
                message = self.write_message(centered, proto, active, p)
                fields.append(message)
                aw = qa[:, None]*active
                ds = ds+(distance*aw).sum(-1)/aw.sum(-1).clamp_min(1e-20)
                dn = dn+(aw.sum(-1) > 0)
                if details:
                    regions.append(dict(transport=transport, original_transport=original, beta=beta,
                                        value=vr[:, ix], query_value=vq[:, ix], metric=w, sim=sim,
                                        competition=competition, available=available, evidence=centered,
                                        message=message, prototype=proto, orientation=orientation,
                                        hypothesis_transport=plans, **comparison_trace))
            field = torch.cat(fields, 1)
            mixed = torch.einsum('bshw,bsc->bchw', write[:, p], field)
            updates.append(.1*self.output[p](mixed))
            sums.append(ds)
            counts.append(dn)
            if details:
                traces.append(dict(regions=regions, mixed=mixed, message=field))
        count = torch.stack(counts).sum(0)
        local = (-torch.stack(sums).sum(0)/count.clamp_min(1)/.1/self.value[0].out_features).masked_fill(count <= 0, -1e4)
        return torch.stack(updates, 1), local, count > 0, traces


def install(network, geometry='axis', coupling='paired'):
    previous = network.matcher
    parameter = next(previous.parameters())
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        current = AxisCoupledMatch(classes=previous.classes, geometry=geometry, coupling=coupling)
    current.to(device=parameter.device, dtype=parameter.dtype)
    loaded = current.load_state_dict(previous.state_dict(), strict=False)
    assert loaded.missing_keys == ['position_strength_raw'] and not loaded.unexpected_keys
    flags = dict(previous.named_parameters())
    for name, value in current.named_parameters():
        if name in flags:
            value.requires_grad_(flags[name].requires_grad)
    current.train(previous.training)
    network.matcher = current
    return current
