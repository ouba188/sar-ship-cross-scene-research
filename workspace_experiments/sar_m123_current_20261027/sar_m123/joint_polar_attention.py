"""Region/class-conditioned M1 updates with a shared backbone and separate views."""

import torch
from torch import nn
from torch.nn import functional as F

from .bottleneck_evidence import BottleneckEvidenceResNet50
from .pool_replacement import grid_occupancy


class JointPolarCondition(nn.Module):
    """Exchange descriptors, never pixels or value tokens, between VV and VH.

    Each view supplies region-wise mean/high/low class responses. Per-view MLPs
    condition the existing M1 class/side contributions on these descriptors.
    self is an equal-capacity control using its own descriptor in both slots.
    Empty corresponding regions fall back to the unmodified M1 update.
    """

    def __init__(self, classes=8, hidden=32, mode='joint'):
        super().__init__()
        if mode not in ('baseline', 'self', 'joint'):
            raise ValueError(mode)
        self.classes, self.mode, self.enabled = classes, mode, True
        self.selectors = nn.ModuleList(nn.Sequential(
            nn.Linear(6*classes+7, hidden), nn.GELU(), nn.Linear(hidden, 3*classes))
            for _ in range(2))
        for selector in self.selectors:
            nn.init.zeros_(selector[-1].weight)
            nn.init.zeros_(selector[-1].bias)
        self.requires_grad_(mode != 'baseline')

    @staticmethod
    def summarize(detail, masks):
        response = detail['class_response'].float()
        region = torch.minimum(masks.float().clamp(0, 1), detail['valid'].float())
        response = response.masked_fill(detail['valid'] == 0, 0)
        response = response-response.mean(1, keepdim=True)
        mass = region.sum((-2, -1))
        weights = region.flatten(2)/mass.clamp_min(1e-8)[..., None]
        values = response.flatten(2)[:, None]
        mean = (weights[:, :, None]*values).sum(-1)
        log_mass = weights.clamp_min(1e-20).log()[:, :, None]
        high, low = [], []
        for sign, output in ((1, high), (-1, low)):
            logits = (log_mass+sign*values).masked_fill(weights[:, :, None] == 0, -1e4)
            selected = logits.softmax(-1)*(weights[:, :, None] > 0)
            selected = selected/selected.sum(-1, keepdim=True).clamp_min(1e-8)
            output.append((selected*values).sum(-1))
        descriptor = torch.stack((mean, high[0], low[0]), 2).flatten(2)
        descriptor = F.layer_norm(descriptor, (descriptor.shape[-1],))*(mass > 0)[..., None]
        return descriptor, region, mass

    def forward(self, operators, details, masks, available):
        summary = [self.summarize(details[p], masks[:, p]) for p in range(2)]
        updates = []
        for p, operator in enumerate(operators):
            own, region, mass = summary[p]
            opposite, _, opposite_mass = summary[1-p]
            other = own if self.mode == 'self' else opposite
            active = (mass > 0) & (opposite_mass > 0) & available.bool().all(1)[:, None]
            occupancy = torch.stack((mass, opposite_mass), -1)/(region.shape[-2]*region.shape[-1])
            presence = available[:, [p, 1-p]].to(own)[:, None].expand(-1, 3, -1)
            identity = torch.eye(3, device=own.device, dtype=own.dtype)[None].expand(len(own), -1, -1)
            context = torch.cat((own, other, occupancy, presence, identity), -1)
            logits = self.selectors[p](context).reshape(len(own), 3, 3, self.classes)
            delta = .5*logits.tanh()*active[..., None, None]
            if not self.enabled or self.mode == 'baseline':
                delta = torch.zeros_like(delta)
            # The gate changes class/side contributions, not the evidence reader.
            write = region/region.sum(1, keepdim=True).clamp_min(1)
            field = torch.einsum('brhw,brsk->bskhw', write, delta)
            d = details[p]
            changed = d['write_weights']*field.to(d['write_weights'])
            transformed = torch.stack([layer(d['tokens'][:, s])
                                       for s, layer in enumerate(operator.transforms)], 1)
            mixed = torch.einsum('bskhw,bskd->bdhw', changed.to(transformed), transformed)
            update = operator.gamma*operator.output(mixed)*d['channel_weights']
            d.update(condition_gate=1+delta, condition_delta=update,
                     condition_descriptor=own, condition_other_descriptor=other,
                     condition_active=active, condition_region_weights=write)
            updates.append(update)
        return updates


class JointPolarBottleneckResNet50(BottleneckEvidenceResNet50):
    """Only M1 conditioning differs; M2/M3 and local training remain unchanged."""

    def __init__(self, weight_path, conditioning='joint', classes=8):
        super().__init__(weight_path, position='pre3x3', policy='local', classes=classes)
        self.cross_condition = JointPolarCondition(classes, mode=conditioning)

    def intervene(self, z, valid, available, field, slots, energy, aux, trace):
        b = len(valid)
        z = z.reshape(b, 2, *z.shape[1:])
        first, m1s, regions, bands = [], [], [], []
        for p in range(2):
            occupancy = grid_occupancy(valid[:, p], z.shape[-2:])
            value, m1 = self.internal_m1[p](z[:, p], occupancy, field[:, p], True)
            region, band = self.region_integrals(slots[:, p], energy[:, p])
            first.append(value)
            m1s.append(m1)
            regions.append(region)
            bands.append(band)
        additions = self.cross_condition(self.internal_m1, m1s, torch.stack(regions, 1), available)
        output, logits, details = [], [], []
        for p in range(2):
            value = first[p]+additions[p]
            m1s[p]['unconditioned_update'] = m1s[p]['update']
            m1s[p]['update'] = m1s[p]['update']+additions[p]
            value, m2 = self.internal_m2[p](value, regions[p], valid[:, p], bands[p])
            output.append(value)
            logits.append(m1s[p]['aux_logits']*available[:, p, None])
            details.append(dict(m1=m1s[p], m2=m2))
        aux.append(torch.stack(logits, 1).sum(1)/available.sum(1, keepdim=True).clamp_min(1))
        trace['bottleneck'] = details
        return torch.stack(output, 1).flatten(0, 1)
