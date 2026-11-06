"""Reference transport with separately selectable layer and frequency evidence.

Comparison anchors are frozen raw features, independent of M1/value learning.
Frequency weights pool image features; frequency distance is not a class prior.
"""
import torch
from torch import nn
from torch.nn import functional as F

from . import tokens
from .m3 import sinkhorn
from .selective_readout import EvidencePool
from .pool_replacement import grid_occupancy


class SelectiveReferenceMatch(nn.Module):
    def __init__(self, channels=1024, dynamic=True, frequency=True, temperature=.1):
        super().__init__()
        self.frequency, self.dynamic, self.temperature = frequency, dynamic, temperature
        self.reader2 = EvidencePool(512, adaptive=False)
        self.reader3 = EvidencePool(1024, adaptive=False)
        self.reader2.requires_grad_(False)
        self.reader3.requires_grad_(False)
        self.selector = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 5))
        nn.init.zeros_(self.selector[-1].weight)
        nn.init.zeros_(self.selector[-1].bias)
        self.branch_bias = nn.Parameter(torch.tensor([.5, .25, 1/12, 1/12, 1/12]).log())
        self.selector.requires_grad_(dynamic)
        self.project = nn.Conv2d(channels, 32, 1, bias=False)
        self.output = nn.Conv2d(32, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.gain = nn.Parameter(torch.tensor(.1))

    def descriptors(self, anchors, local, slots, local_slots, valid, energy, local_energy):
        with torch.no_grad():
            deep = self.reader3(anchors.detach(), slots, valid, energy)
            fine = self.reader2(local.detach(), local_slots, valid, local_energy)
        streams = [deep['mean'], fine['mean']]+list(fine['bands'].unbind(2))
        return [F.normalize(t, dim=-1) for t in streams], deep['mass']

    def forward(self, x, anchor, local, slots, local_slots, valid, energy, local_energy,
                reference, pair_valid, return_details=False):
        qs, qm = self.descriptors(anchor, local, slots, local_slots, valid, energy, local_energy)
        rs, rm = self.descriptors(reference['anchor'], reference['local'], reference['slots'],
                                 reference['local_slots'], reference['valid'], reference['energy'], reference['local_energy'])
        source = reference['features']
        read_masks = torch.minimum(reference['slots'].clamp(0, 1), grid_occupancy(reference['valid'].to(source), source.shape[-2:]).clamp(0, 1))
        z = F.layer_norm(source.permute(0, 2, 3, 1), (source.shape[1],)).permute(0, 3, 1, 2)
        value = torch.einsum('kshw,kdhw->ksd', read_masks, self.project(z))/rm.clamp_min(1e-8)[..., None]
        parts, all_weights, ref_scores, ref_live = [], [], [], []
        traces = {k: [] for k in ('transport', 'similarity', 'branch_similarity', 'branch_weights')}
        for region in tokens.REGIONS:
            ix = region.tolist()
            evidence = torch.stack([torch.einsum('bid,kjd->bkij', q[:, ix], r[:, ix]) for q, r in zip(qs, rs)], -1)
            if not self.frequency:
                evidence = evidence.clone()
                evidence[..., 2:] = 0
            logits = self.branch_bias.expand_as(evidence)
            if self.dynamic:
                logits = logits + self.selector(evidence)
            if not self.frequency:
                logits = logits.clone()
                logits[..., 2:] = -torch.inf
            branch_weights = logits.softmax(-1)
            sim = (branch_weights*evidence).sum(-1)
            qa, ra = qm[:, None, ix], rm[None, :, ix]
            allowed = (qa > 0)[..., :, None] & (ra > 0)[..., None, :] & pair_valid[..., None, None]
            transport = sinkhorn(sim, qa, ra)*allowed.to(sim)
            row_mass = transport.sum(-1)
            score = (transport*sim).sum(-1)/row_mass.clamp_min(1e-20)
            available = row_mass > 0
            weights = (score/self.temperature).masked_fill(~available, -1e4).softmax(1)*available
            weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-20)
            moved = torch.einsum('bkij,kjd->bkid', transport, value[:, ix])
            offered = qa/qa.sum(-1, keepdim=True).clamp_min(1e-20)
            moved = moved/offered[..., None].clamp_min(1e-20)
            parts.append((weights[..., None]*moved).sum(1))
            all_weights.append(weights)
            mass = row_mass.sum(-1)
            ref_scores.append((score*row_mass).sum(-1)/mass.clamp_min(1e-20))
            ref_live.append(mass > 0)
            if return_details:
                for key, t in zip(traces, (transport, sim, evidence, branch_weights)):
                    traces[key].append(t)
        live = torch.stack(ref_live, -1)
        relation = (torch.stack(ref_scores, -1)*live).sum(-1)/live.sum(-1).clamp_min(1)
        relation = (relation/self.temperature).masked_fill(~live.any(-1), -1e4)
        write_masks = torch.minimum(slots.clamp(0, 1), grid_occupancy(valid.to(x), x.shape[-2:]).clamp(0, 1))
        write = write_masks/write_masks.sum(1, keepdim=True).clamp_min(1)
        mixed = torch.einsum('bshw,bsd->bdhw', write, torch.cat(parts, 1))
        update = self.gain*self.output(mixed)
        detail = dict(update=update, reference_logits=relation, reference_valid=live.any(-1),
                      reference_weights=torch.cat(all_weights, -1), **{k: tuple(v) for k, v in traces.items()})
        return x+update, detail
