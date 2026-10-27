"""M1/M2 inside a stride-one bottleneck; M3 keeps protected comparison keys."""
import copy

import torch
from torch import nn

from .selective_backbone import SelectiveBackbone
from .selective_readout import EvidencePool, SpectralClassResponseFeedback
from .pool_replacement import grid_occupancy


class RegionPoolWriteback(nn.Module):
    """Pool three named regions and return each token only to its own footprint.

    This is context pooling/writeback, not spatial downsampling. Zero output
    projections preserve the pretrained path initially without blocking their
    first-step gradients. A zero scalar gain is deliberately not stacked on top.
    """
    def __init__(self, channels, hidden=32):
        super().__init__()
        self.pool = EvidencePool(channels)
        self.transforms = nn.ModuleList(nn.Sequential(nn.Linear(channels, hidden),
            nn.GELU(), nn.Linear(hidden, channels, bias=False)) for _ in range(3))
        for transform in self.transforms:
            nn.init.zeros_(transform[-1].weight)
        self.enabled = True

    def forward(self, x, masks, valid, energy):
        d = self.pool(x, masks, valid, energy, return_details=True)
        values = torch.stack([f(d['pooled'][:, r]) for r, f in enumerate(self.transforms)], 1)
        occupancy = grid_occupancy(valid.to(x), x.shape[-2:]).clamp(0, 1)
        weights = torch.minimum(masks.to(x).clamp(0, 1), occupancy)
        weights = weights/weights.sum(1, keepdim=True).clamp_min(1)
        update = torch.einsum('brhw,brc->bchw', weights, values)
        d.update(write_weights=weights, update=update)
        return x+update if self.enabled else x, d


class BottleneckEvidenceResNet50(SelectiveBackbone):
    """One insertion before/after conv2 of layer3[-1], or original stage layout.

    Only the last bottleneck may be fine-tuned. Its frozen pretrained copy
    supplies raw M3 anchors, preventing accidental comparison-key drift.
    Query/reference use the same current live block and per-polarization M1/M2.
    """
    def __init__(self, weight_path, position='pre3x3', policy='frozen', classes=8):
        if position not in ('stage', 'pre3x3', 'post3x3'):
            raise ValueError(position)
        if policy not in ('frozen', 'local'):
            raise ValueError(policy)
        super().__init__(weight_path, classes, m1_stages=(3,) if position == 'stage' else (),
                         m2_deep=True)
        self.position, self.policy = position, policy
        self.anchor_last = copy.deepcopy(self.backbone.model.layer3[-1]).requires_grad_(False)
        self.anchor_last.eval()
        self.internal_m1 = nn.ModuleList()
        self.internal_m2 = nn.ModuleList()
        if position != 'stage':
            self.internal_m1.extend(SpectralClassResponseFeedback(256, classes) for _ in range(2))
            self.internal_m2.extend(RegionPoolWriteback(256) for _ in range(2))
        if policy == 'local':
            self.backbone.model.layer3[-1].requires_grad_(True)
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.anchor_last.eval()
        return self

    def intervene(self, z, valid, available, field, slots, energy, aux, trace):
        b = len(valid)
        z = z.reshape(b, 2, *z.shape[1:])
        output, logits, details = [], [], []
        for p in range(2):
            occupancy = grid_occupancy(valid[:, p], z.shape[-2:])
            value, m1 = self.internal_m1[p](z[:, p], occupancy, field[:, p], True)
            regions, bands = self.region_integrals(slots[:, p], energy[:, p])
            value, m2 = self.internal_m2[p](value, regions, valid[:, p], bands)
            output.append(value)
            logits.append(m1['aux_logits']*available[:, p, None])
            details.append(dict(m1=m1, m2=m2))
        aux.append(torch.stack(logits, 1).sum(1)/available.sum(1, keepdim=True).clamp_min(1))
        trace['bottleneck'] = details
        return torch.stack(output, 1).flatten(0, 1)

    def live_last(self, z, valid, available, field, slots, energy, aux, trace):
        block = self.backbone.model.layer3[-1]
        identity = z
        value = block.relu(block.bn1(block.conv1(z)))
        if self.position == 'pre3x3':
            value = self.intervene(value, valid, available, field, slots, energy, aux, trace)
        value = block.relu(block.bn2(block.conv2(value)))
        if self.position == 'post3x3':
            value = self.intervene(value, valid, available, field, slots, energy, aux, trace)
        value = block.bn3(block.conv3(value))
        if block.downsample is not None:
            identity = block.downsample(identity)
        return block.relu(value+identity).reshape(len(valid), 2, 1024, 10, 10)

    def stage3_paths(self, prefix, valid, available, field2, field3, reference,
                     rv, slot_masks, frequency, trace, aux):
        with torch.no_grad():
            q, r = prefix.flatten(0, 1), reference['prefix'].flatten(0, 1)
            for block in list(self.backbone.model.layer3.children())[:-1]:
                q, r = block(q), block(r)
            raw3 = self.anchor_last(q).reshape(len(prefix), 2, 1024, 10, 10)
            rr3 = self.anchor_last(r).reshape(len(rv), 2, 1024, 10, 10)
        ignored = []
        x = self.live_last(q, valid, available, field3, slot_masks, frequency, aux, trace)
        ref = self.live_last(r, rv, reference['available'], reference['field3'],
                             reference['slot_masks'], reference['frequency'], ignored, {})
        if self.position == 'stage':
            x = self.apply_m1(x, valid, field3, 3, trace, aux, available)
            ref = self.apply_m1(ref, rv, reference['field3'], 3, {}, ignored, reference['available'])
        return prefix, x, ref, raw3, rr3
