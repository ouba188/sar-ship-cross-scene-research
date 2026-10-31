"""Bounded M1/M2/M3 experiment with native-grid, per-polarization operators."""
import torch
from torch import nn

from . import tokens
from .pool_replacement import PoolReplacementResNet50, grid_occupancy, regional_mean
from .selective_readout import EvidencePool, SpectralClassResponseFeedback
from .selective_matching import SelectiveReferenceMatch


class SelectiveBackbone(PoolReplacementResNet50):
    cache_start = 2

    def __init__(self, weight_path, classes=8, m1_stages=(3,), m3_count=1,
                 m2_select=True, dynamic=True, frequency=True, m2_deep=False):
        super().__init__(weight_path, classes=classes, mode='readout24_region')
        if any(s not in (2, 3) for s in m1_stages) or m3_count not in (0, 1, 2):
            raise ValueError('This pilot supports M1 at layer2/3 and 0/1/2 M3 blocks after layer3.')
        self.m1_stages, self.m3_count = tuple(m1_stages), m3_count
        self.m2_select, self.m2_deep = m2_select, m2_deep
        self.m1 = nn.ModuleDict({f'{s}_{p}': SpectralClassResponseFeedback({2:512, 3:1024}[s], classes)
                                 for s in m1_stages for p in range(2)})
        self.m2 = nn.ModuleList(EvidencePool(512, adaptive=dynamic) for _ in range(2))
        self.deep_pool = nn.ModuleList(EvidencePool(1024, adaptive=dynamic) for _ in range(2)) if m2_deep else nn.ModuleList()
        self.deep_output = nn.ModuleList(nn.Linear(1024, 512, bias=False) for _ in range(2)) if m2_deep else nn.ModuleList()
        for layer in self.deep_output:
            nn.init.zeros_(layer.weight)
        self.m3 = nn.ModuleList(nn.ModuleList(SelectiveReferenceMatch(dynamic=dynamic, frequency=frequency)
                                             for _ in range(2)) for _ in range(m3_count))

    @staticmethod
    def region_integrals(slots, energy):
        ordered = [tokens.REGIONS[i] for i in (0, 2, 1)]
        return (torch.stack([slots[:, r.tolist()].sum(1) for r in ordered], 1),
                torch.stack([energy[:, r.tolist()].sum(1) for r in ordered], 1))

    def apply_m1(self, x, valid, field, stage, trace, auxiliary, available):
        if stage not in self.m1_stages:
            return x
        output, aux, details = [], [], []
        for p in range(2):
            y, d = self.m1[f'{stage}_{p}'](x[:, p], grid_occupancy(valid[:, p], x.shape[-2:]),
                                          energy_field=field[:, p], return_details=True)
            output.append(y)
            aux.append(d['aux_logits']*available[:, p, None])
            details.append(d)
        auxiliary.append(torch.stack(aux, 1).sum(1)/available.sum(1, keepdim=True).clamp_min(1))
        trace[f'm1_{stage}'] = details
        return torch.stack(output, 1)

    def stage3_paths(self, prefix, valid, available, field2, field3, reference,
                     rv, slot_masks, frequency, trace, aux,
                     local_slot_masks=None, local_frequency=None):
        """Build protected comparison anchors and the differentiable value path."""
        b, p = prefix.shape[:2]
        # The comparison path never passes through trainable M1/M3 blocks.
        with torch.no_grad():
            raw3 = self.backbone.model.layer3(prefix.flatten(0, 1)).reshape(b, p, 1024, 10, 10)
            rr3 = self.backbone.model.layer3(reference['prefix'].flatten(0, 1)).reshape(-1, p, 1024, 10, 10)
        ignored = []
        x2 = self.apply_m1(prefix, valid, field2, 2, trace, aux, available)
        r2 = self.apply_m1(reference['prefix'], rv, reference['field2'], 2, {}, ignored, reference['available'])
        if 2 in self.m1_stages:
            x = self.backbone.model.layer3(x2.flatten(0, 1)).reshape_as(raw3)
            r = self.backbone.model.layer3(r2.flatten(0, 1)).reshape_as(rr3)
        else:
            x, r = raw3, rr3
        x = self.apply_m1(x, valid, field3, 3, trace, aux, available)
        r = self.apply_m1(r, rv, reference['field3'], 3, {}, ignored, reference['available'])
        return x2, x, r, raw3, rr3, prefix, reference['prefix']

    def forward_from(self, prefix, masks, valid, available, start=2, *, slot_masks, frequency,
                     local_slot_masks, local_frequency, field2, field3, reference, pair_valid,
                     return_details=False):
        if start != self.cache_start:
            raise ValueError(f'This architecture requires a frozen layer{self.cache_start} prefix.')
        b, p = prefix.shape[:2]
        valid = valid*available[..., None, None, None]
        rv = reference['valid']*reference['available'][..., None, None, None]
        trace, aux = {}, []
        x2, x, r, raw3, rr3, comparison2, reference_comparison2 = self.stage3_paths(
            prefix, valid, available, field2, field3, reference, rv, slot_masks,
            frequency, trace, aux, local_slot_masks, local_frequency)
        relations = []
        for block, operators in enumerate(self.m3):
            output, details = [], []
            for pol in range(p):
                bank = dict(features=r[:, pol], anchor=rr3[:, pol], local=reference_comparison2[:, pol],
                            slots=reference['slot_masks'][:, pol], local_slots=reference['local_slot_masks'][:, pol],
                            valid=rv[:, pol], energy=reference['frequency'][:, pol], local_energy=reference['local_frequency'][:, pol])
                legal = pair_valid & available[:, pol, None].bool() & reference['available'][None, :, pol].bool()
                y, d = operators[pol](x[:, pol], raw3[:, pol], comparison2[:, pol], slot_masks[:, pol],
                    local_slot_masks[:, pol], valid[:, pol], frequency[:, pol], local_frequency[:, pol],
                    bank, legal, return_details=return_details)
                output.append(y)
                details.append(d)
                relations.append((d['reference_logits'], d['reference_valid']))
            x = torch.stack(output, 1)
            trace[f'm3_{block}'] = details
        deep = self.backbone.model.layer4(x.flatten(0, 1))
        glob = regional_mean(deep, valid.flatten(0, 1), valid.flatten(0, 1))[0][:, 0].reshape(b, p, -1)
        pooled, pool_trace = [], []
        for pol in range(p):
            region, energy = self.region_integrals(local_slot_masks[:, pol], local_frequency[:, pol])
            d = self.m2[pol](x2[:, pol], region, valid[:, pol], energy)
            value = d['pooled'] if self.m2_select else d['mean']
            if self.m2_deep:
                rma, ren = self.region_integrals(slot_masks[:, pol], frequency[:, pol])
                extra = self.deep_pool[pol](x[:, pol], rma, valid[:, pol], ren)
                value = value+self.deep_output[pol](extra['pooled'] if self.m2_select else extra['mean'])
            pooled.append(value.flatten(1))
            pool_trace.append(d)
        values = torch.cat((glob, torch.stack(pooled, 1)), -1)*available[..., None]
        score = self.head(torch.cat((values.flatten(1), available.to(values)), -1))
        return (score, dict(aux_logits=aux, relations=relations, operators=trace, pooling=pool_trace)) if return_details else score

    def forward(self, images, masks, valid, available, *, reference, **kwargs):
        bank = {**reference, 'prefix': self.prefix(reference['images'], stop=self.cache_start)}
        return self.forward_from(self.prefix(images, stop=self.cache_start), masks, valid,
                                 available, start=self.cache_start, reference=bank, **kwargs)
