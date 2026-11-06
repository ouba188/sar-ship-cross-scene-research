"""Early M2 feedback with fixed comparison anchors and live value paths."""
import torch
from torch import nn

from .bottleneck_evidence import BottleneckEvidenceResNet50, RegionPoolWriteback
from .pool_replacement import regional_mean


class ShallowM2(nn.Module):
    """Add feedback after layer1/layer2 without modifying the selected model.

    Cache only frozen pre-feedback layer1 and original comparison descriptors.
    Frozen downstream convolutions remain differentiable for the early adapter.
    Query and reference share the same live encoding and polarization operators.
    """
    def __init__(self, state, stages=()):
        super().__init__()
        self.base = BottleneckEvidenceResNet50(None, position='pre3x3', policy='local')
        self.base.load_state_dict(state, strict=True)
        self.base.requires_grad_(False)
        self.stages = tuple(stages)
        self.early = nn.ModuleDict({f'{s}_{p}': RegionPoolWriteback({1:256, 2:512}[s])
                                   for s in stages for p in range(2)})
        for i in (1, 2, 3):
            getattr(self.base.backbone.model.layer3[-1], f'conv{i}').weight.requires_grad_(True)
        self.eval()

    def early_feedback(self, x, data, stage, trace):
        if stage not in self.stages:
            return x
        out, details = [], []
        for p in range(2):
            if stage == 2:
                mask, band = self.base.region_integrals(data['local_slot_masks'][:, p],
                                                        data['local_frequency'][:, p])
            else:
                mask, band = data['regions1'][:, p], data['energy1'][:, p]
            y, d = self.early[f'{stage}_{p}'](x[:, p], mask, data['valid'][:, p], band)
            out.append(y)
            details.append(d)
        trace[f'early_{stage}'] = details
        return torch.stack(out, 1)

    def encode(self, data, trace, aux, propagate=True):
        x1 = self.early_feedback(data['stage1'], data, 1, trace)
        if 1 in self.stages:
            x2 = self.base.backbone.model.layer2(x1.flatten(0, 1)).reshape(len(x1), 2, 512, 20, 20)
        else:
            x2 = data['raw2']
        x2 = self.early_feedback(x2, data, 2, trace)
        x = (x2 if propagate else data['raw2']).flatten(0, 1)
        for block in list(self.base.backbone.model.layer3.children())[:-1]:
            x = block(x)
        x3 = self.base.live_last(x, data['valid'], data['available'], data['field3'],
                                data['slot_masks'], data['frequency'], aux, trace)
        return x2, x3

    def forward(self, query, reference, legal, details=False, route='full'):
        if route not in ('full', 'backbone_only', 'readout_only'):
            raise ValueError(route)
        trace, aux = {}, []
        x2, x = self.encode(query, trace, aux, propagate=route!='readout_only')
        _, ref = self.encode(reference, {}, [], propagate=route!='readout_only')
        relations = []
        for operators in self.base.m3:
            outputs, match_trace = [], []
            for p in range(2):
                bank = dict(features=ref[:, p], anchor=reference['anchor3'][:, p],
                    local=reference['raw2'][:, p], slots=reference['slot_masks'][:, p],
                    local_slots=reference['local_slot_masks'][:, p], valid=reference['valid'][:, p],
                    energy=reference['frequency'][:, p], local_energy=reference['local_frequency'][:, p])
                allowed = legal & query['available'][:, p, None].bool() & reference['available'][None, :, p].bool()
                y, d = operators[p](x[:, p], query['anchor3'][:, p], query['raw2'][:, p],
                    query['slot_masks'][:, p], query['local_slot_masks'][:, p], query['valid'][:, p],
                    query['frequency'][:, p], query['local_frequency'][:, p], bank, allowed, return_details=details)
                outputs.append(y)
                match_trace.append(d)
                relations.append((d['reference_logits'], d['reference_valid']))
            x = torch.stack(outputs, 1)
            trace['m3'] = match_trace
        deep = self.base.backbone.model.layer4(x.flatten(0, 1))
        glob = regional_mean(deep, query['valid'].flatten(0, 1), query['valid'].flatten(0, 1))[0][:, 0]
        glob = glob.reshape(len(x), 2, -1)
        regional = []
        for p in range(2):
            mask, band = self.base.region_integrals(query['local_slot_masks'][:, p], query['local_frequency'][:, p])
            readout_input = query['raw2'] if route == 'backbone_only' else x2
            shallow = self.base.m2[p](readout_input[:, p], mask, query['valid'][:, p], band)['pooled']
            mask, band = self.base.region_integrals(query['slot_masks'][:, p], query['frequency'][:, p])
            deep_region = self.base.deep_pool[p](x[:, p], mask, query['valid'][:, p], band)['pooled']
            regional.append((shallow+self.base.deep_output[p](deep_region)).flatten(1))
        values = torch.cat((glob, torch.stack(regional, 1)), -1)*query['available'][..., None]
        score = self.base.head(torch.cat((values.flatten(1), query['available']), -1))
        return (score, dict(trace=trace, aux_logits=aux, relations=relations)) if details else score
