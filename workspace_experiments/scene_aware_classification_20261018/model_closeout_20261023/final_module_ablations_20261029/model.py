"""Matched train-time removals of the final spatial-M12 backbone."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from sar_m123.current.spatial_m12 import SpatialM12ResNet50
from sar_m123.current.model import CLASSES
from sar_m123.pool_replacement import PoolReplacementResNet50, grid_occupancy, regional_mean

ARMS = ('plain', 'no_m1', 'no_m2', 'no_m3')


class AblationModel(SpatialM12ResNet50):
    architecture = 'native160_spatial_m12_train_time_ablation_v1'

    def __init__(self, arm, classes=8):
        if arm not in (*ARMS, 'full'):
            raise ValueError(arm)
        super().__init__(None, 'pre3x3', 'local', classes)
        self.arm = arm
        if arm in ('plain', 'no_m1'):
            self.internal_m1 = nn.ModuleList()
            self.m1 = nn.ModuleDict()
        if arm in ('plain', 'no_m2'):
            self.internal_m2 = nn.ModuleList()
            self.m2 = nn.ModuleList()
            self.deep_pool = nn.ModuleList()
            self.deep_output = nn.ModuleList()
            self.head = nn.Linear(4098, classes)
        if arm in ('plain', 'no_m3'):
            self.m3 = nn.ModuleList()

    def intervene(self, z, valid, available, field, slots, energy, aux, trace):
        z = z.reshape(len(valid), 2, *z.shape[1:])
        output, logits, details = [], [], []
        for p in range(2):
            value, d = z[:, p], {}
            if self.internal_m1:
                value, d['m1'] = self.internal_m1[p](value,
                    grid_occupancy(valid[:, p], value.shape[-2:]), field[:, p], True)
                logits.append(d['m1']['aux_logits']*available[:, p, None])
            if self.internal_m2:
                regions, bands = self.region_integrals(slots[:, p], energy[:, p])
                value, d['m2'] = self.internal_m2[p](value, regions, valid[:, p], bands)
            output.append(value)
            details.append(d)
        if logits:
            aux.append(torch.stack(logits, 1).sum(1)/available.sum(1, keepdim=True).clamp_min(1))
        trace['bottleneck'] = details
        return torch.stack(output, 1).flatten(0, 1)

    def forward_from(self, prefix, masks, valid, available, start=2, *, slot_masks, frequency,
                     local_slot_masks, local_frequency, field2, field3, reference, pair_valid,
                     return_details=False):
        if self.arm == 'plain':
            self.pool_mode = 'native'
            score = PoolReplacementResNet50.forward_from(self, prefix, masks, valid, available, start)
            return (score, dict(aux_logits=[], relations=[])) if return_details else score
        if self.arm != 'no_m2':
            return super().forward_from(prefix, masks, valid, available, start,
                slot_masks=slot_masks, frequency=frequency, local_slot_masks=local_slot_masks,
                local_frequency=local_frequency, field2=field2, field3=field3,
                reference=reference, pair_valid=pair_valid, return_details=return_details)
        # Same live feature/matching path as SelectiveBackbone; no regional head inputs.
        valid = valid*available[..., None, None, None]
        rv = reference['valid']*reference['available'][..., None, None, None]
        trace, aux, relations = {}, [], []
        _, x, r, raw3, rr3 = self.stage3_paths(prefix, valid, available, field2, field3,
            reference, rv, slot_masks, frequency, trace, aux)
        for block, operators in enumerate(self.m3):
            outputs, details = [], []
            for p, operator in enumerate(operators):
                bank = dict(features=r[:, p], anchor=rr3[:, p], local=reference['prefix'][:, p],
                    slots=reference['slot_masks'][:, p], local_slots=reference['local_slot_masks'][:, p],
                    valid=rv[:, p], energy=reference['frequency'][:, p],
                    local_energy=reference['local_frequency'][:, p])
                legal = pair_valid & available[:, p, None].bool() & reference['available'][None, :, p].bool()
                value, d = operator(x[:, p], raw3[:, p], prefix[:, p], slot_masks[:, p],
                    local_slot_masks[:, p], valid[:, p], frequency[:, p], local_frequency[:, p],
                    bank, legal, return_details=return_details)
                outputs.append(value)
                details.append(d)
                relations.append((d['reference_logits'], d['reference_valid']))
            x = torch.stack(outputs, 1)
            trace[f'm3_{block}'] = details
        deep = self.backbone.model.layer4(x.flatten(0, 1))
        glob = regional_mean(deep, valid.flatten(0, 1), valid.flatten(0, 1))[0][:, 0]
        values = glob.reshape(len(prefix), 2, -1)*available[..., None]
        score = self.head(torch.cat((values.flatten(1), available.to(values)), 1))
        return (score, dict(aux_logits=aux, relations=relations, operators=trace,
                            pooling=[])) if return_details else score


def fresh(arm, init_path, device='cuda', class_indices=None):
    source = torch.load(init_path, map_location='cpu', weights_only=True)['state']
    indexes = list(range(8)) if class_indices is None else list(class_indices)
    model = AblationModel(arm, len(indexes))
    state = {}
    for name, expected in model.state_dict().items():
        value = source[name]
        if name.startswith('head.') or (name.startswith('internal_m1.') and '.reader.' in name):
            value = value[indexes]
        if name == 'head.weight' and expected.shape[1] == 4098:
            value = torch.cat((value[:, :2048], value[:, 3584:5632], value[:, -2:]), 1)
        state[name] = value
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def save(model, path, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(architecture=model.architecture, arm=model.arm, classes=model.head.out_features,
        state={k: v.detach().cpu() for k, v in model.state_dict().items()}, metadata=metadata), path)


def load(path, device='cuda'):
    blob = torch.load(path, map_location='cpu', weights_only=True)
    assert blob['architecture'] == AblationModel.architecture
    model = AblationModel(blob['arm'], blob['classes'])
    model.load_state_dict(blob['state'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def fitting_loss(score, detail, labels, weights, reference_labels):
    ce = (F.cross_entropy(score, labels, reduction='none')*weights).mean()
    aux = torch.stack([(F.cross_entropy(z, labels, reduction='none')*weights).mean()
                       for z in detail['aux_logits']]).mean() if detail['aux_logits'] else ce.new_zeros(())
    terms, active = [], []
    for logits, valid in detail['relations']:
        same = labels[:, None].eq(reference_labels[None]) & valid
        exists = same.any(1) & valid.any(1)
        logp = logits.masked_fill(~valid, -1e4).log_softmax(1)
        terms.append(-(logp*same).sum(1)/same.sum(1).clamp_min(1)*exists)
        active.append(exists)
    rel = (torch.stack(terms).sum(0)/torch.stack(active).sum(0).clamp_min(1)*weights).mean() if terms else ce.new_zeros(())
    return ce+.1*aux+.1*rel, dict(ce=ce, auxiliary=aux, relation=rel)
