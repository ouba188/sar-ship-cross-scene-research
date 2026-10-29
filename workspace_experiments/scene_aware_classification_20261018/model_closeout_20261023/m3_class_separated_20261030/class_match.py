"""Preserve reference-class identity until aligned local distances are formed."""
import torch
from torch import nn
from torch.nn import functional as F
from sar_m123 import tokens
from sar_m123.selective_matching import SelectiveReferenceMatch
from sar_m123.pool_replacement import grid_occupancy


def region_evidence(query, reference, transport, weights, ref_mass, labels, classes, mode):
    row_mass = transport.sum(-1)
    available = row_mass > 0
    if mode == 'aligned':
        moved = torch.einsum('bkij,kjd->bkid', transport, reference)
        moved = moved / row_mass.clamp_min(1e-20)[..., None]
    elif mode == 'pooled':
        mean = (reference * ref_mass[..., None]).sum(1) / ref_mass.sum(1).clamp_min(1e-20)[:, None]
        moved = mean[None, :, None].expand(query.shape[0], -1, query.shape[1], -1)
    else:
        raise ValueError(mode)
    membership = F.one_hot(labels, classes).to(query)
    weighted = weights * available
    mass = torch.einsum('bki,kc->bci', weighted, membership)
    prototype = torch.einsum('bki,bkid,kc->bcid', weighted, moved, membership)
    prototype = prototype / mass.clamp_min(1e-20)[..., None]
    live = mass > 0
    distance = (prototype - query[:, None]).square().sum(-1)
    score = -distance
    mean = (score * live).sum(1, keepdim=True) / live.sum(1, keepdim=True).clamp_min(1)
    return (score - mean) * live, distance, live


class ClassSeparatedMatch(SelectiveReferenceMatch):
    def __init__(self, labels, classes, mode, channels=1024, **kwargs):
        super().__init__(channels=channels, **kwargs)
        self.mode = mode
        self.register_buffer('extra_labels', torch.as_tensor(labels, dtype=torch.long))
        self.register_buffer('extra_scale', torch.ones(()))
        self.extra_output = nn.Conv2d(classes, channels, 1, bias=False)
        nn.init.zeros_(self.extra_output.weight)
        self.extra_control = 'normal'
        self.diagnostic = None

    @staticmethod
    def value_tokens(x, slots, valid):
        masks = torch.minimum(slots.clamp(0, 1), grid_occupancy(valid.to(x), x.shape[-2:]).clamp(0, 1))
        mass = masks.sum((-1, -2))
        z = F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)
        values = torch.einsum('bshw,bdhw->bsd', masks, z) / mass.clamp_min(1e-20)[..., None]
        return F.normalize(values, dim=-1), mass, masks

    def forward(self, x, anchor, local, slots, local_slots, valid, energy,
                local_energy, reference, pair_valid, return_details=False):
        original, detail = super().forward(x, anchor, local, slots, local_slots, valid,
            energy, local_energy, reference, pair_valid, True)
        with torch.no_grad():
            query, _, masks = self.value_tokens(x, slots, valid)
            ref, ref_mass, _ = self.value_tokens(reference['features'], reference['slots'], reference['valid'])
            fields, distances, validity = [], [], []
            offset = 0
            for region, transport in zip(tokens.REGIONS, detail['transport']):
                ix = region.tolist()
                count = len(ix)
                weight = detail['reference_weights'][..., offset:offset+count]
                field, distance, live = region_evidence(query[:, ix], ref[:, ix], transport,
                    weight, ref_mass[:, ix], self.extra_labels, self.extra_output.in_channels, self.mode)
                fields.append(field)
                distances.append(distance)
                validity.append(live)
                offset += count
            evidence = torch.cat(fields, -1)
            write = masks / masks.sum(1, keepdim=True).clamp_min(1)
            spatial = torch.einsum('bshw,bcs->bchw', write, evidence)
            self.diagnostic = dict(selected_square=spatial.square().sum(), selected_numel=spatial.numel(),
                class_evidence=evidence, class_distance=torch.cat(distances, -1),
                class_valid=torch.cat(validity, -1))
        correction = .1 * self.extra_output(spatial / self.extra_scale.clamp_min(1e-8))
        if self.extra_control == 'off':
            correction = correction * 0
        detail['extra_update'] = correction
        detail['update'] = detail['update'] + correction
        if return_details:
            detail['class_evidence'] = evidence
        return original + correction, detail


def attach(net, mode, labels):
    for block in net.m3:
        for pol, old in enumerate(block):
            new = ClassSeparatedMatch(labels, net.head.out_features, mode,
                channels=old.project.in_channels, dynamic=old.dynamic,
                frequency=old.frequency, temperature=old.temperature).to(old.gain.device)
            missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
            assert set(missing) == {'extra_labels', 'extra_scale', 'extra_output.weight'} and not unexpected
            block[pol] = new
    return net.eval().requires_grad_(False)


def configure(net):
    net.eval().requires_grad_(False)
    selected = [(n, p) for n, p in net.named_parameters() if n.startswith('m3.') and '.extra_output.' in n]
    for _, p in selected:
        p.requires_grad_(True)
    return selected
