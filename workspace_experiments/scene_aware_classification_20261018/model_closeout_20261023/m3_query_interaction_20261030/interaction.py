"""Frozen original M3 with a zero-start query/reference interaction writer."""
import torch
from torch import nn
from torch.nn import functional as F
from sar_m123 import tokens
from sar_m123.selective_matching import SelectiveReferenceMatch
from sar_m123.pool_replacement import grid_occupancy


class QueryInteractionMatch(SelectiveReferenceMatch):
    def __init__(self, mode='interaction', channels=1024, **kwargs):
        super().__init__(channels=channels, **kwargs)
        self.mode = mode
        self.extra_output = nn.Conv2d(32, channels, 1, bias=False)
        nn.init.zeros_(self.extra_output.weight)
        self.register_buffer('extra_scale', torch.ones(()))
        self.output.register_forward_pre_hook(self.capture_message)
        self.message = None
        self.diagnostic = None
        self.extra_control = 'normal'

    def capture_message(self, module, arguments):
        self.message = arguments[0].detach()

    def uniform_message(self, reference, detail, masks):
        source = reference['features']
        read = torch.minimum(reference['slots'].clamp(0, 1),
            grid_occupancy(reference['valid'].to(source), source.shape[-2:]).clamp(0, 1))
        mass = read.sum((-1, -2))
        z = F.layer_norm(source.permute(0, 2, 3, 1), (source.shape[1],)).permute(0, 3, 1, 2)
        value = torch.einsum('kshw,kdhw->ksd', read, self.project(z))/mass.clamp_min(1e-8)[..., None]
        query_mass = masks.sum((-1, -2))
        parts = []
        for region, transport in zip(tokens.REGIONS, detail['transport']):
            ix = region.tolist()
            live = transport.sum(-1) > 0
            weights = live.to(z)/live.sum(1, keepdim=True).clamp_min(1)
            offered = query_mass[:, None, ix]
            offered = offered/offered.sum(-1, keepdim=True).clamp_min(1e-20)
            moved = torch.einsum('bkij,kjd->bkid', transport, value[:, ix])
            moved = moved/offered[..., None].clamp_min(1e-20)
            parts.append((weights[..., None]*moved).sum(1))
        write = masks/masks.sum(1, keepdim=True).clamp_min(1)
        return torch.einsum('bshw,bsd->bdhw', write, torch.cat(parts, 1))

    def forward(self, x, anchor, local, slots, local_slots, valid, energy,
                local_energy, reference, pair_valid, return_details=False):
        original, detail = super().forward(x, anchor, local, slots, local_slots,
            valid, energy, local_energy, reference, pair_valid, return_details)
        with torch.no_grad():
            masks = torch.minimum(slots.clamp(0, 1),
                grid_occupancy(valid.to(x), x.shape[-2:]).clamp(0, 1))
            write = masks/masks.sum(1, keepdim=True).clamp_min(1)
            live = detail['reference_weights'].sum(1) > 0
            footprint = torch.einsum('bshw,bs->bhw', write, live.to(x))[:, None]
            q = F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)
            query = self.project(q)*footprint
            message = self.message
            if self.extra_control == 'uniform':
                # Only the added writer changes; the original trained M3 stays intact.
                if not detail['transport']:
                    _, transport_detail = super().forward(x, anchor, local, slots,
                        local_slots, valid, energy, local_energy, reference, pair_valid, True)
                else:
                    transport_detail = detail
                message = self.uniform_message(reference, transport_detail, masks)
            selected = query if self.mode == 'query_only' else message if self.mode == 'ref_only' else query*message
            self.diagnostic = dict(selected_square=selected.square().sum(), selected_numel=selected.numel(),
                query_rms=query.square().mean((1, 2, 3)).sqrt(),
                message_rms=message.square().mean((1, 2, 3)).sqrt())
        correction = .1*self.extra_output(selected/self.extra_scale.clamp_min(1e-8))
        if self.extra_control == 'off':
            correction = correction*0
        detail['extra_update'] = correction
        detail['update'] = detail['update']+correction
        self.message = None
        return original+correction, detail


def attach(net, mode):
    device = next(net.parameters()).device
    for block in net.m3:
        for p, old in enumerate(block):
            new = QueryInteractionMatch(mode, channels=old.project.in_channels,
                dynamic=old.dynamic, frequency=old.frequency, temperature=old.temperature).to(device)
            missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
            assert set(missing) == {'extra_output.weight', 'extra_scale'} and not unexpected
            block[p] = new
    net.eval().requires_grad_(False)
    return net


def configure(net):
    net.eval().requires_grad_(False)
    selected = []
    for name, value in net.named_parameters():
        if name.startswith('m3.') and '.extra_output.' in name:
            value.requires_grad_(True)
            selected.append((name, value))
    return selected
