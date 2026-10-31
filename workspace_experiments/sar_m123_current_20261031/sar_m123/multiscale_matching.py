"""Separate comparison evidence from learnable reference values.

Keys retain the frozen backbone coordinates. Frequency describes native Gabor
energy, not electromagnetic scattering mechanisms. No query labels enter here.
"""
import torch
from torch import nn
from torch.nn import functional as F

from . import tokens
from .m3 import sinkhorn
from .pool_replacement import grid_occupancy


class SeparateEvidenceMatch(nn.Module):
    def __init__(self, channels=1024, local_channels=512, use_local=False,
                 use_frequency=False, transport_eps=.1):
        super().__init__()
        self.channels, self.local_channels = channels, local_channels
        self.use_local, self.use_frequency = use_local, use_frequency
        self.transport_eps = transport_eps
        self.project = nn.Conv2d(channels, 32, 1, bias=False)
        self.output = nn.Conv2d(32, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.gain = nn.Parameter(torch.tensor(.1))
        self.gate = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.)
        self.gate.requires_grad_(use_frequency)
        self.register_buffer('freq_center', torch.zeros(20, 6))
        self.register_buffer('freq_scale', torch.ones(20, 6))

    @torch.no_grad()
    def set_frequency_stats(self, center, scale):
        self.freq_center.copy_(center)
        self.freq_scale.copy_(scale.clamp_min(1e-4))

    @staticmethod
    def describe(x, slots, valid, energy):
        size = x.shape[-2:]
        if slots.shape != (len(x), 20, *size) or energy.shape != (len(x), 20, 3, *size):
            raise ValueError('Expected aligned native slot and slot-energy integrals.')
        occupancy = slots.to(x).clamp(0, 1)
        masks = torch.minimum(occupancy, grid_occupancy(valid.to(x), size).clamp(0, 1))
        mass = masks.sum((-2, -1))
        den = mass.clamp_min(1e-8)
        # Preserve original feature coordinates for comparison; learning values
        # must not rotate all comparison keys toward a common direction.
        key = torch.einsum('bshw,bchw->bsc', masks, x.detach()) / den[..., None]
        cells = energy.to(x) / occupancy[:, :, None].clamp_min(1e-8)
        mean = (cells * masks[:, :, None]).sum((-2, -1)) / den[..., None]
        var = ((cells - mean[..., None, None]).square() * masks[:, :, None]).sum((-2, -1)) / den[..., None]
        freq = torch.cat((mean, var.clamp_min(0).sqrt()), -1)
        freq = freq * (mass > 0)[..., None]
        return masks, mass, key, freq

    def forward(self, x, slot_masks, valid, frequency, reference, pair_valid=None,
                return_details=False, *, local_features=None, local_slot_masks=None,
                local_frequency=None):
        b, k = len(x), len(reference['features'])
        if pair_valid is None:
            pair_valid = torch.ones((b, k), device=x.device, dtype=torch.bool)
        elif pair_valid.shape != (b, k):
            raise ValueError('pair_valid must have shape [B,K].')
        pair_valid = pair_valid.to(device=x.device, dtype=torch.bool)
        if k == 0:
            empty = tuple(x.new_zeros(b, 0, len(r), len(r)) for r in tokens.REGIONS)
            details = dict(update=torch.zeros_like(x), reference_weights=x.new_zeros(b, 0, 20),
                           transport=empty, similarity=empty, spatial_similarity=empty,
                           frequency_similarity=empty, gates=empty)
            return (x, details) if return_details else x
        masks, qm, q, qf = self.describe(x, slot_masks, valid, frequency)
        smasks, sm, s, sf = self.describe(reference['features'], reference['slot_masks'],
                                        reference['valid'], reference['frequency'])
        ql = sl = None
        if self.use_local:
            if local_features is None or local_slot_masks is None or local_frequency is None:
                raise ValueError('Local comparison requires actual layer2 features and integrals.')
            _, _, ql, qf = self.describe(local_features, local_slot_masks, valid, local_frequency)
            _, _, sl, sf = self.describe(reference['local_features'], reference['local_slot_masks'],
                                         reference['valid'], reference['local_frequency'])
            ql, sl = F.normalize(ql, dim=-1), F.normalize(sl, dim=-1)
        qu, su = F.normalize(q, dim=-1), F.normalize(s, dim=-1)
        qf = ((qf-self.freq_center)/self.freq_scale).clamp(-5, 5)
        sf = ((sf-self.freq_center)/self.freq_scale).clamp(-5, 5)
        source = reference['features']
        normalized = F.layer_norm(source.permute(0, 2, 3, 1), (self.channels,)).permute(0, 3, 1, 2)
        values_map = self.project(normalized)
        source_values = torch.einsum('kshw,kdhw->ksd', smasks, values_map) / sm.clamp_min(1e-8)[..., None]
        values, weights_all = [], []
        traces = {key: [] for key in ('transport', 'similarity', 'spatial_similarity', 'frequency_similarity', 'gates')}
        for region in tokens.REGIONS:
            indices = region.tolist()
            spatial = torch.einsum('bid,kjd->bkij', qu[:, indices], su[:, indices])
            if self.use_local:
                spatial = .5 * (spatial + torch.einsum('bid,kjd->bkij', ql[:, indices], sl[:, indices]))
            difference = qf[:, None, indices, None] - sf[None, :, None, indices]
            freq_sim = -difference.square().mean(-1).clamp_max(4.) / 4.
            gate = .5 * self.gate(torch.stack((spatial, freq_sim), -1)).sigmoid().squeeze(-1)
            sim = spatial + gate * freq_sim if self.use_frequency else spatial
            qa, sa = qm[:, None, indices], sm[None, :, indices]
            allowed = ((qa > 0)[..., :, None] & (sa > 0)[..., None, :] & pair_valid[..., None, None])
            transport = sinkhorn(sim, qa, sa, eps=self.transport_eps) * allowed.to(sim)
            row_mass = transport.sum(-1)
            score = (transport * sim).sum(-1) / row_mass.clamp_min(1e-20)
            available = row_mass > 0
            weights = score.masked_fill(~available, torch.finfo(score.dtype).min).softmax(1) * available.to(score)
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-20)
            moved = torch.einsum('bkij,kjd->bkid', transport, source_values[:, indices])
            offered = qa / qa.sum(-1, keepdim=True).clamp_min(1e-20)
            moved = moved / offered[..., None].clamp_min(1e-20)
            values.append((weights[..., None] * moved).sum(1))
            weights_all.append(weights)
            if return_details:
                for key, value in zip(traces, (transport, sim, spatial, freq_sim, gate)):
                    traces[key].append(value)
        write = masks / masks.sum(1, keepdim=True).clamp_min(1)
        mixed = torch.einsum('bshw,bsd->bdhw', write, torch.cat(values, 1))
        update = self.gain * self.output(mixed)
        details = dict(update=update, reference_weights=torch.cat(weights_all, -1),
                       **{key: tuple(value) for key, value in traces.items()})
        return (x+update, details) if return_details else x+update
