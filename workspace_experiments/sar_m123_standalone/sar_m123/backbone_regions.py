"""Experimental region pooling/feedback inside a frozen ResNet-50.

This is a new trainable backbone adapter, not the retained terminal M2.
Masks are feature-cell occupancy weights, not receptive-field segmentation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Backbone


class RegionFeedback(nn.Module):
    """Pool region tokens, then return content-gated updates to their regions."""

    def __init__(self, channels, hidden=32, regions=3, temperature=1., gate_mode='dot', update_mode='broadcast'):
        super().__init__()
        self.channels, self.regions = channels, regions
        if temperature <= 0:
            raise ValueError('Feedback temperature must be positive.')
        self.temperature = float(temperature)
        if gate_mode not in ('dot', 'fixed', 'cosine'):
            raise ValueError(gate_mode)
        self.gate_mode = gate_mode
        if update_mode not in ('broadcast', 'local', 'additive', 'modulate'):
            raise ValueError(update_mode)
        self.update_mode = update_mode
        self.query = nn.Conv2d(channels, hidden, 1, bias=False)
        self.key = nn.Linear(channels, hidden, bias=False)
        if gate_mode == 'fixed':
            self.query.requires_grad_(False)
            self.key.requires_grad_(False)
        self.value = nn.Linear(channels, hidden, bias=False)
        self.region_value = nn.Parameter(torch.eye(hidden).repeat(regions, 1, 1))
        self.output = nn.Conv2d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.gain = nn.Parameter(torch.tensor(.1))

    def forward(self, x, masks, valid, *, return_details=False):
        if masks.shape[:2] != (len(x), self.regions):
            raise ValueError('Expected one mask per region and batch item.')
        valid = F.interpolate(valid.to(x), size=x.shape[-2:], mode='area').clamp(0, 1)
        masks = F.interpolate(masks.to(x), size=x.shape[-2:], mode='area').clamp(0, 1)
        # Input masks already exclude invalid native pixels. Multiplying again
        # would square fractional boundary occupancy; clip to the valid extent.
        masks = torch.minimum(masks, valid)
        mass = masks.sum((-2, -1))
        z = F.layer_norm(x.permute(0, 2, 3, 1), (self.channels,)).permute(0, 3, 1, 2)
        tokens = torch.einsum('brhw,bchw->brc', masks, z) / mass.clamp_min(1e-6)[..., None]
        if self.gate_mode == 'fixed':
            gates = torch.full_like(masks, .5)
        else:
            q, k = self.query(z), self.key(tokens)
            if self.gate_mode == 'cosine':
                q, k = F.normalize(q, dim=1), F.normalize(k, dim=-1)
                logits = torch.einsum('bdhw,brd->brhw', q, k) / self.temperature
            else:
                logits = torch.einsum('bdhw,brd->brhw', q, k) / (math.sqrt(q.shape[1]) * self.temperature)
            gates = torch.sigmoid(logits)
        weights = gates * masks / masks.sum(1, keepdim=True).clamp_min(1)
        values = torch.einsum('brd,rde->bre', self.value(tokens), self.region_value)
        if self.update_mode == 'broadcast':
            mixed = torch.einsum('brhw,brd->bdhw', weights, values)
        else:
            local = F.conv2d(z, self.value.weight[..., None, None])
            local = torch.einsum('bdhw,rde->brehw', local, self.region_value)
            condition = .5 * torch.tanh(values)[..., None, None]
            if self.update_mode == 'additive':
                local = local + condition
            elif self.update_mode == 'modulate':
                local = local * (1 + condition)
            mixed = torch.einsum('brhw,brdhw->bdhw', weights, local)
        update = self.output(mixed)
        out = x + self.gain * update
        if return_details:
            return out, dict(weights=weights, mass=mass, update=self.gain * update, gates=gates)
        return out


class RegionResNet50(nn.Module):
    """Independent per-polarization M2 instances at any selected ResNet stages.

    Backbone parameters and BN statistics stay frozen, but gradients through
    downstream stages remain enabled. No reference bank or M1/M3 is used here.
    """

    def __init__(self, weight_path, classes=8, stages=(2,), mode='regional', hidden=32, temperature=1., gate_mode='dot', update_mode='broadcast'):
        super().__init__()
        if mode not in ('regional', 'merged', 'none'):
            raise ValueError(mode)
        if any(s not in (1, 2, 3, 4) for s in stages):
            raise ValueError('ResNet stages must be 1, 2, 3, or 4.')
        self.backbone = Backbone(weight_path)
        self.mode = mode
        self.stages = tuple(sorted(set(stages))) if mode != 'none' else ()
        channels = {1: 256, 2: 512, 3: 1024, 4: 2048}
        self.blocks = nn.ModuleDict({f'{p}_{s}': RegionFeedback(channels[s], hidden, temperature=temperature, gate_mode=gate_mode, update_mode=update_mode)
                                    for s in self.stages for p in range(2)})
        self.head = nn.Linear(4098, classes)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def prefix(self, images, stop=2):
        """Caching is legal only before the first adapter, with frozen weights."""
        if any(s < stop for s in self.stages):
            raise ValueError('Cannot cache through a trainable earlier adapter.')
        b, p = images.shape[:2]
        x = self.backbone.model.maxpool(self.backbone.stem(images.flatten(0, 1)))
        for s in range(1, stop + 1):
            x = getattr(self.backbone.model, f'layer{s}')(x)
        return x.reshape(b, p, *x.shape[1:])

    def forward_from(self, prefix, masks, valid, available, start=2):
        if any(s < start for s in self.stages):
            raise ValueError('Cached entry would skip an earlier trainable adapter.')
        b, p = prefix.shape[:2]
        x = prefix
        for s in range(start, 5):
            if s > start:
                z = getattr(self.backbone.model, f'layer{s}')(x.flatten(0, 1))
                x = z.reshape(b, p, *z.shape[1:])
            if s in self.stages:
                region = masks
                if self.mode == 'merged':
                    # Same observed support and parameter count, no region identity.
                    union = masks.sum(2, keepdim=True).clamp(0, 1)
                    region = union.expand(-1, -1, 3, -1, -1)
                x = torch.stack([self.blocks[f'{pol}_{s}'](
                    x[:, pol], region[:, pol], valid[:, pol]) for pol in range(p)], 1)
        footprint = F.interpolate(valid.flatten(0, 1).to(x), size=x.shape[-2:], mode='area')
        footprint = footprint.reshape(b, p, 1, *x.shape[-2:])
        features = (x * footprint).sum((-2, -1)) / footprint.sum((-2, -1)).clamp_min(1e-6)
        features = features * available[..., None].to(x)
        return self.head(torch.cat((features.flatten(1), available.to(x)), 1))

    def forward(self, images, masks, valid, available):
        # Full differentiable path supports independently repeated adapters.
        x = self.prefix(images, stop=1)
        return self.forward_from(x, masks, valid, available, start=1)
