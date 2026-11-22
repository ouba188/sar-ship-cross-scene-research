"""Independent-polarization ConvNeXt stages 1--3 and supervised spatial readout.

The encoder is trained without H. ``freeze_for_h`` exports both Fc and the
already-supervised G(Fc); no trainable spatial layer is inserted by stage two.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np  # Import before torch on the supported Windows environment.
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny

from .contracts import EncodedShip, weighted_pool


@dataclass
class Stage1Config:
    classes: int
    weights_path: str | None = None  # Local torchvision ConvNeXt-T state_dict only.
    fusion_max: float = 1.0
    fusion_init: float = 0.1
    gate_rho_min: float = 0.5
    gate_rho_max: float = 0.95
    gate_temperature: float = 1.0
    stochastic_depth: float = 0.0
    axis_symmetric: bool = True
    fusion_enabled: bool = True
    gate_enabled: bool = True

    def __post_init__(self):
        if self.classes < 2:
            raise ValueError("classes must be >= 2")
        if self.fusion_enabled and not 0 < self.fusion_init < self.fusion_max:
            raise ValueError("fusion_init must lie strictly between 0 and fusion_max")
        if not 0 < self.gate_rho_min <= self.gate_rho_max < 1:
            raise ValueError("gate quotas must satisfy 0 < min <= max < 1")
        if self.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")


class JointGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.channel = nn.Sequential(nn.Linear(256, 16), nn.ReLU(), nn.Linear(16, 256))
        self.local = nn.Sequential(nn.Conv2d(256, 32, 1), nn.ReLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(32, 256, 1))

    def forward(self, features, valid, hard=False):
        support = (valid > 0).to(features.dtype)
        channel = self.channel(weighted_pool(features, valid[:, 0]))[..., None, None]
        soft = torch.sigmoid((channel + self.local(features)) / self.config.gate_temperature)
        soft = soft * support
        if not hard:
            return soft
        binary = torch.zeros_like(soft, memory_format=torch.contiguous_format)
        for b in range(len(features)):
            eligible = support[b].expand_as(soft[b]).reshape(-1).bool()
            indices = eligible.nonzero(as_tuple=False).flatten()
            n = indices.numel()
            if not n:
                continue
            values = soft[b].flatten()[indices]
            lower = max(1, math.ceil(n * self.config.gate_rho_min))
            upper = min(n - 1, math.floor(n * self.config.gate_rho_max))
            if lower > upper:
                raise ValueError("gate quota interval contains no integer count")
            count = min(upper, max(lower, int((values >= 0.5).sum().item())))
            # Stable sorting makes ties deterministic in channel-major pixel order.
            selected = torch.argsort(values, descending=True, stable=True)[:count]
            binary[b].view(-1)[indices[selected]] = 1
        return binary + (soft - soft.detach())


class SpatialReadout(nn.Module):
    """G: supervised 256->64 map, applied BEFORE region pooling."""
    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(256, 64, 1)
        self.depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.pointwise = nn.Conv2d(64, 64, 1)

    def forward(self, features, valid):
        mask = (valid > 0).to(features.dtype)
        z = F.gelu(self.project(features)) * mask
        local = F.gelu(self.depthwise(z)) * mask
        local = self.pointwise(local) * mask
        return (z + local) * mask


class ResidualClassDiscriminator(nn.Module):
    """Joint six-map Fs discriminator, independent of background regressors."""
    def __init__(self, classes):
        super().__init__()
        self.projects = nn.ModuleList([nn.Conv2d(256, 32, 1) for _ in range(6)])
        self.head = nn.Sequential(nn.Linear(6 * (32 * 16 + 16 + 1), 64),
                                  nn.ReLU(), nn.Linear(64, classes))

    def forward(self, residual, valid):
        parts = []
        for level in range(3):
            for pol in range(2):
                x, weight = residual[level][:, pol], valid[level][:, pol]
                z = F.relu(self.projects[level * 2 + pol](x)) * (weight > 0)
                coverage = F.adaptive_avg_pool2d(weight, 4)
                pooled = F.adaptive_avg_pool2d(z * weight, 4) / coverage.clamp_min(1e-12)
                flags = (coverage > 0).to(x.dtype)
                parts.extend((pooled.flatten(1), flags.flatten(1),
                              (weight.sum((1, 2, 3)) > 0).to(x.dtype)[:, None]))
        return self.head(torch.cat(parts, dim=1))


class Stage1(nn.Module):
    def __init__(self, config: Stage1Config):
        super().__init__()
        self.config = config
        base = convnext_tiny(weights=None, stochastic_depth_prob=config.stochastic_depth)
        if config.weights_path is not None:
            state = torch.load(Path(config.weights_path), map_location="cpu", weights_only=True)
            base.load_state_dict(state.get("state_dict", state))
        old = base.features[0][0]
        stem = nn.Conv2d(1, 96, 4, stride=4)
        with torch.no_grad():
            stem.weight.copy_(old.weight.sum(dim=1, keepdim=True))
            stem.bias.copy_(old.bias)
        base.features[0][0] = stem
        # Discard fourth-stage downsample, fourth stage, and final classifier.
        features = nn.Sequential(*list(base.features.children())[:6])
        self.backbones = nn.ModuleList([deepcopy(features), deepcopy(features)])
        self.projections = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(c, 256, 1) for c in (96, 192, 384)]) for _ in range(2)])
        self.integrators = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(256, 256, 3, padding=1) for _ in range(3)]) for _ in range(2)])
        if config.fusion_enabled:
            initial = math.log(config.fusion_init / (config.fusion_max - config.fusion_init))
            self.fusion_injection_logits = nn.Parameter(torch.full((2, 2), initial))
            self.fusion_mixture_logits = nn.Parameter(torch.zeros(2, 2))
        self.gates = nn.ModuleList([JointGate(config) for _ in range(6)])
        self.spatial_readouts = nn.ModuleList([SpatialReadout() for _ in range(6)])
        self.class_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(390, 64), nn.GELU(), nn.Linear(64, config.classes))
            for _ in range(3)])
        self.scene_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 2)) for _ in range(6)])
        self.discriminator = ResidualClassDiscriminator(config.classes)
        # Keep construction/RNG order identical to the gated arm so the same seed
        # initializes every shared parameter identically; inactive heads do not train.
        if not config.gate_enabled:
            self.gates.requires_grad_(False)
            self.scene_heads.requires_grad_(False)
            self.discriminator.requires_grad_(False)
        self._frozen_for_h = False

    def train(self, mode=True):
        # Once exported, parent H.train() cannot re-enable stochastic paths.
        return super().train(False if self._frozen_for_h else mode)

    def forward(self, images, valid, *, hard_gate=False):
        if images.ndim != 5 or images.shape[1:3] != (2, 1) or valid.shape != images.shape:
            raise ValueError("images and valid must both be [B,2,1,S,S]")
        if images.shape[-2] != images.shape[-1] or images.shape[-1] % 16:
            raise ValueError("canvas must be square and divisible by 16")
        if self._frozen_for_h and not hard_gate:
            raise ValueError("frozen Stage1 requires deterministic hard_gate=True")
        masks = tuple(F.avg_pool2d(valid.flatten(0, 1).to(images.dtype), s, s)
                      .unflatten(0, (len(images), 2)) for s in (4, 8, 16))
        output = {key: [[] for _ in range(3)] for key in
                  ("content", "spatial", "residual", "gates")}
        for pol in range(2):
            x = images[:, pol] * (valid[:, pol] > 0)
            native = []
            for level in range(3):
                x = self.backbones[pol][2 * level](x)
                x = self.backbones[pol][2 * level + 1](x)
                x = x * (masks[level][:, pol] > 0)
                native.append(self.projections[pol][level](x) * (masks[level][:, pol] > 0))
            def align(source, target):
                return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)
            if self.config.fusion_enabled:
                strength = self.config.fusion_max * self.fusion_injection_logits[pol].sigmoid()
                beta = self.fusion_mixture_logits[pol].softmax(0)
                fused = (native[0] + strength[0] * (beta[0] * align(native[1], native[0])
                                                  + beta[1] * align(native[2], native[0])),
                         native[1] + strength[1] * align(native[2], native[1]), native[2])
            else:
                fused = native
            for level in range(3):
                weight = masks[level][:, pol]
                support = (weight > 0).to(images.dtype)
                p = self.integrators[pol][level](fused[level] * support) * support
                index = 2 * level + pol
                gate = (self.gates[index](p, weight, hard=hard_gate)
                        if self.config.gate_enabled else support.expand_as(p))
                fc, fs = gate * p, (1 - gate) * p
                spatial = self.spatial_readouts[index](fc, weight)
                for key, value in zip(output, (fc, spatial, fs, gate)):
                    output[key][level].append(value)
        result = {key: tuple(torch.stack(pols, dim=1) for pols in levels)
                  for key, levels in output.items()}
        result["valid"] = masks
        return result

    @staticmethod
    def class_valid(geometries):
        return torch.stack([torch.stack([
            (level.slot_valid.bool() & (level.slots.sum((-2, -1)) > 0)).any()
            for level in g.levels]) for g in geometries]).bool()

    def class_logits(self, output, geometries):
        if len(geometries) != output["content"][0].shape[0]:
            raise ValueError("one geometry is required per ship")
        logits = []
        for level in range(3):
            features = output["spatial"][level]
            slots = torch.stack([g.levels[level].slots for g in geometries]).to(features)
            flags = torch.stack([g.levels[level].slot_valid for g in geometries]).to(features)
            flags = flags * (slots.sum((-2, -1)) > 0)
            pooled = weighted_pool(features[:, :, None], slots) * flags[..., None]
            packed = torch.cat((pooled.flatten(1), flags.flatten(1)), dim=1)
            prediction = self.class_heads[level](packed)
            if self.config.axis_symmetric:
                order = [0, 2, 1]
                swapped = torch.cat((pooled[:, :, order].flatten(1), flags[:, :, order].flatten(1)), dim=1)
                prediction = (prediction + self.class_heads[level](swapped)) / 2
            logits.append(prediction * flags.any(dim=(1, 2))[:, None])
        return torch.stack(logits, dim=1)

    def scene_predictions(self, output, background_weights):
        predictions, usable = [], []
        for level in range(3):
            features = output["residual"][level]
            weights = background_weights[level].to(features)
            if weights.ndim == 5:
                weights = weights.squeeze(2)
            weight = weights * (output["valid"][level][:, :, 0] > 0)
            values = weighted_pool(features, weight)
            predictions.append(torch.stack([self.scene_heads[2 * level + p](values[:, p])
                                             for p in range(2)], dim=1))
            usable.append(weight.sum((-2, -1)) > 0)
        return torch.stack(predictions, dim=1), torch.stack(usable, dim=1)

    def freeze_for_h(self):
        self._frozen_for_h = True
        self.requires_grad_(False)
        self.eval()
        return self

    @torch.no_grad()
    def encode(self, images, valid, geometries):
        if not self._frozen_for_h:
            raise RuntimeError("call freeze_for_h() before encoding H features")
        if len(geometries) != len(images):
            raise ValueError("one geometry is required per ship")
        output = self(images, valid, hard_gate=True)
        return [EncodedShip(tuple(x[i].detach() for x in output["content"]),
                            tuple(x[i].detach() for x in output["spatial"]),
                            geometries[i].to(images.device)) for i in range(len(images))]
