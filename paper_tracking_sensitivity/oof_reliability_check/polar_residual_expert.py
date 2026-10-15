"""Feature-space contrast of stretched VV/VH chips, not a backscatter ratio.

Dual-polarization fusion and attention are established mechanisms; this module
only changes the visual features supplied to the existing frozen expert.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def pair_directions(geometry, classes):
    """Detached, unnormalized derivative of prior[top2] - prior[top1]."""
    with torch.no_grad(), torch.autocast(
            device_type=geometry.expert_coef.device.type, enabled=False):
        coef = geometry.expert_coef
        if coef.dtype in (torch.float16, torch.bfloat16):
            coef = coef.float()
        classes = classes.to(device=coef.device)
        difference = coef[classes[:, 1], 128:]-coef[classes[:, 0], 128:]
        return (difference/geometry.expert_scale.to(coef)[128:]) @ (
            geometry.ssl_components.to(coef))


class PolarResidualExpert(nn.Module):
    """Shared e206 layer4 followed by a zero-initialized polar residual.

    encode accepts cached Bx256x8x8 layer3 maps and returns Bx512x4x4 maps.
    fuse accumulates in the module parameter dtype (normally float32), including
    when encode ran under bfloat16 autocast. No classifier is trained here.
    """

    def __init__(self, backbone_state, mode='contrast'):
        super().__init__()
        if mode not in ('contrast', 'pooled'):
            raise ValueError("mode must be 'contrast' or 'pooled'")
        net = resnet18(weights=None)
        missing, unexpected = net.load_state_dict(backbone_state, strict=False)
        if set(missing) != {'fc.weight', 'fc.bias'} or unexpected:
            raise ValueError('expected e206 backbone without classification head')
        self.layer4 = net.layer4
        self.mode = mode
        # Instantiate in identical order in both modes for matched random seeds.
        self.common_norm = nn.LayerNorm(512)
        self.delta_norm = nn.LayerNorm(512)
        self.query = nn.Linear(512, 64, bias=False)
        self.key = nn.Linear(1024, 64, bias=False)
        self.content = nn.Linear(1024, 1, bias=False)
        self.out_proj = nn.Linear(512, 512, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        for module in self.layer4.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def encode(self, vh, vv):
        maps = self.layer4(torch.cat((vh, vv), dim=0))
        return maps.split(vh.shape[0], dim=0)

    def fuse(self, h_map, v_map, xs, geometry, disable_contrast=False,
             return_details=False):
        with torch.autocast(device_type=h_map.device.type, enabled=False):
            h = h_map.to(dtype=self.out_proj.weight.dtype).flatten(2).transpose(1, 2)
            v = v_map.to(dtype=self.out_proj.weight.dtype).flatten(2).transpose(1, 2)
            base_features = h.mean(dim=1)
            delta = v-h
            global_delta = delta.mean(dim=1)
            contrast = torch.zeros_like(global_delta)
            attention_details = {}
            if self.mode == 'contrast' and not disable_contrast:
                with torch.no_grad():
                    prior = geometry(xs.to(base_features), base_features)[0]
                    classes = prior.topk(2, dim=1).indices.detach()
                    direction = F.normalize(
                        pair_directions(geometry, classes).to(base_features), dim=1)
                tokens = torch.cat((self.common_norm((h+v)/2),
                                    self.delta_norm(delta)), dim=-1)
                query = self.query(direction)
                pair = (self.key(tokens)*query.unsqueeze(1)).sum(dim=-1)/math.sqrt(64)
                content = self.content(tokens).squeeze(-1)
                a_plus = (content+pair).softmax(dim=1)
                a_minus = (content-pair).softmax(dim=1)
                contrast = ((a_plus-a_minus).unsqueeze(-1)*delta).sum(dim=1)
                if return_details:
                    attention_details = dict(a_plus=a_plus, a_minus=a_minus,
                                             expert_classes=classes)
            residual = self.out_proj(global_delta+contrast)
            features = base_features+residual
            if return_details:
                details = dict(residual_norm=residual.norm(dim=1),
                               contrast_norm=contrast.norm(dim=1),
                               base_features=base_features, global_delta=global_delta,
                               **attention_details)
                return features, details
            return features

    def forward(self, vh, vv, xs, geometry):
        h_map, v_map = self.encode(vh, vv)
        return self.fuse(h_map, v_map, xs, geometry)
