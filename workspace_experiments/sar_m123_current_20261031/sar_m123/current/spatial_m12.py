"""Closeout M1/M2 without spectral branches; M3 remains unchanged."""
import torch
from torch import nn

from ..backbone_attention import ClassResponseFeedback
from ..bottleneck_evidence import BottleneckEvidenceResNet50
from ..pool_replacement import grid_occupancy, regional_mean


class SpatialClassResponseFeedback(ClassResponseFeedback):
    """Mean/high/low attention only, using the existing backbone call contract."""

    def forward(self, x, valid, energy_field=None, return_details=False):
        return super().forward(x, valid, return_details=return_details)


class SpatialEvidencePool(nn.Module):
    """Regional means without band descriptors, selectors or spectral gains."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x, masks, valid, energy=None, return_details=False):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError('Expected x with shape [B,channels,H,W].')
        occupancy = grid_occupancy(valid.to(x), x.shape[-2:]).clamp(0, 1)
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            local = x.to(dtype).masked_fill(occupancy == 0, 0)
            mean, mass = regional_mean(local, masks.to(dtype), valid.to(dtype))
        result = dict(mean=mean.to(x.dtype), pooled=mean.to(x.dtype), mass=mass)
        if return_details:
            result['valid'] = occupancy
        return result


class SpatialM12ResNet50(BottleneckEvidenceResNet50):
    """Keep region identities, spatial writeback and all M3 comparison paths."""

    architecture = 'native160_pre3x3_spatial_m12_m3'

    def __init__(self, weight_path=None, position='pre3x3', policy='local', classes=8):
        super().__init__(weight_path, position, policy, classes)
        for name, old in list(self.m1.items()):
            self.m1[name] = SpatialClassResponseFeedback(old.channels, classes)
        for i, old in enumerate(self.internal_m1):
            self.internal_m1[i] = SpatialClassResponseFeedback(old.channels, classes)
        for writer in self.internal_m2:
            writer.pool = SpatialEvidencePool(writer.pool.channels)
        self.m2 = nn.ModuleList(SpatialEvidencePool(old.channels) for old in self.m2)
        self.deep_pool = nn.ModuleList(SpatialEvidencePool(old.channels) for old in self.deep_pool)


class SpatialM123ResNet50(SpatialM12ResNet50):
    """Spatial M1/M2 and spatial-only M3; intervention selection stays intact."""

    architecture = 'native160_pre3x3_spatial_m12_spatial_m3'

    def __init__(self, weight_path=None, position='pre3x3', policy='local', classes=8):
        super().__init__(weight_path, position, policy, classes)
        for operators in self.m3:
            for operator in operators:
                operator.frequency = False


def from_legacy(model):
    """Transfer surviving weights, not an output-equivalent checkpoint conversion."""
    result = SpatialM12ResNet50(None, model.position, model.policy, model.internal_m1[0].classes
                              if len(model.internal_m1) else next(iter(model.m1.values())).classes)
    old_state = model.state_dict()
    result.load_state_dict({name: old_state[name] for name in result.state_dict()}, strict=True)
    return result


def without_m3_frequency(model):
    """Copy spatial M1/M2 weights without altering the historical checkpoint."""
    result = SpatialM123ResNet50(None, model.position, model.policy, model.internal_m1[0].classes
                                 if len(model.internal_m1) else next(iter(model.m1.values())).classes)
    result.load_state_dict(model.state_dict(), strict=True)
    return result
