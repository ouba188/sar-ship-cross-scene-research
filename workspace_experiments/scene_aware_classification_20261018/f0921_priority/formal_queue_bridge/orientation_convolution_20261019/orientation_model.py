"""Region-masked orientation sampling on already contextualized frozen maps.

Angles are radians in image coordinates (x right, y down); radii are feature
pixels, clipped to [0, 4]. D receives its perturbed LOS from the caller.
CUDA grid_sample backward can be nondeterministic. The runner must explicitly
choose CPU or permit that CUDA operation; this module never changes global
determinism settings. Masking here does not undo backbone context mixing.
"""
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       'learned_geometry_crosspass_20261019'))
from model import Reader, centered


class OrientationReader(Reader):
    def __init__(self, arm):
        if arm not in ('A', 'B', 'C', 'D'):
            raise ValueError('arm must be A, B, C or D')
        super().__init__(arm)
        self.pre_norm = nn.LayerNorm(32)
        # Template, center/along/cross, channel; opposite points share weights.
        self.spatial_weight = nn.Parameter(torch.zeros(2, 3, 32))

    def reference_angles(self, geometry):
        ship, los, ship_valid, radar_valid = geometry[:, :4].unbind(-1)
        first = torch.full_like(ship, math.pi / 4)
        second = torch.zeros_like(los)
        if self.arm != 'A':
            first = torch.where(ship_valid > 0, ship, first)
        if self.arm in ('C', 'D'):
            second = torch.where(radar_valid > 0, los, second)
        return torch.stack((first, second), -1)

    def local_response(self, projected, masks, geometry):
        """Return [N,2,3,32,H,W] before normalization and center-area pooling.

        projected is [N,2,32,H,W]. Features are masked BEFORE interpolation;
        the denominator is the sum of sampled mask masses, independent of
        learned signed coefficients. Returned values are not area-weighted.
        """
        n, p, c, height, width = projected.shape
        geometry = geometry.to(device=projected.device, dtype=projected.dtype)
        masks = masks.to(device=projected.device, dtype=projected.dtype)
        angles = self.reference_angles(geometry)
        along_radius, across_radius = geometry[:, 4:6].clamp(0, 4).unbind(-1)
        along = torch.stack((angles.cos(), angles.sin()), -1) * along_radius[:, None, None]
        cross = torch.stack((-angles.sin(), angles.cos()), -1) * across_radius[:, None, None]
        offsets = torch.stack((torch.zeros_like(along), along, -along, cross, -cross), -2)
        y, x = torch.meshgrid(
            torch.arange(height, device=projected.device, dtype=projected.dtype),
            torch.arange(width, device=projected.device, dtype=projected.dtype),
            indexing='ij')
        centers = torch.stack((x, y), -1)
        grid = centers[None, :, :, None, None] + offsets[:, None, None]
        scale = projected.new_tensor((width, height))
        grid = 2 * (grid + 0.5) / scale - 1
        grid = grid[:, None].expand(-1, 3, -1, -1, -1, -1, -1)
        grid = grid.reshape(n * 3, height, width * 10, 2)

        # Both polarizations and mask share one interpolation per region.
        masked = projected[:, None] * masks[:, :, None, None]
        source = torch.cat((masked.reshape(n, 3, p * c, height, width),
                            masks[:, :, None]), dim=2).reshape(n * 3, p * c + 1, height, width)
        sampled = F.grid_sample(source, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        sampled = sampled.reshape(n, 3, p * c + 1, height, width, 2, 5)
        values = sampled[:, :, :p * c].reshape(n, 3, p, c, height, width, 2, 5)
        mass = sampled[:, :, -1].sum(-1)
        point_weights = self.spatial_weight[:, [0, 1, 1, 2, 2], :]
        numerator = torch.einsum('nrpchwtk,tkc->nrpchwt', values, point_weights)
        response = numerator / mass[:, :, None, None].clamp_min(1e-8)
        response = torch.where(mass[:, :, None, None] > 1e-8,
                               response, torch.zeros_like(response)).sum(-1)
        residual = projected[:, None] * (masks[:, :, None, None] > 0)
        return (residual + response).permute(0, 2, 1, 3, 4, 5)

    def forward(self, maps, masks, position, geometry, available, drop=None):
        n = len(maps)
        projected = self.content(maps.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        local = self.local_response(projected, masks, geometry)
        local = F.gelu(self.pre_norm(local.permute(0, 1, 2, 4, 5, 3)))
        local = local.permute(0, 1, 2, 5, 3, 4)
        masks = masks.to(dtype=local.dtype, device=local.device)
        region_mass = F.avg_pool2d(masks, 4)
        summed = F.avg_pool2d((local * masks[:, None, :, None]).reshape(-1, 32, 32, 32), 4)
        summed = summed.reshape(n, 2, 3, 32, 8, 8)
        # Merge region sums only after each region's nonlinear local readout.
        summed = torch.stack((summed[:, :, 0], summed[:, :, 1] + summed[:, :, 2]), 2)
        mass = torch.stack((region_mass[:, 0], region_mass[:, 1] + region_mass[:, 2]), 1)
        local = summed / mass[:, None, :, None].clamp_min(1e-8)
        local = local.permute(0, 1, 2, 4, 5, 3).reshape(n, 2, 128, 32)
        h = F.gelu(self.norm(local + self.position(position)[:, None])).reshape(n, 2, 2, 64, 32)
        mass = mass.reshape(n, 1, 2, 64).expand(-1, 2, -1, -1)
        summary = (h * mass[..., None]).sum((2, 3)) / mass.sum((2, 3))[..., None].clamp_min(1e-8)
        h = self.interact(h, self.context(summary)[:, :, None, None])
        output, weights = self.pool(h, mass)
        if drop is not None:
            kept = weights * (~drop)
            kept = kept / kept.sum(-1, keepdim=True).clamp_min(1e-8)
            output = (kept[..., None] * self.value(h)).sum(-2)
        present = mass.sum(-1) > 1e-8
        output = output * present[..., None] * available[:, None, None, None]
        return output.flatten(2), weights
