"""Tensor contract for the native-canvas, frozen-G two-stage model.

Geometry is computed before the network; it never reads class or attention.
All per-ship tensors omit a batch axis and retain polarization order VV, VH.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class LevelGeometry:
    valid: torch.Tensor          # [2,1,h,w], cell coverage
    slots: torch.Tensor          # [2,3,h,w], root/negative-half/positive-half weights
    slot_valid: torch.Tensor     # [2,3], half slots valid as a pair per polarization
    regions: torch.Tensor        # [N,2,h,w], coverage weights for cut regions
    match_valid: torch.Tensor    # [N,2], root pool or non-root observable

    def to(self, device):
        return LevelGeometry(*(getattr(self, k).to(device) for k in self.__dataclass_fields__))


@dataclass
class ShipGeometry:
    levels: tuple[LevelGeometry, ...]
    area: torch.Tensor           # [N], effective union area, not sum of pol areas
    intervals: torch.Tensor      # [N,2], normalized axial center and length
    coverage: torch.Tensor       # [N,2,3], per-branch effective area / region area
    axis_valid: bool
    atom_count: int
    canvas_size: int
    union_slots: torch.Tensor | None = None   # [3,h1,w1], exact union-polarization coverage
    union_regions: torch.Tensor | None = None # [N,h1,w1], exact union-polarization coverage

    @property
    def mass(self):
        return self.area / self.area.sum()

    def to(self, device):
        return ShipGeometry(tuple(x.to(device) for x in self.levels),
                            self.area.to(device), self.intervals.to(device),
                            self.coverage.to(device), self.axis_valid,
                            self.atom_count, self.canvas_size,
                            None if self.union_slots is None else self.union_slots.to(device),
                            None if self.union_regions is None else self.union_regions.to(device))


@dataclass
class EncodedShip:
    content: tuple[torch.Tensor, ...]  # three [2,256,h,w] frozen Fc
    spatial: tuple[torch.Tensor, ...]  # three [2,64,h,w] frozen G(Fc)
    geometry: ShipGeometry

    def to(self, device):
        return EncodedShip(tuple(x.to(device) for x in self.content),
                           tuple(x.to(device) for x in self.spatial), self.geometry.to(device))


def weighted_pool(features, weights):
    """features [...,C,H,W], weights [...,H,W]; returns [...,C]."""
    denominator = weights.sum((-2, -1), keepdim=False)
    result = (features * weights.unsqueeze(-3)).sum((-2, -1))
    return result / denominator.clamp_min(1e-12).unsqueeze(-1)
