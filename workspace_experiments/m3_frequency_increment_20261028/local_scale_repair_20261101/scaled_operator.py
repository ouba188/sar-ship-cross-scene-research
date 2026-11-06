"""Change auxiliary classification temperature only; keep feature writeback exact."""
from pathlib import Path
import sys
import torch

PREVIOUS = Path(__file__).resolve().parent.parent/'class_competitive_matching_20261031'
sys.path.insert(0, str(PREVIOUS))
from operator_block import CompetitiveMatch
from sar_m123.m3 import sinkhorn


class ScaledLocalMatch(CompetitiveMatch):
    def forward(self, *args, **kwargs):
        delta, local, valid, traces = super().forward(*args, **kwargs)
        local = torch.where(valid, local / self.value[0].out_features, local)
        return delta, local, valid, traces


class MaskedCostLocalMatch(ScaledLocalMatch):
    def transport(self, cost, qa, ra, allowed):
        # Empty slots must not backpropagate through clamped Sinkhorn marginals.
        cost = torch.where(allowed, cost, torch.zeros_like(cost))
        return sinkhorn(cost, qa[:, None], ra[None], steps=60) * allowed
