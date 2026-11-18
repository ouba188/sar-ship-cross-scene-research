"""Reference-class similarity context; preserve M3 slots, metric and writeback."""
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE/'sar_m123_current_20261031'))
sys.path.insert(0, str(WORKSPACE/'m3_frequency_increment_20261028/local_scale_repair_20261101'))

import torch
from torch.nn import functional as F
from scaled_operator import MaskedCostLocalMatch


def relation_context(mu, available):
    """Cosine/softmax over legal *other* classes, independently per batch item."""
    normalized = F.normalize(mu, dim=-1)
    similarity = normalized @ normalized.transpose(-1, -2)
    diagonal = torch.eye(mu.shape[1], device=mu.device, dtype=torch.bool)
    allowed = available[:, :, None] & available[:, None, :] & ~diagonal
    # Finite sentinel keeps empty/single-class rows finite before explicit masking.
    weights = similarity.masked_fill(~allowed, torch.finfo(mu.dtype).min).softmax(-1)
    weights = weights * allowed
    total = weights.sum(-1, keepdim=True)
    weights = weights / torch.where(total > 0, total, torch.ones_like(total))
    return weights @ mu, weights


class RelationContextMatch(MaskedCostLocalMatch):
    def __init__(self, seed=17, classes=8, mode='relation'):
        super().__init__(seed=seed, classes=classes)
        if mode not in ('relation', 'uniform'):
            raise ValueError(mode)
        self.mode = mode
        self.record_context = False
        self.context_traces = []

    def other_class_context(self, mu, available):
        if self.mode == 'uniform':
            return super().other_class_context(mu, available)
        other, weights = relation_context(mu, available)
        if self.record_context:
            self.context_traces.append(dict(weights=weights.detach(), available=available.detach(),
                                            context=other.detach()))
        return other

    def forward(self, *args, **kwargs):
        self.context_traces = []
        return super().forward(*args, **kwargs)


def install(network, mode='relation'):
    """Replace a live M3 before optimizer/CUDA-graph creation, without changing weights."""
    previous = network.matcher
    parameter = next(previous.parameters())
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        current = RelationContextMatch(classes=previous.classes, mode=mode)
    current.to(device=parameter.device, dtype=parameter.dtype)
    current.load_state_dict(previous.state_dict(), strict=True)
    flags = dict(previous.named_parameters())
    for name, value in current.named_parameters():
        value.requires_grad_(flags[name].requires_grad)
    current.train(previous.training)
    network.matcher = current
    return current
