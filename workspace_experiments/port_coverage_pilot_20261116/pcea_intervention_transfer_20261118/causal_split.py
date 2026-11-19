"""CausalDRL-inspired complementary features; no claim of identified causality."""
import torch
from torch import nn
from torch.nn import functional as F


class PolarChannelSplit(nn.Module):
    """Shared across samples/positions, separate for VV and VH, fixed capacity.

    Forward uses an exact top-k mask in both train and eval. The sigmoid
    straight-through derivative trains its scores without a hard-mask dead end.
    """
    def __init__(self, channels=256, keep_ratio=.5, seed=20261119):
        super().__init__()
        if not 0 < keep_ratio <= 1:
            raise ValueError('keep_ratio must be in (0, 1]')
        self.keep = max(1, round(channels*keep_ratio))
        generator = torch.Generator().manual_seed(seed)
        self.scores = nn.Parameter(.01*torch.randn(2, channels, generator=generator))

    def get_extra_state(self):
        return {'keep': self.keep}

    def set_extra_state(self, state):
        self.keep = int(state['keep'])

    def mask(self):
        soft = self.scores.sigmoid()
        indices = self.scores.topk(self.keep, dim=-1).indices
        hard = torch.zeros_like(soft).scatter(-1, indices, 1.)
        return hard + (soft-soft.detach())

    def forward(self, features):
        if features.ndim != 5 or features.shape[1:3] != self.scores.shape:
            raise ValueError('Expected [batch, 2, channels, height, width]')
        mask = self.mask()[None, :, :, None, None]
        return features*mask, features*(1-mask)


def donor_permutation(available, generator):
    """No labels used. Singletons are excluded, never disguised as swaps."""
    available = available.detach().cpu().bool()
    donor = torch.arange(len(available))
    eligible = torch.zeros(len(available), dtype=torch.bool)
    for code in range(1, 4):
        indices = torch.where(available[:, 0].long()+2*available[:, 1].long() == code)[0]
        if len(indices) < 2:
            continue
        order = indices[torch.randperm(len(indices), generator=generator)]
        donor[order] = order.roll(1)
        eligible[order] = True
    return donor, eligible


def combine(causal, bias, donor, valid, available):
    """Feature-space intervention, not a physically simulated SAR observation.

    Inject only where recipient and donor have real image support. Coordinates
    are image-grid coordinates; this does not claim ship-part registration.
    """
    donor = donor.to(causal.device)
    if valid.ndim == 4:
        valid = valid[:, :, None]
    occupancy = F.adaptive_avg_pool2d(valid.flatten(0, 1).float(), causal.shape[-2:])
    occupancy = occupancy.reshape(len(causal), 2, 1, *causal.shape[-2:])
    occupancy = occupancy*available[:, :, None, None, None]
    overlap = torch.minimum(occupancy, occupancy[donor])
    return causal + bias[donor]*overlap


def classification(logits, auxiliary, labels, weights, available):
    ce = (F.cross_entropy(logits, labels, reduction='none')*weights).mean()
    per_polar = F.cross_entropy(auxiliary.flatten(0, 1),
        labels[:, None].expand(-1, 2).flatten(), reduction='none').reshape(-1, 2)
    aux = ((per_polar*available).sum(1)/available.sum(1).clamp_min(1)*weights).mean()
    return ce+.02*aux


def swap_consistency(changed, causal, eligible):
    errors = (changed.softmax(-1)-causal.detach().softmax(-1)).square().mean(-1)
    weights = eligible.to(errors)
    return (errors*weights).sum()/weights.sum().clamp_min(1)


def bias_prior_loss(logits, prior, available):
    errors = -(prior.detach()*logits.log_softmax(-1)).sum(-1)
    weights = available.any(1).to(errors)
    return (errors*weights).sum()/weights.sum().clamp_min(1)


def effective_prior(labels, weights, classes):
    counts = torch.bincount(labels.cpu(), weights=weights.cpu().double(), minlength=classes)
    return (counts/counts.sum()).float()
