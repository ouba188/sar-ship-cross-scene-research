"""Content-conditioned parent-or-children evidence, without token averaging after selection."""
import torch
from torch import nn
from torch.nn import functional as F

# Each parent stays within one physical region and one end/side where applicable.
GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 10), (9, 11),
          (12, 14), (13, 15), (16, 17), (18, 19))
FINE_INDEX = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25)
COARSE_INDEX = (8, 9, 18, 19, 20, 21, 26, 27)
PARENT = tuple(next(g for g, ids in enumerate(GROUPS) if i in ids) for i in range(20))
REGIONS = (torch.arange(10), torch.arange(10, 22), torch.arange(22, 28))


def pack(fine, coarse, axis):
    order = list(FINE_INDEX) + list(COARSE_INDEX)
    inverse = torch.tensor(sorted(range(28), key=order.__getitem__), device=fine.device)
    return torch.cat((fine, coarse), axis).index_select(axis, inverse)


def parent_sum(x, axis):
    return torch.stack([x.index_select(axis, torch.tensor(ids, device=x.device)).sum(axis)
                        for ids in GROUPS], axis)


class ContentScaleSelector(nn.Module):
    def __init__(self, channels=768, hidden=32):
        super().__init__()
        self.gates = nn.ModuleList(nn.Sequential(nn.Linear(channels*2, hidden), nn.GELU(),
                                                 nn.Linear(hidden, 1)) for _ in range(2))
        self.register_buffer('parent', torch.tensor(PARENT), persistent=False)

    def forward(self, fine, mass, write, eligible, mode='auto', override=None):
        """fine[B,2,20,D], mass[B,2,20], write[B,2,20,H,W]. g=1 means children.

        Soft relaxation during training; hard mutually exclusive choices at inference.
        The same selector is used for queries and references. No class labels enter it.
        """
        if mode == 'auto':
            mode = 'soft' if self.training else 'hard'
        if mode not in ('soft', 'hard', 'fine', 'coarse'):
            raise ValueError(mode)
        total = parent_sum(mass, 2)
        coarse = parent_sum(fine*mass[..., None], 2)/total[..., None].clamp_min(1e-12)
        normalized = F.layer_norm(fine, (fine.shape[-1],))
        mean = parent_sum(normalized*mass[..., None], 2)/total[..., None].clamp_min(1e-12)
        parent_mean = mean.index_select(2, self.parent)
        variance = parent_sum((normalized-parent_mean).square()*mass[..., None], 2)
        variance = variance/total[..., None].clamp_min(1e-12)
        content = torch.cat((mean, (variance+1e-6).sqrt()), -1)
        probability = torch.stack([self.gates[p](content[:, p]).squeeze(-1) for p in range(2)], 1).sigmoid()
        eligible = eligible.bool() & (total > 0)
        if override is not None:
            if override.shape != probability.shape:
                raise ValueError('override must have shape [batch,2,8]')
            choice = override.to(probability)
            if not bool(((choice >= 0) & (choice <= 1)).all()):
                raise ValueError('override probabilities must lie in [0,1]')
        elif mode == 'soft':
            choice = probability
        elif mode == 'hard':
            choice = (probability >= .5).to(probability)
        else:
            choice = torch.ones_like(probability) if mode == 'fine' else torch.zeros_like(probability)
        choice = choice*eligible
        child_choice = choice.index_select(2, self.parent)
        coarse_write = parent_sum(write, 2)
        selected_mass = pack(mass*child_choice, total*(1-choice), 2)
        selected_write = pack(write*child_choice[..., None, None],
                              coarse_write*(1-choice)[..., None, None], 2)
        descriptor = pack(fine, coarse, 2)
        # Expected additional active tokens, not measured FLOPs or runtime.
        children = parent_sum((mass > 0).to(mass), 2)
        extra = (children-1).clamp_min(0)*eligible
        cost = (probability*extra).sum()/extra.sum().clamp_min(1)
        trace = dict(probability=probability, choice=choice, eligible=eligible,
                     active_tokens=(selected_mass > 0).sum(-1),
                     expected_tokens=((total > 0).to(mass)+probability*extra).sum(-1))
        return dict(desc=descriptor, mass=selected_mass, write=selected_write,
                    scale_cost=cost, scale_trace=trace)
