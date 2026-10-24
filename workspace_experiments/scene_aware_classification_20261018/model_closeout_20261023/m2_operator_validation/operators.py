"""Role-conditioned convolution and mass-preserving within-role pooling."""
import torch
from torch import nn
from torch.nn import functional as F

SLOT_GROUPS = (slice(0, 8), slice(16, 20), slice(8, 16))


def shift(x, dy, dx):
    h, w = x.shape[-2:]
    return F.pad(x, (1, 1, 1, 1))[..., 1+dy:1+dy+h, 1+dx:1+dx+w]


class RoleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(32, 16, 1, bias=False)
        self.expand = nn.Conv2d(16, 32, 1, bias=False)
        self.gate = nn.Sequential(nn.Linear(44, 12), nn.GELU(), nn.Linear(12, 4))
        self.basis = nn.Parameter(torch.randn(9, 4, 16) / 3)
        nn.init.zeros_(self.expand.weight)

    def forward(self, x, region, theta):
        norm = F.layer_norm(x.movedim(1, -1), (32,)).movedim(-1, 1)
        v = self.reduce(norm)
        mass = region.sum(1, keepdim=True)
        role = region / mass.clamp_min(1e-12)
        eligible = (mass > 0).to(x.dtype)
        center = v.movedim(1, -1)
        out = torch.zeros_like(v)
        for k, (dy, dx) in enumerate(( (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) )):
            other = shift(v, dy, dx)
            rel = (role[:, :, None] * shift(role, dy, dx)[:, None]).flatten(1, 2).movedim(1, -1)
            along = dx*theta.cos() + dy*theta.sin()
            across = -dx*theta.sin() + dy*theta.cos()
            angle = torch.stack((along.square(), across.square(), along*across), -1)
            angle = angle[:, None, None].expand(-1, x.shape[2], x.shape[3], -1)
            weight = self.gate(torch.cat((center, other.movedim(1, -1), rel, angle), -1)).softmax(-1)
            filt = (weight @ self.basis[k]).movedim(-1, 1)
            out = out + filt * other * shift(eligible, dy, dx)
        return self.expand(out / 9) * eligible


class GroupPool(nn.Module):
    def __init__(self, iterations=24):
        super().__init__()
        self.project = nn.Linear(32, 4, bias=False)
        self.score = nn.Sequential(nn.Linear(12, 8), nn.GELU(), nn.Linear(8, 1, bias=False))
        nn.init.zeros_(self.score[-1].weight)
        self.iterations = iterations

    def assignment(self, x, a0):
        # The same source content is compared to each geometrically anchored slot.
        f = F.layer_norm(x.movedim(1, -1), (32,)).flatten(1, 2)
        h = self.project(f)
        result = []
        for slots in SLOT_GROUPS:
            prior = a0[:, slots].flatten(2).transpose(1, 2)
            total = prior.sum((1, 2), keepdim=True).clamp_min(1e-20)
            prior = prior / total
            a, b = prior.sum(2), prior.sum(1)
            anchor = torch.bmm(prior.transpose(1, 2), h) / b.clamp_min(1e-20)[..., None]
            q = h[:, :, None].expand(-1, -1, anchor.shape[1], -1)
            s = anchor[:, None].expand(-1, h.shape[1], -1, -1)
            ell = .5*self.score(torch.cat((q+s, (q-s).abs(), q*s), -1)).squeeze(-1).tanh()
            kernel = prior * ell.exp()
            v = torch.ones_like(b)
            for _ in range(self.iterations):
                u = a / torch.bmm(kernel, v[..., None]).squeeze(-1).clamp_min(1e-20)
                v = b / torch.bmm(kernel.transpose(1, 2), u[..., None]).squeeze(-1).clamp_min(1e-20)
            allocation = u[..., None]*kernel*v[:, None]
            result.append((allocation, a, b, prior))
        return result


def class_summary(response, weights, tau):
    """Response B,C,P and mass B,R,P -> B,R,C,(mean,high/low spread,bias)."""
    norm = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
    mu = torch.einsum('brp,bcp->brc', norm, response)
    q = response[:, None] - mu[..., None]
    q = q - q.mean(2, keepdim=True)
    logw = norm.clamp_min(1e-30).log()[:, :, None]
    logw = logw.masked_fill(norm[:, :, None] <= 0, -1e4)
    t = tau[:, None, None, None]
    hi = ((logw + q/t).softmax(-1)*q).sum(-1)
    lo = ((logw - q/t).softmax(-1)*q).sum(-1)
    out = torch.stack((mu, (hi-lo)/2, (hi+lo)/2), -1)
    return out * (weights.sum(-1) > 0)[:, :, None, None]


def pooled_summary(response, a0, tau, assignments=None):
    parts = []
    response = response.flatten(2)
    for r, slots in enumerate(SLOT_GROUPS):
        if assignments is None:
            a = a0[:, slots].flatten(2).transpose(1, 2)
            a = a / a.sum((1, 2), keepdim=True).clamp_min(1e-20)
            b = a.sum(1)
        else:
            a, _, b, _ = assignments[r]
        tokens = torch.bmm(response, a) / b.clamp_min(1e-20)[:, None]
        parts.append(class_summary(tokens, b[:, None], tau)[:, 0])
    return torch.stack(parts, 1)


class RegionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(69, 12), nn.GELU(), nn.Linear(12, 1))

    def forward(self, x):
        y = self.net(x).squeeze(-1)
        return y-y.mean(1, keepdim=True)


def regional_raw(base, summary, quality):
    center = lambda x: x-x.mean(1, keepdim=True)
    cols = [center(base)]
    for p in range(2):
        for r in range(3):
            for k in range(3):
                value = summary[:, p, r, :, k]
                cols.extend((value, center(value)))
    return torch.cat((torch.stack(cols, -1), quality[:, None].expand(-1, 8, -1)), -1)


class Candidate(nn.Module):
    def __init__(self, arm, ck, wproject, tau):
        super().__init__()
        self.arm = arm
        self.conv = RoleConv() if arm.startswith('conv_') else None
        self.pool = GroupPool() if 'learned_pool' in arm else None
        self.head = RegionHead()
        self.head.load_state_dict(ck['state'])
        self.register_buffer('mu', torch.as_tensor(ck['mu'], dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(ck['scale'], dtype=torch.float32))
        self.register_buffer('wproject', torch.as_tensor(wproject, dtype=torch.float32))
        self.register_buffer('tau', torch.as_tensor(tau, dtype=torch.float32))

    def forward(self, data, return_summary=False):
        response, region = data['response'], data['region']
        n = len(response)
        response = response.flatten(0, 2)
        region = region.flatten(0, 2)
        pol = torch.arange(2, device=response.device)[None, :, None].expand(n, 2, 4).reshape(-1)
        if self.conv is not None or self.pool is not None:
            x = data['x'].flatten(0, 2)
        if self.conv is not None:
            delta = self.conv(x, region, data['theta'].reshape(-1))
            response = response + torch.einsum('bcd,bdhw->bchw', self.wproject[pol], delta)
            x = x + delta
        if 'pool' in self.arm:
            a0 = data['a0'].flatten(0, 2)
            assignments = self.pool.assignment(x, a0) if self.pool is not None else None
            summary = pooled_summary(response, a0, self.tau[pol], assignments)
        else:
            summary = class_summary(response.flatten(2), region.flatten(2), self.tau[pol])
        summary = summary.reshape(n, 2, 4, 3, 8, 3).mean(2)
        summary = summary * data['region_valid'][:, :, :, None, None]
        raw = regional_raw(data['s0'], summary, data['quality'])
        score = data['rest'] + self.head((raw-self.mu)/self.scale)
        return (score, summary) if return_summary else score
