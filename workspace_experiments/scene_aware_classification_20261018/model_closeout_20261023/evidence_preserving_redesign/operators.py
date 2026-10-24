"""Executable operator prototypes, not an integrated or trained classifier."""
import math

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
from torch.nn import functional as F


def native_bands(image):
    """Original chip before placement/masking; uint8 -> [0,1], float already [0,1]."""
    image = np.asarray(image)
    image = image.astype(np.float64)/255. if image.dtype == np.uint8 else image.astype(np.float64)
    g1 = gaussian_filter(image, 1, mode="reflect")
    g2 = gaussian_filter(image, 2, mode="reflect")
    return np.stack((image-g1, g1-g2, g2))


def sample_offset(x, dx, dy):
    n, _, h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                           torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    xx = xx[None] + dx[:, None, None]
    yy = yy[None] + dy[:, None, None]
    grid = torch.stack((2*xx/(w-1)-1, 2*yy/(h-1)-1), -1)
    return F.grid_sample(x, grid.expand(n, -1, -1, -1), align_corners=True,
                         mode="bilinear", padding_mode="zeros")


class AxisPairConv(nn.Module):
    """Native-scale symmetric offset pairs; type = polarization * 3 + band.

    Inputs are N,C,H,W, not interpolated layer1 maps. Each type retains its own
    identity stream. The residual samples both sides of an undirected ship axis.
    No radar axis is fabricated when it is unavailable.
    """
    def __init__(self, channels=8):
        super().__init__()
        self.embedding = nn.Embedding(6, 4)
        self.gate = nn.Sequential(nn.Linear(3*channels+20, 12), nn.GELU(),
                                  nn.Linear(12, channels))
        self.basis = nn.Parameter(torch.randn(6, 4, channels)*.1)
        self.output = nn.Parameter(torch.zeros(6, channels, channels))
        self.register_buffer("radii", torch.tensor([1., 2., 4., 1., 2., 4.]))

    def forward(self, x, regions, valid, theta, type_id):
        role = torch.cat((regions, (1-regions.sum(1, keepdim=True)).clamp_min(0)), 1)
        role = role / role.sum(1, keepdim=True).clamp_min(1e-12)
        center = x.movedim(1, -1)
        emb = self.embedding(type_id)[:, None, None].expand(-1, x.shape[2], x.shape[3], -1)
        update = torch.zeros_like(x)
        for k, angle in enumerate((0., math.pi/2, math.pi/4, -math.pi/4)):
            dx = self.radii[type_id] * (theta+angle).cos()
            dy = self.radii[type_id] * (theta+angle).sin()
            plus = sample_offset(x*valid, dx, dy)
            minus = sample_offset(x*valid, -dx, -dy)
            vp, vm = sample_offset(valid, dx, dy), sample_offset(valid, -dx, -dy)
            plus = plus / vp.clamp_min(1e-12)
            minus = minus / vm.clamp_min(1e-12)
            pair_role = (sample_offset(role, dx, dy)+sample_offset(role, -dx, -dy))*.5
            relation = (role[:, :, None]*pair_role[:, None]).flatten(1, 2).movedim(1, -1)
            features = torch.cat((center, ((plus+minus)*.5).movedim(1, -1),
                                  (plus-minus).abs().movedim(1, -1), relation, emb), -1)
            gate = self.gate(features).sigmoid().movedim(-1, 1)
            coverage = valid*(vp > .999).to(x.dtype)*(vm > .999).to(x.dtype)
            difference = (plus+minus)*.5-x
            update = update + gate*self.basis[type_id, k, :, None, None]*difference*coverage
        delta = torch.einsum("noc,nchw->nohw", self.output[type_id], update/4)
        return x+delta*valid


def moment_pool(response, assignment, tau):
    """Preserve sufficient statistics of the existing two-sided summary.

    response: N,C,L; assignment: N,S,L; tau: N. Slots partition ONE role.
    Nonlinearity is evaluated before any slot averaging. Store log partitions
    and their tilted means so even a weak slot has a stable local decode.
    """
    mass = assignment.sum(-1)
    role = assignment.sum(1)
    total = role.sum(-1)
    raw_sum = torch.einsum("nsl,ncl->nsc", assignment, response)
    mu = raw_sum.sum(1) / torch.where(total > 0, total, torch.ones_like(total))[:, None]
    q = response-mu[..., None]
    q = q-q.mean(1, keepdim=True)
    logit = q/tau[:, None, None]
    stats = dict(mass=mass, raw_sum=raw_sum)
    log_mass = assignment.masked_fill(assignment <= 0, 1).log()
    log_mass = log_mass.masked_fill(assignment <= 0, -torch.inf)
    for sign, key in ((1, "plus"), (-1, "minus")):
        value = log_mass[:, :, None]+sign*logit[:, None]
        nonempty = mass[:, :, None] > 0
        safe_value = torch.where(nonempty[..., None], value, torch.zeros_like(value))
        log_z = torch.logsumexp(safe_value, -1)
        weight = torch.exp(safe_value-log_z[..., None])*nonempty[..., None]
        stats["log_z_"+key] = log_z.masked_fill(~nonempty, -torch.inf)
        stats["q_"+key] = (weight*q[:, None]).sum(-1)
    return stats


def decode_moments(stats, merge=False):
    """Returns mean, half high-low span, half high-low bias; not image bands."""
    mass = stats["mass"].sum(1, keepdim=True) if merge else stats["mass"]
    raw_sum = stats["raw_sum"].sum(1, keepdim=True) if merge else stats["raw_sum"]
    mu = raw_sum/torch.where(mass > 0, mass, torch.ones_like(mass))[..., None]
    means = []
    for key in ("plus", "minus"):
        value = stats["q_"+key]
        if merge:
            nonempty = mass[..., None] > 0
            log_z = torch.where(nonempty, stats["log_z_"+key], torch.zeros_like(value))
            weight = (log_z-log_z.logsumexp(1, keepdim=True)).exp()*nonempty
            value = (weight*value).sum(1, keepdim=True)
        means.append(value)
    hi, lo = means
    return torch.stack((mu, (hi-lo)*.5, (hi+lo)*.5), -1)


class TypedPairMetric(nn.Module):
    """A proposed metric for source-standardized evidence, NOT unit tokens.

    Same polarization/region/band is matched. Magnitude, signed band summaries
    and class-response distribution can be retained in the descriptor. Uniform
    rescaling of a descriptor is intentionally not made invisible by cosine.
    """
    def __init__(self, dim, types=18, hidden=4):
        super().__init__()
        self.project = nn.Linear(dim, hidden, bias=False)
        self.embedding = nn.Embedding(types, 4)
        self.output = nn.Linear(2*hidden+4, dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, q, s, type_id):
        # q N,I,D; s N,J,D, both using the SAME frozen source scaler.
        hq, hs = self.project(q), self.project(s)
        a, b = hq[:, :, None], hs[:, None]
        emb = self.embedding(type_id)[:, None, None].expand(-1, q.shape[1], s.shape[1], -1)
        phi = torch.cat(((a+b)*.5, (a-b).abs()*.5, emb), -1)
        w = 1+.25*self.output(F.gelu(phi)).tanh()
        distance = (w*(q[:, :, None]-s[:, None]).square()).sum(-1)/w.sum(-1)
        return 2*torch.exp(-.5*distance)-1
