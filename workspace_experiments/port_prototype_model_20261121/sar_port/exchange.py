"""Explicit current CCREL and cross-mean PCEA, with source-only statistics.

The layer1 fine residual and class-detail gain variant are not used here.
The existing 20x20 local content term IS retained: it is part of cross_mean
PCEA, independent of the old optional layer1 residual.
"""
import numpy as np  # Load before torch in the supported Windows runtime.
import torch
from torch import nn
from torch.nn import functional as F

from .region_ops import grid_occupancy, region_integrals


def origin0_overlap(size=160, source=20, target=10):
    def edges(count):
        return torch.cat((torch.tensor([-.5]),
                          (torch.arange(count-1)+.5)*(size/count),
                          torch.tensor([size-.5])))
    fine, coarse = edges(source), edges(target)
    overlap = (torch.minimum(coarse[1:, None], fine[None, 1:])-
               torch.maximum(coarse[:-1, None], fine[None, :-1])).clamp_min(0)
    return overlap/(coarse[1:]-coarse[:-1])[:, None]


def regional_reduce(value, mask, overlap):
    numerator = torch.einsum('ih,bchw,jw->bcij', overlap, value*mask, overlap)
    denominator = torch.einsum('ih,bchw,jw->bcij', overlap, mask, overlap)
    return numerator/denominator.clamp_min(1e-8)


class CCREL(nn.Module):
    """Class-response regional readout in body/end/side order, no query GT."""

    def __init__(self, classes=8):
        super().__init__()
        self.classes = classes
        self.value = nn.ModuleList(nn.Conv2d(512, 32, 1, bias=False) for _ in range(2))

    def project_raw(self, raw):
        return torch.stack([F.linear(raw[:, p], self.value[p].weight[:, :, 0, 0])
                            for p in range(2)], 1)

    def forward(self, early):
        raw, effective, local_values = [], [], []
        for p in range(2):
            x = early['f2'][:, p]
            value = F.layer_norm(x.permute(0, 2, 3, 1), (512,)).permute(0, 3, 1, 2)
            local_values.append(self.value[p](value))
            value, mask, response, occupancy = (t.flatten(2) for t in
                (value, early['masks'][:, p], early['responses'][:, p], early['occupancy'][:, p]))
            uniform = mask/mask.sum(-1, keepdim=True).clamp_min(1e-8)
            c = self.classes
            relative = (c*response-response.sum(1, keepdim=True))/(c-1)
            vw = occupancy/occupancy.sum(-1, keepdim=True).clamp_min(1e-8)
            centered = relative-(relative*vw).sum(-1, keepdim=True)
            score = 2*torch.tanh(centered/((centered.square()*vw).sum(-1, keepdim=True)+.01).sqrt())
            logits = score[:, None]+mask.clamp_min(1e-12).log()[:, :, None]
            weight = logits.masked_fill(mask[:, :, None] <= 0, -1e4).softmax(-1)
            weight = weight*(mask[:, :, None] > 0)
            weight = weight/weight.sum(-1, keepdim=True).clamp_min(1e-8)
            mean = torch.einsum('bru,bdu->brd', uniform, value)
            classes = torch.einsum('brku,bdu->brkd', weight, value)
            raw.append(torch.cat((mean[:, :, None], classes), 2))
            effective.append(mask.sum(-1).square()/mask.square().sum(-1).clamp_min(1e-8))
        raw = torch.stack(raw, 1)
        return dict(raw=raw, evidence=self.project_raw(raw),
                    effective=torch.stack(effective, 1), local_value=torch.stack(local_values, 1))


class SourcePolarExchange(nn.Module):
    """Current cross_mean PCEA callable for ShipBackbone.middle().

    Training: encode_source(source_early) returns statistics from CURRENT
    legal source features, retaining their gradient graph. Pass that result
    explicitly to forward; no source bank is stored or mutated on the module.
    The caller's fold/provenance checks establish that source_early is
    source-fit data, never test queries.

    Deployment: detach/save the returned statistic tensors with the encoded
    source bank. Different frozen banks can then be interleaved safely.
    There is no implicit target calibration or guessed default scale.
    """

    def __init__(self, classes=8, use_local_detail=True, *, strength=.1,
                 source_floor=.01, source_relative_floor=.1, source_epsilon=1e-8):
        super().__init__()
        self.classes, self.use_local_detail = classes, use_local_detail
        if source_floor <= 0 or source_epsilon <= 0:
            raise ValueError('Source normalization floor and epsilon must be positive.')
        self.strength = strength
        self.source_floor, self.source_relative_floor = source_floor, source_relative_floor
        self.source_epsilon = source_epsilon
        self.reader = CCREL(classes)
        self.project = nn.ModuleList(nn.Linear(32, 16, bias=False) for _ in range(2))
        self.region = nn.Parameter(torch.zeros(3, 16))
        self.slot = nn.Parameter(torch.zeros(classes+1, 16))
        self.gate = nn.Sequential(nn.Linear(64, 16), nn.SiLU(), nn.Linear(16, 3))
        self.output = nn.ModuleList(nn.Linear((classes+1)*16, 256, bias=False) for _ in range(2))
        self.scale_output = nn.ModuleList(nn.Linear((classes+1)*16, 256, bias=False) for _ in range(2))
        for layer in list(self.output)+list(self.scale_output):
            nn.init.zeros_(layer.weight)
        self.register_buffer('overlap', origin0_overlap(), persistent=False)

    def encode_source(self, source_early, source_weight=None):
        """Return source statistics; optional weights have shape [R,3].

        Pair legality for M3 is separate from this shared source statistic.
        This method does not inspect labels, port names, or current queries.
        The result contains only mean, scale, and region availability; it
        contains no images or source feature maps. Detach at the bank export
        boundary, not here, to preserve joint-training source gradients.
        """
        source = self.reader(source_early)['evidence']
        masks = source_early['masks']
        available = source_early['available'].bool()
        weight = (available.all(1)[:, None] & (masks.sum((-2, -1)) > 0).all(1)).to(source)
        if source_weight is not None:
            weight = weight*source_weight.to(source)
        w = weight[:, None, :, None, None]
        count = w.sum(0)
        mean = (source*w).sum(0)/count.clamp_min(self.source_epsilon)
        variance = ((source-mean).square()*w).sum(0)/count.clamp_min(self.source_epsilon)
        moment2 = variance+mean.square()
        floor = (self.source_relative_floor*moment2.mean(-1, keepdim=True).sqrt()).clamp_min(self.source_floor)
        scale = torch.maximum(variance.clamp_min(1e-12).sqrt(), floor)
        return dict(mean=mean, scale=scale, source_active=weight.sum(0) > 0)

    def joint(self, u, v):
        b, r, k, n, d = u.shape
        er = self.region[None, :, None, None].expand(b, r, k, n, d)
        ek = self.slot[None, None, :, None].expand(b, r, k, n, d)
        raw = self.gate(torch.cat((u, v, er, ek), -1))
        alpha, beta = raw[..., :2].softmax(-1), raw[..., 2:].sigmoid()
        return alpha[..., :1]*u+alpha[..., 1:]*v+beta*u*v

    def forward(self, x, early, data, source):
        read = self.reader(early)
        evidence, local_value = read['evidence'], read['local_value']
        mean, scale = source['mean'], source['scale']
        t = (evidence-mean)/scale
        reader_masks = early['masks']
        masks = reader_masks.flatten(-2)
        mass = masks/masks.sum(-1, keepdim=True).clamp_min(1e-8)
        av = data['available']
        valid10 = grid_occupancy(data['valid'].flatten(0, 1), (10, 10)).reshape(len(x), 2, 1, 10, 10)
        writer_masks = torch.stack([torch.minimum(region_integrals(data['slots'][:, p]), valid10[:, p])
                                    for p in range(2)], 1)*av[..., None, None, None]
        uv = []
        # The historical detail denominator is stop-gradient. The evidence
        # standardization above keeps source gradients. No mutable buffers
        # are used, including across clean/intervened training forwards.
        detail_scale = scale.detach()
        for p in range(2):
            center = evidence[:, p, :, 0, :, None, None]
            # The original local-detail denominator is the stored source scale,
            # while standardized evidence above keeps its live source gradient.
            unit = detail_scale[p, :, 0, :, None, None]
            detail = .5*((local_value[:, p, None]-center)/unit).tanh()
            if not self.use_local_detail:
                detail = torch.zeros_like(detail)
            local = t[:, p, :, :, None, :]+detail.flatten(-2).permute(0, 1, 3, 2)[:, :, None]
            uv.append(self.project[p](local).tanh())
        active = (av.bool().all(1)[:, None] & (reader_masks.sum((-2, -1)) > 0).all(1)
                  & (writer_masks.sum((-2, -1)) > 0).all(1) & source['source_active'][None])
        writer = writer_masks/writer_masks.sum(2, keepdim=True).clamp_min(1)
        updates = []
        for p in range(2):
            own, other = uv[p], uv[1-p]
            other_mass = mass[:, 1-p, :, None, :, None]
            context = (other*other_mass).sum(-2, keepdim=True).expand_as(own)
            h = self.joint(own, context) if p == 0 else self.joint(context, own)
            batch, regions, slots, _, rank = h.shape
            h = h.permute(0, 1, 2, 4, 3).reshape(batch*regions*slots, rank, 20, 20)
            rm = reader_masks[:, p, :, None].expand(-1, -1, slots, -1, -1).reshape(-1, 1, 20, 20)
            h = regional_reduce(h, rm, self.overlap.to(h))
            h = h.reshape(batch, regions, slots, rank, 10, 10).permute(0, 1, 4, 5, 2, 3).flatten(-2)
            a = self.scale_output[p](h).permute(0, 1, 4, 2, 3)
            b = self.output[p](h).permute(0, 1, 4, 2, 3)
            mask = writer_masks[:, p].flatten(2)
            w = mask/mask.sum(-1, keepdim=True).clamp_min(1e-8)
            region = torch.einsum('bru,bcu->brc', w, x[:, p].flatten(2))
            amplitude = region.square().mean(-1, keepdim=True).sqrt().detach()
            local_x = F.layer_norm(x[:, p].permute(0, 2, 3, 1), (256,)).permute(0, 3, 1, 2).tanh()
            content = (a*local_x[:, None]+b).tanh()
            delta = self.strength*amplitude[..., None, None]*content*active[..., None, None, None]
            updates.append((delta*writer[:, p, :, None]).sum(1))
        return x+torch.stack(updates, 1)
