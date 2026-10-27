"""Trainable backbone M3 candidate, not an equivalent use of old M3 weights.

The caller supplies a fixed class-balanced reference bank without query labels.
Frequency fields come from raw-chip Gabor extraction on the native 160 canvas
(possibly area-reduced upstream), never from an FFT of backbone features.
"""
import torch
from torch import nn
from torch.nn import functional as F

from . import tokens
from .m3 import PairMetric, sinkhorn


class ReferenceMatchFeedback(nn.Module):
    """Match body/side/end slots separately and write reference values back.

    ``hidden`` is fixed to 32 by PairMetric's 96 -> 4 -> 32 contract. Query and
    reference feature grids may differ. Slots and frequency share a canvas;
    valid may have a different resolution. All grids cover the same native
    canvas. Slot masks are occupancy fractions already clipped to footprints.

    Details contain the residual ``update`` [B,C,H,W], ``reference_weights``
    [B,K,20] (normalized across K per query slot), and tuples of differentiable
    ``transport`` / ``similarity`` [B,K,R,R] for R=8,8,4. Missing pairs, slots
    and regions have exactly zero transport and reference weights.
    """

    def __init__(self, channels, hidden=32, transport_eps=.1, use_frequency=True):
        super().__init__()
        if hidden != 32:
            raise ValueError('PairMetric requires hidden=32 and 96 covariance dimensions.')
        self.channels = channels
        if transport_eps <= 0:
            raise ValueError('Transport epsilon must be positive.')
        self.transport_eps = float(transport_eps)
        self.use_frequency = bool(use_frequency)
        self.project = nn.Conv2d(channels, hidden, 1, bias=False)
        self.metric = PairMetric()
        if not self.use_frequency:
            self.metric.requires_grad_(False)
        self.output = nn.Conv2d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.gain = nn.Parameter(torch.tensor(.1))

    def _describe(self, x, slot_masks, valid, frequency):
        if slot_masks.shape[:2] != (len(x), 20):
            raise ValueError('Expected 20 slot masks per feature map.')
        if frequency.shape != (len(x), 3, *slot_masks.shape[-2:]):
            raise ValueError('Three Gabor fields must share the slot-mask canvas.')
        size = x.shape[-2:]
        masks = slot_masks.to(x).clamp(0, 1)
        occupancy = F.interpolate(masks, size=size, mode='area')
        footprint = F.interpolate(valid.to(x), size=size, mode='area').clamp(0, 1)
        masks = torch.minimum(occupancy, footprint)
        mass = masks.sum((-2, -1))
        den = mass.clamp_min(1e-8)
        z = F.layer_norm(x.permute(0, 2, 3, 1), (self.channels,))
        z = self.project(z.permute(0, 3, 1, 2))
        mean = torch.einsum('bshw,bdhw->bsd', masks, z) / den[..., None]

        # Integrate slot * energy before reducing resolution: multiplying two
        # independently averaged maps would mix energy across slot boundaries.
        energy = slot_masks.to(x).clamp(0, 1)[:, :, None] * frequency.to(x)[:, None]
        energy = F.interpolate(energy.flatten(1, 2), size=size, mode='area')
        energy = energy.reshape(len(x), 20, 3, *size)
        energy = energy * (masks / occupancy.clamp_min(1e-8))[:, :, None]
        band_mean = energy.sum((-2, -1)) / den[..., None]
        cross = torch.einsum('bsfhw,bdhw->bsfd', energy, z) / den[..., None, None]
        covariance = cross - band_mean[..., None] * mean[:, :, None]
        return masks, mass, mean, covariance.flatten(-2)

    def forward(self, x, slot_masks, valid, frequency, reference,
                pair_valid=None, return_details=False):
        b, k = len(x), len(reference['features'])
        if pair_valid is None:
            pair_valid = torch.ones((b, k), dtype=torch.bool, device=x.device)
        else:
            if pair_valid.shape != (b, k):
                raise ValueError('pair_valid must have shape [B,K].')
            pair_valid = pair_valid.to(device=x.device, dtype=torch.bool)
        if k == 0:
            if return_details:
                empty = tuple(x.new_zeros(b, 0, len(r), len(r)) for r in tokens.REGIONS)
                return x, dict(update=torch.zeros_like(x),
                               reference_weights=x.new_zeros(b, 0, 20),
                               transport=empty, similarity=empty)
            return x

        masks, qm, q, qc = self._describe(x, slot_masks, valid, frequency)
        _, sm, s, sc = self._describe(reference['features'], reference['slot_masks'],
                                    reference['valid'], reference['frequency'])
        qh, sh = self.metric.project(qc), self.metric.project(sc)
        qu, su = F.normalize(q, dim=-1), F.normalize(s, dim=-1)
        values, reference_weights, transports, similarities = [], [], [], []
        for region in tokens.REGIONS:
            indices = region.tolist()
            qi, sj = qu[:, None, indices, None, :], su[None, :, None, indices, :]
            hi, hj = qh[:, None, indices, None, :], sh[None, :, None, indices, :]
            w = self.metric.weights(hi, hj) if self.use_frequency else torch.ones_like(qi * sj)
            numerator = (w * qi * sj).sum(-1)
            denominator = ((w * qi.square()).sum(-1) *
                           (w * sj.square()).sum(-1)).clamp_min(1e-20).sqrt()
            sim = numerator / denominator
            qa, sa = qm[:, None, indices], sm[None, :, indices]
            allowed = ((qa > 0)[..., :, None] & (sa > 0)[..., None, :] &
                       pair_valid[..., None, None])
            t = sinkhorn(sim, qa, sa, eps=self.transport_eps) * allowed.to(sim)
            row_mass = t.sum(-1)
            score = (t * sim).sum(-1) / row_mass.clamp_min(1e-20)
            available = row_mass > 0
            # A finite floor keeps the all-missing case defined; multiplication
            # and renormalization make every excluded reference exactly zero.
            logits = score.masked_fill(~available, torch.finfo(score.dtype).min)
            weights = logits.softmax(dim=1) * available.to(score)
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-20)
            transported = torch.einsum('bkij,kjd->bkid', t, s[:, indices])
            # Divide by offered query mass, not matched mass, retaining the
            # dustbin's suppression when evidence is poorly matched.
            offered = qa / qa.sum(-1, keepdim=True).clamp_min(1e-20)
            transported = transported / offered[..., None].clamp_min(1e-20)
            values.append((weights[..., None] * transported).sum(1))
            reference_weights.append(weights)
            if return_details:
                transports.append(t)
                similarities.append(sim)

        values = torch.cat(values, dim=1)
        write_weights = masks / masks.sum(1, keepdim=True).clamp_min(1)
        mixed = torch.einsum('bshw,bsd->bdhw', write_weights, values)
        update = self.gain * self.output(mixed)
        out = x + update
        if return_details:
            return out, dict(update=update, reference_weights=torch.cat(reference_weights, dim=-1),
                             transport=tuple(transports), similarity=tuple(similarities))
        return out
