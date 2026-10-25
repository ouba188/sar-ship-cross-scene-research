"""Frozen rebuilt M1 attention and M2 regional pooling, without legacy imports.

SummaryReader accepts a single VV/VH pair and separately forwarded C4 maps.
M12 accepts single or batched summaries and returns NumPy scores/details.
Only trusted local exported checkpoints should be loaded.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .regions import cell_mass, region_quality, soft_regions


def centered_response(response, norm):
    mu = torch.einsum('bru,bcu->brc', norm, response)
    r = response[:, None]-mu[..., None]
    r = r-r.mean(2, keepdim=True)
    return mu, r


def attention(response, weights, tau, recenter=False):
    norm = weights/weights.sum(-1, keepdim=True).clamp_min(1e-300)
    mu, r = centered_response(response, norm)
    if recenter:
        r = r-(r*norm[:, :, None]).sum(-1, keepdim=True)
    logw = torch.where(norm > 0, norm.log(), -torch.inf)[:, :, None]
    empty = weights.sum(-1) == 0
    logw = torch.where(empty[:, :, None, None], torch.zeros_like(logw), logw)
    qp = ((logw+r/tau).softmax(-1)*r).sum(-1)
    qm = ((logw-r/tau).softmax(-1)*r).sum(-1)
    return torch.stack([mu, (qp-qm)/2, (qp+qm)/2], -1)


class SummaryReader:
    """Cache source-only readers/temperatures. No labels or cohort files needed."""

    def __init__(self, assets_root):
        root = Path(assets_root)/'upstream'
        self.tau = json.loads((root/'scales.json').read_text(encoding='utf-8'))['tau']
        self.readers = {}
        for layer, dim in (('layer1', 256), ('layer3', 1024)):
            with np.load(root/'models'/f'reader_{layer}.npz') as z:
                # Intercept is deliberately NOT added to spatial responses.
                self.readers[layer] = z['W'].reshape(8, 2, dim).transpose(1, 0, 2).copy()

    @torch.inference_mode()
    def summarize(self, prepared, maps):
        if len(prepared) != 2 or len(maps) != 2:
            raise ValueError('Expected VV/VH lists of length two.')
        out = dict(mean3=np.zeros((2, 8)), hb3=np.zeros((2, 8, 2)),
                   summary1=np.zeros((2, 3, 8, 3)), quality=np.zeros((2, 3, 4)),
                   geometry=np.zeros((2, 4)), valid=np.zeros(2, bool),
                   region_valid=np.zeros((2, 3), bool))
        for p, (item, features) in enumerate(zip(prepared, maps)):
            if item is None:
                if features is not None:
                    raise ValueError('Absent polarization must have absent feature maps.')
                continue
            if features is None:
                raise ValueError('Present polarization is missing feature maps.')
            fp, g = np.asarray(item['footprint']), item['geometry']
            if fp.shape != (160, 160) or not np.isfinite(fp).all() or fp.sum() <= 0:
                raise ValueError('Present polarization needs a nonempty 160x160 footprint.')
            masks = soft_regions(g)
            q = region_quality(masks, fp, g)
            flags = (q['area'] > 0) & (q['min_neff'] >= 4)
            out['valid'][p] = True
            out['region_valid'][p] = flags
            out['quality'][p] = np.stack([q['area'], q['coverage'], q['min_neff'], flags], -1)
            if g is not None:
                out['geometry'][p] = np.r_[g[2:4], g[5], 1.]
            for layer, dim, side in (('layer1', 256, 40), ('layer3', 1024, 10)):
                fmap = features[layer]
                if tuple(fmap.shape) != (4, dim, side, side):
                    raise ValueError(f'{layer} must be four separately forwarded C4 maps.')
                W = torch.as_tensor(self.readers[layer][p:p+1], device=fmap.device, dtype=torch.float64)
                tau = torch.tensor(self.tau[layer][p], device=fmap.device, dtype=torch.float64)
                for k in range(4):
                    z = fmap[k:k+1].flatten(2).double()
                    response = torch.einsum('bck,bku->bcu', W, z)
                    if layer == 'layer3':
                        # Exact 160->10 bins avoid CUDA rotated single-channel area interpolation.
                        w = np.rot90(fp, k).reshape(10, 16, 10, 16).mean((1, 3))
                        weight = torch.tensor(w.reshape(1, 1, -1), device=fmap.device, dtype=torch.float64)
                        packed = attention(response, weight, tau, recenter=True)[0, 0].cpu().numpy()
                        out['mean3'][p] += packed[:, 0]/4
                        out['hb3'][p] += packed[:, 1:]/4
                    else:
                        w = cell_mass(np.rot90(masks*fp, k, (-2, -1)))
                        weight = torch.tensor(w.reshape(1, 3, -1), device=fmap.device, dtype=torch.float64)
                        packed = attention(response, weight, tau)[0].cpu().numpy()
                        packed[~flags] = 0
                        out['summary1'][p] += packed/4
        return out


def summarize(prepared, maps, assets_root):
    """Convenience entry; retain SummaryReader across chips to avoid reloading assets."""
    return SummaryReader(assets_root).summarize(prepared, maps)


def center(x):
    return x-x.mean(1, keepdims=True)


def attention_raw(base, means, hb, valid):
    cols = [center(base)]
    for p in range(2):
        for value in (means[:, p], hb[:, p, :, 0], hb[:, p, :, 1]):
            cols.extend([value, center(value)])
    cols.extend([np.broadcast_to(valid[:, p, None], base.shape) for p in range(2)])
    return np.stack(cols, -1)


def regional_raw(base, summary, quality, geometry):
    cols = [center(base)]
    for p in range(2):
        for r in range(3):
            for stat in range(3):
                v = summary[:, p, r, :, stat]
                cols.extend([v, center(v)])
    fields = []
    for p in range(2):
        g, q = geometry[:, p], quality[:, p]
        fields.extend([np.log1p(g[:, 0]), np.log1p(g[:, 1]), g[:, 2], g[:, 3]])
        for r in range(3):
            fields.extend([np.log1p(q[:, r, 0]), q[:, r, 1], np.log1p(q[:, r, 2]), q[:, r, 3]])
    q = np.stack(fields, -1)
    return np.concatenate((np.stack(cols, -1), np.broadcast_to(q[:, None], (len(base), 8, 32))), -1)


def branch(dim):
    return nn.Sequential(nn.Linear(dim, 12), nn.GELU(), nn.Linear(12, 1))


class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = branch(7)
        self.local = branch(11)

    def forward(self, x):
        a = self.mean(x[..., [0, 1, 2, 7, 8, 13, 14]])
        b = self.local(x[..., [0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]])
        z = (a+b).squeeze(-1)
        return z-z.mean(1, keepdim=True)


class RegionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = branch(69)

    def forward(self, x):
        z = self.net(x).squeeze(-1)
        return z-z.mean(1, keepdim=True)


class M12(nn.Module):
    """CPU frozen heads with float64 score accumulation matching rebuilt pipeline.

    base8: (8,) or (N,8) NumPy array; summaries carry matching batch shape.
    forward returns (s12, details), with s0/delta_m1/delta_m2 and selected_steps.
    """

    def __init__(self, assets_root, seed=17):
        super().__init__()
        self.heads = nn.ModuleDict({'attention': AttentionHead(), 'full': RegionHead()})
        self.scalers, self.selected_steps = {}, {}
        for name, head in self.heads.items():
            ck = torch.load(Path(assets_root)/'upstream'/'models'/f'{name}_seed{seed}.pt',
                            map_location='cpu', weights_only=False)
            head.load_state_dict(ck['state'])
            self.scalers[name] = (np.asarray(ck['mu']), np.asarray(ck['scale']))
            self.selected_steps[name] = int(ck['epoch'])
        self.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, base8, summaries):
        base = np.asarray(base8, dtype=np.float64)
        single = base.ndim == 1
        base = base[None] if single else base
        if base.ndim != 2 or base.shape[1] != 8 or len(base) == 0:
            raise ValueError('base8 must have shape (8,) or nonempty (N,8).')
        s = {k: np.asarray(v)[None] if single else np.asarray(v) for k, v in summaries.items()}
        score, details = base.copy(), {}
        for name in ('attention', 'full'):
            raw = (attention_raw(score, s['mean3'], s['hb3'], s['valid']) if name == 'attention'
                   else regional_raw(score, s['summary1'], s['quality'], s['geometry']))
            mu, sd = self.scalers[name]
            x = ((raw-mu)/sd).astype(np.float32)
            head = self.heads[name]
            device = next(head.parameters()).device
            delta = np.concatenate([head(torch.tensor(x[i:i+256], device=device)).cpu().numpy()
                                    for i in range(0, len(x), 256)]).astype(float)
            score = score+delta
            details['delta_m1' if name == 'attention' else 'delta_m2'] = delta[0] if single else delta
            if name == 'attention':
                details['s0'] = score[0].copy() if single else score.copy()
        details['selected_steps'] = self.selected_steps.copy()
        return (score[0] if single else score), details
