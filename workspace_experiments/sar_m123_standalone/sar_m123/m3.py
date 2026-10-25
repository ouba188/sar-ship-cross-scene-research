"""Selected M3: both metric/transport, old VV, centered VH, anchored gate.

Runtime loads only assets_root/m3. Source references are frozen 8 per class;
each new query retrieves 2 per candidate class without its label or identity.
"""
from pathlib import Path

import numpy as np
from scipy.special import softmax
import torch
from torch import nn

from . import tokens


def standardize(raw, scaler):
    return np.where(scaler['active'], (raw-scaler['mu'])/scaler['scale'], 0).astype(np.float32)


def sinkhorn(sim, qa, sa, steps=60, eps=.1):
    qa = qa/qa.sum(-1, keepdim=True).clamp_min(1e-20)
    sa = sa/sa.sum(-1, keepdim=True).clamp_min(1e-20)
    a = torch.cat([qa, 2-qa.sum(-1, keepdim=True)], -1)
    b = torch.cat([sa, 2-sa.sum(-1, keepdim=True)], -1)
    loga, logb = a.clamp_min(1e-20).log(), b.clamp_min(1e-20).log()
    logk = nn.functional.pad(sim/eps, (0, 1, 0, 1))
    u, v = torch.zeros_like(a), torch.zeros_like(b)
    for _ in range(steps):
        u = loga-torch.logsumexp(logk+v[..., None, :], -1)
        v = logb-torch.logsumexp(logk+u[..., :, None], -2)
    return (logk+u[..., :, None]+v[..., None, :]).exp()[..., :-1, :-1]


class PairMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(96, 4, bias=False)
        self.output = nn.Linear(8, 32, bias=False)

    def weights(self, q, s):
        phi = torch.cat([(q+s)*.5, (q-s).abs()*.5], -1)
        return 1+.25*self.output(nn.functional.gelu(phi)).tanh()


class PolarHead(nn.Module):
    def __init__(self, pol, centered=False, null=None):
        super().__init__()
        self.centered = centered
        self.register_buffer('indices', torch.tensor(list(range(pol*6, pol*6+6))+[12, 13, 14]))
        if centered:
            self.register_buffer('null', torch.as_tensor(null, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(9, 12), nn.GELU(), nn.Linear(12, 1))

    def forward(self, x):
        value = self.net(x[..., self.indices]).squeeze(-1)
        if self.centered:
            empty = x.clone()
            empty[..., :12] = self.null[:12]
            value = value-self.net(empty[..., self.indices]).squeeze(-1)
        return value-value.mean(-1, keepdim=True)


class AnchoredGate(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = float(alpha)
        self.net = nn.Sequential(nn.Linear(185, 12), nn.GELU(), nn.Linear(12, 1))

    def forward(self, x, available):
        r = self.net(x).mean((1, 2))
        alpha = (self.alpha+.25*r.tanh()).clamp(0, 1)
        alpha = torch.where(available.bool().all(1), alpha, torch.full_like(alpha, self.alpha))
        return torch.stack((alpha, 1-alpha), 1)*available, alpha


def local_features(local, base):
    z = base-base.mean(-1, keepdims=True)
    other = np.stack([np.delete(z, c, axis=1).max(1) for c in range(8)], 1)
    return np.concatenate((local.reshape(len(local), 8, 12),
                           np.stack((z, other, z-other), -1)), -1)


def pair_stats(v, valid):
    den = np.maximum(valid.sum(2), 1)
    mean = (v*valid).sum(2)/den
    var = (np.square(v-mean[:, :, None])*valid).sum(2)/den
    return mean, np.sqrt(var)


def regional_descriptors(features, covariance_scaler):
    mass = features['mass']
    valid = np.stack([mass[:, :, r].sum(2) for r in tokens.REGIONS], -1) > 0
    x = tokens.unit(tokens.region_pool(features['mean'], mass))
    joint = standardize(features['joint'], covariance_scaler)
    joint[mass <= 0] = 0
    cov = tokens.unit(tokens.region_pool(joint, mass).reshape(len(mass), 2, 3, 3, 32))
    energy = features['e_image'][..., :12].reshape(len(mass), 2, 20, 3, 4).mean(-1)
    return dict(valid=valid, x=x, cov=cov, energy=tokens.region_pool(energy, mass),
                coverage=tokens.region_pool(features['coverage'], mass))


def gate_features(query, bank, refs, base, dv, dh):
    qvalid, rvalid = query['valid'], bank['valid'][refs]
    valid = qvalid[:, None, None] & rvalid
    has = valid.any(2)
    q, s = query['x'][:, None, None], bank['x'][refs]
    mean, std = pair_stats((q*s).sum(-1), valid)
    agreement = (s[:, :, 0]*s[:, :, 1]).sum(-1)*valid.all(2)
    spatial = np.stack([mean, std, agreement, valid.mean(2),
                       np.broadcast_to(qvalid[:, None], has.shape)], -1).reshape(len(base), 8, -1)
    bandvalid = np.broadcast_to(valid[..., None], valid.shape+(3,))
    cm, cs = pair_stats((query['cov'][:, None, None]*bank['cov'][refs]).sum(-1), bandvalid)
    em, es = pair_stats(bank['energy'][refs], bandvalid)
    ed, _ = pair_stats(abs(query['energy'][:, None, None]-bank['energy'][refs]), bandvalid)
    cover, _ = pair_stats(bank['coverage'][refs], bandvalid)
    qe = np.broadcast_to(query['energy'][:, None], em.shape)
    qc = np.broadcast_to(query['coverage'][:, None], em.shape)
    frequency = np.stack([qe, em, ed, es, cm, cs, qc, cover], -1)
    frequency *= has[..., None, None]
    frequency = frequency.reshape(len(base), 8, -1)
    available = qvalid.any(-1)
    vv, vh = base+dv, base+dh
    values = [base-base.mean(1, keepdims=True), vv-base, vh-base]
    for score in (base, vv, vh):
        other = np.stack([np.delete(score, c, axis=1).max(1) for c in range(8)], 1)
        values.append(score-other)
    values.extend(softmax(score, axis=1) for score in (base, vv, vh))
    values.extend(np.broadcast_to(available[:, p, None], base.shape) for p in range(2))
    score = np.stack(values, -1).astype(np.float32)
    return np.concatenate((score, spatial, frequency), -1).astype(np.float32), available


class M3:
    """Inference wrapper. extract -> forward supports new chips, no cached queries.

    forward accepts an unbatched extracted dictionary (mass [2,20]) or a stacked
    dictionary (mass [N,2,20]); s12 must be [8] or [N,8], respectively. Returns
    numpy float32 logits and details. References indices are in the 64-row bank.
    """
    def __init__(self, assets_root, device='cpu'):
        self.root = Path(assets_root)/'m3'
        self.device = torch.device(device)
        with np.load(self.root/'projection.npz') as z:
            self.projection = {k: z[k].copy() for k in z.files}
        with np.load(self.root/'bank.npz') as z:
            self.bank = {k: z[k].copy() for k in z.files}
        with np.load(self.root/'covariance_scaler.npz') as z:
            self.covariance_scaler = {k: z[k].copy() for k in z.files}
        if self.bank['mass'].shape != (64, 2, 20):
            raise ValueError('M3 requires the frozen 8-class x 8-reference bank')
        self.bank_body = tokens.body_descriptor(self.bank['x'], self.bank['mass']).reshape(8, 8, 64)
        self.bank_regions = regional_descriptors(self.bank, self.covariance_scaler)
        self.models = {}

    def extract(self, prepared, maps):
        return tokens.extract(prepared, maps, self.projection)

    def _load(self, seed):
        if seed in self.models:
            return self.models[seed]
        ck = torch.load(self.root/f'seed_{seed}.pt', map_location='cpu', weights_only=False)
        metric = PairMetric()
        metric.load_state_dict(ck['metric_state'])
        vv = PolarHead(0)
        vv.load_state_dict(ck['vv_state'])
        vh = PolarHead(1, True, ck['vh_state']['null'])
        vh.load_state_dict(ck['vh_state'])
        gate = AnchoredGate(ck['alpha_anchor'])
        gate.load_state_dict(ck['gate_state'])
        for model in (metric, vv, vh, gate):
            model.to(self.device).eval().requires_grad_(False)
        result = dict(ck=ck, metric=metric, vv=vv, vh=vh, gate=gate)
        self.models[seed] = result
        return result

    def retrieve(self, features):
        body = tokens.body_descriptor(features['x'], features['mass'])
        cosine = np.einsum('nd,ckd->nck', body, self.bank_body)
        selected = np.argsort(-cosine, axis=-1)[..., :2]
        return np.arange(8)[None, :, None]*8+selected

    def _tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    def match(self, features, refs, model):
        ck, metric = model['ck'], model['metric']
        q, s = self._tensor(features['x']), self._tensor(self.bank['x'][refs])
        a, b = self._tensor(features['mass']), self._tensor(self.bank['mass'][refs])
        qd = standardize(features['joint'], ck['metric_scaler'])
        sd = standardize(self.bank['joint'], ck['metric_scaler'])
        qd[features['mass'] <= 0] = 0
        sd[self.bank['mass'] <= 0] = 0
        hq, hs = metric.project(self._tensor(qd)), metric.project(self._tensor(sd))[refs]
        groups = []
        for region in tokens.REGIONS:
            qi, sj = q[:, :, region][:, None, None, :, :, None, :], s[..., region, :][..., None, :, :]
            hi, hj = hq[:, :, region][:, None, None, :, :, None, :], hs[..., region, :][..., None, :, :]
            w = metric.weights(hi, hj)
            numerator = (w*qi*sj).sum(-1)
            denominator = ((w*qi.square()).sum(-1)*(w*sj.square()).sum(-1)).clamp_min(1e-20).sqrt()
            similarity = numerator/denominator
            qa, sa = a[:, None, None, :, region], b[..., region]
            t = sinkhorn(similarity, qa, sa)
            t = t*((qa > 0)[..., :, None] & (sa > 0)[..., None, :])
            groups.append(torch.stack([(t*similarity).sum((-1, -2)), t.sum((-1, -2))], -1))
        return torch.stack(groups, -2).mean(2).cpu().numpy()

    @torch.inference_mode()
    def forward(self, features, s12, seed=17):
        single = features['mass'].ndim == 2
        if single:
            features = {k: v[None] for k, v in features.items()}
        base = np.asarray(s12, dtype=np.float32).reshape(-1, 8)
        if len(base) != len(features['mass']):
            raise ValueError('s12 and feature batch sizes differ')
        model = self._load(seed)
        refs = self.retrieve(features)
        local = self.match(features, refs, model)
        x = self._tensor(standardize(local_features(local, base), model['ck']['readout_scaler']))
        dv = model['vv'](x).cpu().numpy()
        dh = model['vh'](x).cpu().numpy()
        query_regions = regional_descriptors(features, self.covariance_scaler)
        raw, available = gate_features(query_regions, self.bank_regions, refs, base, dv, dh)
        gx = self._tensor(standardize(raw, model['ck']['gate_scaler']))
        g, alpha = model['gate'](gx, self._tensor(available))
        g, alpha = g.cpu().numpy(), alpha.cpu().numpy()
        score = base+g[:, :1]*dv+g[:, 1:]*dh
        details = dict(refs=refs, reference_ids=self.bank['source_index'][refs], local=local,
                       delta_vv=dv, delta_vh=dh, gate_raw=raw, available=available,
                       polar_weights=g, alpha=alpha)
        if single:
            return score[0], {k: v[0] for k, v in details.items()}
        return score, details

    __call__ = forward
