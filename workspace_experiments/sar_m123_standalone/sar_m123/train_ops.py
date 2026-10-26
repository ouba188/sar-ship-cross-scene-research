"""Frozen public-four-class training operators, migrated without math changes.

Provenance: oss1_m123_replication_20261026/train_m3.py.
No historical experiment directory is needed to import or execute this module.
"""
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import softmax, logsumexp
from . import tokens
from .m3 import PairMetric, PolarHead, sinkhorn, regional_descriptors, pair_stats

C = 4


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def frozen_model(model, checkpoint, device):
    model.load_state_dict(checkpoint['state'])
    return model.to(device).eval().requires_grad_(False)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def metrics(y, score):
    correct = score.argmax(1) == y
    lp = score-logsumexp(score, axis=1, keepdims=True)
    oa = float(correct.mean())
    ba = float(np.mean([correct[y == c].mean() for c in np.unique(y)]))
    return dict(n=len(y), OA=oa, BA=ba, J=(oa+ba)/2,
                NLL=float(-lp[np.arange(len(y)), y].mean()),
                Brier=float(((np.exp(lp)-np.eye(C)[y])**2).sum(1).mean()))


def exact_j(y, score):
    correct = score.argmax(1) == y
    return (Fraction(int(correct.sum()), len(y))+
            sum(Fraction(int(correct[y == c].sum()), int((y == c).sum()))
                for c in np.unique(y))/len(np.unique(y)))/2


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def scale_fit(raw, ids, axes=(0, 1)):
    mu, sd = raw[ids].mean(axes), raw[ids].std(axes)
    active = sd > 1e-6
    return dict(mu=mu, scale=np.where(active, sd, 1), active=active)


def transform(raw, sc):
    return np.where(sc['active'], (raw-sc['mu'])/sc['scale'], 0).astype(np.float32)


def local_raw(local, base):
    z = base-base.mean(1, keepdims=True)
    competitor = np.stack([np.delete(z, c, axis=1).max(1) for c in range(C)], 1)
    return np.concatenate((local.reshape(len(base), C, 12),
                           np.stack((z, competitor, z-competitor), -1)), -1)


def gate_raw(query, bank, refs, base, dv, dh):
    qvalid, rvalid = query['valid'], bank['valid'][refs]
    valid = qvalid[:, None, None] & rvalid
    has = valid.any(2)
    q, s = query['x'][:, None, None], bank['x'][refs]
    mean, std = pair_stats((q*s).sum(-1), valid)
    agreement = (s[:, :, 0]*s[:, :, 1]).sum(-1)*valid.all(2)
    spatial = np.stack((mean, std, agreement, valid.mean(2),
                        np.broadcast_to(qvalid[:, None], has.shape)), -1).reshape(len(base), C, 30)
    bv = np.broadcast_to(valid[..., None], valid.shape+(3,))
    cm, cs = pair_stats((query['cov'][:, None, None]*bank['cov'][refs]).sum(-1), bv)
    em, es = pair_stats(bank['energy'][refs], bv)
    ed, _ = pair_stats(abs(query['energy'][:, None, None]-bank['energy'][refs]), bv)
    cover, _ = pair_stats(bank['coverage'][refs], bv)
    qe, qc = (np.broadcast_to(query[k][:, None], em.shape) for k in ('energy', 'coverage'))
    freq = np.stack((qe, em, ed, es, cm, cs, qc, cover), -1)
    freq *= has[..., None, None]
    avail = qvalid.any(-1)
    values = [base-base.mean(1, keepdims=True), dv, dh]
    scores = (base, base+dv, base+dh)
    for score in scores:
        other = np.stack([np.delete(score, c, axis=1).max(1) for c in range(C)], 1)
        values.append(score-other)
    values.extend(softmax(s, axis=1) for s in scores)
    values.extend(np.broadcast_to(avail[:, p, None], base.shape) for p in range(2))
    raw = np.concatenate((np.stack(values, -1), spatial, freq.reshape(len(base), C, 144)), -1)
    assert raw.shape == (len(base), C, 185)
    return raw.astype(np.float32), avail.astype(np.float32)


class JointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(15, 12), nn.GELU(), nn.Linear(12, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        d = self.net(x).squeeze(-1)
        return d-d.mean(-1, keepdim=True)


class Gate(nn.Module):
    def __init__(self, dim, alpha):
        super().__init__()
        self.alpha = float(alpha)
        self.net = nn.Sequential(nn.Linear(dim, 12), nn.GELU(), nn.Linear(12, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, available):
        a = (self.alpha+.25*self.net(x).mean((1, 2)).tanh()).clamp(0, 1)
        a = torch.where(available.bool().all(1), a, torch.full_like(a, self.alpha))
        return torch.stack((a, 1-a), 1)*available


def balanced_weights(y, ids, device):
    count = np.bincount(y[ids], minlength=C)
    assert (count > 0).all(), count
    return torch.tensor(.5/len(ids)+.5/(C*count[y[ids]]), dtype=torch.float32, device=device)


def fit_head(name, model, x, base, y, tr, va, out, steps=200):
    device = next(model.parameters()).device
    xt, bt = torch.as_tensor(x, device=device), torch.as_tensor(base, device=device)
    yt = torch.as_tensor(y[tr], device=device)
    wt = balanced_weights(y, tr, device)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    best, selected, chosen = exact_j(y[va], base[va]), 0, state(model)
    curves = [dict(step=0, **metrics(y[va], base[va]))]
    for step in range(1, steps+1):
        opt.zero_grad(set_to_none=True)
        loss = (F.cross_entropy(bt[tr]+model(xt[tr]), yt, reduction='none')*wt).sum()
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        if step % 5 == 0:
            with torch.no_grad():
                score = (bt[va]+model(xt[va])).cpu().numpy()
            j = exact_j(y[va], score)
            curves.append(dict(step=step, loss=float(loss.detach()), **metrics(y[va], score)))
            if j > best:
                best, selected, chosen = j, step, state(model)
    model.load_state_dict(chosen)
    model.eval().requires_grad_(False)
    pd.DataFrame(curves).to_csv(out/f'curve_{name}.csv', index=False)
    torch.save(dict(state=chosen, selected_step=selected), out/f'{name}.pt')
    with torch.inference_mode():
        delta = model(xt).cpu().numpy()
    print(name, 'step', selected, 'source_J', float(best), flush=True)
    return delta, selected


class Matcher:
    def __init__(self, features, refs, scaler, device):
        self.device = device
        self.refs = torch.as_tensor(refs, device=device)
        self.x = torch.as_tensor(features['x'], device=device)
        self.mass = torch.as_tensor(features['mass'], device=device)
        d = transform(features['joint'], scaler)
        d[features['mass'] <= 0] = 0
        self.d = torch.as_tensor(d, device=device)

    def local(self, idx, model):
        idx = torch.as_tensor(idx, device=self.device)
        ref = self.refs[idx]
        q, s = self.x[idx], self.x[ref]
        a, b = self.mass[idx], self.mass[ref]
        h = model.project(self.d)
        hq, hs = h[idx], h[ref]
        groups = []
        for r in tokens.REGIONS:
            qi, sj = q[:, :, r][:, None, None, :, :, None, :], s[..., r, :][..., None, :, :]
            hi, hj = hq[:, :, r][:, None, None, :, :, None, :], hs[..., r, :][..., None, :, :]
            w = model.weights(hi, hj)
            sim = (w*qi*sj).sum(-1)/((w*qi.square()).sum(-1)*(w*sj.square()).sum(-1)).clamp_min(1e-20).sqrt()
            qa, sa = a[:, None, None, :, r], b[..., r]
            t = sinkhorn(sim, qa, sa)
            t = t*((qa > 0)[..., :, None] & (sa > 0)[..., None, :])
            groups.append(torch.stack(((t*sim).sum((-1, -2)), t.sum((-1, -2))), -1))
        return torch.stack(groups, -2).mean(2)

    def predict(self, idx, model, batch=48):
        with torch.inference_mode():
            return np.concatenate([self.local(idx[i:i+batch], model).cpu().numpy()
                                   for i in range(0, len(idx), batch)])


def choose_alpha(base, dv, dh, avail, y, ids):
    choices = []
    for k in range(11):
        a = k/10
        s = base+a*avail[:, :1]*dv+(1-a)*avail[:, 1:]*dh
        choices.append((a, exact_j(y[ids], s[ids])))
    return min(choices, key=lambda v: (-v[1], abs(v[0]-.5), v[0]))[0]


def fit_gate(arm, seed, raw, base, dv, dh, avail, y, tr, va, alpha, out):
    dim = 41 if arm == 'spatial' else 185
    scaler = scale_fit(raw[..., :dim], tr)
    x = torch.tensor(transform(raw[..., :dim], scaler))
    b, v, h, a = (torch.tensor(z) for z in (base, dv, dh, avail))
    torch.manual_seed(seed)
    model = Gate(dim, alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    ytr, wt = torch.tensor(y[tr]), balanced_weights(y, tr, 'cpu')
    def prediction(ids):
        w = model(x[ids], a[ids])
        return b[ids]+w[:, :1]*v[ids]+w[:, 1:]*h[ids], w
    with torch.no_grad():
        initial = prediction(va)[0].numpy()
    best, selected, chosen = exact_j(y[va], initial), 0, state(model)
    curves = [dict(step=0, **metrics(y[va], initial))]
    for step in range(1, 101):
        opt.zero_grad(set_to_none=True)
        loss = (F.cross_entropy(prediction(tr)[0], ytr, reduction='none')*wt).sum()
        loss.backward()
        opt.step()
        if step % 5 == 0:
            with torch.no_grad():
                s = prediction(va)[0].numpy()
            j = exact_j(y[va], s)
            curves.append(dict(step=step, loss=float(loss.detach()), **metrics(y[va], s)))
            if j > best:
                best, selected, chosen = j, step, state(model)
    model.load_state_dict(chosen)
    with torch.inference_mode():
        s, w = prediction(np.arange(len(base)))
    torch.save(dict(state=chosen, scaler=scaler, alpha=alpha, selected_step=selected), out/f'{arm}.pt')
    pd.DataFrame(curves).to_csv(out/f'curve_{arm}.csv', index=False)
    return s.numpy(), w.numpy(), selected
