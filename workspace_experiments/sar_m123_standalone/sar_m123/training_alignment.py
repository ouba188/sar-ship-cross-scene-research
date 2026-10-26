"""Accepted metric alignment against the actual frozen decision path.

Provenance: m3_aligned_training_20261026/run.py. Numerical code unchanged.
"""
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from . import train_ops as m3
from . import train_ops as tr


class FrozenDecision:
    """Differentiate actual polar fusion, including the base-dependent gate fields."""
    def __init__(self, folder, cached, base, device):
        self.device = device
        self.b = torch.tensor(base, device=device)
        self.av = torch.tensor(cached['available'], device=device)
        self.tail = torch.tensor(cached['gate_raw'][..., 11:], device=device)
        sc = tr.load_checkpoint(folder/'M3_readout_scaler.pt')
        self.sc = {k: torch.as_tensor(v, device=device) for k, v in sc.items()}
        self.polar = []
        for p, name in enumerate(('vv', 'vh_centered')):
            ck = tr.load_checkpoint(folder/f'{name}.pt')
            h = m3.PolarHead(p, centered=bool(p), null=ck['state'].get('null'))
            self.polar.append(tr.frozen_model(h, ck, device))
        ck = tr.load_checkpoint(folder/'frequency.pt')
        self.gs = {k: torch.as_tensor(v, device=device) for k, v in ck['scaler'].items()}
        self.gate = tr.frozen_model(m3.Gate(185, ck['alpha']), ck, device)

    def __call__(self, local, ids):
        b, av = self.b[ids], self.av[ids]
        z = b-b.mean(1, keepdim=True)
        rival = torch.stack([torch.cat((z[:, :c], z[:, c+1:]), 1).max(1).values for c in range(4)], 1)
        raw = torch.cat((local.reshape(len(ids), 4, 12), torch.stack((z, rival, z-rival), -1)), -1)
        x = torch.where(self.sc['active'], (raw-self.sc['mu'])/self.sc['scale'], 0).float()
        dv, dh = (h(x) for h in self.polar)
        vals = [z, dv, dh]
        for s in (b, b+dv, b+dh):
            other = torch.stack([torch.cat((s[:, :c], s[:, c+1:]), 1).max(1).values for c in range(4)], 1)
            vals.append(s-other)
        vals.extend(torch.softmax(s, 1) for s in (b, b+dv, b+dh))
        vals.extend(av[:, p, None].expand_as(b) for p in range(2))
        gr = torch.cat((torch.stack(vals, -1), self.tail[ids]), -1)
        gx = torch.where(self.gs['active'], (gr-self.gs['mu'])/self.gs['scale'], 0).float()
        w = self.gate(gx, av)
        return b+w[:, :1]*dv+w[:, 1:]*dh


def fit_metric(matcher, metric, decision, y, fit, val, out):
    def predict(ids):
        with torch.no_grad():
            return np.concatenate([decision(matcher.local(ids[k:k+48], metric), ids[k:k+48]).cpu().numpy()
                                   for k in range(0, len(ids), 48)])
    initial = predict(val)
    best, selected, chosen = m3.exact_j(y[val], initial), 0, m3.state(metric)
    curve = [dict(step=0, **m3.metrics(y[val], initial))]
    wt = m3.balanced_weights(y, fit, matcher.device)
    yt = torch.tensor(y[fit], device=matcher.device)
    opt = torch.optim.AdamW(metric.parameters(), lr=.003, weight_decay=.01)
    tick = time.perf_counter()
    gradient_max = 0.
    for step in range(1, 101):
        opt.zero_grad(set_to_none=True)
        total = 0.
        for k in range(0, len(fit), 24):
            ix = fit[k:k+24]
            scores = decision(matcher.local(ix, metric), ix)
            loss = (F.cross_entropy(scores, yt[k:k+24], reduction='none')*wt[k:k+24]).sum()
            assert torch.isfinite(loss)
            loss.backward()
            total += float(loss.detach())
        grad = max(float(p.grad.abs().max()) for p in metric.parameters() if p.grad is not None)
        gradient_max = max(gradient_max, grad)
        # An exactly zero frozen correction has no gradient path to matching.
        if step == 1 and grad == 0:
            curve.append(dict(step=1, loss=total, gradient_max=grad, **m3.metrics(y[val], initial)))
            break
        opt.step()
        if step in (5, 10, 15, 20, 30, 50, 75, 100):
            scores = predict(val)
            j = m3.exact_j(y[val], scores)
            curve.append(dict(step=step, loss=total, gradient_max=grad, **m3.metrics(y[val], scores)))
            if j > best:
                best, selected, chosen = j, step, m3.state(metric)
            print('METRIC', out.parent.name, out.name, step, 'J', float(j), 'seconds', round(time.perf_counter()-tick), flush=True)
    metric.load_state_dict(chosen)
    metric.eval().requires_grad_(False)
    torch.save(dict(state=chosen, selected_step=selected), out/'aligned_metric.pt')
    pd.DataFrame(curve).to_csv(out/'metric_curve.csv', index=False)
    return selected, gradient_max, time.perf_counter()-tick
