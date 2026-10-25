"""Source-only four-class refit of the frozen M3 comparison protocol.

No 24-port supervised assets are read. All four arms share a source-trained
frontend, source reference bank, local comparison and target observations.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import softmax, logsumexp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path('E:/临时会话/sar_m123_standalone')))
from sar_m123 import tokens
from sar_m123.m3 import PairMetric, PolarHead, sinkhorn, regional_descriptors, pair_stats

C = 4
SEEDS = (17, 29, 43)


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


def fit_metric(matcher, model, head, scaler, base, y, tr, va, out):
    device = matcher.device
    mu, sd, active = (torch.as_tensor(scaler[k], device=device) for k in ('mu', 'scale', 'active'))
    bt = torch.as_tensor(base, device=device)
    yt = torch.as_tensor(y[tr], device=device)
    wt = balanced_weights(y, tr, device)
    context = torch.as_tensor(local_raw(np.zeros((len(base), C, 2, 3, 2)), base)[..., 12:], device=device)
    def delta(local, ids):
        raw = torch.cat((local.reshape(len(ids), C, 12), context[ids]), -1)
        x = torch.where(active, (raw-mu)/sd, 0).float()
        return head(x)
    def predict(ids):
        parts = []
        with torch.no_grad():
            for i in range(0, len(ids), 48):
                ix = ids[i:i+48]
                parts.append((bt[ix]+delta(matcher.local(ix, model), ix)).cpu().numpy())
        return np.concatenate(parts)
    initial = predict(va)
    best, selected, chosen = exact_j(y[va], initial), 0, state(model)
    curves = [dict(step=0, **metrics(y[va], initial))]
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    tick = time.perf_counter()
    for step in range(1, 101):
        opt.zero_grad(set_to_none=True)
        total = 0.
        # Accumulate the exact full-source objective without storing all OT graphs.
        for i in range(0, len(tr), 24):
            ix = tr[i:i+24]
            score = bt[ix]+delta(matcher.local(ix, model), ix)
            loss = (F.cross_entropy(score, yt[i:i+24], reduction='none')*wt[i:i+24]).sum()
            assert torch.isfinite(loss)
            loss.backward()
            total += float(loss.detach())
        opt.step()
        if step in (5, 10, 15, 20, 30, 50, 75, 100):
            score = predict(va)
            j = exact_j(y[va], score)
            curves.append(dict(step=step, loss=total, **metrics(y[va], score)))
            if j > best:
                best, selected, chosen = j, step, state(model)
            print('both_metric', step, 'source_J', float(j), 'seconds', round(time.perf_counter()-tick), flush=True)
    model.load_state_dict(chosen)
    model.eval().requires_grad_(False)
    torch.save(dict(state=chosen, scaler=scaler, selected_step=selected), out/'both_metric.pt')
    pd.DataFrame(curves).to_csv(out/'curve_both_metric.csv', index=False)
    return selected


def prepare_local(raw, table, fit, out):
    mass = raw['mass'].astype(np.float32)
    source = raw['layer1'][fit][mass[fit] > 0]
    pca = PCA(n_components=32, svd_solver='randomized', random_state=17).fit(source)
    mean = ((raw['layer1']-pca.mean_)@pca.components_.T).astype(np.float32)
    x = tokens.unit(mean).astype(np.float32)
    x[mass <= 0] = 0
    joint = (raw['joint_raw']@pca.components_.T).reshape(len(table), 2, 20, 96).astype(np.float32)
    features = dict(x=x, mean=mean, mass=mass, joint=joint,
                    e_image=raw['e_image'], coverage=raw['coverage'])
    np.savez_compressed(out/'projection.npz', mean=pca.mean_, components=pca.components_, fit=fit)
    np.savez_compressed(out/'local.npz', **features)
    body = tokens.body_descriptor(x, mass)
    eligible = fit[mass[fit, :, :8].sum((1, 2)) > 0]
    unique = table.iloc[eligible].drop_duplicates('mmsi').index.to_numpy()
    bank = []
    for c in range(C):
        ids = unique[table.iloc[unique].class_id.to_numpy() == c]
        assert len(ids) >= 8, (c, len(ids))
        km = KMeans(n_clusters=8, random_state=17, n_init=10).fit(body[ids])
        selected = []
        for ctr in km.cluster_centers_:
            order = np.argsort(((body[ids]-ctr)**2).sum(1))
            selected.append(next(int(ids[j]) for j in order if int(ids[j]) not in selected))
        bank.append(selected)
    bank = np.array(bank)
    table.iloc[bank.ravel()][['obs_id', 'mmsi', 'class_id', 'role']].to_csv(out/'reference_bank.csv', index=False)
    score = np.einsum('nd,ckd->nck', body, body[bank])
    same_ship = table.mmsi.to_numpy()[:, None, None] == table.mmsi.to_numpy()[bank][None]
    score[same_ship] = -np.inf
    pick = np.argsort(-score, axis=-1)[..., :2]
    refs = np.take_along_axis(np.broadcast_to(bank, (len(table), C, 8)), pick, axis=2)
    assert not (table.mmsi.to_numpy()[:, None, None] == table.mmsi.to_numpy()[refs]).any()
    sc = scale_fit(joint.reshape(len(table), -1, 96), fit)
    np.savez_compressed(out/'references.npz', bank=bank, refs=refs, **sc)
    return features, refs, sc


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--frontend', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    table = pd.read_csv(args.table, dtype={'mmsi': str})
    y = table.class_id.to_numpy(int)
    fit = np.flatnonzero(table.role.eq('fit_train'))
    tr = np.flatnonzero(table.role.eq('utility_train'))
    va = np.flatnonzero(table.role.eq('selection'))
    te = np.flatnonzero(table.role.eq('evaluation'))
    assert all(len(z) for z in (fit, tr, va, te)), 'Explicit five-role protocol required'
    for a, b in ((fit, tr), (fit, va), (fit, te), (tr, va), (tr, te), (va, te)):
        for key in ('mmsi', 'pass_group'):
            assert not set(table.iloc[a][key]) & set(table.iloc[b][key])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out/'table.csv', index=False)
    with np.load(args.raw) as z:
        raw = {k: z[k].copy() for k in z.files}
    np.testing.assert_array_equal(raw['obs_id'].astype(str), table.obs_id.astype(str).to_numpy())
    features, refs, cov_sc = prepare_local(raw, table, fit, out)
    metric_sc = scale_fit(features['joint'].reshape(len(table), -1, 96), tr)
    matcher = Matcher(features, refs, metric_sc, device)
    regions = regional_descriptors(features, cov_sc)
    available = regions['valid'].any(-1).astype(np.float32)
    results, receipts = [], []
    tick = time.perf_counter()
    for seed in args.seeds:
        dest = out/f'seed{seed}'
        dest.mkdir(exist_ok=True)
        with np.load(args.frontend/f'scores_seed{seed}.npz') as z:
            np.testing.assert_array_equal(z['obs_id'].astype(str), table.obs_id.astype(str).to_numpy())
            base = z['s12'].astype(np.float32)
        assert base.shape == (len(table), C)
        torch.manual_seed(seed)
        model = PairMetric().to(device)
        nn.init.zeros_(model.output.weight)
        plain = matcher.predict(np.arange(len(table)), model)
        head_sc = scale_fit(local_raw(plain, base), tr)
        x = transform(local_raw(plain, base), head_sc)
        torch.manual_seed(seed)
        head = JointHead().to(device)
        _, step_h = fit_head('joint_head', head, x, base, y, tr, va, dest)
        step_m = fit_metric(matcher, model, head, head_sc, base, y, tr, va, dest)
        local = matcher.predict(np.arange(len(table)), model)
        xx = transform(local_raw(local, base), head_sc)
        with torch.inference_mode():
            old_both = base+head(torch.tensor(xx, device=device)).cpu().numpy()
        polar_sc = scale_fit(local_raw(local, base), tr)
        xp = transform(local_raw(local, base), polar_sc)
        null = transform(np.zeros((1, 1, 15)), polar_sc).reshape(15)
        delta, steps = [], []
        for pol in range(2):
            torch.manual_seed(seed)
            polar = PolarHead(pol, centered=bool(pol), null=null).to(device)
            nn.init.zeros_(polar.net[-1].weight)
            nn.init.zeros_(polar.net[-1].bias)
            d, step = fit_head('vv' if pol == 0 else 'vh_centered', polar, xp, base, y, tr, va, dest)
            delta.append(d)
            steps.append(step)
        dv, dh = delta
        alpha = choose_alpha(base, dv, dh, available, y, va)
        weights = np.array([alpha, 1-alpha], np.float32)*available
        fixed = base+weights[:, :1]*dv+weights[:, 1:]*dh
        gr, av = gate_raw(regions, regions, refs, base, dv, dh)
        scores = dict(old_both=old_both, B_fixed=fixed)
        ws = dict(B_fixed=weights)
        gate_steps = {}
        for arm in ('spatial', 'frequency'):
            scores[arm], ws[arm], gate_steps[arm] = fit_gate(arm, seed, gr, base, dv, dh, av, y, tr, va, alpha, dest)
        np.savez_compressed(dest/'predictions.npz', **scores, base=base, delta_vv=dv, delta_vh=dh,
                            local=local, refs=refs, gate_raw=gr, available=available,
                            **{f'weight_{k}': v for k, v in ws.items()})
        torch.save(dict(head_scaler=head_sc, polar_scaler=polar_sc, metric_scaler=metric_sc), dest/'scalers.pt')
        receipt = dict(seed=seed, joint_step=step_h, metric_step=step_m,
                       vv_step=steps[0], vh_step=steps[1], alpha=alpha, gate_steps=gate_steps)
        write_json(dest/'selection.json', receipt)
        receipts.append(receipt)
        # Only after all arms are selected and saved are evaluation labels used.
        for arm, s in scores.items():
            results.append(dict(seed=seed, arm=arm, **metrics(y[te], s[te])))
        print('EVALUATION', seed, results[-4:], flush=True)
    pd.DataFrame(results).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(results).groupby('arm')[['OA', 'BA', 'NLL', 'Brier']].agg(['mean', 'std']).to_csv(out/'summary.csv')
    write_json(out/'complete.json', dict(seconds=time.perf_counter()-tick, n=len(table), evaluation=len(te),
               classes=4, seeds=args.seeds, selections=receipts, source_only=True,
               training_adaptation='Fresh four-class stagewise source fit; held-out source selection reused across stages.'))


if __name__ == '__main__':
    main()
