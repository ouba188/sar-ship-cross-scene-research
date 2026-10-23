"""Source-only paired trial of shared-evidence competitive matching."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UP = ROOT.parent / 'regional_utility_pilot/upstream'
sys.path.insert(0, str(ROOT))
from match_readout import unit
from train_readout import metrics, exact_j
from solver import transport_values

EPS = .1
PENALTY = .1
ARMS = ('fixed', 'uncoupled', 'coupled')


class Matching:
    def __init__(self, sim, a, b, available):
        # Regional batch, support class, query token, support token.
        self.a = a
        self.b = b
        self.available = available
        self.valid = available & (available.sum(-1, keepdim=True) >= 2)
        self.classes = sim.shape[1]
        aa = torch.cat([a, 2 - a.sum(-1, keepdim=True)], -1)
        bb = torch.cat([b, 2 - b.sum(-1, keepdim=True)], -1)
        self.bb = bb
        self.ss = torch.nn.functional.pad(sim, (0, 1, 0, 1))
        self.ref = aa[:, None, :, None] * bb[:, :, None, :] / 2
        self.kernel = (self.ss / EPS).exp() * self.ref
        # The row-unconstrained control has the same null alternative and b.
        self.free = EPS * torch.logsumexp(
            self.ss[:, :, :-1] / EPS + (bb[:, :, None] / 2).clamp_min(1e-30).log(), -1)
        self.max_error = 0.

    def values(self, w, coupled=True, return_plan=False):
        # w: batch, candidate hypothesis, query token. All rivals see this w.
        if not coupled:
            value = torch.einsum('bhn,bcn->bhc', w, self.free)
            grad = self.free[:, None].expand(-1, w.shape[1], -1, -1)
            return value, grad
        ww = torch.cat([w, 2 - w.sum(-1, keepdim=True)], -1)
        k = self.kernel[:, None].expand(-1, w.shape[1], -1, -1, -1)
        shape = k.shape
        k = k.reshape(-1, shape[-2], shape[-1])
        rows = ww[:, :, None].expand(-1, -1, self.classes, -1).reshape(-1, shape[-2])
        cols = self.bb[:, None].expand(-1, w.shape[1], -1, -1).reshape(-1, shape[-1])
        v = torch.ones_like(cols)
        for step in range(240):
            u = rows / torch.bmm(k, v[..., None]).squeeze(-1).clamp_min(1e-30)
            v = cols / torch.bmm(k.transpose(1, 2), u[..., None]).squeeze(-1).clamp_min(1e-30)
            if step in (59, 119, 239):
                t = u[..., None] * k * v[:, None]
                err = max(float((t.sum(-1) - rows).abs().max()),
                          float((t.sum(-2) - cols).abs().max()))
                if err < 2e-6:
                    break
        self.max_error = max(self.max_error, err)
        if err >= 2e-6:
            raise RuntimeError(f'Inner transport not converged: {err}')
        t = t.reshape(shape)
        ref = self.ref[:, None]
        kl = (t * (t.clamp_min(1e-30).log() - ref.clamp_min(1e-30).log()) - t + ref).sum((-1, -2))
        value = (t * self.ss[:, None]).sum((-1, -2)) - EPS * kl
        grad = -EPS * u.clamp_min(1e-30).log().reshape(*shape[:-2], shape[-2])[..., :-1]
        if return_plan:
            return value, grad, t
        return value, grad

    def objective(self, w, coupled=True):
        value, grad = self.values(w, coupled)
        c = self.classes
        own = value.diagonal(dim1=1, dim2=2)
        own_grad = grad[:, torch.arange(c), torch.arange(c)]
        mask = self.available[:, None] & ~torch.eye(c, device=w.device, dtype=torch.bool)[None]
        wrong = value.masked_fill(~mask, -1e6)
        rival = wrong.argmax(-1)
        rival_value = wrong.gather(-1, rival[..., None]).squeeze(-1)
        rival_grad = grad.gather(2, rival[..., None, None].expand(-1, -1, 1, w.shape[-1])).squeeze(2)
        logratio = w.clamp_min(1e-30).log() - self.a[:, None].clamp_min(1e-30).log()
        score = own - rival_value - PENALTY * (w * logratio).sum(-1)
        gradient = own_grad - rival_grad - PENALTY * (logratio + 1)
        score = torch.where(self.valid, score, 0)
        gradient = torch.where(self.valid[..., None] & (self.a[:, None] > 0), gradient, 0)
        return score, gradient, rival

    def optimize(self, coupled=True, rounds=20):
        w = self.a[:, None].expand(-1, self.classes, -1).clone()
        score, grad, rival = self.objective(w, coupled)
        initial = score.clone()
        start_rival = rival.clone()
        accepted_count = torch.zeros_like(score, dtype=torch.int32)
        active = self.valid.clone()
        for _ in range(rounds):
            if not bool(active.any()):
                break
            rate = torch.ones_like(score)
            pending = active.clone()
            accepted = torch.zeros_like(active)
            for _ in range(8):
                proposal = w * (rate[..., None] * (grad - grad.amax(-1, keepdim=True))).exp()
                cap = 2 * self.a[:, None]
                high = torch.where(cap > 0, cap / proposal.clamp_min(1e-30), 0).amax(-1)
                low = torch.zeros_like(high)
                target = self.a.sum(-1)[:, None]
                for _ in range(32):
                    middle = (high + low) / 2
                    total = torch.minimum(cap, middle[..., None] * proposal).sum(-1)
                    low = torch.where(total < target, middle, low)
                    high = torch.where(total >= target, middle, high)
                trial = torch.minimum(cap, high[..., None] * proposal)
                trial = torch.where(pending[..., None], trial, w)
                value, gradient, other = self.objective(trial, coupled)
                take = pending & (value > score + 1e-5)
                w = torch.where(take[..., None], trial, w)
                score = torch.where(take, value, score)
                grad = torch.where(take[..., None], gradient, grad)
                rival = torch.where(take, other, rival)
                accepted_count += take
                accepted |= take
                pending &= ~take
                if not bool(pending.any()):
                    break
                rate *= .5
            active &= accepted
        return score, w, rival, initial, accepted_count, start_rival


def check_gpu():
    torch.set_num_threads(2)
    rng = np.random.default_rng(17)
    s = rng.uniform(-.2, .8, (3, 4, 5))
    a = np.array([.1, .2, .3, .4])
    b = rng.uniform(.1, 1, (3, 5)); b /= b.sum(-1, keepdims=True)
    expected = transport_values(s, a, b, a)
    device = 'cuda'
    model = Matching(torch.tensor(s[None], device=device, dtype=torch.float32),
                     torch.tensor(a[None], device=device, dtype=torch.float32),
                     torch.tensor(b[None], device=device, dtype=torch.float32),
                     torch.ones((1, 3), device=device, dtype=torch.bool))
    value, gradient = model.values(model.a[:, None])
    verror = float(abs(value[0, 0].cpu().numpy() - expected[0]).max())
    d = np.array([.2, -.1, .1, -.2])
    gerror = float(abs(gradient[0, 0].cpu().numpy() @ d - expected[1] @ d).max())
    assert verror < 3e-6 and gerror < 3e-6, (verror, gerror)
    report = dict(value_error=verror, simplex_gradient_error=gerror, marginal_error=model.max_error)
    print('GPU_CHECK', report, flush=True)
    return report


def regional_inputs(x, m, source_x, source_m, refs):
    rows = []; masses = []; support = []; available = []
    for p in range(2):
        for start, stop in ((0, 8), (8, 16), (16, 20)):
            q = x[:, p, start:stop]
            s = source_x[refs, p, start:stop]
            qa = m[:, p, start:stop]
            sb = source_m[refs, p, start:stop]
            qa = qa / np.maximum(qa.sum(-1, keepdims=True), 1e-20)
            has = sb.sum(-1) > 0
            sb = sb / np.maximum(sb.sum(-1, keepdims=True), 1e-20)
            sb = sb / np.maximum(has.sum(-1, keepdims=True)[..., None], 1)
            s = s.reshape(len(x), 8, -1, x.shape[-1])
            sb = sb.reshape(len(x), 8, -1)
            sim = np.einsum('bnd,bcmd->bcnm', q, s)
            sim = np.pad(sim, ((0, 0), (0, 0), (0, 8 - q.shape[1]), (0, 16 - s.shape[2])))
            qa = np.pad(qa, ((0, 0), (0, 8 - qa.shape[1])))
            sb = np.pad(sb, ((0, 0), (0, 0), (0, 16 - sb.shape[2])))
            rows.append(sim); masses.append(qa); support.append(sb)
            available.append(has.any(-1) & (qa.sum(-1, keepdims=True) > 0))
    return (np.stack(rows, 1).reshape(-1, 8, 8, 16),
            np.stack(masses, 1).reshape(-1, 8),
            np.stack(support, 1).reshape(-1, 8, 16),
            np.stack(available, 1).reshape(-1, 8))


def extract(limit=0):
    tick = time.perf_counter()
    check = check_gpu()
    table = pd.read_csv(UP / 'table.csv')
    idx = np.flatnonzero(table.role.isin(['utility_train', 'selection', 'evaluation']))
    if limit:
        idx = idx[:limit]
    data = np.load(ROOT / 'tokens.npz'); pr = np.load(ROOT / 'projection_layer1.npz')
    x = unit((data['layer1'] - pr['mean']) @ pr['components'].T).astype(np.float32)
    x[data['mass'] <= 0] = 0
    refs = np.load(ROOT / 'scores_layer1.npz')['refs'][idx]
    scores = {arm: [] for arm in ARMS}
    weights = []; basemass = []; availability = []; changes = []; steps = []
    errors = []
    with torch.inference_mode():
        for first in range(0, len(idx), 16):
            ids = idx[first:first + 16]
            args = regional_inputs(x[ids], data['mass'][ids], x, data['mass'], refs[first:first + 16])
            sim, a, b = [torch.tensor(v, device='cuda', dtype=torch.float32) for v in args[:3]]
            valid = torch.tensor(args[3], device='cuda', dtype=torch.bool)
            model = Matching(sim, a, b, valid)
            fixed = model.objective(a[:, None].expand(-1, 8, -1))[0]
            free = model.optimize(coupled=False)
            coupled = model.optimize()
            for arm, value in [('fixed', fixed), ('uncoupled', free[0]), ('coupled', coupled[0])]:
                scores[arm].append(value.cpu().numpy().reshape(len(ids), 2, 3, 8).transpose(0, 3, 1, 2))
            weights.append(coupled[1].cpu().numpy().reshape(len(ids), 2, 3, 8, 8).transpose(0, 3, 1, 2, 4))
            basemass.append(args[1].reshape(len(ids), 2, 3, 8))
            availability.append(model.valid.cpu().numpy().reshape(len(ids), 2, 3, 8).transpose(0, 3, 1, 2))
            changes.append((coupled[2] != coupled[5]).cpu().numpy())
            steps.append(coupled[4].cpu().numpy())
            errors.append(model.max_error)
            print('EXTRACT', first + len(ids), '/', len(idx), 'seconds', round(time.perf_counter() - tick, 2), flush=True)
    name = 'smoke' if limit else 'scores'
    np.savez_compressed(HERE / f'{name}.npz', idx=idx, refs=refs,
                        **{arm: np.concatenate(v) for arm, v in scores.items()},
                        coupled_weights=np.concatenate(weights), baseline_mass=np.concatenate(basemass),
                        available=np.concatenate(availability),
                        rival_changed=np.concatenate(changes), accepted_steps=np.concatenate(steps))
    receipt = dict(stage=name, samples=len(idx), seconds=time.perf_counter() - tick,
                   maximum_inner_error=max(errors), gpu_check=check,
                   source_only=True, eps=EPS, penalty=PENALTY, rounds=20, cap_ratio=2,
                   outer_acceptance_gain=1e-5, source_roles=table.iloc[idx].role.value_counts().to_dict())
    (HERE / f'{name}_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')


def fixed_free():
    cache = np.load(HERE / 'scores.npz'); idx = cache['idx']; refs = cache['refs']
    data = np.load(ROOT / 'tokens.npz'); pr = np.load(ROOT / 'projection_layer1.npz')
    x = unit((data['layer1'] - pr['mean']) @ pr['components'].T).astype(np.float32)
    x[data['mass'] <= 0] = 0
    output = []
    with torch.inference_mode():
        for first in range(0, len(idx), 64):
            ids = idx[first:first + 64]
            args = regional_inputs(x[ids], data['mass'][ids], x, data['mass'], refs[first:first + 64])
            sim, a, b = [torch.tensor(v, device='cuda', dtype=torch.float32) for v in args[:3]]
            valid = torch.tensor(args[3], device='cuda', dtype=torch.bool)
            model = Matching(sim, a, b, valid)
            value = model.objective(a[:, None].expand(-1, 8, -1), coupled=False)[0]
            output.append(value.cpu().numpy().reshape(len(ids), 2, 3, 8).transpose(0, 3, 1, 2))
    np.save(HERE / 'uncoupled_fixed_scores.npy', np.concatenate(output))


def evaluate():
    data = np.load(HERE / 'scores.npz'); idx = data['idx']
    table = pd.read_csv(UP / 'table.csv'); labels = table.class_id.to_numpy()
    role = table.iloc[idx].role.to_numpy()
    tr = np.flatnonzero(role == 'utility_train'); va = np.flatnonzero(role == 'selection')
    raw = {}; transforms = {}
    arms = ('fixed', 'uncoupled_fixed', 'uncoupled', 'coupled')
    if not (HERE / 'uncoupled_fixed_scores.npy').exists():
        fixed_free()
    for arm in arms:
        value = np.load(HERE / 'uncoupled_fixed_scores.npy') if arm == 'uncoupled_fixed' else data[arm].copy()
        valid = data['available']
        # Missing regions remain neutral after scaling; no negative fill values.
        counts = valid[tr].sum((0, 1))
        scale = np.sqrt((value[tr] ** 2 * valid[tr]).sum((0, 1)) / np.maximum(counts, 1))
        scale = np.maximum(scale, 1e-6)
        value = value / scale[None, None]
        combined = (value * valid).sum((2, 3)) / np.maximum(valid.sum((2, 3)), 1)
        raw[arm] = combined - combined.mean(-1, keepdims=True)
        transforms[arm] = scale.tolist()
    lambdas = (0., .1, .25, .5, 1., 2.)
    rows = []; selections = []; direct = []
    for arm in arms:
        for r in ('utility_train', 'selection', 'evaluation'):
            use = np.flatnonzero(role == r)
            direct.append(dict(arm=arm, role=r, **metrics(labels[idx[use]], raw[arm][use])))
    for seed in (17, 29, 43):
        base = np.load(UP / f'strategies_seed{seed}.npz')['logits'][:, 2].astype(np.float32)
        predictions = {'full': base}
        for arm in arms:
            best = exact_j(labels[idx[va]], base[idx[va]].argmax(-1)); selected = 0.
            for weight in lambdas:
                value = base[idx[va]] + weight * raw[arm][va]
                objective = exact_j(labels[idx[va]], value.argmax(-1))
                selections.append(dict(seed=seed, arm=arm, weight=weight,
                                       **metrics(labels[idx[va]], value)))
                if objective > best:
                    best = objective; selected = weight
            score = base.copy(); score[idx] += selected * raw[arm]
            predictions[arm] = score
            for r in ('utility_train', 'selection', 'evaluation'):
                use = idx[role == r]
                rows.append(dict(seed=seed, arm=arm, role=r, weight=selected,
                                 **metrics(labels[use], score[use])))
        for r in ('utility_train', 'selection', 'evaluation'):
            use = idx[role == r]
            rows.append(dict(seed=seed, arm='full', role=r, weight=0., **metrics(labels[use], base[use])))
        np.savez_compressed(HERE / f'predictions_{seed}.npz', **predictions)
    pd.DataFrame(rows).to_csv(HERE / 'source_metrics.csv', index=False)
    pd.DataFrame(selections).to_csv(HERE / 'selection_curve.csv', index=False)
    pd.DataFrame(direct).to_csv(HERE / 'direct_metrics.csv', index=False)
    (HERE / 'transforms.json').write_text(json.dumps(transforms, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).query("role == 'evaluation'").to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=int, default=0)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
    else:
        extract(args.smoke)
