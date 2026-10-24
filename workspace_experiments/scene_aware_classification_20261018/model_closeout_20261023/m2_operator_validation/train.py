"""Source-only matched operator fits over the frozen complete three-module model."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import json
import time
from fractions import Fraction
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.special import logsumexp
from operators import Candidate, class_summary, RoleConv, GroupPool, pooled_summary

HERE = Path(__file__).resolve().parent
UP = HERE.parent/'regional_utility_pilot/upstream'
LOCAL = HERE.parent/'local_evidence_transport'
ARMS = ('grid_refit', 'static_pool', 'conv_grid', 'conv_static_pool', 'learned_pool', 'conv_learned_pool')
SEEDS = (17, 29, 43)
CHECKPOINTS = (0, 10, 20, 40, 60, 80, 100)


def metrics(y, score):
    pred = score.argmax(1)
    lp = score-logsumexp(score, axis=1, keepdims=True)
    recall = [(pred[y == c] == c).mean() for c in np.unique(y)]
    oa, ba = float((pred == y).mean()), float(np.mean(recall))
    return dict(n=len(y), OA=oa, BA=ba, J=(oa+ba)/2,
                NLL=float(-lp[np.arange(len(y)), y].mean()),
                Brier=float(((np.exp(lp)-np.eye(8)[y])**2).sum(1).mean()))


def rank(y, score):
    p = score.argmax(1)
    return (Fraction(int((p == y).sum()), len(y)) + sum(
        Fraction(int((p[y == c] == c).sum()), int((y == c).sum())) for c in np.unique(y))/len(np.unique(y)))/2


class Data:
    def __init__(self, seed, device='cuda'):
        self.device = device
        self.table = pd.read_csv(HERE/'table.csv', dtype={'mmsi': str})
        self.up = self.table.up_index.to_numpy(int)
        self.maps = {k: np.load(HERE/f'{k}.npy', mmap_mode='r') for k in ('x', 'response', 'region', 'a0')}
        summary = np.load(UP/'summaries.npz')
        self.flags = summary['region_valid'][self.up].astype(np.float32)
        self.original_summary = summary['summary1'][self.up]
        source = np.load(UP/f'strategies_seed{seed}.npz')
        self.s0 = source['logits'][self.up, 0].astype(np.float32)
        old_m2 = source['logits'][self.up, 2]-source['logits'][self.up, 0]
        self.quality = source['regional_raw'][self.up, 0, -32:].astype(np.float32)
        compact = pd.read_csv(LOCAL/'paired_scale_readout/table.csv')
        mapping = {int(i): j for j, i in enumerate(compact.table_index)}
        self.compact = np.array([mapping[int(i)] for i in self.up])
        self.baseline = np.load(LOCAL/'pair_conditioned_joint_metric/results'/f'pair_joint_{seed}.npz')['score'][self.compact].astype(np.float32)
        self.rest = (self.baseline-old_m2).astype(np.float32)
        self.ck = torch.load(UP/'models'/f'full_seed{seed}.pt', map_location='cpu', weights_only=False)
        pca = np.load(LOCAL/'projection_layer1.npz')['components']
        w = np.load(UP/'models/reader_layer1.npz')['W'].reshape(8, 2, 256).transpose(1, 0, 2)
        self.wproject = w@pca.T
        self.tau = json.loads((UP/'scales.json').read_text())['tau']['layer1']
        meta = np.load(HERE/'metadata.npz')
        g = meta['geometry']
        self.theta = g[:, :, 4, None]-np.arange(4)[None, None]*np.pi/2
        self.tr = np.flatnonzero(self.table.role.eq('utility_train'))
        self.va = np.flatnonzero(self.table.role.eq('selection'))
        self.te = np.flatnonzero(self.table.role.eq('evaluation'))

    def batch(self, ids, arm):
        ids = np.asarray(ids, int)
        regions = np.stack([self.maps['region'][i] for i in ids])
        occupied = regions.sum((0, 1, 2, 3)) > 0
        yy, xx = np.nonzero(occupied)
        if len(yy):
            ys = slice(max(0, yy.min()-1), min(40, yy.max()+2))
            xs = slice(max(0, xx.min()-1), min(40, xx.max()+2))
        else:
            ys = xs = slice(0, 3)
        arrays = {'region': regions[..., ys, xs]}
        needed = ['response']
        if arm.startswith('conv_') or 'learned_pool' in arm:
            needed.append('x')
        if 'pool' in arm:
            needed.append('a0')
        for key in needed:
            arrays[key] = np.stack([self.maps[key][i, ..., ys, xs] for i in ids])
        arrays.update(s0=self.s0[ids], rest=self.rest[ids], quality=self.quality[ids],
                      theta=self.theta[ids], region_valid=self.flags[ids])
        return {k: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32, device=self.device)
                for k, v in arrays.items()}


@torch.inference_mode()
def predict(model, data, ids, size=8, summaries=False):
    scores, desc = [], []
    for first in range(0, len(ids), size):
        s, d = model(data.batch(ids[first:first+size], model.arm), True)
        scores.append(s.cpu().numpy())
        if summaries:
            desc.append(d.cpu().numpy())
    score = np.concatenate(scores)
    return (score, np.concatenate(desc)) if summaries else score


def synthetic_checks(device='cpu'):
    torch.manual_seed(123)
    x = torch.randn(2, 32, 5, 6, device=device)
    reg = torch.rand(2, 3, 5, 6, device=device)
    reg[..., 0, :] = 0
    theta = torch.tensor([.3, 1.7], device=device)
    conv = RoleConv().to(device)
    initial = float(conv(x, reg, theta).abs().max())
    assert initial == 0
    target = torch.randn_like(x)
    (conv(x, reg, theta)*target).sum().backward()
    gradient = float(conv.expand.weight.grad.norm())
    assert gradient > 0
    with torch.no_grad():
        conv.expand.weight.normal_(0, .03)
    z = conv(x, reg, theta)
    axis_error = float((z-conv(x, reg, theta+torch.pi)).abs().max())
    assert axis_error < 2e-6 and float(z[..., 0, :].abs().max()) == 0
    pool = GroupPool().to(device)
    a0 = torch.rand(2, 20, 5, 6, device=device)
    a0[..., 0, :] = 0
    allocations = pool.assignment(x, a0)
    initial_error = max(float((a-prior).abs().max()) for a, _, _, prior in allocations)
    assert initial_error < 1e-7
    response = torch.randn(2, 8, 5, 6, device=device)
    tau = torch.ones(2, device=device)
    static = pooled_summary(response, a0, tau)
    learned = pooled_summary(response, a0, tau, allocations)
    assert float((static-learned).abs().max()) < 1e-6
    (learned*torch.randn_like(learned)).sum().backward()
    pool_gradient = float(pool.score[-1].weight.grad.norm())
    assert pool_gradient > 1e-8
    with torch.no_grad():
        pool.score[-1].weight.normal_(0, .2)
    allocations = pool.assignment(x, a0)
    row_error = max(float((a.sum(-1)-row).abs().max()) for a, row, b, prior in allocations)
    col_error = max(float((a.sum(1)-b).abs().max()) for a, row, b, prior in allocations)
    movement = max(float((a-prior).abs().max()) for a, row, b, prior in allocations)
    assert max(row_error, col_error) < 1e-6 and movement > 1e-7
    updated = pooled_summary(response, a0, tau, allocations)
    mean_error = float((updated[..., 0]-static[..., 0]).abs().max())
    assert mean_error < 1e-6
    return dict(conv_identity=initial, conv_gradient=gradient, axis_sign_error=axis_error,
                pool_identity=initial_error, pool_gradient=pool_gradient, row_error=row_error,
                column_error=col_error, assignment_change=movement, conserved_mean_error=mean_error)


def fit(data, arm, seed, steps=100):
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = Candidate(arm, data.ck, data.wproject, data.tau).to(data.device)
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ytr = data.table.iloc[data.tr].class_id.to_numpy(int)
    yv = data.table.iloc[data.va].class_id.to_numpy(int)
    counts = np.bincount(ytr, minlength=8)
    weights = .5+.5*len(ytr)/(8*counts[ytr])
    labels = torch.tensor(ytr, device=data.device)
    weights = torch.tensor(weights, device=data.device, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(data.tr), size=(steps, 64))
    best = rank(yv, data.baseline[data.va])
    chosen, best_state = -1, initial_state
    rows = [dict(seed=seed, arm=arm, step=-1, train_loss=None, **metrics(yv, data.baseline[data.va]))]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    for step in range(steps+1):
        loss_value = None
        if step:
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for first in range(0, 64, 8):
                take = draws[step-1, first:first+8]
                score = model(data.batch(data.tr[take], arm))
                loss = (F.cross_entropy(score, labels[take], reduction='none')*weights[take]).sum()/64
                assert torch.isfinite(loss), (arm, step)
                loss.backward()
                total += float(loss.detach())
            optimizer.step()
            loss_value = total
        if step not in CHECKPOINTS and step != steps:
            continue
        score = predict(model, data, data.va)
        m = metrics(yv, score)
        rows.append(dict(seed=seed, arm=arm, step=step, train_loss=loss_value, **m))
        current = rank(yv, score)
        if current > best:
            best, chosen = current, step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print('FIT', seed, arm, step, 'J', round(m['J'], 6), 'selected', chosen,
              'seconds', round(time.perf_counter()-started, 1), flush=True)
    receipt = dict(seed=seed, arm=arm, selected_step=chosen, seconds=time.perf_counter()-started,
                   parameters=sum(p.numel() for p in model.parameters()), steps=steps, examples_per_step=64)
    torch.save(dict(state=best_state, receipt=receipt, head_mu=data.ck['mu'], head_scale=data.ck['scale']),
               HERE/'models'/f'{arm}_{seed}.pt')
    pd.DataFrame(rows).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv', index=False)
    del model, optimizer
    torch.cuda.empty_cache()
    return receipt


def real_checks(data):
    torch.manual_seed(17)
    ids = data.tr[:8]
    model = Candidate('grid_refit', data.ck, data.wproject, data.tau).to(data.device)
    score, summary = predict(model, data, ids, summaries=True)
    se = float(abs(summary-data.original_summary[ids]).max())
    pe = float(abs(score-data.baseline[ids]).max())
    assert se < 3e-4 and pe < 5e-4, (se, pe)
    del model
    batch = data.batch(ids[:2], 'conv_learned_pool')
    x, a0 = batch['x'].flatten(0, 2), batch['a0'].flatten(0, 2)
    pool = GroupPool().to(data.device)
    with torch.no_grad():
        pool.score[-1].weight.normal_(0, .3)
    alloc = pool.assignment(x, a0)
    err = max(float((a.sum(-1)-row).abs().max()) for a, row, b, prior in alloc)
    movement = max(float((a-prior).abs().max()) for a, row, b, prior in alloc)
    assert err < 2e-4 and movement > 1e-8, (err, movement)
    for arm in ('conv_grid', 'learned_pool', 'conv_learned_pool'):
        torch.manual_seed(17)
        m = Candidate(arm, data.ck, data.wproject, data.tau).to(data.device)
        b = data.batch(ids[:2], arm)
        loss = F.cross_entropy(m(b), torch.tensor(data.table.iloc[ids[:2]].class_id.to_numpy(), device=data.device))
        loss.backward()
        grad = float(m.head.net[-1].weight.grad.norm())
        assert np.isfinite(grad) and grad > 0
        if m.pool is not None:
            g = float(m.pool.score[-1].weight.grad.norm())
            assert g > 1e-12, (arm, g)
        if m.conv is not None:
            g = float(m.conv.expand.weight.grad.norm())
            assert g > 1e-12, (arm, g)
        del m, b
    return dict(grid_summary_replay=se, baseline_replay=pe, real_assignment_mass_error=err, real_assignment_change=movement)


def main(mode, arms=ARMS, seeds=SEEDS):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for folder in ('results', 'models'):
        (HERE/folder).mkdir(exist_ok=True)
    if mode == 'synthetic':
        checks = synthetic_checks()
        (HERE/'synthetic_checks.json').write_text(json.dumps(checks, indent=2))
        print(checks, flush=True)
        return
    started = time.perf_counter()
    data = Data(17)
    checks = real_checks(data)
    (HERE/'real_checks.json').write_text(json.dumps(checks, indent=2))
    print('REAL_CHECKS', checks, flush=True)
    if mode == 'check':
        return
    del data
    receipts = []
    for seed in seeds:
        data = Data(seed)
        for arm in arms:
            path = HERE/'models'/f'{arm}_{seed}.pt'
            if path.exists():
                receipts.append(torch.load(path, map_location='cpu', weights_only=False)['receipt'])
            else:
                receipts.append(fit(data, arm, seed))
            pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv', index=False)
        del data
    # Evaluation outcomes are inspected only after the source-selected fits finish.
    rows = []
    for seed in seeds:
        data = Data(seed)
        for arm in arms:
            ck = torch.load(HERE/'models'/f'{arm}_{seed}.pt', map_location='cpu', weights_only=False)
            model = Candidate(arm, data.ck, data.wproject, data.tau).cuda()
            model.load_state_dict(ck['state'])
            score = data.baseline.copy() if ck['receipt']['selected_step'] < 0 else predict(model, data, np.arange(len(data.table)))
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz', score=score, up_index=data.up)
            for role, ids in (('utility_train', data.tr), ('selection', data.va), ('evaluation', data.te)):
                rows.append(dict(seed=seed, arm=arm, role=role, selected_step=ck['receipt']['selected_step'],
                                 **metrics(data.table.iloc[ids].class_id.to_numpy(int), score[ids])))
            del model
        for role, ids in (('utility_train', data.tr), ('selection', data.va), ('evaluation', data.te)):
            rows.append(dict(seed=seed, arm='old_both', role=role, selected_step=-1,
                             **metrics(data.table.iloc[ids].class_id.to_numpy(int), data.baseline[ids])))
        del data
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv', index=False)
    (HERE/'complete.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started, fits=len(receipts),
                                                    arms=list(arms), seeds=list(seeds), checks=checks), indent=2))
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA','BA']].mean(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['synthetic', 'check', 'run'])
    p.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS)
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    a = p.parse_args()
    main(a.mode, a.arms, a.seeds)
