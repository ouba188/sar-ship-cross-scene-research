"""Marginal-preserving local pairing contrast before class aggregation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import itertools
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location('pairing_relative', ROOT/'reference_relative_metric/run.py')
relative = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relative)
learned, edge = relative.learned, relative.edge
Metric = relative.Metric
ARMS = ('joint', 'excess', 'covariance', 'excess_shuffle101', 'excess_shuffle307')
SEEDS = (17, 29, 43)
BATCH = 128


def aggregate(B, N, qm, sm, mode):
    """B/N: batch,class,reference,query-edge,support-edge, normalized logits."""
    q = qm[:, None, None].to(B.dtype)
    s = sm[None].to(B.dtype)
    nq = q.sum(-1).clamp_min(1)
    ns = s.sum(-1).clamp_min(1)
    valid = (qm.sum(-1)[:, None, None] >= 2) & (sm.sum(-1)[None] >= 2)
    if mode == 'covariance':
        def center(v):
            row = (v*s[..., None, :]).sum(-1, keepdim=True)/ns[..., None, None]
            col = (v*q[..., :, None]).sum(-2, keepdim=True)/nq[..., None, None]
            mean = (row*q[..., :, None]).sum(-2, keepdim=True)/nq[..., None, None]
            return v-row-col+mean
        products = center(B)*center(N)
        by_edge = (products*s[..., None, :]).sum(-1)/ns[..., None]
        score = (by_edge*q).sum(-1)/nq
        return torch.where(valid, score, 0.), by_edge, None
    # i keeps the query body; k replaces only its neighborhood endpoint.
    logits = B[..., :, None, :] + N[..., None, :, :]
    logits = logits.masked_fill(~sm[None, ..., None, None, :], -1e4)
    C = torch.logsumexp(logits, -1)-ns[..., None, None].log()
    diagonal = C.diagonal(dim1=-2, dim2=-1)
    null = (C*q[..., None, :]).sum(-1)/nq[..., None]
    by_edge = diagonal if mode == 'joint' else diagonal-null
    score = (by_edge*q).sum(-1)/nq
    return torch.where(valid, score, 0.), by_edge, C


class Reader:
    def __init__(self, x, mass, bank, arm, divisor=None, device='cuda', dtype=torch.float32):
        shuffle = int(arm.split('shuffle')[1]) if 'shuffle' in arm else None
        b, n, legal, mapping = edge.edges(x, mass, shuffle)
        self.arm, self.bank, self.mapping = arm, bank, mapping
        self.b = torch.as_tensor(b, device=device, dtype=dtype)
        self.n = torch.as_tensor(n, device=device, dtype=dtype)
        self.legal = torch.as_tensor(legal, device=device)
        self.sb, self.sn = self.b[bank], self.n[bank]
        self.sm = self.legal[bank]
        self.divisor = torch.as_tensor(np.ones(4) if divisor is None else divisor, device=device, dtype=dtype)
        self.reference = torch.full((len(x), 8, 4), float('nan'), device=device, dtype=dtype)
        self.reference_ready = np.zeros(len(x), bool)

    def full_scores(self, ids, W, trace=False):
        output, valids, traces = [], [], []
        for g in range(4):
            B = torch.einsum('bid,crjd->bcrij', self.b[ids, g]@W, self.sb[:, :, g])/(2*edge.TAU)
            N = torch.einsum('bid,crjd->bcrij', self.n[ids, g]@W, self.sn[:, :, g])/(2*edge.TAU)
            qm, sm = self.legal[ids, g], self.sm[:, :, g]
            ref_valid = (sm.sum(-1) >= 2)
            value, by_edge, C = aggregate(B, N, qm, sm, self.arm)
            raw = value.sum(-1)/ref_valid.sum(-1).clamp_min(1)[None]
            valid = (qm.sum(-1) >= 2)[:, None] & ref_valid.any(-1)[None]
            output.append(raw)
            valids.append(valid)
            if trace:
                traces.append(dict(B=B.detach().cpu().numpy(), N=N.detach().cpu().numpy(),
                                   C=None if C is None else C.detach().cpu().numpy(),
                                   by_edge=by_edge.detach().cpu().numpy(),
                                   reference_score=value.detach().cpu().numpy(),
                                   reference_valid=ref_valid.detach().cpu().numpy()))
        raw, valid = torch.stack(output, -1), torch.stack(valids, -1)
        center = torch.where(valid, raw, 0.).sum(1, keepdim=True)/valid.sum(1, keepdim=True).clamp_min(1)
        z = torch.where(valid, (raw-center)/self.divisor, 0.)
        return (z, raw, valid, traces) if trace else (z, raw, valid)

    def cache_reference(self, ids):
        missing = np.asarray(ids)[~self.reference_ready[ids]]
        I = torch.eye(32, device=self.b.device, dtype=self.b.dtype)
        with torch.inference_mode():
            for first in range(0, len(missing), BATCH):
                part = missing[first:first+BATCH]
                self.reference[part] = self.full_scores(part, I)[0]
        self.reference_ready[missing] = True

    def scores(self, ids, W):
        z, raw, valid = self.full_scores(ids, W)
        return z-self.reference[ids], raw, valid


def smoke(data):
    table, x, mass, bank, idx, tr, va, te, _ = data
    rng = np.random.default_rng(17)
    C = rng.normal(size=(3, 3))
    null = np.mean([C[np.arange(3), p].mean() for p in itertools.permutations(range(3))])
    a = C.diagonal().mean()-C.mean()
    nuisance = rng.normal(size=(3, 1))+rng.normal(size=(1, 3))
    b = (C+nuisance).diagonal().mean()-(C+nuisance).mean()
    checks = dict(permutation_null_error=abs(null-C.mean()), row_column_error=abs(a-b))
    utility = idx[tr]
    missing = utility[mass[utility, 1].sum(-1) == 0]
    ids = np.unique(np.r_[utility[:2], missing[:1]])
    reader = Reader(x, mass, bank, 'excess', device='cpu', dtype=torch.float64)
    W = torch.eye(32, dtype=torch.float64, requires_grad=True)
    z, raw, valid, trace = reader.full_scores(ids, W, True)
    scalar_error = 0.
    for g, state in enumerate(trace):
        for t, query in enumerate(ids):
            qi = np.flatnonzero(reader.legal[query, g].numpy())
            for c in range(8):
                values = []
                for r in range(8):
                    sj = np.flatnonzero(reader.sm[c, r, g].numpy())
                    if len(qi) < 2 or len(sj) < 2:
                        continue
                    from scipy.special import logsumexp
                    matrix = np.array([[logsumexp(state['B'][t, c, r, i, sj]+state['N'][t, c, r, k, sj])-np.log(len(sj)) for k in qi] for i in qi])
                    values.append(matrix.diagonal().mean()-matrix.mean())
                expected = np.mean(values) if values else 0.
                scalar_error = max(scalar_error, abs(expected-float(raw[t, c, g].detach())))
    checks['scalar_replay_error'] = scalar_error
    if len(missing):
        checks['missing_VH_max'] = float(raw[ids == missing[0], :, 2:].abs().max().detach())
    weights = torch.arange(z.numel(), dtype=torch.float64).reshape(z.shape)/z.numel()
    objective = (z*weights).sum()
    objective.backward()
    pos = np.unravel_index(int(W.grad.abs().argmax()), (32, 32))
    exact = float(W.grad[pos])
    eps = 1e-5
    with torch.no_grad():
        W[pos] += eps
        plus = float((reader.full_scores(ids, W)[0]*weights).sum())
        W[pos] -= 2*eps
        minus = float((reader.full_scores(ids, W)[0]*weights).sum())
        W[pos] += eps
    checks['gradient_error'] = abs(exact-(plus-minus)/(2*eps))
    flip = Reader(x[:, :, edge.REV], mass[:, :, edge.REV], bank, 'excess', device='cpu', dtype=torch.float64)
    checks['axis_sign_flip_error'] = float((reader.full_scores(ids, W)[0]-flip.full_scores(ids, W)[0]).abs().max().detach())
    # Constant neighborhood cannot contribute a pairing excess.
    B = torch.randn(1, 1, 1, 3, 4, dtype=torch.float64)
    N = torch.randn(1, 1, 1, 1, 4, dtype=torch.float64).expand_as(B)
    qm, sm = torch.ones((1, 3), dtype=torch.bool), torch.ones((1, 1, 4), dtype=torch.bool)
    checks['constant_neighbor_error'] = float(aggregate(B, N, qm, sm, 'excess')[0].abs().max())
    for key, value in checks.items():
        assert value < 1e-7, (key, value)
    checks['utility_indices'] = ids.tolist()
    checks['evaluation_used'] = False
    (HERE/'smoke.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks), flush=True)


def main(data, selected_arms):
    table, x, mass, bank, idx, tr, va, te, _ = data
    y = table.class_id.to_numpy(int)
    allids = np.arange(len(table))
    relative.HERE = HERE
    relative.BATCH = learned.BATCH = BATCH
    results, receipts, histories, class_rows, pair_rows, scales = [], [], [], [], [], []
    weights = {}
    predictions = {seed: dict(np.load(ROOT/f'reference_dependence/frozen_marginal/predictions_{seed}.npz')) for seed in SEEDS}
    features = {seed: {} for seed in SEEDS}
    start = time.perf_counter()
    for arm in selected_arms:
        reader = Reader(x, mass, bank, arm)
        initial, validity = [], []
        with torch.inference_mode():
            for first in range(0, len(tr), BATCH):
                z, _, valid = reader.full_scores(idx[tr[first:first+BATCH]], torch.eye(32, device='cuda'))
                initial.append(z.cpu().numpy())
                validity.append(valid.cpu().numpy())
        initial, validity = np.concatenate(initial), np.concatenate(validity)
        rms = np.sqrt(np.sum(initial**2, axis=(0, 1))/np.maximum(validity.sum((0, 1)), 1)).clip(1e-6)
        reader.divisor.copy_(torch.as_tensor(rms, device='cuda'))
        scales.extend(dict(arm=arm, feature=g, divisor=float(v)) for g, v in enumerate(rms))
        reader.cache_reference(idx[np.r_[tr, va]])
        for seed in SEEDS:
            tick = time.perf_counter()
            anchor = predictions[seed]['anchor'].astype(np.float32)
            model, state, selected, history = relative.fit(reader, anchor, table, idx, tr, va, seed, arm)
            reader.cache_reference(allids)
            score, z, _ = learned.infer(reader, model, allids, anchor)
            predictions[seed][arm] = score
            features[seed][arm] = z
            weights.update({f'{arm}_{seed}_{key}':value for key, value in state.items()})
            receipts.append(dict(arm=arm, seed=seed, selected_step=selected, seconds=time.perf_counter()-tick, parameters=260))
            histories.extend(history)
            for role, loc in [('utility_train', tr), ('selection', va), ('evaluation', te)]:
                ids = idx[loc]
                for name, values in [(arm, score), ('anchor', anchor)]:
                    metrics = edge.dep.metrics(y[ids], values[ids])
                    results.append(dict(arm=name, seed=seed, role=role, **metrics))
                old, new = anchor[ids].argmax(-1) == y[ids], score[ids].argmax(-1) == y[ids]
                pair_rows.append(dict(arm=arm, seed=seed, role=role, rescued=int((new & ~old).sum()), harmed=int((old & ~new).sum()), net=int(new.sum()-old.sum())))
                for c in range(8):
                    mask = y[ids] == c
                    class_rows.append(dict(arm=arm, seed=seed, role=role, class_id=c, n=int(mask.sum()), anchor_correct=int(old[mask].sum()), correct=int(new[mask].sum()), net=int(new[mask].sum()-old[mask].sum())))
            np.savez_compressed(HERE/f'predictions_{seed}.npz', **predictions[seed])
            np.savez_compressed(HERE/f'features_{seed}.npz', **features[seed])
            np.savez_compressed(HERE/'weights.npz', **weights)
            pd.DataFrame(results).drop_duplicates(['arm', 'seed', 'role']).to_csv(HERE/'metrics.csv', index=False)
            pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv', index=False)
            pd.DataFrame(histories).to_csv(HERE/'training_curve.csv', index=False)
            pd.DataFrame(class_rows).to_csv(HERE/'class_results.csv', index=False)
            pd.DataFrame(pair_rows).to_csv(HERE/'paired_results.csv', index=False)
            pd.DataFrame(scales).to_csv(HERE/'scales.csv', index=False)
        del reader
    results = pd.DataFrame(results).drop_duplicates(['arm', 'seed', 'role'])
    summary = results[results.role.eq('evaluation')].groupby('arm')[['OA', 'BA', 'J', 'NLL', 'balanced_NLL', 'Brier']].mean()
    summary.to_csv(HERE/'summary.csv')
    (HERE/'receipt.json').write_text(json.dumps(dict(completed=True, fits=len(receipts), arms=selected_arms, seconds=time.perf_counter()-start,
        source_utility=len(tr), source_selection=len(va), source_evaluation=len(te), evaluation_tuning=False, external=False), indent=2), encoding='utf-8')
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS)
    args = parser.parse_args()
    data = learned.source_data()
    smoke(data)
    if not args.smoke_only:
        main(data, args.arms)
