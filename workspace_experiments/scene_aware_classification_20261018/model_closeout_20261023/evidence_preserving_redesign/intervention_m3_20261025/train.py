"""Source intervention-utility supervision for bounded query-level M3 modulation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import time
import importlib.util
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


anchor = load('intervention_anchor', HERE.parent/'anchored_m3_20261025/train.py')
Data, metrics, write_json = anchor.Data, anchor.metrics, anchor.write_json
prior = anchor.prior
SEEDS = (17, 29, 43)
ARMS = ('ce_only', 'effect', 'shuffled')
DOSE = 1.
CHECKPOINTS = (0, 20, 40, 60, 80, 100)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(32, 4, bias=False)
        self.embedding = nn.Embedding(18, 4)
        self.hidden = nn.Linear(38, 16)
        self.output = nn.Linear(16, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, valid, context):
        b = len(x)
        h = self.project(x).flatten(-2)
        types = self.embedding(torch.arange(18, device=x.device)).reshape(1, 2, 3, 3, 4)
        types = types.expand(b, -1, -1, -1, -1)
        context = context[:, None, None, None].expand(-1, 2, 3, 3, -1)
        z = torch.cat((h, valid.float(), context, types), -1)
        utility = self.output(F.gelu(self.hidden(z))).squeeze(-1).tanh()
        active = valid[..., 0] & (valid[..., 1] | valid[..., 2])
        utility = utility*active
        gate = utility.clamp_min(0)
        gate = gate/gate.sum(-1, keepdim=True).clamp_min(1)
        return gate, utility


def make_model(seed):
    torch.manual_seed(seed)
    return Model().cuda()


class Reader:
    def __init__(self, data, seed):
        tick = time.perf_counter()
        self.data, self.seed = data, seed
        self.old = anchor.Reader(data, seed)
        self.baseline = self.old.baseline
        self.refs = self.old.refs
        self.ix = torch.tensor(self.old.ix[:data.n], device='cuda')
        self.kernels = []
        self._prepare()
        self.prepare_seconds = time.perf_counter()-tick

    @torch.no_grad()
    def _prepare(self):
        old, metric = self.old.old, self.old.frozen
        n, idx = self.data.n, self.ix
        source = self.ix[self.data.tr]
        ref = old.refs[idx]
        means, pools, poolmass = [], [], []
        for r in prior.old.REGIONS:
            mass = old.mass[:, :, r]
            desc = old.d[:, :, r]
            polmean = torch.stack([desc[source, p][mass[source, p] > 0].mean(0) for p in range(2)])
            means.append(polmean)
            pools.append((desc*mass[..., None]).sum(2)/mass.sum(2).clamp_min(1e-20)[..., None])
            poolmass.append(mass.sum(2))
            saved = {k: [] for k in ('k0', 'dk', 'qa', 'sa')}
            for start in range(0, n, 64):
                ii = idx[start:start+64]; rr = old.refs[ii]
                q = old.x[ii][:, :, r][:, None, None, :, :, None, :]
                s = old.x[rr][..., r, :][..., None, :, :]
                dq, ds = old.d[ii][:, :, r], old.d[rr][..., r, :]
                hq, hs = metric.project(dq), metric.project(ds)
                k0 = prior.similarity(q, s, metric.weights(hq[:, None, None, :, :, None], hs[..., None, :, :]))
                variants = []
                for band in range(3):
                    qnew, snew = dq.clone(), ds.clone()
                    sl = slice(32*band, 32*(band+1))
                    qnew[..., sl] = polmean[None, :, None, sl]
                    snew[..., sl] = polmean[None, None, None, :, None, sl]
                    # Empty tokens are not converted into invented valid observations.
                    qnew *= old.mass[ii][:, :, r].gt(0)[..., None]
                    snew *= old.mass[rr][..., r].gt(0)[..., None]
                    hq, hs = metric.project(qnew), metric.project(snew)
                    ka = prior.similarity(q, s, metric.weights(hq[:, None, None, :, :, None], hs[..., None, :, :]))
                    variants.append(ka-k0)
                saved['k0'].append(k0)
                saved['dk'].append(torch.stack(variants, 4))
                saved['qa'].append(old.mass[ii][:, None, None, :, r])
                saved['sa'].append(old.mass[rr][..., r])
            self.kernels.append({k: torch.cat(v) for k, v in saved.items()})
        self.replacement_means = torch.stack(means).cpu().numpy()
        pool = torch.stack(pools, 2).reshape(len(old.ids), 2, 3, 3, 32)
        mass = torch.stack(poolmass, 2)
        q, qm = pool[idx], mass[idx]
        rs, rm = pool[ref], mass[ref]
        support = (rs*rm[..., None, None]).sum(2)/rm.sum(2).clamp_min(1e-20)[..., None, None]
        valid_support = rm.sum(2).gt(0)
        bscore = torch.tensor(self.baseline, device='cuda')
        prob = bscore.softmax(-1)
        top = bscore.topk(2, -1).indices
        row = torch.arange(n, device='cuda')
        a, b = support[row, top[:, 0]], support[row, top[:, 1]]
        va, vb = valid_support[row, top[:, 0]], valid_support[row, top[:, 1]]
        self.x = torch.stack((q, a, b, q.flip(1), a.flip(1), b.flip(1)), -2)
        validity = torch.stack((qm.gt(0), va, vb, qm.gt(0).flip(1), va.flip(1), vb.flip(1)), -1)
        self.valid = validity[:, :, :, None].expand(-1, -1, -1, 3, -1)
        self.context = torch.stack((prob[row, top[:, 0]], prob[row, top[:, 1]],
                                    -(prob*prob.clamp_min(1e-20).log()).sum(-1)/np.log(8),
                                    (bscore[row, top[:, 0]]-bscore[row, top[:, 1]]).tanh()), -1)
        self.local0 = torch.tensor(self.old.local0, device='cuda')

    def _group(self, ids, r, gate, mode='both'):
        saved = self.kernels[r]
        k0, dk = saved['k0'][ids], saved['dk'][ids]
        change = DOSE*(dk*gate[:, None, None, :, :, None, None]).sum(4)
        k1 = k0+change
        kt = k0 if mode == 'score' else k1
        ks = k0 if mode == 'transport' else k1
        qa, sa = saved['qa'][ids], saved['sa'][ids]
        t, _, _, _ = prior.old.sinkhorn(kt, qa, sa)
        return prior.old.pair_score(t, ks, qa, sa).mean(2)

    def read(self, ids, local):
        idx = self.ix[ids]
        old = self.old.old
        return old.anchor[idx]+old.delta(local, idx)-old.old_delta[idx]

    def forward(self, ids, model, mode='both', return_local=False, override=None):
        ids = torch.as_tensor(ids, device='cuda')
        gate, _ = model(self.x[ids], self.valid[ids], self.context[ids])
        if override is not None:
            gate = override
        if mode == 'initial':
            gate = gate*0
        groups = [self._group(ids, r, gate[:, :, r], mode) for r in range(3)]
        local = torch.stack(groups, -2)
        score = self.read(ids, local)
        return (score, local, gate) if return_local else score

    @torch.no_grad()
    def predict(self, ids, model, mode='both', batch=32, return_local=False):
        out = [self.forward(ids[i:i+batch], model, mode, return_local) for i in range(0, len(ids), batch)]
        if return_local:
            return tuple(np.concatenate([v[j].cpu().numpy() for v in out]) for j in range(3))
        return np.concatenate([v.cpu().numpy() for v in out])

    @torch.no_grad()
    def teaching(self, ids, return_scores=False):
        ids = np.asarray(ids)
        all_effect, all_scores = [], []
        for first in range(0, len(ids), 64):
            take = ids[first:first+64]
            ii = torch.tensor(take, device='cuda')
            y = torch.tensor(self.data.table.iloc[take].class_id.to_numpy(int), device='cuda')
            base = torch.tensor(self.baseline[take], device='cuda')
            rival = base.clone(); rival[torch.arange(len(ii), device='cuda'), y] = -torch.inf
            rival = rival.argmax(-1)
            ar = torch.arange(len(ii), device='cuda')
            m0 = base[ar, y]-base[ar, rival]
            effect, scores = [], []
            for p in range(2):
                for r in range(3):
                    saved = self.kernels[r]
                    for band in range(3):
                        k = saved['k0'][ii, :, :, p]+DOSE*saved['dk'][ii, :, :, p, band]
                        qa, sa = saved['qa'][ii, :, :, p], saved['sa'][ii, :, :, p]
                        t, _, _, _ = prior.old.sinkhorn(k, qa, sa)
                        ev = prior.old.pair_score(t, k, qa, sa).mean(2)
                        local = self.local0[ii].clone(); local[:, :, p, r] = ev
                        score = self.read(ii, local)
                        effect.append(score[ar, y]-score[ar, rival]-m0)
                        if return_scores: scores.append(score)
            all_effect.append(torch.stack(effect, -1).cpu().numpy())
            if return_scores: all_scores.append(torch.stack(scores, 1).cpu().numpy())
        eff = np.concatenate(all_effect)
        return (eff, np.concatenate(all_scores)) if return_scores else eff


def restore(reader, arm, seed):
    model = make_model(seed)
    ck = torch.load(HERE/'models'/f'{arm}_{seed}.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state']); model.eval()
    return model, ck['receipt']


def checks(reader):
    ids = reader.data.tr[:3]
    model = make_model(reader.seed)
    with torch.no_grad():
        score, local, gate = reader.forward(ids, model, return_local=True)
    error = float(abs(score-torch.tensor(reader.baseline[ids], device='cuda')).max())
    assert error < 3e-5 and float(gate.abs().max()) == 0
    effects = reader.teaching(ids)
    y = reader.data.table.iloc[ids].class_id.to_numpy(int)
    s0 = reader.baseline[ids]; rival = np.where(np.arange(8)[None] == y[:, None], -np.inf, s0).argmax(1)
    replay = []
    for t in (0, 8, 9, 17):
        override = torch.zeros((3, 2, 3, 3), device='cuda'); override.reshape(3, 18)[:, t] = 1
        with torch.no_grad(): s = reader.forward(ids, model, override=override).cpu().numpy()
        d = s[np.arange(3), y]-s[np.arange(3), rival]-(s0[np.arange(3), y]-s0[np.arange(3), rival])
        replay.append(float(abs(d-effects[:, t]).max()))
    assert max(replay) < 3e-5, replay
    # Test away from the deliberate non-smooth zero gate boundary.
    with torch.no_grad(): model.output.bias.fill_(.3)
    yt = torch.tensor(y, device='cuda')
    F.cross_entropy(reader.forward(ids, model), yt).backward()
    auto = float(model.output.bias.grad)
    vals = []
    with torch.no_grad():
        for delta in (.01, -.01):
            model.output.bias.fill_(.3+delta)
            vals.append(float(F.cross_entropy(reader.forward(ids, model), yt)))
    numeric = (vals[0]-vals[1])/.02
    assert abs(auto-numeric) < max(2e-5, abs(auto)*.15), (auto, numeric)
    assert all(p.grad is None for p in reader.old.frozen.parameters())
    assert all(p.grad is None for p in reader.old.old.head.parameters())
    return dict(zero_replay=error, teacher_endpoint_replay=max(replay), gradient_auto=auto,
                gradient_numeric=numeric, parameters=sum(p.numel() for p in model.parameters()))


def fit(reader, arm, seed, raw_effect):
    data = reader.data; tick = time.perf_counter()
    model = make_model(seed)
    y = data.table.class_id.to_numpy(int)
    yt = torch.tensor(y, device='cuda')
    scale = max(float(np.sqrt(np.mean(raw_effect[data.tr]**2))), 1e-6)
    target = np.tanh(raw_effect/scale).reshape(data.n, 2, 3, 3)
    if arm == 'shuffled':
        rng = np.random.default_rng(seed+9000)
        groups = {}
        vv = reader.valid[:, :, 0, 0, 0].cpu().numpy()
        for j in data.tr:
            groups.setdefault((y[j], bool(vv[j, 0]), bool(vv[j, 1])), []).append(j)
        original = target.copy()
        for ids in groups.values(): target[ids] = original[rng.permutation(ids)]
    target = torch.tensor(target, device='cuda')
    counts = np.bincount(y[data.tr], minlength=8)
    weights = torch.tensor(.5+.5*len(data.tr)/(8*counts), device='cuda', dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    rng = np.random.default_rng(seed)
    best = prior.old.exact_j(y[data.va], reader.baseline[data.va].argmax(1))
    chosen = 0; state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    curve = [dict(seed=seed, arm=arm, step=0, loss=None, ce=None, auxiliary=None,
                  **metrics(y[data.va], reader.baseline[data.va]))]
    for step in range(1, 101):
        batch = rng.choice(data.tr, 32, replace=False)
        opt.zero_grad(set_to_none=True); totals = np.zeros(3)
        for start in range(0, 32, 8):
            ids = batch[start:start+8]
            score = reader.forward(ids, model)
            ce = (F.cross_entropy(score, yt[ids], reduction='none')*weights[yt[ids]]).sum()/32
            _, utility = model(reader.x[ids], reader.valid[ids], reader.context[ids])
            active = reader.valid[ids, ..., 0] & (reader.valid[ids, ..., 1] | reader.valid[ids, ..., 2])
            aux = F.smooth_l1_loss(utility, target[ids], reduction='none')
            aux = ((aux*active).sum((1,2,3))/active.sum((1,2,3)).clamp_min(1)*weights[yt[ids]]).sum()/32
            loss = ce + (0 if arm == 'ce_only' else .1)*aux
            assert torch.isfinite(loss)
            loss.backward(); totals += [float(loss.detach()), float(ce.detach()), float(aux.detach())]
        opt.step()
        if step in CHECKPOINTS:
            score = reader.predict(data.va, model, batch=64)
            m = metrics(y[data.va], score)
            j = prior.old.exact_j(y[data.va], score.argmax(1))
            curve.append(dict(seed=seed, arm=arm, step=step, loss=totals[0], ce=totals[1], auxiliary=totals[2], **m))
            if j > best:
                best = j; chosen = step
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('FIT', seed, arm, step, 'source_J', round(m['J'], 6), 'seconds', round(time.perf_counter()-tick, 1), flush=True)
    receipt = dict(seed=seed, arm=arm, selected_step=chosen, source_effect_rms=scale,
                   steps=100, batch=32, auxiliary_weight=0 if arm=='ce_only' else .1,
                   trainable_parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-tick)
    torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    receipt=receipt, diagnostic_only=True), HERE/'models'/f'final_{arm}_{seed}.pt')
    torch.save(dict(state=state, receipt=receipt), HERE/'models'/f'{arm}_{seed}.pt')
    pd.DataFrame(curve).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv', index=False)
    return receipt


def main(check_only=False):
    tick = time.perf_counter(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    for name in ('models', 'results'): (HERE/name).mkdir(exist_ok=True)
    data = Data(); receipts = []; verification = {}
    for seed in SEEDS:
        reader = Reader(data, seed)
        verification[seed] = checks(reader)
        write_json(HERE/'checks.json', verification)
        print('CHECK', seed, verification[seed], 'prepare_seconds', reader.prepare_seconds, flush=True)
        if check_only: return
        ids = np.r_[data.tr, data.va]
        raw, interventions = reader.teaching(ids, return_scores=True)
        np.savez_compressed(HERE/'results'/f'source_interventions_{seed}.npz', ids=ids, effect=raw,
                            score=interventions, replacement_mean=reader.replacement_means)
        full = np.zeros((data.n,18), np.float32); full[ids] = raw
        for arm in ARMS:
            receipts.append(fit(reader, arm, seed, full))
            pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv', index=False)
        del reader; torch.cuda.empty_cache()
    # Only after every source-selected state is frozen do we open development outcomes.
    rows = []; diagnostics = []
    for seed in SEEDS:
        reader = Reader(data, seed)
        np.savez_compressed(HERE/'results'/f'old_full_{seed}.npz', score=reader.baseline,
                            local=reader.old.local0, up_index=data.up[:data.n])
        for arm in ('old_full',)+ARMS:
            if arm == 'old_full': score = reader.baseline
            else:
                model, receipt = restore(reader, arm, seed)
                torch.cuda.synchronize(); start=time.perf_counter()
                score, local, gate = reader.predict(np.arange(data.n), model, batch=64, return_local=True)
                torch.cuda.synchronize()
                diagnostics.append(dict(seed=seed, arm=arm, seconds=time.perf_counter()-start,
                    prepare_seconds=reader.prepare_seconds, mean_gate=float(gate.mean()), max_gate=float(gate.max())))
                np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz', score=score, local=local,
                                    gate=gate, up_index=data.up[:data.n])
            for role, ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
                rows.append(dict(seed=seed, arm=arm, role=role, **metrics(data.table.iloc[ids].class_id.to_numpy(int), score[ids])))
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(HERE/'inference_receipts.csv', index=False)
    write_json(HERE/'complete.json', dict(fits=len(receipts), seconds=time.perf_counter()-tick,
        note='Frozen model representation interventions, not raw-pixel or physical causal interventions.'))
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean().to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check-only', action='store_true')
    main(parser.parse_args().check_only)
