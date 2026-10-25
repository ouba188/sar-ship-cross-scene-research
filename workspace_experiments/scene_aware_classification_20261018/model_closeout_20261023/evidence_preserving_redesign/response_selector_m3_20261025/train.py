"""Frozen M3 single-action selection from summaries versus actual responses."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'intervention_m3_20261025'
SEEDS = (17, 29, 43)
LEARNED = ('summary', 'response', 'combined')
ARMS = ('fixed_action',) + LEARNED
CHECKPOINTS = (0, 20, 40, 60, 80, 100)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


previous = load('selector_previous', PREVIOUS / 'train.py')
Data, Reader = previous.Data, previous.Reader
metrics, write_json = previous.metrics, previous.write_json
BASE, LOCAL = previous.anchor.BASE, previous.anchor.LOCAL


@torch.inference_mode()
def candidates(reader, ids):
    """No labels: recompute each exact action through the frozen full head."""
    out = []
    for start in range(0, len(ids), 64):
        ii = torch.as_tensor(ids[start:start+64], device='cuda')
        scores = [torch.tensor(reader.baseline[ii.cpu().numpy()], device='cuda')]
        for p in range(2):
            for r in range(3):
                saved = reader.kernels[r]
                for band in range(3):
                    k = saved['k0'][ii, :, :, p] + saved['dk'][ii, :, :, p, band]
                    qa, sa = saved['qa'][ii, :, :, p], saved['sa'][ii, :, :, p]
                    t, _, _, _ = previous.prior.old.sinkhorn(k, qa, sa)
                    ev = previous.prior.old.pair_score(t, k, qa, sa).mean(2)
                    local = reader.local0[ii].clone()
                    local[:, :, p, r] = ev
                    scores.append(reader.read(ii, local))
        out.append(torch.stack(scores, 1).cpu().numpy())
    return np.concatenate(out)


def features(reader):
    n = reader.data.n
    x = reader.x.reshape(n, 18, 6, 32)
    x = torch.cat((x.mean(1, keepdim=True), x), 1).detach()
    v = reader.valid.reshape(n, 18, 6).float()
    v = torch.cat((v.mean(1, keepdim=True), v), 1)
    context = reader.context[:, None].expand(-1, 19, -1)
    active = []
    for p in range(2):
        for r in range(3):
            saved = reader.kernels[r]
            q = saved['qa'].sum(-1)[:, 0, 0, p] > 0
            s = saved['sa'].sum((1, 2, 4))[:, p] > 0
            active.extend([q & s] * 3)
    valid = torch.cat((torch.ones(n, 1, device='cuda', dtype=torch.bool),
                       torch.stack(active, 1)), 1)
    return x, torch.cat((v, context), -1).detach(), valid


def response_features(score, tr, scaler=None):
    s = score - score.mean(-1, keepdims=True)
    raw = np.concatenate((np.repeat(s[:, :1], 19, axis=1), s-s[:, :1]), -1)
    if scaler is None:
        mu = raw[tr].mean((0, 1))
        sd = raw[tr].std((0, 1))
        scaler = (mu, np.where(sd > 1e-6, sd, 1.))
    mu, sd = scaler
    return torch.tensor((raw-mu)/sd, device='cuda', dtype=torch.float32), scaler


class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.project = nn.Linear(32, 4, bias=False)
        self.embedding = nn.Embedding(19, 4)
        self.hidden = nn.Linear(54, 16)
        self.output = nn.Linear(16, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, response, extra, valid):
        h = self.project(x).flatten(-2)
        if self.arm == 'response':
            h = h * 0
        if self.arm == 'summary':
            response = response * 0
        emb = self.embedding(torch.arange(19, device=x.device))[None].expand(len(x), -1, -1)
        z = torch.cat((h, response, extra, emb), -1)
        return self.output(F.gelu(self.hidden(z))).squeeze(-1).masked_fill(~valid, -torch.inf)


def make_model(arm, seed):
    torch.manual_seed(seed)
    return Model(arm).cuda()


@torch.inference_mode()
def choose(model, x, response, extra, valid, score, ids):
    ids = np.asarray(ids)
    logits = model(x[ids], response[ids], extra[ids], valid[ids])
    action = logits.argmax(-1).cpu().numpy()
    return score[ids, action], action, logits.softmax(-1).cpu().numpy()


def risk_targets(score, y, tr):
    s = torch.tensor(score, device='cuda')
    yt = torch.tensor(y, device='cuda')
    ce = -s.log_softmax(-1).gather(-1, yt[:, None, None].expand(-1, 19, 1)).squeeze(-1)
    counts = np.bincount(y[tr], minlength=8)
    w = torch.tensor(.5 + .5*len(tr)/(8*counts), device='cuda', dtype=torch.float32)[yt]
    delta = (ce-ce[:, :1])*w[:, None]
    scale = max(float(delta[tr, 1:].square().mean().sqrt()), 1e-6)
    return ce.detach(), (delta/scale).detach(), scale


def fit(data, seed, arm, score, x, response, extra, valid):
    tick = time.perf_counter()
    y = data.table.iloc[:data.n].class_id.to_numpy(int)
    ce, risk, scale = risk_targets(score, y, data.tr)
    model = make_model(arm, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    rng = np.random.default_rng(seed)
    best = previous.prior.old.exact_j(y[data.va], score[data.va, 0].argmax(-1))
    selected = 0
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    curve = []
    for step in range(101):
        if step:
            ids = rng.choice(data.tr, 32, replace=False)
            opt.zero_grad(set_to_none=True)
            logit = model(x[ids], response[ids], extra[ids], valid[ids])
            loss = (logit.softmax(-1)*risk[ids]).sum(-1).mean()
            loss.backward()
            assert torch.isfinite(loss)
            opt.step()
        if step in CHECKPOINTS:
            pred, action, prob = choose(model, x, response, extra, valid, score, data.va)
            m = metrics(y[data.va], pred)
            j = previous.prior.old.exact_j(y[data.va], pred.argmax(-1))
            soft_ce = float((prob*ce[data.va].cpu().numpy()).sum(-1).mean())
            curve.append(dict(seed=seed, arm=arm, step=step, loss=None if step == 0 else float(loss.detach()),
                              soft_NLL=soft_ce, active_fraction=float((action != 0).mean()), **m))
            if j > best:
                best, selected = j, step
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print('FIT', seed, arm, step, m['J'], 'action_rate', (action != 0).mean(), flush=True)
    receipt = dict(seed=seed, arm=arm, selected_step=selected, seconds=time.perf_counter()-tick,
                   parameters=sum(p.numel() for p in model.parameters()), source_risk_rms=scale,
                   source_class_counts=np.bincount(y[data.tr], minlength=8))
    torch.save(dict(state=model.state_dict(), receipt=receipt, source_diagnostic_only=True),
               HERE/'models'/f'final_{arm}_{seed}.pt')
    torch.save(dict(state=state, receipt=receipt), HERE/'models'/f'{arm}_{seed}.pt')
    pd.DataFrame(curve).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv', index=False)
    return receipt


def fixed_choice(data, score, valid):
    y = data.table.class_id.to_numpy(int)
    rows = []
    for a in range(19):
        actions = np.where(valid[data.va, a].cpu().numpy(), a, 0)
        pred = score[data.va, actions]
        j = previous.prior.old.exact_j(y[data.va], pred.argmax(-1))
        rows.append((j, a))
    return max(rows, key=lambda z: (z[0], -z[1]))[1]


def checks(data, reader, score, x, response, extra, valid):
    ids = data.tr[:3]
    model = make_model('combined', reader.seed)
    pred, action, _ = choose(model, x, response, extra, valid, score, ids)
    np.testing.assert_array_equal(action, 0)
    np.testing.assert_allclose(pred, reader.baseline[ids], atol=1e-6)
    direct = candidates(reader, ids)
    err = float(abs(direct-score[ids]).max())
    assert err < 3e-5, err
    # Independently exercise the actual single-action gate path.
    dummy = previous.make_model(reader.seed)
    replay = []
    for a in (1, 9, 10, 18):
        gate = torch.zeros((len(ids), 2, 3, 3), device='cuda')
        gate.reshape(len(ids), 18)[:, a-1] = 1
        with torch.no_grad():
            actual = reader.forward(ids, dummy, override=gate).cpu().numpy()
        replay.append(float(abs(actual-score[ids, a]).max()))
    assert max(replay) < 3e-5, replay
    mask = valid[ids].clone(); mask[:, 1:] = False
    masked = model(x[ids], response[ids], extra[ids], mask).softmax(-1)
    assert float(masked[:, 0].min()) == 1 and float(masked[:, 1:].sum()) == 0
    # Finite difference a learned weight against the expected-risk objective.
    y = data.table.iloc[:data.n].class_id.to_numpy(int)
    _, risk, _ = risk_targets(score, y, data.tr)
    loss = (model(x[ids], response[ids], extra[ids], valid[ids]).softmax(-1)*risk[ids]).sum(-1).mean()
    loss.backward()
    grad = model.output.weight.grad
    col = int(grad.abs().argmax()); auto = float(grad[0, col])
    vals = []
    with torch.no_grad():
        for d in (.001, -.001):
            model.output.weight[0, col] = d
            vals.append(float((model(x[ids], response[ids], extra[ids], valid[ids]).softmax(-1)*risk[ids]).sum(-1).mean()))
    numerical = (vals[0]-vals[1])/.002
    assert abs(auto-numerical) < max(2e-4, .03*abs(auto)), (auto, numerical)
    assert all(p.grad is None for p in reader.old.frozen.parameters())
    return dict(cache_replay=err, full_action_replay=max(replay), gradient_auto=auto, gradient_numeric=numerical)


def main(check_only=False):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for folder in ('models', 'results', 'analysis'):
        (HERE/folder).mkdir(exist_ok=True)
    data = Data(); receipts = []; configs = {}; checks_all = {}; costs = []
    for seed in SEEDS:
        reader = Reader(data, seed)
        x, extra, valid = features(reader)
        ids = np.r_[data.tr, data.va]
        with np.load(PREVIOUS/'results'/f'source_interventions_{seed}.npz') as z:
            np.testing.assert_array_equal(z['ids'], ids)
            score = np.zeros((data.n, 19, 8), np.float32)
            score[:, 0] = reader.baseline
            score[ids, 1:] = z['score']
        response, scaler = response_features(score, data.tr)
        checks_all[seed] = checks(data, reader, score, x, response, extra, valid)
        write_json(HERE/'checks.json', checks_all)
        if check_only:
            print(checks_all, flush=True); return
        configs[seed] = dict(fixed_action=fixed_choice(data, score, valid))
        np.savez_compressed(HERE/'results'/f'inputs_{seed}.npz', x=x.cpu().numpy(), extra=extra.cpu().numpy(),
            valid=valid.cpu().numpy(), response_mu=scaler[0], response_sd=scaler[1], up_index=data.up[:data.n])
        for arm in LEARNED:
            receipts.append(fit(data, seed, arm, score, x, response, extra, valid))
            pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv', index=False)
        # Save source candidates only; no development actions generated before all fits freeze.
        np.savez_compressed(HERE/'results'/f'source_candidates_{seed}.npz', ids=ids, score=score[ids])
        del reader; torch.cuda.empty_cache()
    write_json(HERE/'frozen_selection.json', configs)
    rows = []
    for seed in SEEDS:
        reader = Reader(data, seed)
        x, extra, valid = features(reader)
        with np.load(HERE/'results'/f'inputs_{seed}.npz') as z:
            scaler = (z['response_mu'], z['response_sd'])
        source = np.load(HERE/'results'/f'source_candidates_{seed}.npz')
        score = np.zeros((data.n, 19, 8), np.float32)
        score[source['ids']] = source['score']
        candidates(reader, data.te[:8])
        torch.cuda.synchronize(); tick = time.perf_counter()
        score[data.te] = candidates(reader, data.te)
        torch.cuda.synchronize(); elapsed = time.perf_counter()-tick
        costs.append(dict(seed=seed, stage='18_endpoints_from_precomputed_kernels', n=len(data.te), seconds=elapsed))
        response, _ = response_features(score, data.tr, scaler)
        np.savez_compressed(HERE/'results'/f'candidates_{seed}.npz', score=score, up_index=data.up[:data.n])
        for arm in ('old_full',) + ARMS:
            if arm == 'old_full':
                action = np.zeros(data.n, dtype=int)
            elif arm == 'fixed_action':
                a = configs[seed]['fixed_action']
                action = np.where(valid[:, a].cpu().numpy(), a, 0)
            else:
                ck = torch.load(HERE/'models'/f'{arm}_{seed}.pt', map_location='cpu', weights_only=False)
                model = make_model(arm, seed); model.load_state_dict(ck['state']); model.eval()
                choose(model, x, response, extra, valid, score, data.te[:8])
                torch.cuda.synchronize(); tick = time.perf_counter()
                _, action, _ = choose(model, x, response, extra, valid, score, np.arange(data.n))
                torch.cuda.synchronize()
                costs.append(dict(seed=seed, stage=arm+'_selector_only', n=data.n, seconds=time.perf_counter()-tick))
            pred = score[np.arange(data.n), action]
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz', score=pred, action=action, up_index=data.up[:data.n])
            for role, ii in (('utility_train', data.tr), ('selection', data.va), ('evaluation', data.te)):
                rows.append(dict(seed=seed, arm=arm, role=role, action_fraction=float((action[ii] != 0).mean()),
                    **metrics(data.table.iloc[ii].class_id.to_numpy(int), pred[ii])))
        # Old cached-feature forward, not a cache lookup, for a bounded cost comparison.
        ii = torch.tensor(reader.old.ix[data.te], device='cuda')
        reader.old.old.forward(ii[:8], reader.old.frozen)
        torch.cuda.synchronize(); tick = time.perf_counter()
        with torch.no_grad():
            base = np.concatenate([reader.old.old.forward(ii[j:j+64], reader.old.frozen).cpu().numpy()
                                   for j in range(0, len(ii), 64)])
        torch.cuda.synchronize()
        costs.append(dict(seed=seed, stage='old_m3_from_cached_features', n=len(ii), seconds=time.perf_counter()-tick))
        np.testing.assert_allclose(base, score[data.te, 0], atol=3e-5, rtol=1e-5)
        del reader; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv', index=False)
    pd.DataFrame(costs).to_csv(HERE/'inference_cost.csv', index=False)
    write_json(HERE/'complete.json', dict(fits=len(receipts), note='All selections frozen before development endpoint evaluation.'))
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean().to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    main(parser.parse_args().check_only)
