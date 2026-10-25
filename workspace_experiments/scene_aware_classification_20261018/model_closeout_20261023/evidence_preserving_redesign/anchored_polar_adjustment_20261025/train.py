"""Bounded query-dependent changes around source-selected polar fusion B."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
from pathlib import Path
import importlib.util
import time
HERE = Path(__file__).resolve().parent
PREF = HERE.parent / 'polar_preference_20261025'
spec = importlib.util.spec_from_file_location('anchor_previous', PREF / 'train.py')
prev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prev)
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
Data, metrics, write_json, load = prev.Data, prev.metrics, prev.write_json, prev.load
BASE, LOCAL, SEEDS, STEPS = prev.BASE, prev.LOCAL, prev.SEEDS, prev.STEPS
ARMS = ('constant', 'spatial', 'spatial_wide', 'frequency')
RADIUS = .25
inputs, folds = prev.inputs, prev.folds


def choose_alpha(z, y, ids):
    a, grid = prev.prev.common.previous.select(
        (z['base']+z['delta_vv'])[ids].numpy(),
        (z['base']+z['delta_vh'])[ids].numpy(), y[ids])
    return float(a), grid


class Gate(nn.Module):
    def __init__(self, arm, alpha):
        super().__init__()
        self.arm, self.alpha = arm, float(alpha)
        self.dim = 185 if arm == 'frequency' else 41
        if arm == 'constant':
            self.logit = nn.Parameter(torch.zeros(()))
        else:
            width = 52 if arm == 'spatial_wide' else 12
            self.net = nn.Sequential(nn.Linear(self.dim, width), nn.GELU(), nn.Linear(width, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, available):
        r = self.logit.expand(len(x)) if self.arm == 'constant' else self.net(x[..., :self.dim]).mean((1, 2))
        alpha = (self.alpha+RADIUS*r.tanh()).clamp(0, 1)
        alpha = torch.where(available.bool().all(1), alpha, torch.full_like(alpha, self.alpha))
        weight = torch.stack((alpha, 1-alpha), 1)*available
        return weight, alpha, r


def model_for(arm, seed, alpha):
    torch.manual_seed(seed)
    return Gate(arm, alpha)


def prediction(model, x, z, ids):
    g, alpha, r = model(x[ids], z['available'][ids])
    score = z['base'][ids]+g[:, :1]*z['delta_vv'][ids]+g[:, 1:]*z['delta_vh'][ids]
    return score, g, alpha, r


def train_steps(arm, seed, raw, z, y, tr, va, alpha, total=100):
    x, scaler = prev.prev.normalize(raw, tr)
    model = model_for(arm, seed, alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    count = np.bincount(y[tr], minlength=8)
    assert (count > 0).all()
    wt = torch.tensor(.5/len(tr)+.5/(8*count[y[tr]]), dtype=torch.float32)
    yt = torch.tensor(y[tr], dtype=torch.long)
    snapshots, curve = {}, []
    for step in range(total+1):
        if step:
            opt.zero_grad(set_to_none=True)
            score = prediction(model, x, z, tr)[0]
            loss = (F.cross_entropy(score, yt, reduction='none')*wt).sum()
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
        if step in STEPS or step == total:
            with torch.no_grad():
                vals = prediction(model, x, z, va)
                snapshots[step] = tuple(v.numpy().copy() for v in vals)
            curve.append(dict(step=step, train_loss=None if not step else float(loss.detach()),
                              alpha_anchor=alpha, **metrics(y[va], snapshots[step][0])))
    return model, scaler, snapshots, curve


def checks(data, raw, z, alpha):
    ids = data.va
    x, _ = prev.prev.normalize(raw, ids)
    model = model_for('frequency', 17, alpha)
    score, weights, a, _ = prediction(model, x, z, ids)
    expected = z['base'][ids]+alpha*z['delta_vv'][ids]+(1-alpha)*z['delta_vh'][ids]
    err = float((score-expected).abs().max().detach())
    assert err < 3e-6, err
    test = model_for('frequency', 17, .6)
    yi = torch.tensor(data.table.iloc[ids].class_id.to_numpy(), dtype=torch.long)
    loss = F.cross_entropy(prediction(test, x, z, ids)[0], yi)
    loss.backward()
    grad = test.net[-1].weight.grad
    ij = np.unravel_index(int(grad.abs().argmax()), grad.shape)
    auto = float(grad[ij]); vals = []
    assert abs(auto) > 1e-8
    with torch.no_grad():
        for sign in (1, -1):
            test.net[-1].weight[ij] = sign*.001
            vals.append(float(F.cross_entropy(prediction(test, x, z, ids)[0], yi)))
        test.net[-1].weight[ij] = 0
        test.net[-1].bias.fill_(5)
        g, aa, _ = test(x, z['available'])
        paired = z['available'].bool().all(1)
        assert torch.allclose(g[paired].sum(1), torch.ones_like(aa[paired]))
        assert torch.all(aa[~paired] == .6)
        assert torch.all(g[~z['available'].bool()] == 0)
        assert abs(aa-.6).max() <= RADIUS+1e-6
        assert g.min() >= 0 and g.max() <= 1
    numeric = (vals[0]-vals[1])/.002
    assert abs(auto-numeric) < max(.0003, .05*abs(auto))
    inward = []
    for anchor, sign in ((0., -1.), (1., 1.)):
        edge = model_for('constant', 17, anchor)
        opt = torch.optim.SGD(edge.parameters(), lr=.1)
        _, aa, _ = edge(x[:2], torch.ones((2, 2)))
        (sign*aa.mean()).backward(); opt.step()
        moved = float(edge(x[:2], torch.ones((2, 2)))[1].mean().detach())
        assert 0 < moved < 1
        inward.append(dict(anchor=anchor, moved=moved))
    return dict(initial_replay_error=err, gradient_auto=auto, gradient_numeric=numeric,
                missing_unchanged=True, inward_endpoint_steps=inward,
                parameters={a: sum(p.numel() for p in model_for(a,17,alpha).parameters()) for a in ARMS})


def restore(seed, arm, raw):
    ck = torch.load(HERE/'models'/f'{arm}_{seed}.pt', map_location='cpu', weights_only=False)
    model = model_for(arm, seed, ck['receipt']['alpha_anchor'])
    model.load_state_dict(ck['state']); model.eval()
    sc = ck['scaler']
    x = torch.tensor(np.where(sc['active'], (raw-sc['mu'])/sc['scale'], 0).astype(np.float32))
    return model, x, ck


def main():
    torch.set_num_threads(2)
    for folder in ('models', 'results', 'analysis'):
        (HERE/folder).mkdir(exist_ok=True)
    data = Data(); a, b = folds(data)
    y = data.table.iloc[:data.n].class_id.to_numpy(int)
    receipts, cv_rows, grid_rows, all_checks = [], [], [], {}
    for seed in SEEDS:
        raw, z = inputs(data, seed)
        anchors = {}
        for name, tr in (('A_to_B', a), ('B_to_A', b), ('all_source', data.va)):
            anchors[name], grid = choose_alpha(z, y, tr)
            grid_rows.extend(dict(seed=seed, fold=name, **r) for r in grid)
        alpha = anchors['all_source']
        all_checks[seed] = checks(data, raw, z, alpha)
        bscore = z['base']+alpha*z['delta_vv']+(1-alpha)*z['delta_vh']
        old = prev.prev.common.scores(data, seed)['hybrid_source']
        replay = float(abs(bscore.numpy()-old).max()); assert replay < 3e-6
        all_checks[seed]['historical_B_replay'] = replay
        for arm in ARMS:
            tick = time.perf_counter()
            oof = {s: [np.zeros((data.n,8),np.float32), np.zeros((data.n,2),np.float32),
                       np.zeros(data.n,np.float32), np.zeros(data.n,np.float32)] for s in STEPS}
            for name, tr, va in (('A_to_B', a, b), ('B_to_A', b, a)):
                _, _, snapshots, curve = train_steps(arm,seed,raw,z,y,tr,va,anchors[name])
                for step, values in snapshots.items():
                    for dst, value in zip(oof[step], values):
                        dst[va] = value
                pd.DataFrame(curve).to_csv(HERE/'results'/f'curve_{arm}_{seed}_{name}.csv', index=False)
            selected = 0
            best = prev.prev.common.exact_j(y[data.va], oof[0][0][data.va].argmax(1))
            for step in STEPS:
                j = prev.prev.common.exact_j(y[data.va], oof[step][0][data.va].argmax(1))
                cv_rows.append(dict(seed=seed, arm=arm, step=step, **metrics(y[data.va],oof[step][0][data.va])))
                if j > best:
                    selected, best = step, j
            model, scaler, _, curve = train_steps(arm,seed,raw,z,y,data.va,data.va,alpha,total=selected)
            receipt = dict(seed=seed,arm=arm,selected_step=selected,alpha_anchor=alpha,fold_anchors=anchors,
                           source_cv_J=float(best),parameters=sum(p.numel() for p in model.parameters()),
                           seconds=time.perf_counter()-tick)
            torch.save(dict(state=model.state_dict(),scaler=scaler,receipt=receipt),HERE/'models'/f'{arm}_{seed}.pt')
            np.savez_compressed(HERE/'results'/f'source_cv_{arm}_{seed}.npz',
                **{k:v[data.va] for k,v in zip(('score','weight','alpha','logit'),oof[selected])},
                baseline=oof[0][0][data.va], up_index=data.up[data.va])
            pd.DataFrame(curve).to_csv(HERE/'results'/f'curve_{arm}_{seed}_refit.csv',index=False)
            receipts.append(receipt); print('FIT',receipt,flush=True)
    pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
    pd.DataFrame(cv_rows).to_csv(HERE/'source_cv.csv',index=False)
    pd.DataFrame(grid_rows).to_csv(HERE/'source_alpha_grid.csv',index=False)
    write_json(HERE/'checks.json',all_checks)
    write_json(HERE/'source_choices.json',dict(choices=receipts,development_metrics_computed=False,
        limitation='Historical source gate-level grouped selection; not independent full-pipeline OOF.'))
    rows = []
    for seed in SEEDS:
        raw,z = inputs(data,seed)
        for arm in ARMS:
            model,x,ck = restore(seed,arm,raw)
            with torch.no_grad():
                values = tuple(v.numpy() for v in prediction(model,x,z,np.arange(data.n)))
                scores, weight, alpha, logits = values
                diagnostics = {}
                if arm == 'frequency':
                    changed = x.clone(); changed[...,41:] = 0
                    diagnostics['no_frequency'] = prediction(model,changed,z,np.arange(data.n))[0].numpy()
                paired = z['available'][data.va].bool().all(1).numpy()
                mean_alpha = float(alpha[data.va][paired].mean())
                am = np.where(z['available'].bool().all(1).numpy(),mean_alpha,model.alpha)
                gm = torch.tensor(np.stack((am,1-am),1),dtype=torch.float32)*z['available']
                diagnostics['source_mean'] = (z['base']+gm[:,:1]*z['delta_vv']+gm[:,1:]*z['delta_vh']).numpy()
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',
                **dict(zip(('score','weight','alpha','logit'),values)),up_index=data.up[:data.n])
            np.savez_compressed(HERE/'results'/f'diagnostics_{arm}_{seed}.npz',**diagnostics,up_index=data.up[:data.n])
            for role,ids in (('source_refit',data.va),('evaluation',data.te)):
                rows.append(dict(seed=seed,arm=arm,role=role,**metrics(y[ids],scores[ids])))
            for name,s in diagnostics.items():
                rows.append(dict(seed=seed,arm=arm+'__'+name,role='evaluation',**metrics(y[data.te],s[data.te])))
        for arm in ('base','old_full','prior_late','hybrid_equal','hybrid_source'):
            s = prev.prev.common.scores(data,seed)[arm]
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',score=s,up_index=data.up[:data.n])
            rows.append(dict(seed=seed,arm=arm,role='evaluation',**metrics(y[data.te],s[data.te])))
    frame = pd.DataFrame(rows); frame.to_csv(HERE/'metrics.csv',index=False)
    frame.query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].agg(['mean','std']).to_csv(HERE/'summary.csv')
    write_json(HERE/'complete.json',dict(status='complete',cv_fits=24,final_refits=12,source=612,development=485,
                                      all_models_frozen_before_development=True))
    print(frame.query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean().to_string(),flush=True)


if __name__ == '__main__':
    main()
