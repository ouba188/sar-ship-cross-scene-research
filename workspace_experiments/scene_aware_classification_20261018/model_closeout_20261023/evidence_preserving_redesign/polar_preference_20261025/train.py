"""Separate common correction strength from supervised polar preference."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
from pathlib import Path
import importlib.util
import time
import json
HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'query_polar_gate_20261025'
sys.path.insert(0, str(PREVIOUS))
spec = importlib.util.spec_from_file_location('previous_gate_train', PREVIOUS / 'train.py')
prev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prev)
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
Data, metrics, write_json, load = prev.Data, prev.metrics, prev.write_json, prev.load
BASE, LOCAL, SEEDS = prev.BASE, prev.LOCAL, prev.SEEDS
ARMS = ('common', 'ce', 'aux', 'aux_wide', 'aux_freq', 'globalpref')
STEPS = prev.STEPS
COMPARATORS = ('old_full', 'prior_late', 'hybrid_equal', 'hybrid_source', 'base', 'spatial', 'frequency')


def inputs(data, seed):
    return prev.inputs(data, seed)


def folds(data):
    source = pd.read_csv(PREVIOUS / 'source_folds.csv').set_index('up_index')
    names = source.loc[data.up[data.va], 'fold'].to_numpy()
    a, b = data.va[names == 'A'], data.va[names == 'B']
    for field in ('mmsi', 'pass_group'):
        assert not set(data.table.iloc[a][field]) & set(data.table.iloc[b][field])
    return a, b


class Gate(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.dim = 185 if arm in ('aux_freq', 'conflict_freq') else 41
        width = 51 if arm == 'aux_wide' else 12
        outputs = 1 if arm in ('common', 'globalpref') else 2
        self.net = nn.Sequential(nn.Linear(self.dim, width), nn.GELU(), nn.Linear(width, outputs))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if arm == 'globalpref':
            self.preference = nn.Parameter(torch.zeros(()))

    def forward(self, x, available, mode=None):
        logits = self.net(x[..., :self.dim]).mean(1)
        t = logits[:, 0].sigmoid()
        r = (self.preference.expand_as(t) if self.arm == 'globalpref' else
             torch.zeros_like(t) if self.arm == 'common' else logits[:, 1])
        if mode == 'no_preference':
            r = torch.zeros_like(r)
        if mode == 'fixed_strength':
            t = torch.full_like(t, .5)
        d = (r/2).tanh()
        span = torch.minimum(t, 1-t)
        weight = torch.stack((t+span*d, t-span*d), 1)*available
        return weight, t, r


def model_for(arm, seed):
    torch.manual_seed(seed)
    return Gate(arm)


def prediction(model, x, z, ids, mode=None):
    g, t, r = model(x[ids], z['available'][ids], mode)
    score = z['base'][ids]+g[:, :1]*z['delta_vv'][ids]+g[:, 1:]*z['delta_vh'][ids]
    return score, g, t, r


def targets(z, y, ids):
    labels = torch.tensor(y[ids], dtype=torch.long)
    lv = F.cross_entropy((z['base']+z['delta_vv'])[ids], labels, reduction='none')
    lh = F.cross_entropy((z['base']+z['delta_vh'])[ids], labels, reduction='none')
    q = (lh-lv).sigmoid().detach()
    confidence = abs(2*q-1)*z['available'][ids].prod(1)
    return labels, q, confidence


def train_steps(arm, seed, raw, z, y, tr, va, total=100):
    x, scaler = prev.normalize(raw, tr)
    model = model_for(arm, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    count = np.bincount(y[tr], minlength=8)
    assert (count > 0).all()
    wt = torch.tensor(.5/len(tr)+.5/(8*count[y[tr]]), dtype=torch.float32)
    yt, q, confidence = targets(z, y, tr)
    if arm.startswith('conflict_'):
        correct_v = (z['base']+z['delta_vv'])[tr].argmax(1) == yt
        correct_h = (z['base']+z['delta_vh'])[tr].argmax(1) == yt
        confidence = ((correct_v != correct_h) & z['available'][tr].all(1)).float()
        q = correct_v.float()
        assert confidence.sum() > 0
    pw = wt*confidence
    pw = pw/pw.sum().clamp_min(1e-8)
    snapshots, curve = {}, []
    for step in range(total+1):
        if step:
            opt.zero_grad(set_to_none=True)
            score, _, _, r = prediction(model, x, z, tr)
            closs = (F.cross_entropy(score, yt, reduction='none')*wt).sum()
            ploss = (F.binary_cross_entropy_with_logits(r, q, reduction='none')*pw).sum()
            loss = closs + (.25*ploss if arm in ('aux', 'aux_wide', 'aux_freq', 'globalpref') or arm.startswith('conflict_') else 0)
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
        if step in STEPS or step == total:
            with torch.no_grad():
                values = prediction(model, x, z, va)
                snapshots[step] = tuple(v.numpy().copy() for v in values)
            curve.append(dict(step=step, classification_loss=None if not step else float(closs.detach()),
                preference_loss=None if not step else float(ploss.detach()), **metrics(y[va], snapshots[step][0])))
    return model, scaler, snapshots, curve


def checks(data, raw, z):
    ids = data.va[:32]
    x, _ = prev.normalize(raw, data.va)
    model = model_for('aux', 17)
    s, g, t, r = prediction(model, x, z, ids)
    expected = z['base'][ids]+.5*z['available'][ids, :1]*z['delta_vv'][ids]+.5*z['available'][ids, 1:]*z['delta_vh'][ids]
    err = float((s-expected).abs().max().detach()); assert err < 2e-6
    y = data.table.iloc[:data.n].class_id.to_numpy(int)
    yt, q, confidence = targets(z, y, ids)
    assert torch.all(confidence[z['available'][ids].prod(1) == 0] == 0)
    # A positive loss advantage must favor VV; differentiate the complete task loss.
    loss = F.cross_entropy(s, yt)
    loss.backward()
    grad = model.net[-1].weight.grad
    ij = np.unravel_index(int(grad.abs().argmax()), grad.shape)
    auto = float(grad[ij]); vals = []
    with torch.no_grad():
        for sign in (1, -1):
            model.net[-1].weight[ij] = sign*.001
            vals.append(float(F.cross_entropy(prediction(model, x, z, ids)[0], yt)))
        model.net[-1].weight[ij] = 0
        model.net[-1].bias[1] = 2
        gg, _, _ = model(x, z['available'])
        paired = z['available'].prod(1).bool()
        assert (gg[paired, 0] > gg[paired, 1]).all()
        assert gg.min() >= 0 and gg.max() <= 1
        assert (gg[~z['available'].bool()] == 0).all()
    numeric = (vals[0]-vals[1])/.002
    assert abs(auto-numeric) < max(.0003, .05*abs(auto))
    return dict(initial_replay_error=err, gradient_auto=auto, gradient_numeric=numeric,
        positive_preference_favors_VV=True, missing_polar_mask=True,
        parameters={a: sum(p.numel() for p in model_for(a,17).parameters()) for a in ARMS})


def restore(seed, arm, raw):
    ck = torch.load(HERE / 'models' / f'{arm}_{seed}.pt', map_location='cpu', weights_only=False)
    model = model_for(arm, seed); model.load_state_dict(ck['state']); model.eval()
    sc = ck['scaler']
    x = torch.tensor(np.where(sc['active'], (raw-sc['mu'])/sc['scale'], 0).astype(np.float32))
    return model, x, ck


def main():
    torch.set_num_threads(2)
    for folder in ('models', 'results', 'analysis'):
        (HERE / folder).mkdir(exist_ok=True)
    data = Data(); a, b = folds(data)
    y = data.table.iloc[:data.n].class_id.to_numpy(int)
    receipts, rows, fold_rows, all_checks = [], [], [], {}
    for seed in SEEDS:
        raw, z = inputs(data, seed)
        all_checks[seed] = checks(data, raw, z)
        for arm in ARMS:
            start = time.perf_counter()
            oof = {s: [np.zeros((data.n,8),np.float32), np.zeros((data.n,2),np.float32),
                       np.zeros(data.n,np.float32), np.zeros(data.n,np.float32)] for s in STEPS}
            for fold, tr, va in (('A_to_B',a,b),('B_to_A',b,a)):
                _, _, snapshots, curve = train_steps(arm,seed,raw,z,y,tr,va)
                for step, vals in snapshots.items():
                    for dest, value in zip(oof[step], vals):
                        dest[va] = value
                    fold_rows.append(dict(seed=seed,arm=arm,fold=fold,step=step,**metrics(y[va], vals[0])))
                pd.DataFrame(curve).to_csv(HERE/'results'/f'curve_{arm}_{seed}_{fold}.csv',index=False)
            selected = 0
            best = prev.common.exact_j(y[data.va], oof[0][0][data.va].argmax(1))
            for step in STEPS:
                vals = oof[step]
                j = prev.common.exact_j(y[data.va], vals[0][data.va].argmax(1))
                rows.append(dict(seed=seed,arm=arm,step=step,**metrics(y[data.va],vals[0][data.va])))
                if j > best:
                    selected,best = step,j
            model,scaler,_,curve = train_steps(arm,seed,raw,z,y,data.va,data.va,total=selected)
            receipt = dict(seed=seed,arm=arm,selected_step=selected,source_cv_J=float(best),
                           parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start)
            torch.save(dict(state=model.state_dict(),scaler=scaler,receipt=receipt),HERE/'models'/f'{arm}_{seed}.pt')
            np.savez_compressed(HERE/'results'/f'source_cv_{arm}_{seed}.npz',
                **{k:v[data.va] for k,v in zip(('score','weight','strength','preference'),oof[selected])},up_index=data.up[data.va])
            receipts.append(receipt)
            print('FIT',receipt,flush=True)
    pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
    pd.DataFrame(rows).to_csv(HERE/'source_cv.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(HERE/'source_cv_folds.csv',index=False)
    write_json(HERE/'checks.json',all_checks)
    write_json(HERE/'source_choices.json',dict(choices=receipts,development_metrics_computed=False,
        limitation='Gate-level cross-selection on historical early-stop source; not clean full-pipeline OOF.'))
    result_rows = []
    for seed in SEEDS:
        raw,z = inputs(data,seed)
        for arm in ARMS:
            model,x,ck = restore(seed,arm,raw)
            with torch.no_grad():
                vals = tuple(v.numpy() for v in prediction(model,x,z,np.arange(data.n)))
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',
                **dict(zip(('score','weight','strength','preference'),vals)),up_index=data.up[:data.n])
            result_rows.append(dict(seed=seed,arm=arm,**metrics(y[data.te],vals[0][data.te])))
            diagnostics = {}
            with torch.no_grad():
                for mode in ('no_preference','fixed_strength'):
                    diagnostics[mode] = prediction(model,x,z,np.arange(data.n),mode)[0].numpy()
                if arm == 'aux_freq':
                    changed = x.clone(); changed[...,41:]=0
                    diagnostics['no_frequency'] = prediction(model,changed,z,np.arange(data.n))[0].numpy()
            np.savez_compressed(HERE/'results'/f'diagnostics_{arm}_{seed}.npz',**diagnostics,up_index=data.up[:data.n])
        for arm in COMPARATORS:
            with np.load(PREVIOUS/'results'/f'{arm}_{seed}.npz') as f:
                np.testing.assert_array_equal(f['up_index'],data.up[:data.n])
                score=f['score'].copy()
            result_rows.append(dict(seed=seed,arm=arm,**metrics(y[data.te],score[data.te])))
    frame=pd.DataFrame(result_rows);frame.to_csv(HERE/'metrics.csv',index=False)
    frame.groupby('arm')[['OA','BA','NLL','Brier']].agg(['mean','std']).to_csv(HERE/'summary.csv')
    write_json(HERE/'complete.json',dict(status='complete',cv_fits=36,final_refits=18,source=612,development=485,
        all_models_frozen_before_development=True))
    print(frame.groupby('arm')[['OA','BA','NLL','Brier']].mean().to_string(),flush=True)


if __name__ == '__main__':
    main()
