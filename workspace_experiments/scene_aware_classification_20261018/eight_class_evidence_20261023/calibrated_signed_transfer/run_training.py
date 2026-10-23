"""Fixed eight-class calibration followed by frozen-baseline residual learning."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '2')
import argparse
import json
import time
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
SEEDS = [17, 29, 43]
ARMS = ['signed_only', 'conditional_std', 'score_only']


class Head(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        width = {'score_only': 8, 'signed_only': 5, 'conditional_std': 3}[arm]
        self.net = nn.Sequential(nn.Linear(width, 16), nn.GELU(), nn.Linear(16, 8 if arm == 'score_only' else 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, moments):
        value = self.net(x)
        if self.arm == 'score_only':
            return value
        delta = value.tanh().squeeze(-1)*moments[..., 1 if self.arm == 'signed_only' else 0]
        return delta-delta.mean(1, keepdim=True)


def raw_input(arm, base, moments):
    centered = base-base.mean(1, keepdims=True)
    if arm == 'score_only':
        return centered
    sigma, kappa = moments[..., 0], moments[..., 1]
    cols = [centered, sigma, sigma-sigma.mean(1, keepdims=True)]
    if arm == 'signed_only':
        cols += [kappa, kappa-kappa.mean(1, keepdims=True)]
    return np.stack(cols, -1)


def assessment(y, score):
    pred = score.argmax(1)
    present = np.unique(y)
    lp = score-logsumexp(score, axis=1, keepdims=True)
    loss = -lp[np.arange(len(y)), y]
    brier = ((np.exp(lp)-np.eye(8)[y])**2).sum(1)
    oa = 100*float(np.mean(pred == y))
    ba = 100*float(np.mean([np.mean(pred[y == c] == c) for c in present]))
    return dict(n=len(y), OA=oa, BA=ba, J=(oa+ba)/200, NLL=float(loss.mean()),
                balanced_NLL=float(np.mean([loss[y == c].mean() for c in present])),
                Brier=float(brier.mean()), balanced_Brier=float(np.mean([brier[y == c].mean() for c in present])))


def exact_j(y, pred):
    assert len(np.unique(y)) == 8
    return Fraction(int((pred == y).sum()), 2*len(y))+sum(
        (Fraction(int(((pred == c)&(y == c)).sum()), 16*int((y == c).sum())) for c in range(8)), Fraction())


def joint_loss(score, y):
    loss = nn.functional.cross_entropy(score, y, reduction='none')
    return .5*loss.mean()+.5*torch.stack([loss[y == c].mean() for c in range(8)]).mean()


def frozen_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def load_table():
    table = pd.read_csv(PARENT/'cohort.csv', dtype={'mmsi': str})
    table['pass_group'] = table['product'].map(lambda s: '_'.join([s.split('_')[0], s.split('_')[6], s.split('_')[7]]))
    return table


def load_fold(fold, table, cache):
    keys = ['base', 'oof', 'train', 'validation', 'test', 'instance_ids']
    with np.load(PARENT/'baseline'/f'fold{fold}_inputs.npz') as z:
        d = {k: z[k] for k in keys}
    assert np.array_equal(d['instance_ids'], table.instance_id)
    assert np.array_equal(cache['instance_ids'], table.instance_id)
    names = list(cache['reader_names'])
    outer = cache['moments'][names.index(f'fold{fold}_outer')]
    oof = np.zeros_like(outer)
    count = np.zeros(len(table), int)
    for inner in range(4):
        name = f'fold{fold}_inner{inner}'
        with np.load(PARENT/'baseline/models'/f'{name}.npz') as z:
            held, fit = z['held'], z['fit']
        assert set(held) <= set(d['train']) and set(fit) <= set(d['train'])
        assert not set(table.iloc[fit].mmsi)&set(table.iloc[held].mmsi)
        oof[held] = cache['moments'][names.index(name), held]
        count[held] += 1
    assert (count[d['train']] == 1).all() and count.sum() == len(d['train'])
    return d, outer, oof


def checkpoint_path(stage, fold, seed, arm):
    return HERE/stage/'models'/f'fold{fold}_{arm}_seed{seed}.pt'


def restore(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = Head(ck['arm']).eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck, model


def apply_head(ck, model, base, moments):
    x = raw_input(ck['arm'], base, moments)
    with torch.inference_mode():
        delta = model(torch.tensor((x-ck['mu'])/ck['scale'], dtype=torch.float32),
                      torch.tensor(moments, dtype=torch.float32)).numpy().astype(float)
    return base.astype(float)+delta


def calibrated(fold, seed, d, outer, oof):
    ck, model = restore(checkpoint_path('calibration', fold, seed, 'score_only'))
    return apply_head(ck, model, d['base'], outer), apply_head(ck, model, d['oof'], oof), ck['epoch']


def main(stage, evaluate_only=False):
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out = HERE/stage
    (out/'models').mkdir(parents=True, exist_ok=True)
    table = load_table()
    cache = dict(np.load(HERE/'signed_features.npz'))
    assert json.loads((HERE/'extraction.json').read_text())['status'] == 'complete'
    y = table.class_id.to_numpy()
    yy = torch.tensor(y, dtype=torch.long)
    curves, selections, checks, costs = [], [], [], []
    for fold in ([] if evaluate_only else range(3)):
        d, outer, oof = load_fold(fold, table, cache)
        tr, va = d['train'], d['validation']
        for seed in SEEDS:
            base, train_base, ce = (d['base'], d['oof'], -1) if stage == 'calibration' else calibrated(fold, seed, d, outer, oof)
            for arm in ['score_only'] if stage == 'calibration' else ARMS:
                tick = time.perf_counter()
                raw, rawv = raw_input(arm, train_base[tr], oof[tr]), raw_input(arm, base[va], outer[va])
                axes = 0 if arm == 'score_only' else (0, 1)
                mu, sd = raw.mean(axes), raw.std(axes).clip(1e-8)
                x, xv = [torch.tensor((r-mu)/sd, dtype=torch.float32) for r in [raw, rawv]]
                b, bv = [torch.tensor(v, dtype=torch.float32) for v in [train_base[tr], base[va]]]
                m, mv = [torch.tensor(v, dtype=torch.float32) for v in [oof[tr], outer[va]]]
                torch.manual_seed(seed)
                model = Head(arm)
                parameters = sum(p.numel() for p in model.parameters())
                assert parameters == {'score_only': 280, 'signed_only': 113, 'conditional_std': 81}[arm]
                with torch.no_grad():
                    zero = model(xv, mv).numpy()
                assert not zero.any() and np.array_equal(base[va]+zero, base[va])
                best, best_epoch = frozen_state(model), 0
                best_j = exact_j(y[va], bv.numpy().argmax(1))
                meta = dict(stage=stage, fold=fold, seed=seed, arm=arm)
                trace = [dict(**meta, epoch=0, train_loss=np.nan, **assessment(y[va], bv.numpy().astype(float)))]
                opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                for epoch in range(1, 201):
                    opt.zero_grad()
                    loss = joint_loss(b+model(x, m), yy[tr])
                    assert torch.isfinite(loss)
                    loss.backward()
                    opt.step()
                    if epoch % 5:
                        continue
                    with torch.no_grad():
                        score = (bv+model(xv, mv)).numpy().astype(float)
                    j = exact_j(y[va], score.argmax(1))
                    trace.append(dict(**meta, epoch=epoch, train_loss=float(loss.detach()), **assessment(y[va], score)))
                    if j > best_j:
                        best_j, best_epoch, best = j, epoch, frozen_state(model)
                path = checkpoint_path(stage, fold, seed, arm)
                torch.save(dict(**meta, state=best, mu=mu, scale=sd, epoch=best_epoch, calibrator_epoch=ce,
                                class_count=8, parameters=parameters), path)
                ck, replay = restore(path)
                with torch.inference_mode():
                    replay_score = (bv+replay(xv, mv)).numpy().astype(float)
                chosen = next(r for r in trace if r['epoch'] == best_epoch)
                assert exact_j(y[va], replay_score.argmax(1)) == best_j
                infer_score = apply_head(ck, replay, base[va], outer[va])
                changes = int(np.sum(infer_score.argmax(1) != replay_score.argmax(1)))
                assert changes == 0
                curves.extend(trace)
                selections.append(dict(**meta, epoch=best_epoch, calibrator_epoch=ce, parameters=parameters,
                                       validation_J=float(best_j), validation_NLL=chosen['NLL'], checkpoint=str(path)))
                checks.append(dict(**meta, zero_head_error=0, validation_replay_J=float(best_j),
                                   fp64_inference_top1_changes=changes, fp64_max_error=float(abs(infer_score-replay_score).max())))
                costs.append(dict(**meta, steps=200, seconds=time.perf_counter()-tick))
                print(stage, fold, seed, arm, 'epoch', best_epoch, 'valJ', round(float(best_j), 6), flush=True)
    if evaluate_only:
        assert (out/'selection_frozen.json').exists()
        selections = pd.read_csv(out/'selections.csv').to_dict('records')
        costs = pd.read_csv(out/'costs.csv').to_dict('records')
    else:
        pd.DataFrame(selections).to_csv(out/'selections.csv', index=False)
        pd.DataFrame(curves).to_csv(out/'selection_curves.csv', index=False)
        pd.DataFrame(checks).to_csv(out/'checks.csv', index=False)
        pd.DataFrame(costs).to_csv(out/'costs.csv', index=False)
        (out/'selection_frozen.json').write_text(json.dumps(dict(heads=len(selections), test_evaluated=False)), encoding='utf-8')
    predictions = []
    for fold in range(3):
        d, outer, oof = load_fold(fold, table, cache)
        for seed in SEEDS:
            base = d['base'] if stage == 'calibration' else calibrated(fold, seed, d, outer, oof)[0]
            for arm in ['score_only'] if stage == 'calibration' else ARMS:
                ck, model = restore(checkpoint_path(stage, fold, seed, arm))
                for role in ['validation', 'test']:
                    ix = d[role]
                    score = apply_head(ck, model, base[ix], outer[ix])
                    assert np.isfinite(score).all()
                    g = table.iloc[ix][['instance_id', 'class_id', 'class_name', 'mmsi', 'product', 'port', 'pass_group']].copy()
                    for k, v in dict(stage=stage, fold=fold, seed=seed, arm=arm, epoch=ck['epoch'], role=role).items():
                        g[k] = v
                    g['valid_vv'], g['valid_vh'] = cache['valid'][ix].T
                    g['prediction'], g['base_prediction'], g['A_prediction'] = score.argmax(1), base[ix].argmax(1), d['base'][ix].argmax(1)
                    g['net'] = (g.prediction == g.class_id).astype(int)-(g.base_prediction == g.class_id).astype(int)
                    g['outcome'] = np.select([g.net.eq(1), g.net.eq(-1), g.prediction.eq(g.class_id)], ['rescued', 'harmed', 'CC'], default='WW')
                    for c in range(8):
                        g[f's{c}'], g[f'b{c}'], g[f'a{c}'] = score[:, c], base[ix, c], d['base'][ix, c]
                    predictions.append(g)
    p = pd.concat(predictions, ignore_index=True)
    p.to_csv(out/'predictions.csv', index=False)
    pooled = []
    for (role, arm, seed), g in p.groupby(['role', 'arm', 'seed']):
        if role == 'test':
            assert g.instance_id.is_unique and len(g) == 4744 and (~g.valid_vh).sum() == 308
        y = g.class_id.to_numpy()
        before = assessment(y, g[[f'b{c}' for c in range(8)]].to_numpy())
        after = assessment(y, g[[f's{c}' for c in range(8)]].to_numpy())
        pooled.append(dict(stage=stage, role=role, arm=arm, seed=seed, net=int(g.net.sum()),
                           rescued=int(g.net.eq(1).sum()), harmed=int(g.net.eq(-1).sum()), unique_observations=g.instance_id.nunique(), **after,
                           baseline_OA=before['OA'], baseline_BA=before['BA'],
                           delta_OA=after['OA']-before['OA'], delta_BA=after['BA']-before['BA']))
    pd.DataFrame(pooled).to_csv(out/'pooled.csv', index=False)
    summary = dict(stage=stage, trajectories=len(selections), steps=len(selections)*200,
                   train_seconds=sum(r['seconds'] for r in costs), invocation_seconds=time.perf_counter()-start,
                   evaluation_only=evaluate_only,
                   classes=8, test_selection=False, second_level_OOF=False, C_frozen=stage == 'increment',
                   zero_epoch_heads=sum(r['epoch'] == 0 for r in selections))
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.DataFrame(pooled).query("role == 'test'").to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['calibration', 'increment'])
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    main(args.stage, args.evaluate_only)
