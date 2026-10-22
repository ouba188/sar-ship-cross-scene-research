"""Freeze the effective score calibrator and train matched supplemental heads."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'source_utility_alignment/matched_controls'))
import run_controls as ref

CONFIGS = {
    'new': (ref.parent.NEW, ROOT, ROOT/'source_utility_alignment/matched_controls'),
    'old': (ROOT/'old_partition_replication', ROOT/'old_partition_replication', ROOT/'old_partition_replication'),
}
ARMS = ['signed_only', 'conditional_std', 'score_only']
SEEDS = [17, 29, 43]


def load_fold(partition, fold, seed):
    inputs_path, cache_path, models_path = CONFIGS[partition]
    ref.parent.NEW = inputs_path
    cache = np.load(cache_path/'signed_features.npz')
    d, outer, oof, _, _, _ = ref.inputs(fold, cache, list(cache['reader_names']))
    ck = torch.load(models_path/'models'/f'fold{fold}_score_only_J_seed{seed}.pt', map_location='cpu', weights_only=False)
    calibrator = ref.ScoreOnly().eval().requires_grad_(False)
    calibrator.load_state_dict(ck['state'])
    def apply(score):
        raw = score-score.mean(1, keepdims=True)
        with torch.inference_mode():
            delta = calibrator(torch.tensor((raw-ck['mu'])/ck['scale'], dtype=torch.float32), None).numpy().astype(float)
        return score.astype(float)+delta
    d = {**d, 'raw_base': d['base'], 'base': apply(d['base']), 'oof': apply(d['oof'])}
    return d, outer, oof, ck['epoch']


def main():
    assert (HERE/'PROTOCOL_ZH.md').exists()
    start = time.perf_counter()
    torch.set_num_threads(2)
    (HERE/'models').mkdir(exist_ok=True)
    table = pd.read_csv(ref.parent.EXP/'readout/targets.csv', dtype={'mmsi': str})
    y = table.class_id.to_numpy()
    yy = torch.tensor(y, dtype=torch.long)
    selected, curves, checks, costs = [], [], [], []
    for partition, (_, _, models_path) in CONFIGS.items():
        original = pd.read_csv(models_path/'predictions.csv')
        original = original[original.arm.eq('score_only') & original.selector.eq('J')]
        for fold in range(6):
            for seed in SEEDS:
                d, outer, oof, ce = load_fold(partition, fold, seed)
                tr, va, te = [d[k] for k in ['train', 'validation', 'test']]
                expected = original[original.fold.eq(fold) & original.seed.eq(seed)].set_index('instance_id').loc[table.iloc[te].instance_id]
                error = float(abs(expected[['s0','s1','s2']].to_numpy()-d['base'][te]).max())
                assert error < 2e-6
                assert np.array_equal(expected.prediction.to_numpy(), d['base'][te].argmax(1))
                checks.append(dict(partition=partition, fold=fold, seed=seed, check='strong_baseline', error=error))
                b, bv = [torch.tensor(d[k], dtype=torch.float32) for k in ['oof', 'base']]
                m, mv = [torch.tensor(v, dtype=torch.float32) for v in [oof, outer]]
                for arm in ARMS:
                    tick = time.perf_counter()
                    raw = ref.raw_input(arm, d['oof'], oof)
                    rawv = ref.raw_input(arm, d['base'], outer)
                    axes = 0 if arm == 'score_only' else (0, 1)
                    mu, sd = raw[tr].mean(axes), raw[tr].std(axes).clip(1e-8)
                    x, xv = [torch.tensor((r-mu)/sd, dtype=torch.float32) for r in [raw, rawv]]
                    torch.manual_seed(seed)
                    model = ref.make_model(arm)
                    with torch.no_grad():
                        assert model(xv[va], mv[va]).abs().max() == 0
                    value = ref.exact_j(y[va], bv[va].numpy().argmax(1))
                    best, epoch_best = ref.frozen_state(model), 0
                    meta = dict(partition=partition, fold=fold, seed=seed, arm=arm)
                    curves.append(dict(**meta, epoch=0, **ref.assessment(y[va], bv[va].numpy().astype(float))))
                    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                    for epoch in range(1, 201):
                        opt.zero_grad()
                        score = b[tr]+model(x[tr], m[tr])
                        loss = .5*ref.parent.balanced_loss(score, yy[tr])+.5*torch.nn.functional.cross_entropy(score, yy[tr])
                        loss.backward()
                        opt.step()
                        if epoch % 5:
                            continue
                        with torch.no_grad():
                            score = (bv[va]+model(xv[va], mv[va])).numpy().astype(float)
                        j = ref.exact_j(y[va], score.argmax(1))
                        curves.append(dict(**meta, epoch=epoch, **ref.assessment(y[va], score)))
                        if j > value:
                            value, best, epoch_best = j, ref.frozen_state(model), epoch
                    path = HERE/'models'/f'{partition}_fold{fold}_{arm}_seed{seed}.pt'
                    torch.save(dict(state=best, mu=mu, scale=sd, epoch=epoch_best, **meta), path)
                    selected.append(dict(**meta, epoch=epoch_best, validation_J=float(value), checkpoint=str(path),
                                         calibrator_epoch=ce, parameters=sum(p.numel() for p in model.parameters())))
                    costs.append(dict(**meta, seconds=time.perf_counter()-tick, steps=200))
                    print(f'{partition} fold{fold} seed{seed} {arm} epoch={epoch_best}', flush=True)
    pd.DataFrame(selected).to_csv(HERE/'selections.csv', index=False)
    pd.DataFrame(curves).to_csv(HERE/'selection_curves.csv', index=False)
    pd.DataFrame(costs).to_csv(HERE/'trajectory_costs.csv', index=False)
    (HERE/'selection_frozen.json').write_text(json.dumps(dict(heads=len(selected), test_evaluated=False)), encoding='utf8')
    predictions = []
    for partition in CONFIGS:
        for fold in range(6):
            for seed in SEEDS:
                d, outer, _, _ = load_fold(partition, fold, seed)
                te = d['test']
                for arm in ARMS:
                    ck = torch.load(HERE/'models'/f'{partition}_fold{fold}_{arm}_seed{seed}.pt', map_location='cpu', weights_only=False)
                    model = ref.make_model(arm).eval()
                    model.load_state_dict(ck['state'])
                    raw = ref.raw_input(arm, d['base'], outer)
                    with torch.inference_mode():
                        delta = model(torch.tensor((raw[te]-ck['mu'])/ck['scale'], dtype=torch.float32),
                                      torch.tensor(outer[te], dtype=torch.float32)).numpy().astype(float)
                    base, score = d['base'][te], d['base'][te]+delta
                    g = table.iloc[te][['instance_id','class_id','mmsi','port','pass_group']].copy()
                    for key, val in dict(partition=partition, fold=fold, seed=seed, arm=arm, epoch=ck['epoch']).items():
                        g[key] = val
                    g['prediction'], g['base_prediction'] = score.argmax(1), base.argmax(1)
                    g['net'] = (g.prediction == g.class_id).astype(int)-(g.base_prediction == g.class_id).astype(int)
                    g['outcome'] = np.select([g.net.eq(1),g.net.eq(-1),g.prediction.eq(g.class_id)], ['rescued','harmed','CC'], default='WW')
                    for c in range(3):
                        g[f's{c}'],g[f'b{c}'],g[f'delta{c}'] = score[:,c],base[:,c],delta[:,c]
                    predictions.append(g)
    p = pd.concat(predictions, ignore_index=True)
    p.to_csv(HERE/'predictions.csv', index=False)
    pooled = []
    for (partition, arm, seed), g in p.groupby(['partition','arm','seed']):
        assert g.instance_id.is_unique and len(g) == len(table)
        before = ref.assessment(g.class_id.to_numpy(), g[['b0','b1','b2']].to_numpy())
        after = ref.assessment(g.class_id.to_numpy(), g[['s0','s1','s2']].to_numpy())
        pooled.append(dict(partition=partition,arm=arm,seed=seed,net=int(g.net.sum()),rescued=int(g.net.eq(1).sum()),
                          harmed=int(g.net.eq(-1).sum()), **after, delta_OA=after['OA']-before['OA'],delta_BA=after['BA']-before['BA']))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv', index=False)
    p.groupby(['partition','arm','seed','class_id']).agg(n=('net','size'),net=('net','sum')).to_csv(HERE/'class_effects.csv')
    p.groupby(['partition','arm','seed','pass_group']).agg(n=('net','size'),net=('net','sum')).to_csv(HERE/'pass_effects.csv')
    pd.DataFrame(checks).to_csv(HERE/'baseline_replays.csv', index=False)
    (HERE/'summary.json').write_text(json.dumps(dict(heads=len(selected), steps=200*len(selected),
        train_seconds=sum(r['seconds'] for r in costs), seconds=time.perf_counter()-start,
        frozen_calibrator=True, calibrator_OOF=False, test_selection=False), indent=2), encoding='utf8')
    print(pd.DataFrame(pooled).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
