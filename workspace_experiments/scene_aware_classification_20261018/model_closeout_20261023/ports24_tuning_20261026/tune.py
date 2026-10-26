"""Source-only bounded search, then frozen deployment of selected configurations."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import sys
import time
from fractions import Fraction
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PREV = HERE.parent/'ports24_aligned_20261026'
sys.path.insert(0, str(PREV))
from prepare_bundle import load, write
from run import apply_readouts
from sar_m123 import train_aligned as ta, train_ops as op
from sar_m123.training_alignment import FrozenDecision, fit_metric
from sar_m123.m3 import PairMetric, PolarHead, regional_descriptors

SEEDS = (17, 29, 43)
LAMBDAS = (0., .25, .5, .75, 1., 1.25, 1.5)
CONFIGS = [
    dict(name='default'),
    dict(name='head_lr_1e3', head_lr=.001),
    dict(name='head_lr_3e4', head_lr=.0003),
    dict(name='head_wd_01', head_wd=.1),
    dict(name='head_wd_1', head_wd=1.),
    dict(name='head_lr1e3_wd01', head_lr=.001, head_wd=.1),
    dict(name='head_steps400', head_steps=400),
    dict(name='metric_lr3e4', metric_lr=.0003, metric_steps=100),
    dict(name='metric_lr1e3_steps200', metric_lr=.001, metric_steps=200),
    dict(name='transport_eps005', temperature=.05),
    dict(name='transport_eps02', temperature=.2),
    dict(name='transport_eps04', temperature=.4),
]


def fit_readouts(local, base, features, refs, covariance, table, groups, seed, out, config):
    utility, selection = groups['utility_train'], groups['selection']
    y = table.class_id.to_numpy(int, copy=True)
    assert (y[groups['evaluation']] == -1).all()
    scaler = op.scale_fit(op.local_raw(local, base), utility)
    x = op.transform(op.local_raw(local, base), scaler)
    null = op.transform(np.zeros((1, 1, 15)), scaler).reshape(15)
    ds, steps = [], []
    for p, name in enumerate(('vv', 'vh_centered')):
        torch.manual_seed(seed)
        head = PolarHead(p, bool(p), null)
        nn.init.zeros_(head.net[-1].weight)
        nn.init.zeros_(head.net[-1].bias)
        d, step = op.fit_head(name, head, x, base, y, utility, selection, out,
            steps=config.get('head_steps', 200), lr=config.get('head_lr', .003),
            weight_decay=config.get('head_wd', .01))
        ds.append(d)
        steps.append(step)
    dv, dh = ds
    desc = regional_descriptors(features, covariance)
    gr, av = op.gate_raw(desc, desc, refs, base, dv, dh)
    alpha = op.choose_alpha(base, dv, dh, av, y, selection)
    score, weights, gs = op.fit_gate('frequency', seed, gr, base, dv, dh, av, y,
        utility, selection, alpha, out)
    torch.save(scaler, out/'M3_readout_scaler.pt')
    return dict(frequency=score, bare=base, delta_vv=dv, delta_vh=dh,
        weight_frequency=weights, gate_raw=gr, available=av), dict(
            vv_step=steps[0], vh_step=steps[1], alpha=alpha, frequency_step=gs)


def train():
    data = load(PREV/'bundle/features.npz')
    table = pd.read_csv(PREV/'bundle/table.csv', dtype={'mmsi': str, 'pass_group': str})
    groups = ta.validate_roles(table, data)
    assert (table.loc[table.role.eq('evaluation'), 'class_id'] == -1).all()
    features = {k: data[k] for k in ta.FEATURES}
    covariance = {k: data['covariance_'+k] for k in ('mu', 'scale', 'active')}
    ids = np.flatnonzero(data['local_available'])
    tick = time.perf_counter()
    write(HERE/'search_space.json', dict(configs=CONFIGS, seeds=SEEDS, lambdas=LAMBDAS,
        selection='mean three-seed exact source J; common config and lambda',
        external_features_loaded=False, evaluation_labels_used=False))
    for config in CONFIGS:
        for seed in SEEDS:
            out = HERE/'training'/config['name']/f'seed{seed}'
            out.mkdir(parents=True, exist_ok=True)
            if (out/'complete.json').exists():
                continue
            prior = load(PREV/'training'/f'seed{seed}/predictions.npz')
            base, local = prior['bare'], prior['local'].copy()
            if config['name'] == 'default':
                result, receipt = prior, dict(reused=True)
            else:
                initial = PREV/'initial_assets/warm'/f'seed{seed}'
                metric = PairMetric().cuda()
                metric.load_state_dict(op.load_checkpoint(initial/'both_metric.pt')['state'])
                if 'metric_lr' in config or 'temperature' in config:
                    sc = op.load_checkpoint(initial/'metric_scaler.pt')
                    matcher = op.Matcher(features, data['refs'], sc, 'cuda',
                                         temperature=config.get('temperature'))
                    if 'metric_lr' in config:
                        cached = {**prior}
                        cached['gate_raw'] = prior['gate_raw']
                        decision = FrozenDecision(PREV/'training'/f'seed{seed}/warm', cached, base, 'cuda')
                        check_ids = groups['selection'][:48]
                        with torch.no_grad():
                            initial_scores = decision(torch.tensor(local[check_ids], device='cuda'), check_ids).cpu().numpy()
                        np.testing.assert_allclose(initial_scores, prior['frequency'][check_ids], atol=2e-5, rtol=1e-5)
                        step, grad, seconds = fit_metric(matcher, metric, decision,
                            table.class_id.to_numpy(int), groups['utility_train'], groups['selection'],
                            out, steps=config['metric_steps'], lr=config['metric_lr'], batch_size=128)
                        receipt = dict(metric_step=step, gradient_max=grad, metric_seconds=seconds,
                                       accumulation_chunk=128)
                    else:
                        receipt = dict(metric_step=0, temperature=config['temperature'])
                    local[ids] = matcher.predict(ids, metric)
                    torch.save(dict(state=op.state(metric), **receipt), out/'aligned_metric.pt')
                    del matcher, metric
                    torch.cuda.empty_cache()
                else:
                    receipt = dict(metric_step=0, reused_local=True)
                result, heads = fit_readouts(local, base, features, data['refs'], covariance,
                    table, groups, seed, out, config)
                receipt.update(heads)
            np.savez_compressed(out/'source.npz', **{k: result[k] for k in (
                'bare', 'frequency', 'delta_vv', 'delta_vh', 'weight_frequency', 'available')})
            write(out/'complete.json', dict(config=config, seed=seed, **receipt,
                elapsed_total_seconds=time.perf_counter()-tick, source_only=True))
            print('SOURCE_DONE', config['name'], seed, receipt, flush=True)
    write(HERE/'training_complete.json', dict(configurations=len(CONFIGS)*len(SEEDS),
        elapsed_seconds=time.perf_counter()-tick, external_features_loaded=False))


def select():
    assert (HERE/'training_complete.json').exists()
    table = pd.read_csv(PREV/'bundle/table.csv')
    ids = np.flatnonzero(table.role.eq('selection'))
    y = table.class_id.to_numpy(int)[ids]
    records, grouped, ranks = [], [], []
    for order, config in enumerate(CONFIGS):
        cached = [load(HERE/'training'/config['name']/f'seed{seed}/source.npz') for seed in SEEDS]
        for lam in LAMBDAS:
            js = []
            for seed, z in zip(SEEDS, cached):
                scores = z['bare'][ids]+lam*(z['frequency'][ids]-z['bare'][ids])
                js.append(op.exact_j(y, scores))
                records.append(dict(config=config['name'], seed=seed, correction_scale=lam,
                    **op.metrics(y, scores)))
                for pg, ix in table.iloc[ids].reset_index(drop=True).groupby('pass_group').indices.items():
                    grouped.append(dict(config=config['name'], seed=seed, correction_scale=lam,
                        pass_group=pg, **op.metrics(y[ix], scores[ix])))
            j = sum(js, Fraction(0))/len(js)
            ranks.append((j, order, lam, config['name']))
    ranks.sort(key=lambda r: (-r[0], abs(r[2]-1), r[1], r[2]))
    best = ranks[0]
    runner = next(r for r in ranks if r[3] != best[3])
    amplitude = next(r for r in ranks if r[3] == 'default')
    def entry(row):
        return dict(config=row[3], correction_scale=row[2], source_J=float(row[0]))
    selected = dict(primary=entry(best), runner_up=entry(runner), lambda_only=entry(amplitude),
        selected_without_evaluation_labels=True, shared_across_seeds=True,
        selection_n=len(ids), selection_passes=int(table.iloc[ids].pass_group.nunique()))
    pd.DataFrame(records).to_csv(HERE/'source_grid.csv', index=False)
    pd.DataFrame(grouped).to_csv(HERE/'source_by_pass.csv', index=False)
    pd.DataFrame([entry(r) for r in ranks]).to_csv(HERE/'source_ranking.csv', index=False)
    write(HERE/'selected.json', selected)
    print('SELECTION_FROZEN', selected, flush=True)


@torch.inference_mode()
def infer():
    selected = json.loads((HERE/'selected.json').read_text(encoding='utf-8'))
    data, ext = load(PREV/'bundle/features.npz'), load(PREV/'bundle/external.npz')
    n = len(data['obs_id'])
    features = {k: np.concatenate((data[k], ext[k])) for k in ta.FEATURES}
    refs = np.concatenate((data['refs'], ext['refs']))
    covariance = {k: data['covariance_'+k] for k in ('mu', 'scale', 'active')}
    descriptors = regional_descriptors(features, covariance)
    ids = np.r_[np.flatnonzero(data['local_available']), np.arange(n, len(refs))]
    for seed in SEEDS:
        old = load(PREV/'results'/f'seed{seed}/both.npz')
        base = old['bare']
        for arm in ('primary', 'runner_up', 'lambda_only'):
            selection = selected[arm]
            config = next(c for c in CONFIGS if c['name'] == selection['config'])
            out = HERE/'training'/config['name']/f'seed{seed}'
            if config['name'] == 'default':
                pred = old
            else:
                if 'metric_lr' in config or 'temperature' in config:
                    sc = op.load_checkpoint(PREV/'initial_assets/warm'/f'seed{seed}/metric_scaler.pt')
                    matcher = op.Matcher(features, refs, sc, 'cuda', temperature=config.get('temperature'))
                    metric = PairMetric().cuda().eval()
                    metric.load_state_dict(op.load_checkpoint(out/'aligned_metric.pt')['state'])
                    local = np.zeros_like(old['local'])
                    local[ids] = matcher.predict(ids, metric)
                    del matcher, metric
                    torch.cuda.empty_cache()
                else:
                    local = old['local']
                # Only the frequency endpoint is tuned; do not require a spatial checkpoint.
                scaler = op.load_checkpoint(out/'M3_readout_scaler.pt')
                xx = torch.tensor(op.transform(op.local_raw(local, base), scaler))
                ds = []
                for p, name in enumerate(('vv', 'vh_centered')):
                    ck = op.load_checkpoint(out/f'{name}.pt')
                    head = PolarHead(p, bool(p), ck['state'].get('null')).eval()
                    head.load_state_dict(ck['state'])
                    ds.append(head(xx).numpy())
                gr, av = op.gate_raw(descriptors, descriptors, refs, base, *ds)
                ck = op.load_checkpoint(out/'frequency.pt')
                gate = op.Gate(185, ck['alpha']).eval()
                gate.load_state_dict(ck['state'])
                w = gate(torch.tensor(op.transform(gr, ck['scaler'])), torch.tensor(av)).numpy()
                pred = dict(bare=base, frequency=base+w[:, :1]*ds[0]+w[:, 1:]*ds[1],
                    delta_vv=ds[0], delta_vh=ds[1], weight_frequency=w, available=av)
                expected = load(out/'source.npz')['frequency']
                np.testing.assert_allclose(pred['frequency'][:n], expected, atol=2e-5, rtol=1e-5)
                np.testing.assert_array_equal(pred['frequency'][:n].argmax(1), expected.argmax(1))
            lam = selection['correction_scale']
            score = base+lam*(pred['frequency']-base)
            dest = HERE/'results'/f'seed{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest/f'{arm}.npz', score=score, bare=base, refs=refs,
                delta_vv=pred['delta_vv'], delta_vh=pred['delta_vh'],
                weight_frequency=lam*pred['weight_frequency'], obs_id=np.r_[data['obs_id'], ext['obs_id']])
            print('PREDICTION_FROZEN', seed, arm, selection, flush=True)
    write(HERE/'prediction_complete.json', dict(selected=selected, target_count=len(ext['obs_id']),
        evaluation_labels_used=False, seeds=SEEDS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('train', 'select', 'infer', 'all'), default='all')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.stage in ('train', 'all'):
        train()
    if args.stage in ('select', 'all'):
        select()
    if args.stage in ('infer', 'all'):
        infer()
