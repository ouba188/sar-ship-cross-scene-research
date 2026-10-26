"""Train the retained four-class candidate with a frozen pretrained frontend.

Inputs are portable source features and identity/role metadata, not experiment
modules. No evaluation labels enter optimization or checkpoint selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import train_ops as op
from .m3 import PairMetric, PolarHead, regional_descriptors
from .training_alignment import FrozenDecision, fit_metric

FEATURES = ('x', 'mean', 'mass', 'joint', 'e_image', 'coverage')
ROLES = ('fit_train', 'fit_validation', 'utility_train', 'selection', 'evaluation')


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def source_scaler(x, fit):
    mu, sd = x[fit].mean(0), x[fit].std(0)
    active = np.ptp(x[fit], axis=0) > 0
    return dict(mean=mu, scale=np.where(active, np.maximum(sd, 1e-6), 1), active=active)


def scale_input(x, sc):
    return np.where(sc['active'], (x-sc['mean'])/sc['scale'], 0).astype(np.float32)


def validate_roles(table, data):
    if not table.obs_id.is_unique:
        raise ValueError('obs_id must uniquely identify an observation.')
    np.testing.assert_array_equal(table.obs_id.to_numpy().astype(str), data['obs_id'].astype(str))
    if set(table.role)-set(ROLES):
        raise ValueError('Unknown role: expected source fit/validation/utility/selection/evaluation.')
    groups = {role: np.flatnonzero(table.role.eq(role)) for role in ROLES}
    for role in ROLES[:-1]:
        if not len(groups[role]) or set(table.iloc[groups[role]].class_id) != set(range(4)):
            raise ValueError(f'{role} needs all four source classes.')
    for key in ('mmsi', 'pass_group'):
        if table[key].isna().any() or table[key].astype(str).str.strip().eq('').any():
            raise ValueError(f'Missing {key}; cannot enforce the source split.')
        for i, role in enumerate(ROLES):
            for other in ROLES[i+1:]:
                if set(table.iloc[groups[role]][key]) & set(table.iloc[groups[other]][key]):
                    raise ValueError(f'{key} overlaps {role} and {other}.')
    refs, bank = data['refs'], data['bank'].ravel()
    if refs.shape != (len(table), 4, 2) or data['bank'].shape != (4, 8):
        raise ValueError('Expected 32 source references and two references per candidate class.')
    if not np.isin(refs, bank).all() or not table.iloc[bank].role.eq('fit_train').all():
        raise ValueError('References must belong to the frozen fit-only bank.')
    np.testing.assert_array_equal(table.iloc[bank].class_id, np.repeat(np.arange(4), 8))
    if not (table.class_id.to_numpy()[refs] == np.arange(4)[None, :, None]).all():
        raise ValueError('Retrieved references do not match the candidate class order.')
    if (table.mmsi.to_numpy()[:, None, None] == table.mmsi.to_numpy()[refs]).any():
        raise ValueError('A query retrieves its own vessel.')
    return groups


def fit_base(data, table, groups, seed, out):
    raw, aux = data['raw'], data['aux']
    fit, val = groups['fit_train'], groups['fit_validation']
    sc, sa = source_scaler(raw, fit), source_scaler(aux, fit)
    x = np.c_[scale_input(raw, sc), scale_input(aux, sa)]
    torch.manual_seed(seed)
    initial = nn.Sequential(nn.Linear(1538, 128), nn.GELU(), nn.Linear(128, 4))
    model = nn.Sequential(nn.Linear(1642, 128), nn.GELU(), nn.Linear(128, 4))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].weight[:, :1538].copy_(initial[0].weight)
        model[0].bias.copy_(initial[0].bias)
        model[2].load_state_dict(initial[2].state_dict())
    f = table.iloc[fit]
    ships = f.groupby('class_id').mmsi.transform('nunique').to_numpy()
    repeat = f.groupby('mmsi').mmsi.transform('size').to_numpy()
    wt = torch.tensor(len(f)/(4*ships*repeat), dtype=torch.float32)
    xt, yt = torch.tensor(x), torch.tensor(f.class_id.to_numpy(int))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    best, chosen, selected, curve = None, None, 0, []
    for step in range(1, 201):
        optimizer.zero_grad(set_to_none=True)
        loss = (nn.functional.cross_entropy(model(xt[fit]), yt, reduction='none')*wt).mean()
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            with torch.no_grad():
                score = model(xt[val]).numpy()
            y = table.iloc[val].class_id.to_numpy(int)
            j = op.exact_j(y, score)
            curve.append(dict(step=step, loss=float(loss.detach()), **op.metrics(y, score)))
            if best is None or j > best:
                best, selected, chosen = j, step, op.state(model)
    model.load_state_dict(chosen)
    with torch.no_grad():
        base = model(xt).numpy()
    torch.save(dict(state=chosen, step=selected, raw_dim=1538, aux_dim=104), out/'raw_summary.pt')
    np.savez_compressed(out/'raw_scaler.npz', **sc)
    np.savez_compressed(out/'summary_scaler.npz', **sa)
    pd.DataFrame(curve).to_csv(out/'base_curve.csv', index=False)
    return base


def fit_readouts(local, base, features, refs, covariance, table, groups, seed, out):
    utility, selection = groups['utility_train'], groups['selection']
    y = table.class_id.to_numpy(int, copy=True)
    y[table.role.eq('evaluation')] = -1
    scaler = op.scale_fit(op.local_raw(local, base), utility)
    x = op.transform(op.local_raw(local, base), scaler)
    null = op.transform(np.zeros((1, 1, 15)), scaler).reshape(15)
    deltas, steps = [], []
    for p, name in enumerate(('vv', 'vh_centered')):
        torch.manual_seed(seed)
        head = PolarHead(p, centered=bool(p), null=null)
        nn.init.zeros_(head.net[-1].weight)
        nn.init.zeros_(head.net[-1].bias)
        delta, step = op.fit_head(name, head, x, base, y, utility, selection, out)
        deltas.append(delta)
        steps.append(step)
    dv, dh = deltas
    desc = regional_descriptors(features, covariance)
    gr, av = op.gate_raw(desc, desc, refs, base, dv, dh)
    alpha = op.choose_alpha(base, dv, dh, av, y, selection)
    final, weights, gs = op.fit_gate('frequency', seed, gr, base, dv, dh, av,
                                   y, utility, selection, alpha, out)
    torch.save(scaler, out/'M3_readout_scaler.pt')
    return dict(frequency=final, gate_raw=gr, available=av,
                delta_vv=dv, delta_vh=dh, weight_frequency=weights), dict(
                    vv_step=steps[0], vh_step=steps[1], alpha=alpha, frequency_step=gs)


def train(bundle, assets, out, seeds=(17, 29, 43), device='cpu'):
    """Refit base, warm polar heads, align metric, then refit final polar heads.

The pretrained response readers, PCA, source bank and initial PairMetric remain
explicit input assets. This is staged training, not backbone pretraining.
"""
    bundle, assets, out = Path(bundle), Path(assets), Path(out)
    data = load_npz(bundle/'features.npz')
    table = pd.read_csv(bundle/'table.csv', dtype={'mmsi': str, 'pass_group': str})
    groups = validate_roles(table, data)
    features = {k: data[k] for k in FEATURES}
    covariance = {k: data['covariance_'+k] for k in ('mu', 'scale', 'active')}
    refs = data['refs']
    y = table.class_id.to_numpy(int, copy=True)
    y[table.role.eq('evaluation')] = -1
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        dest = out/f'seed{seed}'
        dest.mkdir(exist_ok=True)
        warm = dest/'warm'
        warm.mkdir(exist_ok=True)
        base = fit_base(data, table, groups, seed, dest)
        initial = assets/'warm'/f'seed{seed}'
        metric = PairMetric().to(device)
        metric.load_state_dict(op.load_checkpoint(initial/'both_metric.pt')['state'])
        sc = op.load_checkpoint(initial/'metric_scaler.pt')
        matcher = op.Matcher(features, refs, sc, device)
        local = matcher.predict(np.arange(len(table)), metric)
        cached, _ = fit_readouts(local, base, features, refs, covariance,
                                  table, groups, seed, warm)
        decision = FrozenDecision(warm, cached, base, device)
        step, gradient, seconds = fit_metric(matcher, metric, decision, y,
            groups['utility_train'], groups['selection'], dest)
        if step:
            local = matcher.predict(np.arange(len(table)), metric)
        outputs, receipt = fit_readouts(local, base, features, refs, covariance,
                                        table, groups, seed, dest)
        receipt.update(seed=seed, metric_step=step, max_gradient=gradient, metric_seconds=seconds)
        np.savez_compressed(dest/'predictions.npz', obs_id=data['obs_id'], bare=base,
                            local=local, refs=refs, **outputs)
        op.write_json(dest/'selection.json', receipt)
        results.append(receipt)
        print('FROZEN', seed, receipt, flush=True)
    op.write_json(out/'training.json', dict(seeds=list(seeds), results=results,
        evaluation_labels_used=False, backbone_frozen=True,
        initialization='source-trained readers, PCA, bank and initial PairMetric',
        stages='raw_summary -> polar warmup -> aligned metric -> polar refit'))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    train(args.bundle, args.assets, args.output, args.seeds, args.device)


if __name__ == '__main__':
    main()
