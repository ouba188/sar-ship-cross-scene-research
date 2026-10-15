"""Few-label target supervision of internal affine responses, matched CPU pilot."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

from channel_operator import (BASE, OUT as PARENT, PORTS, SEED, BATCH,
    build_blocks, metrics, save_json)

OUT = BASE/'target_response_adapter_20261010'
DRAWS = 3
STEPS = 24
RIDGE = 8.
BALANCED_HEAD = False


class ResponseAdapter(nn.Module):
    def __init__(self, channel_scale):
        super().__init__()
        self.log_gain = nn.Parameter(torch.zeros(len(channel_scale)))
        self.shift = nn.Parameter(torch.zeros(len(channel_scale)))
        self.register_buffer('channel_scale', torch.as_tensor(channel_scale, dtype=torch.float32))

    def forward(self, h):
        return (h*self.log_gain.exp()[None, :, None, None]
                + (self.shift*self.channel_scale)[None, :, None, None])

    def bound(self):
        with torch.no_grad():
            self.log_gain.clamp_(-np.log(2.), np.log(2.))
            self.shift.clamp_(-.5, .5)


def residual_head(phi, prior, labels, ridge=RIDGE, balanced=None):
    balanced = BALANCED_HEAD if balanced is None else balanced
    target = 2*torch.eye(8, dtype=phi.dtype, device=phi.device)[labels]-1
    if balanced:
        counts = torch.bincount(labels, minlength=8)
        present = counts > 0
        weights = len(labels)/(present.sum()*counts[labels].to(phi.dtype))
        residual = (target-prior)*present[None, :]
    else:
        weights = torch.ones(len(phi), dtype=phi.dtype)
        residual = target-prior
    root = weights.sqrt()
    weighted_phi = phi*root[:, None]
    kernel = weighted_phi@weighted_phi.T
    inverse = torch.linalg.solve(kernel+ridge*torch.eye(len(phi), dtype=phi.dtype),
                                 torch.eye(len(phi), dtype=phi.dtype))
    alpha = root[:, None]*(inverse@(root[:, None]*residual))
    loo = prior+residual-alpha/(weights*inverse.diagonal())[:, None]
    return phi.T@alpha, loo, target


def support_design(meta, draw):
    rng = np.random.default_rng(SEED+draw)
    identities = np.sort(meta.mmsi.unique())
    selected = rng.choice(identities, 64, replace=False)
    support = np.array([rng.choice(meta.index[meta.mmsi.eq(v)].to_numpy()) for v in selected])
    query = np.flatnonzero(~meta.mmsi.isin(selected).to_numpy())
    return support, query


def phi_prior(features, center, scale, weight, bias):
    x = (features-center)/scale/np.sqrt(512)
    return torch.cat([x, torch.ones((len(x), 1), dtype=x.dtype)], dim=1), x@weight+bias


def encode(block, adapter, maps, rows):
    values = np.empty((len(rows), 512), np.float32)
    with torch.inference_mode():
        for lo in range(0, len(rows), BATCH):
            idx = rows[lo:lo+BATCH]
            h = torch.from_numpy(np.array(maps[idx]))
            values[lo:lo+len(idx)] = block(adapter(h)).mean((2, 3)).numpy()
    return values


def fit_adapter(block, maps, support, labels, source_meta, base_source_scores,
                center, scale, weight, bias, channel_scale, precision, arm, tag):
    adapter = ResponseAdapter(channel_scale)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.01)
    hs = torch.from_numpy(np.array(maps[support]))
    y = torch.as_tensor(labels, dtype=torch.long)
    counts = np.bincount(labels, minlength=8)
    class_weight = np.zeros(8, dtype=np.float32)
    present = counts > 0
    class_weight[present] = len(labels)/(present.sum()*counts[present])
    row_weight = torch.from_numpy(class_weight[labels])
    prec = torch.from_numpy(precision)
    histories = []
    source_labels = source_meta.class_id.to_numpy()
    for step in range(STEPS):
        # Both adapter arms use identical replay draws and target support labels.
        rng = np.random.default_rng(SEED+step)
        replay = np.concatenate([rng.choice(np.flatnonzero(source_labels == c), 4, replace=False)
                                 for c in range(8)])
        hr = torch.from_numpy(np.array(maps[replay]))
        fs = block(adapter(hs)).mean((2, 3))
        phi, prior = phi_prior(fs, center, scale, weight, bias)
        _, loo, target = residual_head(phi, prior, y)
        target_loss = ((loo-target).square().mean(1)*row_weight).mean()
        fr = block(adapter(hr)).mean((2, 3))
        _, source_prior = phi_prior(fr, center, scale, weight, bias)
        source_loss = (source_prior-base_source_scores[replay]).square().mean()
        regularizer = (prec*(adapter.log_gain.square()+adapter.shift.square())).mean()
        loss = target_loss+.25*source_loss+.1*regularizer
        optimizer.zero_grad()
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.))
        optimizer.step()
        adapter.bound()
        if step in (0, 7, 15, STEPS-1):
            record = dict(task=tag, arm=arm, step=step+1, target_loo_loss=float(target_loss.detach()),
                source_preservation_loss=float(source_loss.detach()), regularizer=float(regularizer.detach()),
                gradient_norm=grad, parameter_rms=float(torch.cat([adapter.log_gain.detach(),
                    adapter.shift.detach()]).square().mean().sqrt()))
            histories.append(record)
            print('ADAPT', json.dumps(record), flush=True)
    return adapter.eval(), histories


def run():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Completed outputs must not be overwritten')
    started = time.perf_counter()
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(PARENT/'split.npz') as split:
        train, test = split['train_obs_ids'], split['test_obs_ids']
    n = len(train)
    source_meta = obs.iloc[train].reset_index(drop=True)
    target_meta = obs.iloc[test].reset_index(drop=True)
    maps = np.load(PARENT/'boundary_maps.npy', mmap_mode='r')
    with np.load(PARENT/'baseline.npz') as old:
        center, scale, weight, bias = [torch.from_numpy(old[k].astype(np.float32))
                                     for k in ('center', 'scale', 'coef', 'bias')]
        previous_pred = old['predictions']
    source_maps = np.asarray(maps[:n])
    channel_scale = np.maximum(source_maps.std((0, 2, 3)), 1e-3)
    rank = pd.read_csv(PARENT/'internal_channel_ranking.csv')
    rank_fraction = rank.selection_score.rank(method='average', pct=True).to_numpy()
    allowance = .5+1.5*rank_fraction
    precision = (allowance**-2)
    precision = (precision/precision.mean()).astype(np.float32)
    pd.DataFrame(dict(channel=np.arange(512), source_scene_score=rank.selection_score,
                      guided_precision=precision)).to_csv(OUT/'guidance.csv', index=False)
    save_json(OUT/'PROTOCOL.json', dict(
        purpose='Can64 target labels use internally scene-sensitive information; does source scene guidance improve over ordinary affine adaptation?',
        ports=list(PORTS), draws=DRAWS, support_budget=64,
        supports='New fixed metadata-only sample of64 distinct MMSIs, one random observation each; same supports for every arm',
        exclusion='All observations sharing a support MMSI excluded from query; source excludes all target identities',
        source_labels=n, no_query_fitting=True,
        scene_overlap_boundary='Support/query scenes may overlap; this tests unseen vessels, not unseen target acquisitions',
        arms=['source_only', 'head_only', 'ordinary_adapter', 'scene_guided_adapter'],
        adapter='R18 before final residual block;1024 trainable log-gain/shift parameters; identity initialization; frozen backbone',
        source_guidance='Same1024 parameters; source scene/semantic channel rank changes quadratic prior precision, normalized to mean1',
        target_head='Source prior plus differentiable residual ridge8 fitted to the same64 labels; no new label budget',
        balanced_head=BALANCED_HEAD,
        absent_support_classes='Residual updates zero for absent classes; current-representation source prior retained' if BALANCED_HEAD else 'Legacy unweighted negative updates preserved as interrupted counterevidence',
        leave_one_out='Analytic deletion with weights fixed from the full support set; training objective, not independent validation',
        objective='Class-balanced support leave-one-out ridge MSE +0.25 source prior preservation +0.1 quadratic identity prior',
        optimization=dict(steps=STEPS, lr=.01, optimizer='Adam', replay_per_step=32,
                          gain_bounds=[.5, 2.], normalized_shift_bounds=[-.5, .5]),
        model_selection='All constants fixed before query results; final fixed step, no target tuning',
        limitations='Small source R18 operator pilot, not full dual/main64 pipeline; repeated development datasets; inherited pretraining exposure',
        resources='CPU1; no parent code modifications, no GPU', seed=SEED))
    block = build_blocks()[1]
    identity = ResponseAdapter(channel_scale).eval()
    base_features = encode(block, identity, maps, np.arange(len(maps)))
    with torch.no_grad():
        all_phi, all_prior = phi_prior(torch.from_numpy(base_features), center, scale, weight, bias)
        if (all_prior[n:].argmax(1).numpy() != previous_pred).any():
            raise ValueError('Frozen R18 baseline did not replay')
    base_source_scores = all_prior[:n].clone()
    results, classes, histories, designs, updates, overlaps = [], [], [], [], [], []
    for port in PORTS:
        port_rows = np.flatnonzero(target_meta.port.eq(port).to_numpy())
        pm = target_meta.iloc[port_rows].reset_index(drop=True)
        for draw in range(DRAWS):
            tag = port.replace(' ', '_')+f'_draw{draw}'
            support_local, query_local = support_design(pm, draw)
            support = n+port_rows[support_local]
            query = n+port_rows[query_local]
            sy = pm.iloc[support_local].class_id.to_numpy(copy=True)
            forbidden = set(pm.iloc[support_local].mmsi)
            assert not forbidden & set(pm.iloc[query_local].mmsi)
            designs.append(dict(port=port, draw=draw, support=len(support), query=len(query),
                support_classes=int(np.unique(sy).size), query_classes=int(pm.iloc[query_local].class_id.nunique()),
                removed_same_vessel=int(len(pm)-len(query)-64),
                scene_overlap=len(set(pm.iloc[support_local]['product']) & set(pm.iloc[query_local]['product']))))
            np.savez_compressed(OUT/f'{tag}_split.npz', support_map_rows=support,
                query_map_rows=query, support_obs_ids=test[port_rows[support_local]],
                query_obs_ids=test[port_rows[query_local]], support_labels=sy)
            with torch.no_grad():
                beta, _, _ = residual_head(all_phi[support], all_prior[support], torch.from_numpy(sy))
                head_scores = all_prior[query]+all_phi[query]@beta
            predictions = dict(source_only=all_prior[query].argmax(1).numpy(),
                               head_only=head_scores.argmax(1).numpy())
            saved_states = {}
            for arm, prec in [('ordinary_adapter', np.ones(512, np.float32)),
                              ('scene_guided_adapter', precision)]:
                adapter, records = fit_adapter(block, maps, support, sy, source_meta, base_source_scores,
                    center, scale, weight, bias, channel_scale, prec, arm, tag)
                histories.extend(records)
                # Query labels are not read until both models have finished adaptation.
                rows = np.concatenate([support, query])
                features = encode(block, adapter, maps, rows)
                with torch.no_grad():
                    phi, prior = phi_prior(torch.from_numpy(features), center, scale, weight, bias)
                    beta, loo, target = residual_head(phi[:64], prior[:64], torch.from_numpy(sy))
                    score = prior[64:]+phi[64:]@beta
                predictions[arm] = score.argmax(1).numpy()
                lg, shift = adapter.log_gain.detach().numpy(), adapter.shift.detach().numpy()
                magnitude = lg**2+shift**2
                updates.append(dict(port=port, draw=draw, arm=arm,
                    parameter_rms=float(np.sqrt(magnitude.mean()/2)),
                    guided_top64_update_fraction=float(magnitude[rank.selected.to_numpy()].sum()/max(magnitude.sum(), 1e-12)),
                    support_LOO_BA=metrics(loo.argmax(1).numpy(), sy)['BA'],
                    source_scene_prior_precision_mean=float(prec.mean()),
                    gain_min=float(np.exp(lg).min()), gain_max=float(np.exp(lg).max()),
                    shift_min=float(shift.min()), shift_max=float(shift.max())))
                saved_states[arm+'_log_gain'], saved_states[arm+'_shift'] = lg, shift
                saved_states[arm+'_residual_coef'] = beta.numpy()
            qy = pm.iloc[query_local].class_id.to_numpy()
            ref = predictions['head_only'] == qy
            for arm, pred in predictions.items():
                correct = pred == qy
                results.append(dict(port=port, draw=draw, arm=arm, query=len(query), **metrics(pred, qy),
                    rescue_vs_head=int((~ref & correct).sum()), harm_vs_head=int((ref & ~correct).sum())))
                for c in np.unique(qy):
                    m = qy == c
                    classes.append(dict(port=port, draw=draw, arm=arm, class_id=int(c), n=int(m.sum()),
                        recall=float(100*correct[m].mean()), rescue_vs_head=int((~ref & correct & m).sum()),
                        harm_vs_head=int((ref & ~correct & m).sum())))
            normal = predictions['ordinary_adapter'] == qy
            guided = predictions['scene_guided_adapter'] == qy
            overlaps.append(dict(port=port, draw=draw,
                guidance_rescue=int((~normal & guided).sum()), guidance_harm=int((normal & ~guided).sum())))
            np.savez_compressed(OUT/f'{tag}_models.npz', **saved_states,
                **{k+'_pred': v for k, v in predictions.items()}, source_center=center.numpy(),
                source_scale=scale.numpy(), source_coef=weight.numpy(), source_bias=bias.numpy(),
                channel_scale=channel_scale)
            for name, rows in [('results', results), ('by_class', classes), ('training_history', histories),
                               ('support_design', designs), ('parameter_updates', updates), ('guidance_comparison', overlaps)]:
                pd.DataFrame(rows).to_csv(OUT/f'{name}.csv', index=False)
            print('TASK', tag, json.dumps(results[-4:]), flush=True)
    save_json(OUT/'COMPLETE.json', dict(tasks=len(designs), adapter_fits=2*len(designs),
        supports=64, draws=DRAWS, seconds=time.perf_counter()-started,
        original_prediction_disagreements=0, query_labels_used_for_fitting=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--balanced-head', action='store_true')
    args = parser.parse_args()
    if args.balanced_head:
        BALANCED_HEAD = True
        OUT = BASE/'target_response_adapter_balanced_20261010'
    run()
