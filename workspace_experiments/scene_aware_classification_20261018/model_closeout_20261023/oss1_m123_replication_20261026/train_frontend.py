"""Four-class source frontend: original A/D/T, mean-inclusive M1 and old M2.

All supervised fits/scalers use fit_train; checkpoint/alpha selection uses only
fit_validation. Other roles receive frozen scores. No historical trained asset
is imported. Five's latent width remains 8; only class dimensions become 4.
Stages: base (A/D/T + readers), summaries (reader-dependent C4 extraction),
heads (M1/M2), all. Output scores follow the supplied role table exactly.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler

import extract_frontend as extract

HERE = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
C = 4
spec = importlib.util.spec_from_file_location('oss1_upstream_algorithms',
    HERE.parent / 'regional_utility_pilot/upstream/pipeline.py')
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
from sar_m123.m12 import attention_raw


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def load_table(path, fold=None):
    table = extract.read_table(path, fold)
    tr = np.flatnonzero(table.role.eq('fit_train'))
    va = np.flatnonzero(table.role.eq('fit_validation'))
    allowed = {'fit_train', 'fit_validation', 'utility_train', 'selection', 'evaluation'}
    if set(table.role)-allowed or not len(tr) or not len(va):
        raise ValueError('Expected explicit five-role protocol with nonempty fit_train/fit_validation.')
    if set(table.iloc[tr].class_id) != set(range(C)):
        raise ValueError('fit_train must contain all four remapped classes.')
    for key in ('obs_id', 'mmsi', 'pass_group'):
        for name in ('utility_train', 'selection', 'evaluation', 'fit_validation'):
            if set(table.iloc[tr][key]) & set(table.loc[table.role.eq(name), key]):
                raise ValueError(f'fit_train overlaps {name} in {key}.')
        if set(table.iloc[va][key]) & set(table.loc[~table.role.isin(['fit_train', 'fit_validation']), key]):
            raise ValueError(f'fit_validation overlaps downstream roles in {key}.')
    table['row_index'] = np.arange(len(table))
    # Downstream labels are never available to the training routines below.
    table.loc[~table.role.isin(['fit_train', 'fit_validation']), 'class_id'] = -1
    return table, tr, va


def aligned(path, table):
    with np.load(path) as z:
        idx = pd.Index(z['obs_id']).get_indexer(table.obs_id)
        if (idx < 0).any():
            raise ValueError(f'{path} is missing role-table observations.')
        return {key: z[key][idx] for key in z.files if z[key].ndim and len(z[key]) == len(z['obs_id'])}


def weight(table, tr):
    frame = table.iloc[tr]
    return len(tr)/(C*frame.groupby('mmsi').mmsi.transform('size').to_numpy()*
                    frame.groupby('class_id').mmsi.transform('nunique').to_numpy())


def metrics(y, scores):
    pred = scores.argmax(1)
    lp = scores-logsumexp(scores, axis=1, keepdims=True)
    loss = -lp[np.arange(len(y)), y]
    cls = np.unique(y)
    oa = float((pred == y).mean())
    ba = float(np.mean([(pred[y == c] == c).mean() for c in cls]))
    return dict(n=len(y), classes=len(cls), OA=oa, BA=ba, J=(oa+ba)/2,
                NLL=float(loss.mean()), balanced_NLL=float(np.mean([loss[y == c].mean() for c in cls])),
                Brier=float(((np.exp(lp)-np.eye(C)[y])**2).sum(1).mean()))


def scaled(raw, tr, classwise=False, floor=1e-8):
    axes = (0, 1) if classwise else 0
    mu = raw[tr].mean(axis=axes, keepdims=True)
    sd = np.maximum(raw[tr].std(axis=axes, keepdims=True), floor)
    return ((raw-mu)/sd).astype(np.float32), mu, sd


class Five(upstream.Five):
    def __init__(self):
        # Keep the original 64->8 latent bottleneck, 60 slots, output four classes.
        nn.Module.__init__(self)
        self.content = nn.Linear(64, 8)
        self.output = nn.Linear(60*8, C)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


class Neighborhood(upstream.Neighborhood):
    def __init__(self):
        nn.Module.__init__(self)
        self.content, self.output = nn.Linear(260, 32), nn.Linear(32, C)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


class Calibration(upstream.Calibration):
    def __init__(self):
        nn.Module.__init__(self)
        self.net = upstream.branch(C, 16, C)


def train_head(out, name, seed, model, x, base, table, tr, va, mu, scale, mse=False):
    """Original upstream optimizer, loss, epochs, checkpoint grid and strict ties."""
    tick = time.perf_counter()
    y = torch.tensor(table.iloc[tr].class_id.to_numpy(), dtype=torch.long)
    yv = table.iloc[va].class_id.to_numpy()
    xt, xv = torch.tensor(x[tr]), torch.tensor(x[va])
    bt = torch.tensor(base[tr], dtype=torch.float32)
    initial = upstream.state(model)
    with torch.no_grad():
        if model(xt).any() or model(xv).any():
            raise ValueError('Residual head must start at exact zero.')
    best_state, chosen = initial, 0
    met = metrics(yv, base[va])
    best = (met['BA'], -met['Brier']) if mse else upstream.exact_j(yv, base[va].argmax(1))
    curves = [dict(stage=name, seed=seed, epoch=0, train_loss=None, **met)]
    opt = torch.optim.AdamW(model.parameters(), lr=.001 if mse else .003,
                           weight_decay=.001 if mse else .01)
    rng = np.random.default_rng(seed)
    weights = torch.tensor(weight(table, tr), dtype=torch.float32)
    steps = 150 if mse else 200
    for step in range(1, steps+1):
        opt.zero_grad(set_to_none=True)
        if mse:
            pick = rng.integers(len(tr), size=128)
            target = 2*F.one_hot(y[pick], C).float()-1
            loss = ((bt[pick]+model(xt[pick])-target).square().mean(1)*weights[pick]).mean()
        else:
            loss = upstream.joint_loss(bt+model(xt), y)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss in {name}.')
        loss.backward()
        opt.step()
        if step % (50 if mse else 5):
            continue
        with torch.no_grad():
            score = base[va]+model(xv).numpy().astype(float)
        met = metrics(yv, score)
        curves.append(dict(stage=name, seed=seed, epoch=step, train_loss=float(loss.detach()), **met))
        rank = (met['BA'], -met['Brier']) if mse else upstream.exact_j(yv, score.argmax(1))
        if rank > best:
            best, chosen, best_state = rank, step, upstream.state(model)
    checkpoint = dict(stage=name, seed=seed, state=best_state, initial_state=initial,
        epoch=chosen, mu=mu, scale=scale, class_ids=np.arange(C),
        fit_indices=tr, validation_indices=va,
        fit_obs_id=table.iloc[tr].obs_id.to_numpy(), validation_obs_id=table.iloc[va].obs_id.to_numpy(),
        parameters=sum(p.numel() for p in model.parameters()), upstream_training_predictions_oof=False,
        selected_utc=upstream.utc(), source_algorithm=str(spec.origin),
        optimizer=dict(name='AdamW', lr=.001 if mse else .003, weight_decay=.001 if mse else .01),
        selection='BA then negative Brier' if mse else 'exact (OA+BA)/2; earliest strict maximum')
    path = out / 'models' / f'{name}_seed{seed}.pt'
    torch.save(checkpoint, path)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(restored['state'])
    model.eval()
    result = base+upstream.predict(model, x)
    with torch.no_grad():
        selected_val = base[va]+model(xv).numpy().astype(float)
    met = metrics(yv, selected_val)
    replay = (met['BA'], -met['Brier']) if mse else upstream.exact_j(yv, selected_val.argmax(1))
    if replay != best:
        raise ValueError('Restored checkpoint differs from validation selection.')
    batch_error = float(abs(result[va]-selected_val).max())
    np.testing.assert_array_equal(result[va].argmax(1), selected_val.argmax(1))
    if batch_error >= 2e-5:
        raise ValueError('Prediction batching changed selected scores.')
    pd.DataFrame(curves).to_csv(out / f'curves_{name}_{seed}.csv', index=False)
    write_json(out / f'selection_{name}_{seed}.json', dict(stage=name, seed=seed, epoch=chosen,
        seconds=time.perf_counter()-tick, parameters=checkpoint['parameters'], fit_validation=met,
        replay_selected=True, zero_step_exact=True, saved_vs_selection_batch_max_error=batch_error))
    print(name, seed, 'selected', chosen, 'valJ', met['J'], 'seconds', round(time.perf_counter()-tick, 2), flush=True)
    return result


def fit_reader(out, name, h, table, tr, va):
    scale = StandardScaler().fit(h[tr])
    # Fixed 128 matches the original reader; no test-dependent dimension choices.
    if min(len(tr), h.shape[1]) < 128:
        raise ValueError('Original 128-dimensional reader PCA requires >=128 fit observations.')
    pca = PCA(n_components=128, svd_solver='randomized', random_state=17,
              iterated_power=4).fit(scale.transform(h[tr]))
    post = StandardScaler().fit(pca.transform(scale.transform(h[tr])))
    def project(x):
        return post.transform(pca.transform(scale.transform(x)))
    ytr, yv = table.iloc[tr].class_id.to_numpy(), table.iloc[va].class_id.to_numpy()
    candidates, logs = [], []
    for alpha in (1., 10., 100.):
        head = Ridge(alpha=alpha).fit(project(h[tr]), np.eye(C)[ytr], sample_weight=weight(table, tr))
        met = metrics(yv, head.predict(project(h[va]))*3)
        candidates.append((met['balanced_NLL'], alpha, head))
        logs.append(dict(reader=name, alpha=alpha, **met))
    _, alpha, head = min(candidates, key=lambda item: item[:2])
    W = ((head.coef_/post.scale_) @ pca.components_)/scale.scale_*3
    intercept = head.predict(project(np.zeros((1, h.shape[1]))))[0]*3
    error = float(abs(h @ W.T+intercept-head.predict(project(h))*3).max())
    if error >= 1e-8:
        raise ValueError(f'Reader linearization error {error}.')
    np.savez_compressed(out / 'models' / f'{name}.npz', W=W, intercept=intercept, alpha=alpha,
        scale_mean=scale.mean_, scale_scale=scale.scale_, pca_mean=pca.mean_, pca_components=pca.components_,
        post_mean=post.mean_, post_scale=post.scale_, head_coef=head.coef_, head_intercept=head.intercept_,
        fit_obs_id=table.iloc[tr].obs_id.to_numpy(), validation_obs_id=table.iloc[va].obs_id.to_numpy())
    pd.DataFrame(logs).to_csv(out / f'{name}_alpha.csv', index=False)
    write_json(out / f'{name}_receipt.json', dict(alpha=alpha, linear_replay_max_error=error))
    print(name, 'alpha', alpha, 'linear replay', error, flush=True)


def fit_base(table, tr, va, cache, out, seeds):
    raw = aligned(cache / 'frontend_raw.npz', table)
    for key in ('global1', 'global3'):
        if np.any(raw[key][~raw['valid']]):
            raise ValueError('No-target polarization must be zero before reader fitting.')
    h = raw['axis']
    mu, sd = h[tr].mean(0), np.maximum(h[tr].std(0), 1e-6)
    x = (h-mu)/sd/np.sqrt(h.shape[1])
    candidates, logs = [], []
    for alpha in (.1, 1., 10.):
        model = RidgeClassifier(alpha=alpha, solver='cholesky').fit(
            x[tr], table.iloc[tr].class_id, sample_weight=weight(table, tr))
        met = metrics(table.iloc[va].class_id.to_numpy(), model.decision_function(x[va]))
        candidates.append((met['BA'], -met['Brier'], alpha, model))
        logs.append(dict(alpha=alpha, **met))
    _, _, alpha, model = max(candidates, key=lambda item: item[:2])
    ridge = model.decision_function(x).astype(float)
    np.savez_compressed(out / 'models/axis_ridge.npz', mean=mu, std=sd, alpha=alpha,
        coef=model.coef_, intercept=model.intercept_, fit_obs_id=table.iloc[tr].obs_id.to_numpy(),
        validation_obs_id=table.iloc[va].obs_id.to_numpy())
    pd.DataFrame(logs).to_csv(out / 'axis_alpha.csv', index=False)
    five = raw['five']
    fm, fs = five[tr].mean(0), np.maximum(five[tr].std(0), 1e-6)
    xf = ((five-fm)/fs).astype(np.float32)
    feat, level = np.zeros_like(raw['centered']), np.zeros_like(raw['means'])
    dm, ds, lm, ls = np.zeros((2, 128)), np.ones((2, 128)), np.zeros(2), np.ones(2)
    for p in range(2):
        rows = tr[raw['usable'][tr, p].astype(bool)]
        if not len(rows):
            raise ValueError(f'No source-fit usable neighborhood for polarization {p}.')
        dm[p] = raw['centered'][rows, p].mean(0)
        ds[p] = np.maximum(raw['centered'][rows, p].std(0), 1e-5)
        lm[p], ls[p] = raw['means'][rows, p].mean(), max(raw['means'][rows, p].std(), 1e-5)
        feat[:, p] = (raw['centered'][:, p]-dm[p])/ds[p]*raw['usable'][:, p, None]
        level[:, p] = (raw['means'][:, p]-lm[p])/ls[p]*raw['usable'][:, p]
    xd = np.concatenate((feat.reshape(len(table), -1), level, raw['usable'].astype(np.float32)), 1).astype(np.float32)
    np.savez_compressed(out / 'models/neighborhood_scaler.npz', mean=dm, std=ds,
                        level_mean=lm, level_std=ls, fit_obs_id=table.iloc[tr].obs_id.to_numpy())
    for seed in seeds:
        torch.manual_seed(seed)
        a = train_head(out, 'five', seed, Five(), xf, ridge, table, tr, va, fm, fs, mse=True)
        torch.manual_seed(seed)
        u = train_head(out, 'neighborhood', seed, Neighborhood(), xd, a, table, tr, va, None, None, mse=True)
        xt, tm, ts = scaled(upstream.center(u), tr)
        torch.manual_seed(seed)
        b = train_head(out, 'calibration', seed, Calibration(), xt, u, table, tr, va, tm, ts)
        np.savez_compressed(out / f'base_seed{seed}.npz', obs_id=table.obs_id.to_numpy(),
            axis=ridge, A=a, D=u-a, U=u, B=b, base=b, class_ids=np.arange(C))
    for layer, key in (('layer1', 'global1'), ('layer3', 'global3')):
        fit_reader(out, f'reader_{layer}', raw[key].reshape(len(table), -1).astype(float), table, tr, va)
    write_json(out / 'base_complete.json', dict(seeds=list(seeds), n=len(table), fit=len(tr), val=len(va)))


def regional_raw(base, summary, quality, geometry):
    # Same 69 shared-class features as old OBBpool; only broadcast class count changes.
    cols = [upstream.center(base)]
    for p in range(2):
        for r in range(3):
            for stat in range(3):
                value = summary[:, p, r, :, stat]
                cols.extend([value, upstream.center(value)])
    fields = []
    for p in range(2):
        g, q = geometry[:, p], quality[:, p]
        fields.extend([np.log1p(g[:, 0]), np.log1p(g[:, 1]), g[:, 2], g[:, 3]])
        for r in range(3):
            fields.extend([np.log1p(q[:, r, 0]), q[:, r, 1], np.log1p(q[:, r, 2]), q[:, r, 3]])
    q = np.stack(fields, -1)
    return np.concatenate((np.stack(cols, -1), np.broadcast_to(q[:, None], (len(base), C, 32))), -1)


def fit_heads(table, tr, va, out, seeds):
    sums = aligned(out / 'summaries.npz', table)
    for seed in seeds:
        b = aligned(out / f'base_seed{seed}.npz', table)['B']
        raw = attention_raw(b, sums['mean3'], sums['hb3'], sums['valid'])
        x, mu, sd = scaled(raw, tr, classwise=True)
        torch.manual_seed(seed)
        s1 = train_head(out, 'attention', seed, upstream.AttentionHead(), x, b, table, tr, va, mu, sd)
        raw = regional_raw(s1, sums['summary1'], sums['quality'], sums['geometry'])
        if raw.shape != (len(table), C, 69):
            raise ValueError('Old M2 feature contract failed.')
        x, mu, sd = scaled(raw, tr, classwise=True)
        torch.manual_seed(seed)
        s12 = train_head(out, 'full', seed, upstream.RegionHead(), x, s1, table, tr, va, mu, sd)
        np.savez_compressed(out / f'scores_seed{seed}.npz', obs_id=table.obs_id.to_numpy(),
            sample_id=table.sample_id.to_numpy(str), role=table.role.to_numpy(str),
            base=b, B=b, m1=s1, s1=s1, s12=s12, delta_m1=s1-b, delta_m2=s12-s1,
            class_ids=np.arange(C), seed=seed)
        print('READY', out / f'scores_seed{seed}.npz', s12.shape, flush=True)
    write_json(out / 'complete.json', dict(seeds=list(seeds), n=len(table), fit=len(tr), val=len(va),
        fit_role='fit_train', selection_role='fit_validation', downstream_labels_used=False,
        baseline='axis Ridge + original five-point A + isolated-context D + calibration T',
        M1='mean-inclusive two-sided attention', M2='original 69-12-1 OBBpool',
        feature_weights='ImageNet R50 V2 only', whole_stack_oof=False, test_metrics_computed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'base', 'summaries', 'heads'], nargs='?', default='all')
    parser.add_argument('--table', type=Path, default=HERE / 'stack_table.csv')
    parser.add_argument('--fold')
    parser.add_argument('--cache', type=Path, default=HERE / 'frontend_cache')
    parser.add_argument('--out', type=Path, default=HERE / 'frontend')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--weights', type=Path, default=extract.WEIGHTS)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(HERE):
        raise ValueError('Output must remain in the authorized scope.')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'models').mkdir(exist_ok=True)
    table, tr, va = load_table(args.table, args.fold)
    write_json(args.out / 'config.json', dict(table=str(args.table), cache=str(args.cache),
        obs_id=table.obs_id.tolist(), seeds=args.seeds, classes=C, fit=len(tr), val=len(va),
        legacy_steps=150, new_steps=200, reader_pca_dim=128, backbone_training=False,
        class_map={'0': 0, '2': 1, '3': 2, '4': 3}, downstream_labels_masked=True))
    if args.stage in ('all', 'base'):
        fit_base(table, tr, va, args.cache, args.out, args.seeds)
    if args.stage in ('all', 'summaries'):
        extract.extract_summaries(table, tr, args.cache, args.out / 'models', args.out, args.device, args.weights)
    if args.stage in ('all', 'heads'):
        fit_heads(table, tr, va, args.out, args.seeds)


if __name__ == '__main__':
    main()
