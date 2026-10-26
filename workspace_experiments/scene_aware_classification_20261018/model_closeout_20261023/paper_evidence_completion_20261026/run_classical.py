"""Source-only frozen-R50 baselines on fixed public and cross-dataset cohorts."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import logsumexp
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRANSFER = ROOT / 'public_m123_cross_dataset_20261026'
sys.path.insert(0, str(TRANSFER))
import transfer as tr

SEEDS = (17, 29, 43)
ALPHAS = (.01, .1, 1., 10., 100., 1000.)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def load_selected(path, table, keys):
    with np.load(path) as z:
        ix = pd.Index(z['obs_id']).get_indexer(table.obs_id)
        if (ix < 0).any():
            raise ValueError(f'Missing rows: {path}')
        return {k: z[k][ix] for k in keys}


def dataset(name):
    source = ROOT / f'{name}_m123_replication_20261026'
    other = 'oss2' if name == 'oss1' else 'oss1'
    cross = TRANSFER / f'{name}-to-{other}'
    protocol = tr.read_json(cross / 'protocol.json')
    table = pd.read_csv(source / 'stack_table.csv', dtype={'mmsi': str})
    # Target labels are not exposed during fitting or prediction.
    target = tr.read_table(cross / 'target_table.csv')
    fields = ('global1', 'global3', 'valid')
    raw = load_selected(source / 'frontend_cache/frontend_raw.npz', table, fields)
    ext = load_selected(Path(protocol['target_cache']) / 'frontend_raw.npz', target, fields)
    mfields = ('layer1', 'mass')
    local = load_selected(source / 'frontend_cache/m3_raw.npz', table, mfields)
    local_ext = load_selected(Path(protocol['target_cache']) / 'm3_raw.npz', target, mfields)
    return source, cross, table, target, raw, ext, local, local_ext


def representations(raw, local):
    valid = raw['valid'].astype(np.float32)
    global3 = (raw['global3']*valid[..., None]).reshape(len(valid), -1)
    global1 = (raw['global1']*valid[..., None]).reshape(len(valid), -1)
    chunks = []
    for ids in (np.arange(8), np.arange(8, 16), np.arange(16, 20)):
        mass = local['mass'][:, :, ids]
        values = (local['layer1'][:, :, ids]*mass[..., None]).sum(2)
        chunks.append(values / np.maximum(mass.sum(2)[..., None], 1e-8))
    regional = np.stack(chunks, 2).reshape(len(valid), -1)
    return {
        'global3': np.c_[global3, valid].astype(np.float32),
        'multilevel': np.c_[global1, global3, valid].astype(np.float32),
        'regional': np.c_[regional, valid].astype(np.float32),
    }


def standardize(x, fit):
    mu, sd = x[fit].mean(0), x[fit].std(0)
    active = np.ptp(x[fit], axis=0) > 0
    scale = np.where(active, np.maximum(sd, 1e-6), 1)
    return dict(mean=mu, scale=scale, active=active)


def transform(x, scale):
    return np.where(scale['active'], (x-scale['mean'])/scale['scale'], 0).astype(np.float32)


def sample_weights(table, fit):
    f = table.iloc[fit]
    # A repeated vessel does not receive extra class mass.
    ships = f.groupby('class_id').mmsi.transform('nunique').to_numpy()
    repeat = f.groupby('mmsi').mmsi.transform('size').to_numpy()
    return len(f)/(f.class_id.nunique()*ships*repeat)


def classification_metrics(y, score):
    correct = score.argmax(1) == y
    lp = score-logsumexp(score, axis=1, keepdims=True)
    oa = float(correct.mean())
    ba = float(np.mean([correct[y == c].mean() for c in np.unique(y)]))
    return dict(n=len(y), OA=oa, BA=ba, J=(oa+ba)/2,
                NLL=float(-lp[np.arange(len(y)), y].mean()),
                Brier=float(((np.exp(lp)-np.eye(score.shape[1])[y])**2).sum(1).mean()))


def ranking(y, score):
    return tr.m3.exact_j(y, score)


def train_ridge(x, xe, table, fit, val, out):
    y = table.class_id.to_numpy(int)
    target = 2*np.eye(int(y[fit].max())+1)[y[fit]]-1
    best, chosen, rows = None, None, []
    tick = time.perf_counter()
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver='lsqr', tol=1e-6, max_iter=10000)
        model.fit(x[fit], target, sample_weight=sample_weights(table, fit))
        score = model.predict(x[val])
        j = ranking(y[val], score)
        rows.append(dict(alpha=alpha, **classification_metrics(y[val], score)))
        if best is None or j > best:
            best, chosen = j, model
    source, external = chosen.predict(x), chosen.predict(xe)
    np.savez_compressed(out/'ridge_model.npz', coef=chosen.coef_, intercept=chosen.intercept_)
    pd.DataFrame(rows).to_csv(out/'ridge_selection.csv', index=False)
    return source, external, dict(alpha=float(chosen.alpha), seconds=time.perf_counter()-tick,
                                  parameters=int(chosen.coef_.size+chosen.intercept_.size))


def train_mlp(x, xe, table, fit, val, seed, out):
    torch.manual_seed(seed)
    classes = int(table.iloc[fit].class_id.max())+1
    model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(), nn.Linear(128, classes))
    xt = torch.tensor(x)
    y = table.class_id.to_numpy(int)
    labels = torch.tensor(y[fit])
    weights = torch.tensor(sample_weights(table, fit), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    best, chosen, chosen_step = None, None, None
    curves = []
    tick = time.perf_counter()
    for step in range(1, 201):
        opt.zero_grad(set_to_none=True)
        score = model(xt[fit])
        loss = (nn.functional.cross_entropy(score, labels, reduction='none')*weights).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite source loss')
        loss.backward()
        opt.step()
        if step % 5 == 0:
            with torch.inference_mode():
                v = model(xt[val]).numpy()
            j = ranking(y[val], v)
            curves.append(dict(step=step, loss=float(loss.detach()), **classification_metrics(y[val], v)))
            if best is None or j > best:
                best, chosen_step = j, step
                chosen = {k: z.detach().clone() for k, z in model.state_dict().items()}
    model.load_state_dict(chosen)
    with torch.inference_mode():
        source, external = model(xt).numpy(), model(torch.tensor(xe)).numpy()
    torch.save(dict(state=chosen, step=chosen_step), out/f'mlp_seed{seed}.pt')
    pd.DataFrame(curves).to_csv(out/f'mlp_seed{seed}_selection.csv', index=False)
    return source, external, dict(step=chosen_step, seconds=time.perf_counter()-tick,
                                  parameters=sum(p.numel() for p in model.parameters()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=('oss1', 'oss2'), required=True)
    parser.add_argument('--training-role', choices=('pooled', 'fit_only'), default='pooled')
    args = parser.parse_args()
    torch.set_num_threads(2)
    source, cross, table, target, raw, ext, local, local_ext = dataset(args.source)
    out = HERE / ('classical' if args.training_role == 'pooled' else 'classical_fit_only') / args.source
    out.mkdir(parents=True, exist_ok=True)
    fit_roles = ('fit_train', 'utility_train') if args.training_role == 'pooled' else ('fit_train',)
    val_roles = ('fit_validation', 'selection') if args.training_role == 'pooled' else ('fit_validation',)
    fit = np.flatnonzero(table.role.isin(fit_roles))
    val = np.flatnonzero(table.role.isin(val_roles))
    te = np.flatnonzero(table.role.eq('evaluation'))
    for key in ('mmsi', 'pass_group'):
        assert not set(table.iloc[fit][key]) & set(table.iloc[val][key])
        assert not set(table.iloc[np.r_[fit,val]][key]) & set(table.iloc[te][key])
    X, XE = representations(raw, local), representations(ext, local_ext)
    outputs, receipts = {}, []
    for name, x in X.items():
        dest = out/name
        dest.mkdir(exist_ok=True)
        scale = standardize(x, fit)
        np.savez_compressed(dest/'scaler.npz', **scale)
        x, xe = transform(x, scale), transform(XE[name], scale)
        s, e, receipt = train_ridge(x, xe, table, fit, val, dest)
        outputs[f'{name}_ridge'] = (s, e)
        receipts.append(dict(method=f'{name}_ridge', seed=None, **receipt))
        for seed in SEEDS:
            s, e, receipt = train_mlp(x, xe, table, fit, val, seed, dest)
            outputs[f'{name}_mlp_seed{seed}'] = (s, e)
            receipts.append(dict(method=f'{name}_mlp', seed=seed, **receipt))
        print('FROZEN', args.source, name, flush=True)
    np.savez_compressed(out/'source_predictions.npz', obs_id=table.obs_id.to_numpy(),
                        **{k:v[0] for k,v in outputs.items()})
    np.savez_compressed(out/'cross_predictions.npz', obs_id=target.obs_id.to_numpy(),
                        **{k:v[1] for k,v in outputs.items()})
    pd.DataFrame(receipts).to_csv(out/'training_receipts.csv', index=False)
    # All methods and source checkpoints now frozen; evaluate targets once.
    external_labels = tr.read_table(cross/'target_table.csv', labels=True)
    np.testing.assert_array_equal(target.obs_id, external_labels.obs_id)
    metrics = []
    for key, (s,e) in outputs.items():
        for cohort, yy, score in [('within', table.iloc[te].class_id.to_numpy(int), s[te]),
                                  ('cross', external_labels.class_id.to_numpy(int), e)]:
            metrics.append(dict(source=args.source, cohort=cohort, method=key, **tr.m3.metrics(yy,score)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    write_json(out/'complete.json', dict(source=args.source, train=len(fit), validation=len(val),
         within=len(te), cross=len(target), seeds=SEEDS, target_selection=False,
         conventional_training=args.training_role, native_features=True))
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
