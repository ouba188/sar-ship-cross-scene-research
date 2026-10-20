"""Source-only compact relation probe on frozen native strip tokens."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
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

ROOT = Path(__file__).resolve().parent
STRIP = ROOT.parent / 'axial_strip_sequence_20261020'
spec = importlib.util.spec_from_file_location('strip_previous', STRIP / 'run.py')
strip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strip)
prior, five, LOCAL = strip.prior, strip.five, strip.LOCAL
ARMS = ('marginal', 'factorized', 'joint', 'factorized_condition', 'joint_condition', 'joint_mismatched')


def load_data():
    f = pd.read_csv(prior.ROOT / 'cohort.csv', dtype={'mmsi': str})
    excluded = pd.read_csv(LOCAL / 'EXCLUSIONS.csv').obs_id.to_numpy()
    splits = []
    for fold in range(3):
        splits.append(tuple(ii[~f.iloc[ii].obs_id.isin(excluded).to_numpy()] for ii in prior.indices(f, fold)))
    return f, splits


def relation_moments(x, m):
    # [B, pol, rotation, reference, lane, length, feature/role].
    m = m.to(x.dtype)
    c = x[..., 0, :, :].unsqueeze(-3)
    s = x[..., 1:, :, :]
    q = m[..., 0, :, 0].unsqueeze(-2) * m[..., 1:, :, 1]
    den = q.sum(-1)
    mc = (c * q[..., None]).sum(-2) / den[..., None].clamp_min(1e-8)
    ms = (s * q[..., None]).sum(-2) / den[..., None].clamp_min(1e-8)
    joint = torch.einsum('...ti,...tj,...t->...ij', c.expand_as(s), s, q)
    joint = joint / den[..., None, None].clamp_min(1e-8)
    factor = mc[..., :, None] * ms[..., None, :]
    def pool(z):
        # Keep pol/reference; average sides and complete C4 observations.
        return z.flatten(-2).mean(-2).mean(2)
    marginal = torch.cat((mc, ms), -1).mean(-2).mean(2)
    present = (den > 0).any(-1).any(2)
    return marginal, pool(factor), pool(joint), present, den


def conditions(f):
    views = json.loads((LOCAL / 'view_snapshot.json').read_text(encoding='utf-8'))
    assert [v['obs_id'] for v in views] == f.obs_id.tolist()
    g = np.zeros((len(f), 2, 3), np.float32)
    valid = np.zeros((len(f), 2), bool)
    for i, view in enumerate(views):
        for p, pol in enumerate(('vv', 'vh')):
            v = view['images'][pol]
            geom, _ = five.geometry(v)
            inc = v['priors'].get('radar_incidence')
            if geom[2] and geom[3] and inc is not None and np.isfinite(inc):
                angle = geom[0] - geom[1]
                g[i, p] = (np.cos(2 * angle), np.sin(2 * angle), inc)
                valid[i, p] = True
    return g, valid


def extract():
    torch.set_num_threads(2)
    start = time.perf_counter()
    f, splits = load_data()
    x = np.load(STRIP / 'tokens.npy', mmap_mode='r')
    m = np.load(STRIP / 'masses.npy', mmap_mode='r')
    condition, condition_valid = conditions(f)
    records = []
    for fold, (tr, val, test) in enumerate(splits):
        old = torch.load(STRIP / f'head_f{fold}_strip_conv_s0.pt', map_location='cpu', weights_only=False)
        mean, std = old['mean'].cuda(), old['std'].cuda()
        cov = torch.zeros(64, 64, dtype=torch.float64, device='cuda')
        total = torch.zeros(64, dtype=torch.float64, device='cuda')
        count = 0.
        for first in range(0, len(tr), 16):
            ii = tr[first:first + 16]
            xx = (torch.tensor(np.array(x[ii]), device='cuda', dtype=torch.float32) - mean) / std
            ww = torch.tensor(np.array(m[ii]), device='cuda').sum(-1).clamp_max(1)
            xx = xx.reshape(-1, 64).double(); ww = ww.flatten().double()
            cov += xx.T @ (xx * ww[:, None]); total += (xx * ww[:, None]).sum(0)
            count += ww.sum().item()
        center = total / count
        cov = cov / count - center[:, None] * center[None, :]
        eigen, vectors = torch.linalg.eigh(cov)
        projection = vectors[:, -8:].float(); center = center.float()
        explained = float(eigen[-8:].sum() / eigen.clamp_min(0).sum())
        blocks = {name: [] for name in ('marginal', 'factorized', 'joint', 'present')}
        coverage = []
        for first in range(0, len(f), 32):
            ii = np.arange(first, min(first + 32, len(f)))
            xx = (torch.tensor(np.array(x[ii]), device='cuda', dtype=torch.float32) - mean) / std
            xx = (xx - center) @ projection
            mm = torch.tensor(np.array(m[ii]), device='cuda')
            ma, fa, jo, present, den = relation_moments(xx, mm)
            for name, values in zip(blocks, (ma, fa, jo, present)):
                blocks[name].append(values.cpu().numpy())
            coverage.append(den.mean((1, 2, 3, 4)).cpu().numpy())
        values = {k: np.concatenate(v) for k, v in blocks.items()}
        np.savez_compressed(ROOT / f'features_f{fold}.npz', **values, condition=condition,
                            condition_valid=condition_valid, projection=projection.cpu().numpy(),
                            center=center.cpu().numpy(), mean=mean.cpu().numpy(), std=std.cpu().numpy())
        for name, ii in zip(('train', 'val', 'test'), (tr, val, test)):
            records.append(dict(fold=fold, partition=name, n=len(ii), ships=f.iloc[ii].mmsi.nunique(),
                                pca8_variance=explained, pair_any=float(values['present'][ii].any((1, 2)).mean()),
                                paired_mass_mean=float(np.concatenate(coverage)[ii].mean())))
        pd.DataFrame(records).to_csv(ROOT / 'coverage.csv', index=False)
        print('extracted', fold, 'variance', explained, flush=True)
    (ROOT / 'EXTRACTION.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,
        source_pca=True, rank=8, feature_shapes={k:list(v.shape) for k,v in values.items()}), indent=2), encoding='utf-8')


def standard_block(x, present, tr):
    out = np.zeros_like(x)
    for p in range(2):
        for ref in range(2):
            ii = tr[present[tr, p, ref]]
            if not len(ii):
                continue
            mu = x[ii, p, ref].mean(0)
            sd = np.maximum(x[ii, p, ref].std(0), 1e-5)
            out[:, p, ref] = (x[:, p, ref] - mu) / sd
    return out * present[..., None]


def matrices(data, f, splits, fold):
    tr, val, test = splits
    present = data['present'].astype(bool)
    g = data['condition'].copy(); valid = data['condition_valid'].copy()
    for p in range(2):
        ii = tr[valid[tr, p]]
        mu = g[ii, p].mean(0); sd = np.maximum(g[ii, p].std(0), 1e-5)
        g[:, p] = (g[:, p] - mu) / sd
    g *= valid[..., None]
    wrong = g.copy(); donors = []
    rng = np.random.default_rng(20261020 + fold)
    for part, ii in zip(('train', 'val', 'test'), splits):
        for p in range(2):
            donor = np.arange(len(f))
            subset = f.iloc[ii[valid[ii, p]]]
            for _, group in subset.groupby(['port', 'platform', 'quarter']):
                jj = group.index.to_numpy()
                donor[jj] = rng.permutation(jj)
            wrong[ii, p] = g[donor[ii], p]
            donors.append(dict(fold=fold, partition=part, pol=p,
                changed_fraction=float((np.abs(wrong[ii, p]-g[ii, p]).max(-1)>1e-6).mean())))
    z = {key: standard_block(data[key], present, tr) for key in ('marginal', 'factorized', 'joint')}
    def conditional(base, cond):
        product = (base[..., None] * cond[:, :, None, None, :]).reshape(*base.shape[:-1], -1)
        product = standard_block(product, present & valid[:, :, None], tr)
        return np.concatenate((base, product), -1)
    z.update(factorized_condition=conditional(z['factorized'], g),
             joint_condition=conditional(z['joint'], g),
             joint_mismatched=conditional(z['joint'], wrong))
    # Conditional blocks remain zero when that polarization has no valid geometry.
    z = {key: value.reshape(len(f), -1) for key, value in z.items()}
    return z, donors


def train():
    torch.set_num_threads(2)
    start = time.perf_counter(); f, splits = load_data(); y = f.class_id.to_numpy()
    yy = torch.tensor(y, device='cuda')
    raw = np.load(five.AXIS / 'descriptors.npz')
    raw_base = np.concatenate((raw['base'], raw['axis']), 1)
    raw_five = np.load(strip.PREV / 'keep_refs.npy', mmap_mode='r').reshape(len(f), 60, 64)
    metrics = []; selections = []; predictions = []; controls = []; mismatch = []; replay = []
    for fold, (tr, val, test) in enumerate(splits):
        features, donors = matrices(np.load(ROOT/f'features_f{fold}.npz'), f, splits[fold], fold)
        mismatch.extend(donors)
        features = {k: torch.tensor(v, device='cuda', dtype=torch.float32) for k,v in features.items()}
        weights = torch.tensor(prior.weights(f,tr), device='cuda', dtype=torch.float32)
        for seed in range(3):
            base_np = strip.frozen_scores(fold, seed, raw_base, raw_five)
            base = torch.tensor(base_np, device='cuda')
            previous = np.load(STRIP / f'scores_f{fold}_frozen_fivepoint_s{seed}.npz')
            np.testing.assert_array_equal(test, previous['rows'])
            err = float(np.abs(base_np[test] - previous['scores']).max())
            assert err < 1e-5
            replay.append(dict(fold=fold, seed=seed, max_abs=err))
            np.savez_compressed(ROOT/f'scores_f{fold}_baseline_s{seed}.npz', rows=test, scores=base_np[test], base=base_np[test])
            metrics.append(dict(fold=fold, seed=seed, arm='baseline', partition='test', step=0, **prior.metrics(y[test],base_np[test])))
            for arm in ARMS:
                xx = features[arm]
                torch.manual_seed(20261020+seed)
                head = nn.Linear(xx.shape[1],8,bias=False).cuda(); nn.init.zeros_(head.weight)
                optimizer = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
                rng = np.random.default_rng(20261020+seed+31*fold)
                best = None; state = None; chosen = 0
                for step in range(151):
                    if step in (0,50,100,150):
                        with torch.no_grad():
                            score = (base[val]+head(xx[val])).cpu().numpy()
                        met = prior.metrics(y[val],score)
                        selections.append(dict(fold=fold,seed=seed,arm=arm,step=step,**met))
                        key = (met['ba'],-met['brier'])
                        if best is None or key > best:
                            best=key; chosen=step; state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                        if step==0:
                            np.testing.assert_array_equal(score,base_np[val])
                    if step==150:
                        break
                    take=rng.integers(len(tr),size=128); ii=tr[take]
                    wanted=2*F.one_hot(yy[ii],8).float()-1
                    loss=(((base[ii]+head(xx[ii])-wanted).square().mean(-1))*weights[take]).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError((fold,seed,arm,step))
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                with torch.no_grad():
                    for part,ii in (('terminal_train',tr),('terminal_val',val)):
                        scores=(base[ii]+head(xx[ii])).cpu().numpy()
                        metrics.append(dict(fold=fold,seed=seed,arm=arm,partition=part,step=150,**prior.metrics(y[ii],scores)))
                head.load_state_dict(state)
                torch.save(dict(state=state,step=chosen,dimension=xx.shape[1]),ROOT/f'head_f{fold}_{arm}_s{seed}.pt')
                with torch.no_grad():
                    residual=head(xx).cpu().numpy(); scores=base_np[test]+residual[test]
                for part,ii in (('train',tr),('val',val),('test',test)):
                    metrics.append(dict(fold=fold,seed=seed,arm=arm,partition=part,step=chosen,
                                        **prior.metrics(y[ii],base_np[ii]+residual[ii])))
                constant=residual[tr].mean(0)
                controls.append(dict(fold=fold,seed=seed,arm=arm,step=chosen,
                    image_vs_constant_flips=int(np.sum(scores.argmax(1)!=(base_np[test]+constant).argmax(1))),
                    **prior.metrics(y[test],base_np[test]+constant)))
                np.savez_compressed(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz',rows=test,scores=scores,base=base_np[test],constant=constant)
                for j,i in enumerate(test):
                    rival=int(np.argmax(np.where(np.arange(8)==y[i],-np.inf,base_np[i])))
                    predictions.append(dict(fold=fold,seed=seed,arm=arm,step=chosen,obs_id=int(f.iloc[i].obs_id),
                        mmsi=f.iloc[i].mmsi,port=f.iloc[i].port,class_id=int(y[i]),prediction=int(scores[j].argmax()),
                        base_prediction=int(base_np[i].argmax()),margin=float(scores[j,y[i]]-scores[j,rival]),
                        base_margin=float(base_np[i,y[i]]-base_np[i,rival])))
                print('fit',fold,seed,arm,chosen,metrics[-1]['oa'],metrics[-1]['ba'],flush=True)
            for filename, values in (('metrics.csv',metrics),('validation_selection.csv',selections),
                    ('predictions.csv',predictions),('source_constant.csv',controls),
                    ('condition_mismatch.csv',mismatch),('baseline_replay.csv',replay)):
                pd.DataFrame(values).to_csv(ROOT/filename,index=False)
    (ROOT/'RECEIPT.json').write_text(json.dumps(dict(status='complete',fits=54,seconds=time.perf_counter()-start,
        arms=ARMS,seeds=3,folds=3,steps=150,new_convolution=False,source_only_pca=True,
        scope='frozen native local relation readout; not full support adaptation'),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('extract','train'))
    args=parser.parse_args()
    extract() if args.stage=='extract' else train()
