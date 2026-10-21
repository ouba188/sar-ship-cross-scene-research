"""Separate neighborhood display level from local variation on a frozen model."""
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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


control = load_module('input_controls_level', ROOT.parent / 'crude_input_controls_20261021/input_controls.py')
relation = load_module('paired_relation_level', ROOT.parent / 'paired_relation_probe_20261020/run.py')
prior, strip, five = relation.prior, relation.strip, relation.five
ARMS = ('raw', 'centered', 'separated')


def encode(x, region, centered):
    mu = (x * region).sum((2, 3), keepdim=True) / region.sum((2, 3), keepdim=True).clamp_min(1)
    value = x - mu if centered else x
    return .5 + .5 * value * region, mu.flatten(1)


def descriptor(stem, encoded, region):
    mean = encoded.new_tensor([.485, .456, .406])[None, :, None, None]
    std = encoded.new_tensor([.229, .224, .225])[None, :, None, None]
    result = 0
    for k in range(4):
        x = torch.rot90(encoded, k, (-2, -1)).expand(-1, 3, -1, -1)
        features = F.avg_pool2d(stem((x - mean) / std), 2)
        result = result + control.pool(features, torch.rot90(region, k, (-2, -1))) / 4
    return result


def extract():
    start = time.perf_counter()
    f, views, disabled = control.load_data()
    stem = control.make_stem()
    values = {arm: np.zeros((len(f), 2, 128), np.float32) for arm in ('raw', 'centered')}
    means = np.zeros((len(f), 2), np.float32)
    usable = np.zeros((len(f), 2), bool)
    coverage, checks = [], []
    with torch.inference_mode():
        for first in range(0, len(f), 24):
            images, regions, addresses = [], [], []
            for i in range(first, min(first + 24, len(f))):
                row = f.iloc[i]
                if row.obs_id == 31237:
                    continue
                for p, pol in enumerate(('vv', 'vh')):
                    x, masks, _, _, place = control.prepare_view(row, views[i]['images'][pol], pol)
                    ok = bool(masks[0].any() and masks[2].any() and not (p == 1 and disabled[i]))
                    usable[i, p] = ok
                    coverage.append(dict(obs_id=int(row.obs_id), pol=pol, usable=ok,
                                         alignment_disabled=bool(p == 1 and disabled[i]), **place))
                    if not ok:
                        continue
                    images.append(x); regions.append(masks[2:3]); addresses.append((i, p))
            if not images:
                continue
            x = torch.tensor(np.stack(images), device='cuda')
            r = torch.tensor(np.stack(regions), device='cuda')
            for arm in values:
                z, mu = encode(x, r, arm == 'centered')
                assert float(z.min()) >= -1e-6 and float(z.max()) <= 1 + 1e-6
                feat = descriptor(stem, z, r).cpu().numpy()
                level = mu.cpu().numpy()[:, 0]
                for b, (i, p) in enumerate(addresses):
                    values[arm][i, p] = feat[b]
                    means[i, p] = level[b]
            if first == 0:
                xx, rr = x[:2], r[:2]
                centered, _ = encode(xx, rr, True)
                shifted, _ = encode(xx + .123 * rr, rr, True)
                err = float((centered - shifted).abs().max())
                assert err < 2e-7
                checks.append(dict(test='centered_encoding_additive_shift', max_abs=err))
                err = float((descriptor(stem, centered, rr) - descriptor(stem, shifted, rr)).abs().max())
                assert err < 2e-5
                checks.append(dict(test='centered_descriptor_additive_shift', max_abs=err))
                for centered_flag in (False, True):
                    z, _ = encode(xx, rr, centered_flag)
                    z2, _ = encode(xx * rr + .731 * (1-rr), rr, centered_flag)
                    err = float((z-z2).abs().max())
                    assert err == 0
                    checks.append(dict(test=f'outside_R_isolation_{centered_flag}', max_abs=err))
                d = descriptor(stem, centered, rr)
                d2 = descriptor(stem, torch.rot90(centered, 1, (-2,-1)), torch.rot90(rr, 1, (-2,-1)))
                err = float((d-d2).abs().max())
                assert err < 2e-5
                checks.append(dict(test='C4_joint_rotation', max_abs=err))
            if first % 480 == 0:
                print('extract', first, len(f), round(time.perf_counter()-start, 1), flush=True)
    np.savez_compressed(ROOT/'features.npz', **values, means=means, usable=usable)
    pd.DataFrame(coverage).to_csv(ROOT/'input_coverage.csv', index=False)
    pd.DataFrame(checks).to_csv(ROOT/'input_checks.csv', index=False)
    (ROOT/'EXTRACTION.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,
        rows=len(f), vv_usable=int(usable[:,0].sum()), vh_usable=int(usable[:,1].sum()),
        feature_shape=list(values['raw'].shape), native_canvas=160, resize=False), indent=2), encoding='utf-8')


def standardize(cache, tr):
    usable = cache['usable'].astype(bool)
    out, stats = {}, {}
    for name in ('raw', 'centered'):
        data = cache[name]
        z = np.zeros_like(data); mu = np.zeros((2,128), np.float32); sd = np.ones_like(mu)
        for p in range(2):
            rows = tr[usable[tr,p]]
            mu[p] = data[rows,p].mean(0)
            sd[p] = np.maximum(data[rows,p].std(0), 1e-5)
            z[:,p] = ((data[:,p]-mu[p])/sd[p]) * usable[:,p,None]
        out[name] = z.reshape(len(data), -1)
        stats[name+'_mean'] = mu; stats[name+'_std'] = sd
    level = np.zeros_like(cache['means'])
    level_mean = np.zeros(2, np.float32); level_std = np.ones(2, np.float32)
    for p in range(2):
        rows = tr[usable[tr,p]]
        level_mean[p] = cache['means'][rows,p].mean()
        level_std[p] = max(cache['means'][rows,p].std(), 1e-5)
        level[:,p] = (cache['means'][:,p]-level_mean[p])/level_std[p]*usable[:,p]
    stats.update(level_mean=level_mean, level_std=level_std)
    matrices = {}
    for arm in ARMS:
        content = out['raw' if arm == 'raw' else 'centered']
        levels = level if arm == 'separated' else np.zeros_like(level)
        matrices[arm] = np.concatenate((content, levels, usable.astype(np.float32)), 1)
    np.testing.assert_array_equal(matrices['centered'][:,:256], matrices['separated'][:,:256])
    return matrices, stats


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.content = nn.Linear(260, 32)
        self.output = nn.Linear(32, 8)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(F.gelu(self.content(x)))


def train():
    torch.set_num_threads(2)
    start = time.perf_counter()
    f, splits = relation.load_data(); y = f.class_id.to_numpy()
    yy = torch.tensor(y, device='cuda')
    cache = np.load(ROOT/'features.npz')
    raw = np.load(five.AXIS/'descriptors.npz')
    raw_base = np.concatenate((raw['base'],raw['axis']),1)
    raw_five = np.load(strip.PREV/'keep_refs.npy',mmap_mode='r').reshape(len(f),60,64)
    measures, selections, predictions, diagnostics, replays, partitions = [], [], [], [], [], []
    for fold, (tr, val, test) in enumerate(splits):
        matrices, stats = standardize(cache, tr)
        np.savez_compressed(ROOT/f'normalization_f{fold}.npz', **stats)
        matrices = {k:torch.tensor(v,device='cuda') for k,v in matrices.items()}
        weights = torch.tensor(prior.weights(f,tr),device='cuda',dtype=torch.float32)
        for part, ii in zip(('train','val','test'), (tr,val,test)):
            partitions.extend(dict(split_fold=fold, partition=part, obs_id=int(f.iloc[i].obs_id)) for i in ii)
        for seed in range(3):
            base_np = strip.frozen_scores(fold,seed,raw_base,raw_five)
            base = torch.tensor(base_np,device='cuda')
            old = np.load(relation.ROOT/f'scores_f{fold}_baseline_s{seed}.npz')
            np.testing.assert_array_equal(test, old['rows'])
            err = float(np.abs(base_np[test]-old['scores']).max())
            assert err < 1e-5
            replays.append(dict(fold=fold,seed=seed,max_abs=err))
            np.savez_compressed(ROOT/f'scores_f{fold}_baseline_s{seed}.npz',rows=test,scores=base_np[test])
            for part, ii in zip(('train','val','test'), (tr,val,test)):
                measures.append(dict(fold=fold,seed=seed,arm='baseline',partition=part,step=0,
                                     **control.metrics(y[ii],base_np[ii])))
            for arm in ARMS:
                xx = matrices[arm]
                torch.manual_seed(20261021+seed)
                head = Residual().cuda()
                optimizer = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
                rng = np.random.default_rng(20261021+seed+31*fold)
                best, state, chosen = None, None, 0
                for step in range(151):
                    if step in (0,50,100,150):
                        with torch.no_grad():
                            score = (base[val]+head(xx[val])).cpu().numpy()
                        metric = control.metrics(y[val],score)
                        selections.append(dict(fold=fold,seed=seed,arm=arm,step=step,**metric))
                        key = (metric['ba'],-metric['brier'])
                        if best is None or key > best:
                            best=key; chosen=step
                            state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                        if step==0:
                            np.testing.assert_array_equal(score,base_np[val])
                    if step==150:
                        break
                    take = rng.integers(len(tr),size=128); ii=tr[take]
                    target = 2*F.one_hot(yy[ii],8).float()-1
                    loss = ((base[ii]+head(xx[ii])-target).square().mean(-1)*weights[take]).mean()
                    assert torch.isfinite(loss), (fold,seed,arm,step)
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                with torch.no_grad():
                    for part, ii in (('terminal_train',tr),('terminal_val',val)):
                        scores = (base[ii]+head(xx[ii])).cpu().numpy()
                        measures.append(dict(fold=fold,seed=seed,arm=arm,partition=part,step=150,
                                             **control.metrics(y[ii],scores)))
                head.load_state_dict(state)
                torch.save(dict(state=state,step=chosen,parameters=sum(p.numel() for p in head.parameters())),
                           ROOT/f'head_f{fold}_{arm}_s{seed}.pt')
                with torch.no_grad():
                    residual = head(xx).cpu().numpy()
                    fixed_mean = xx[test].clone(); fixed_mean[:,256:258]=0
                    mean_control = base_np[test]+head(fixed_mean).cpu().numpy()
                for part, ii in zip(('train','val','test'), (tr,val,test)):
                    measures.append(dict(fold=fold,seed=seed,arm=arm,partition=part,step=chosen,
                                         **control.metrics(y[ii],base_np[ii]+residual[ii])))
                scores = base_np[test]+residual[test]
                constant = residual[tr].mean(0)
                for kind, diagnostic in (('source_constant',base_np[test]+constant),('source_mean_level',mean_control)):
                    diagnostics.append(dict(fold=fold,seed=seed,arm=arm,control=kind,
                        changed_predictions=int(np.sum(diagnostic.argmax(1)!=scores.argmax(1))),
                        **control.metrics(y[test],diagnostic)))
                np.savez_compressed(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz',rows=test,scores=scores,
                                    base=base_np[test],source_mean_level=mean_control,
                                    source_constant=base_np[test]+constant)
                wrong = scores.copy(); wrong[np.arange(len(test)),y[test]]=-np.inf
                base_wrong = base_np[test].copy(); base_wrong[np.arange(len(test)),y[test]]=-np.inf
                for j,i in enumerate(test):
                    predictions.append(dict(fold=fold,seed=seed,arm=arm,step=chosen,
                        obs_id=int(f.iloc[i].obs_id),mmsi=f.iloc[i].mmsi,port=f.iloc[i].port,
                        product=f.iloc[i]['product'],class_id=int(y[i]),prediction=int(scores[j].argmax()),
                        base_prediction=int(base_np[i].argmax()),
                        margin=float(scores[j,y[i]]-wrong[j].max()),
                        base_margin=float(base_np[i,y[i]]-base_wrong[j].max())))
                print('fit',fold,seed,arm,chosen,measures[-1]['oa'],measures[-1]['ba'],flush=True)
            for name, rows in (('metrics',measures),('validation_selection',selections),
                               ('predictions',predictions),('fixed_head_controls',diagnostics),
                               ('baseline_replay',replays),('partitions',partitions)):
                pd.DataFrame(rows).to_csv(ROOT/(name+'.csv'),index=False)
    (ROOT/'TRAINING.json').write_text(json.dumps(dict(status='complete',fits=27,folds=3,seeds=3,
        updates=150,parameters=sum(p.numel() for p in head.parameters()),seconds=time.perf_counter()-start,
        scope='native frozen stronger classifier; not full target-support system'),indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('extract','train'))
    args=parser.parse_args()
    extract() if args.stage=='extract' else train()
