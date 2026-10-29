"""Bounded source-only protected-writer experiment, development endpoint only."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE.parents[2]/'sar_m123_standalone'))
spec = importlib.util.spec_from_file_location('delta_refit', ROOT/'spatial_m12_refit_20261029/run.py')
refit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refit)
from sar_m123.current.model import load_model, setup, forward_batch
from sar_m123.current.spatial_m12 import SpatialM12ResNet50
from sar_m123.current.training import class_weights
from sar_m123.current.data import batch, prepare_record, pair_mask, read_manifest
from model import attach, configure


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def metrics(score, y):
    t = torch.as_tensor(score).double()
    pred = score.argmax(1)
    p = t.softmax(1).numpy()
    return dict(oa=100*float(np.mean(pred == y)),
        ba=100*float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])),
        nll=float(F.cross_entropy(t, torch.as_tensor(y))),
        brier=float(np.square(p-np.eye(score.shape[1])[y]).sum(1).mean()))


def save(net, path, arm):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(arm=arm, state={n:v.detach().cpu() for n,v in net.state_dict().items()}), path)


def load(path):
    blob = torch.load(path, map_location='cpu', weights_only=True)
    net = SpatialM12ResNet50(None, 'pre3x3', 'local', 8)
    if blob['arm'] != 'control':
        net = attach(net, blob['arm'],blob['state'].get('m3.0.0.reference_labels'))
    net.load_state_dict(blob['state'], strict=True)
    return net.cuda().eval().requires_grad_(False)


@torch.no_grad()
def calibrate(net, arrays, indices, rows, bank):
    y = np.array([r['label'] for r in rows['fit']])
    chosen = np.concatenate([np.flatnonzero(y == c)[:8] for c in np.unique(y)])
    totals = [dict(square=0., count=0, records=[]) for _ in net.m3[0]]
    for first in range(0, len(chosen), 16):
        ix = chosen[first:first+16]
        group = [rows['fit'][i] for i in ix]
        refit.score(net, arrays, indices['fit'][ix], group, rows['references'], bank)
        for pol, m in enumerate(net.m3[0]):
            d = m.diagnostic
            totals[pol]['square'] += float(d['selected_square'])
            totals[pol]['count'] += d['selected_numel']
            for j,r in enumerate(group):
                totals[pol]['records'].append(dict(id=r['id'], label=r['label'],
                    **{k:float(d[k][j]) for k in ('total_rms','common_rms','delta_rms')}))
    for pol,m in enumerate(net.m3[0]):
        scale = (totals[pol]['square']/totals[pol]['count'])**.5
        assert scale > 1e-8, 'No reference-specific signal to train.'
        m.extra_scale.fill_(scale)
        totals[pol]['scale'] = scale
    return totals


@torch.no_grad()
def add_adapt(rows, arrays, indices):
    rows['adapt'] = read_manifest(refit.RELEASE/'manifests/source_adapt.json')
    net = load_model(ROOT/'spatial_m12_full_20261029/training/spatial_s17/model.pt','cuda')
    chunks = {k:[] for k in arrays['_gpu']}
    for first in range(0,len(rows['adapt']),16):
        data = batch([prepare_record(r) for r in rows['adapt'][first:first+16]],'cuda')
        data['prefix'] = net.prefix(data['images'],stop=2)
        for k in chunks:
            chunks[k].append(data[k])
    start = len(arrays['_gpu']['prefix'])
    indices['adapt'] = -np.arange(1,len(rows['adapt'])+1)
    for j,i in enumerate(indices['adapt']):
        arrays['_lookup'][int(i)] = start+j
    for k in chunks:
        arrays['_gpu'][k] = torch.cat([arrays['_gpu'][k],*chunks[k]])
    del net,data,chunks


def run(seeds, arms, stage='fit', semantic=False, semantic_lr=None):
    setup()
    rows, arrays, indices, bank = refit.load()
    if stage == 'adapt':
        add_adapt(rows,arrays,indices)
    output = HERE if stage == 'fit' else HERE/('semantic_pilot' if semantic else 'adapt_pilot')
    if semantic_lr is not None:
        output=HERE/'semantic_live_pilot'
    y = np.array([r['label'] for r in rows['development']])
    train_rows = rows[stage]
    w = class_weights(train_rows,stage)
    epochs,lr = (3,.001) if stage == 'fit' else (6,.0001)
    if semantic_lr is not None:
        lr=semantic_lr
    for seed in seeds:
        original = load_model(ROOT/'spatial_m12_full_20261029/training'/f'spatial_s{seed}/model.pt','cuda')
        teacher = refit.predict(original, arrays, indices[stage], train_rows, rows['references'], bank)
        base = refit.predict(original, arrays, indices['development'], rows['development'], rows['references'], bank)
        with np.load(ROOT/'spatial_m12_full_20261029/predictions'/f'development_after_spatial_s{seed}.npz') as old:
            assert abs(base-old['score']).max() < 1e-4
        train_metrics = metrics(teacher,np.array([r['label'] for r in train_rows]))
        print('TRAIN_BASELINE',stage,seed,train_metrics,flush=True)
        del original
        for arm in arms:
            out = output/'training'/f'{arm}_s{seed}'
            if (out/'training.json').exists():
                continue
            net = load_model(ROOT/'spatial_m12_full_20261029/training'/f'spatial_s{seed}/model.pt','cuda')
            diagnostic = None
            if arm != 'control':
                labels=[r['label'] for r in rows['references']]
                if 'shuffled' in arm:
                    labels=np.random.default_rng(1030).permutation(labels).tolist()
                net = attach(net, arm, labels)
                diagnostic = calibrate(net, arrays, indices, rows, bank)
            initial = refit.predict(net, arrays, indices['development'], rows['development'], rows['references'], bank)
            initial_error = float(abs(initial-base).max())
            assert initial_error < 1e-5
            params = configure(net, arm)
            fixed = {n:v.detach().cpu().clone() for n,v in net.named_parameters() if not v.requires_grad}
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=.01)
            rng = np.random.default_rng(seed)
            tick = time.perf_counter()
            curve = []
            for epoch in range(1,epochs+1):
                terms = []
                for ix in np.array_split(rng.permutation(len(train_rows)),range(24,len(train_rows),24)):
                    q = [train_rows[i] for i in ix]
                    labels = torch.tensor([r['label'] for r in q],device='cuda')
                    weight = torch.as_tensor(w[ix],device='cuda')
                    target = torch.as_tensor(teacher[ix],device='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    score = refit.score(net,arrays,indices[stage][ix],q,rows['references'],bank)
                    ce = (F.cross_entropy(score,labels,reduction='none')*weight).mean()
                    kl = F.kl_div(score.log_softmax(1),target.log_softmax(1),
                        log_target=True,reduction='none').sum(1)
                    keep = (kl*target.argmax(1).eq(labels)*weight).mean()
                    loss = ce+.1*keep
                    assert torch.isfinite(loss)
                    loss.backward()
                    if epoch == 1 and not terms:
                        gradients = {n:float(p.grad.norm()) for n,p in net.named_parameters() if p.requires_grad and p.grad is not None}
                        assert any(v > 0 for v in gradients.values())
                    optimizer.step()
                    terms.append([float(ce.detach()),float(keep.detach())])
                curve.append(dict(epoch=epoch,ce=float(np.mean(terms,0)[0]),keep=float(np.mean(terms,0)[1])))
                print('EPOCH',arm,seed,curve[-1],round(time.perf_counter()-tick,1),flush=True)
            assert all(torch.equal(dict(net.named_parameters())[n].detach().cpu(),v) for n,v in fixed.items())
            del fixed,optimizer,score,loss
            net.requires_grad_(False)
            after = refit.predict(net,arrays,indices['development'],rows['development'],rows['references'],bank)
            save(net,out/'model.pt',arm)
            (output/'predictions').mkdir(exist_ok=True)
            np.savez(output/'predictions'/f'{arm}_s{seed}.npz',score=after,baseline=base,y=y,
                ids=np.array([str(r['id']) for r in rows['development']]))
            native_rows = rows['development'][:4]
            q = batch([prepare_record(r) for r in native_rows],'cuda')
            refs = batch([prepare_record(r) for r in rows['references']],'cuda')
            with torch.no_grad():
                refs['prefix'] = net.prefix(refs['images'],stop=2)
                native = forward_batch(net,q,refs,pair_mask(native_rows,rows['references'],'cuda')).cpu().numpy()
            native_error = float(abs(native-after[:4]).max())
            assert native_error < 1e-4
            receipt = dict(arm=arm,seed=seed,stage=stage,epochs=epochs,lr=lr,
                steps=epochs*((len(train_rows)+23)//24),source_count=len(train_rows),train_baseline=train_metrics,
                reference_count=len(rows['references']),trainable=sum(p.numel() for p in params),
                before=metrics(base,y),after=metrics(after,y),curve=curve,gradients=gradients,
                initial_error=initial_error,native_error=native_error,nonselected_parameters_unchanged=True,
                diagnostic=diagnostic,seconds=time.perf_counter()-tick,validation_selection=False)
            del net,q,refs,params
            restored = load(out/'model.pt')
            replay = refit.predict(restored,arrays,indices['development'][:4],native_rows,rows['references'],bank)
            receipt['replay_error'] = float(abs(replay-after[:4]).max())
            assert receipt['replay_error'] < 1e-4
            write(out/'training.json',receipt)
            print('COMPLETE',arm,seed,receipt['after'],flush=True)
            del restored
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=[17])
    parser.add_argument('--arms',nargs='+',choices=['total','delta','control','label','label_shuffled','label_live','label_live_shuffled'],default=['total','delta','control'])
    parser.add_argument('--stage',choices=['fit','adapt'],default='fit')
    parser.add_argument('--semantic',action='store_true')
    parser.add_argument('--semantic-lr',type=float)
    args = parser.parse_args()
    run(args.seeds,args.arms,args.stage,args.semantic,args.semantic_lr)
