"""Bounded native-chip M2 pilot; selection never uses evaluation labels."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = Path('E:/临时会话/sar_m123_standalone')
sys.path.insert(0, str(PACKAGE))
from sar_m123.backbone_regions import RegionResNet50
from sar_m123.input import prepare_view
from sar_m123.regions import soft_regions
from sar_m123 import train_ops as op

WEIGHT = PACKAGE/'assets/backbone/resnet50.pth'
SNAPSHOTS = Path('E:/临时会话/same_vessel_gallery_20261019/local_condition_module/view_snapshot.json')
PREVIOUS = ROOT/'ports24_aligned_20261026'
SEEDS = (17, 29, 43)
MODES = ('none', 'merged', 'regional')


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def native(row, snapshots, availability):
    images = np.zeros((2, 1, 160, 160), np.float32)
    masks = np.zeros((2, 3, 160, 160), np.float32)
    valid = np.zeros((2, 1, 160, 160), np.float32)
    geometry = []
    for p, pol in enumerate(('vv', 'vh')):
        if not availability[p]:
            geometry.append(None)
            continue
        priors = snapshots[int(row.obs_id)]['images'][pol]['priors']
        spec = dict(image=row['image_'+pol], detections=priors['detections'],
                    target_index=priors['target_index'])
        v = prepare_view(spec)
        assert v['geometry'] is not None and v['valid_target']
        assert v['placement']['retained_target_pixels'] == v['placement']['native_target_pixels']
        images[p, 0] = v['pixels']/255
        valid[p, 0] = v['footprint']
        masks[p] = soft_regions(v['geometry'])*v['footprint']
        geometry.append(v['geometry'].tolist())
    return images, masks, valid, geometry


def prepare():
    dest = HERE/'cache'
    dest.mkdir(exist_ok=True)
    table = pd.read_csv(PREVIOUS/'bundle/table.csv', dtype={'mmsi': str})
    table = table[table.role.isin(('fit_train', 'fit_validation', 'evaluation'))].reset_index(drop=True)
    assert table[table.role.eq('evaluation')].class_id.eq(-1).all()
    table.to_csv(HERE/'table.csv', index=False)
    for field in ('mmsi', 'pass_group'):
        sets = [set(table.loc[table.role.eq(role), field]) for role in ('fit_train', 'fit_validation', 'evaluation')]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    with np.load(ROOT/'local_evidence_transport/tokens.npz') as z:
        available = z['valid'][table.source_index.to_numpy(int)].astype(np.float32)
    snapshots = {int(x['obs_id']): x for x in json.loads(SNAPSHOTS.read_text(encoding='utf-8'))}
    n = len(table)
    arrays = {name: np.lib.format.open_memmap(dest/f'{name}.npy', mode='w+', dtype=dtype, shape=shape)
              for name, shape, dtype in (
                  ('prefix', (n, 2, 512, 20, 20), 'float16'),
                  ('images', (n, 2, 1, 160, 160), 'uint8'),
                  ('masks', (n, 2, 3, 20, 20), 'float32'),
                  ('valid', (n, 2, 1, 20, 20), 'float32'))}
    model = RegionResNet50(WEIGHT, mode='none').cuda().eval()
    geometries, error = [], 0.
    tick = time.perf_counter()
    with torch.no_grad():
        for first in range(0, n, 16):
            ids = list(range(first, min(n, first+16)))
            batch = [native(table.iloc[i], snapshots, available[i]) for i in ids]
            im, ma, va = [torch.tensor(np.stack([x[k] for x in batch]), device='cuda') for k in range(3)]
            pf = model.prefix(im).cpu().numpy()
            arrays['prefix'][ids] = pf
            arrays['images'][ids] = (im.cpu().numpy()*255).round().astype(np.uint8)
            arrays['masks'][ids] = F.interpolate(ma.flatten(0, 1), (20, 20), mode='area').reshape(len(ids), 2, 3, 20, 20).cpu()
            arrays['valid'][ids] = F.interpolate(va.flatten(0, 1), (20, 20), mode='area').reshape(len(ids), 2, 1, 20, 20).cpu()
            geometries.extend(x[3] for x in batch)
            if first == 0:
                av = torch.tensor(available[ids], device='cuda')
                full = model(im, ma, va, av)
                fresh = model.forward_from(torch.tensor(pf, device='cuda'), ma, va, av)
                half = model.forward_from(torch.tensor(pf.astype(np.float16), device='cuda').float(), ma, va, av)
                assert (full-fresh).abs().max() < 1e-5
                error = float((half-fresh).abs().max())
                assert error < .005
            if first % 160 == 0:
                print('CACHE', min(first+16, n), n, 'seconds', round(time.perf_counter()-tick, 1), flush=True)
    for arr in arrays.values():
        arr.flush()
    np.save(dest/'available.npy', available)
    write(HERE/'geometry.json', geometries)
    write(HERE/'preparation.json', dict(rows=n, roles=table.role.value_counts().to_dict(),
          seconds=time.perf_counter()-tick, half_cache_score_max_error=error,
          missing_views=(available == 0).sum(0).tolist(), input_scale=1., c4=False))


def load():
    table = pd.read_csv(HERE/'table.csv', dtype={'mmsi': str})
    arrays = {key: np.load(HERE/'cache'/f'{key}.npy', mmap_mode='r')
              for key in ('prefix', 'masks', 'valid', 'available')}
    return table, arrays


def batch(arrays, ids):
    return [torch.tensor(np.asarray(arrays[k][ids]), device='cuda', dtype=torch.float32)
            for k in ('prefix', 'masks', 'valid', 'available')]


@torch.no_grad()
def predict(model, arrays, ids):
    model.eval()
    out = []
    for first in range(0, len(ids), 32):
        out.append(model.forward_from(*batch(arrays, ids[first:first+32])).cpu().numpy())
    return np.concatenate(out)


def train():
    table, arrays = load()
    fit = np.flatnonzero(table.role.eq('fit_train'))
    val = np.flatnonzero(table.role.eq('fit_validation'))
    y = table.class_id.to_numpy(int)
    f = table.iloc[fit]
    wt = np.zeros(len(table), np.float32)
    wt[fit] = len(f)/(8*f.groupby('class_id').mmsi.transform('nunique').to_numpy()
                     *f.groupby('mmsi').mmsi.transform('size').to_numpy())
    records = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        shared_head = torch.nn.Linear(4098, 8).state_dict()
        for mode in MODES:
            out = HERE/'training'/f'{mode}_{seed}'
            out.mkdir(parents=True, exist_ok=True)
            if (out/'complete.json').exists():
                records.append(json.loads((out/'complete.json').read_text()))
                continue
            torch.manual_seed(seed)
            model = RegionResNet50(WEIGHT, mode=mode).cuda()
            model.head.load_state_dict(shared_head)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01)
            rng = np.random.default_rng(seed)
            best, curve, tick = -np.inf, [], time.perf_counter()
            for epoch in range(1, 9):
                model.train()
                order = rng.permutation(fit)
                losses = []
                for first in range(0, len(order), 24):
                    ids = order[first:first+24]
                    opt.zero_grad(set_to_none=True)
                    logits = model.forward_from(*batch(arrays, ids))
                    loss = (F.cross_entropy(logits, torch.tensor(y[ids], device='cuda'), reduction='none')
                            *torch.tensor(wt[ids], device='cuda')).mean()
                    loss.backward()
                    assert all(p.grad is None for p in model.backbone.parameters())
                    opt.step()
                    losses.append(float(loss.detach()))
                score = predict(model, arrays, val)
                m = op.metrics(y[val], score)
                j = op.exact_j(y[val], score)
                curve.append(dict(epoch=epoch, loss=float(np.mean(losses)), **m))
                if j > best:
                    best, selected = j, epoch
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith('backbone.')}
                    torch.save(dict(state=state, epoch=epoch), out/'selected.pt')
                pd.DataFrame(curve).to_csv(out/'curve.csv', index=False)
                print('FIT', mode, seed, epoch, m, 'sec', round(time.perf_counter()-tick, 1), flush=True)
            receipt = dict(mode=mode, seed=seed, selected_epoch=selected, validation_j=float(best),
                           seconds=time.perf_counter()-tick,
                           trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
            write(out/'complete.json', receipt)
            records.append(receipt)
            del model, opt
            torch.cuda.empty_cache()
    # Immutable stage boundary: all choices are recorded before loading eval truth.
    write(HERE/'frozen_choices.json', records)


def evaluate():
    records = json.loads((HERE/'frozen_choices.json').read_text())
    assert len(records) == 9
    table, arrays = load()
    ids = np.flatnonzero(table.role.eq('evaluation'))
    truth = pd.read_csv(PREVIOUS/'source_truth.csv').set_index('obs_id')
    y = truth.loc[table.iloc[ids].obs_id, 'class_id'].to_numpy(int)
    rows = []
    for r in records:
        out = HERE/'training'/f"{r['mode']}_{r['seed']}"
        model = RegionResNet50(WEIGHT, mode=r['mode']).cuda().eval()
        ck = torch.load(out/'selected.pt', map_location='cpu', weights_only=True)
        result = model.load_state_dict(ck['state'], strict=False)
        assert not result.unexpected_keys and all(k.startswith('backbone.') for k in result.missing_keys)
        tick = time.perf_counter()
        score = predict(model, arrays, ids)
        torch.cuda.synchronize()
        m = op.metrics(y, score)
        np.savez_compressed(out/'evaluation.npz', ids=ids, obs_id=table.iloc[ids].obs_id.to_numpy(), y=y, score=score)
        rows.append(dict(**r, **m, evaluation_seconds=time.perf_counter()-tick))
        print('EVAL', r['mode'], r['seed'], m, flush=True)
        del model
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'train', 'evaluate', 'all'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name, function in (('prepare', prepare), ('train', train), ('evaluate', evaluate)):
        if args.stage in (name, 'all'):
            function()
