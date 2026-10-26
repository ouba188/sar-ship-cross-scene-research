"""Source-selected factorial comparison of M2 insertion and gate design."""
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
PILOT = HERE.parent
sys.path.insert(0, str(PILOT))
import run as old
sys.path.remove(str(PILOT))
from sar_m123.backbone_regions import RegionResNet50
from sar_m123 import train_ops as op

CONFIGS = [(1, 'none'), (1, 'fixed'), (1, 'cosine'), (2, 'fixed'), (2, 'cosine')]


def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def model_for(stage, gate):
    return RegionResNet50(old.WEIGHT, stages=(stage,), mode='none' if gate == 'none' else 'regional',
                          gate_mode='dot' if gate == 'none' else gate).cuda()


def load(stage):
    table, arrays = old.load()
    if stage == 1:
        arrays.update({key: np.load(HERE/'cache'/f'{key}.npy', mmap_mode='r')
                       for key in ('prefix', 'masks', 'valid')})
    return table, arrays


@torch.no_grad()
def predict(model, arrays, ids, stage):
    model.eval()
    return np.concatenate([model.forward_from(*old.batch(arrays, ids[i:i+24]), start=stage).cpu().numpy()
                           for i in range(0, len(ids), 24)])


def prepare():
    dest = HERE/'cache'
    dest.mkdir(exist_ok=True)
    table, original = old.load()
    snapshots = {int(x['obs_id']): x for x in json.loads(old.SNAPSHOTS.read_text(encoding='utf-8'))}
    n = len(table)
    arrays = {key: np.lib.format.open_memmap(dest/f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
              for key, dtype, shape in [('prefix', 'float16', (n, 2, 256, 40, 40)),
                                       ('masks', 'float32', (n, 2, 3, 40, 40)),
                                       ('valid', 'float32', (n, 2, 1, 40, 40))]}
    model = model_for(1, 'none').eval()
    resolution, parity = [], []
    tick = time.perf_counter()
    with torch.no_grad():
        for first in range(0, n, 16):
            ids = np.arange(first, min(first+16, n))
            items = [old.native(table.iloc[i], snapshots, original['available'][i]) for i in ids]
            im, ma, va = [torch.tensor(np.stack([v[k] for v in items]), device='cuda') for k in range(3)]
            pf = model.prefix(im, stop=1)
            arrays['prefix'][ids] = pf.cpu().numpy()
            arrays['masks'][ids] = F.interpolate(ma.flatten(0, 1), (40, 40), mode='area').reshape(len(ids), 2, 3, 40, 40).cpu()
            arrays['valid'][ids] = F.interpolate(va.flatten(0, 1), (40, 40), mode='area').reshape(len(ids), 2, 1, 40, 40).cpu()
            if first == 0:
                av = torch.tensor(original['available'][ids], device='cuda')
                full = model(im, ma, va, av)
                fresh = model.forward_from(pf, ma, va, av, start=1)
                half = model.forward_from(pf.half().float(), ma, va, av, start=1)
                assert float((full-fresh).abs().max()) < 1e-5
                assert float((half-fresh).abs().max()) < .005
                parity.append(dict(full_cache_error=float((full-fresh).abs().max()), half_error=float((half-fresh).abs().max())))
            # Compare actual masks and channel-normalized tokens, not upsampled stage2.
            features = pf
            for stage in (1, 2, 3):
                if stage > 1:
                    nxt = getattr(model.backbone.model, f'layer{stage}')(features.flatten(0, 1))
                    features = nxt.reshape(len(ids), 2, *nxt.shape[1:])
                masks = F.interpolate(ma.flatten(0, 1), features.shape[-2:], mode='area').reshape(len(ids), 2, 3, *features.shape[-2:])
                z = F.layer_norm(features.permute(0, 1, 3, 4, 2), (features.shape[2],)).permute(0, 1, 4, 2, 3)
                tokens = torch.einsum('bprhw,bpchw->bprc', masks, z)/masks.sum((-2, -1)).clamp_min(1e-6)[..., None]
                mm = F.normalize(masks.flatten(-2), dim=-1)
                tt = F.normalize(tokens, dim=-1)
                for j, idx in enumerate(ids):
                    for p in range(2):
                        if not original['available'][idx, p]:
                            continue
                        r = dict(obs_id=int(table.iloc[idx].obs_id), role=table.iloc[idx].role, stage=stage, pol=('VV', 'VH')[p])
                        for a, name in enumerate(('body', 'end', 'side')):
                            r[name+'_mass'] = float(masks[j, p, a].sum())
                        for a, b, name in ((0, 1, 'body_end'), (0, 2, 'body_side')):
                            r[name+'_mask_cos'] = float((mm[j, p, a]*mm[j, p, b]).sum())
                            r[name+'_token_cos'] = float((tt[j, p, a]*tt[j, p, b]).sum())
                        resolution.append(r)
            if first % 160 == 0:
                print('PREPARE', first, n, round(time.perf_counter()-tick, 1), flush=True)
    for arr in arrays.values():
        arr.flush()
    pd.DataFrame(resolution).to_csv(HERE/'resolution.csv', index=False)
    old.write(HERE/'preparation.json', dict(rows=n, seconds=time.perf_counter()-tick, parity=parity))


def train():
    table, _ = old.load()
    fit, val = [np.flatnonzero(table.role.eq(role)) for role in ('fit_train', 'fit_validation')]
    y = table.class_id.to_numpy(int)
    f = table.iloc[fit]
    wt = np.zeros(len(table), np.float32)
    wt[fit] = len(f)/(8*f.groupby('class_id').mmsi.transform('nunique').to_numpy()*f.groupby('mmsi').mmsi.transform('size').to_numpy())
    records = []
    for stage, gate in CONFIGS:
        _, arrays = load(stage)
        for seed in old.SEEDS:
            out = HERE/'training'/f's{stage}_{gate}_{seed}'
            out.mkdir(parents=True, exist_ok=True)
            if (out/'complete.json').exists():
                records.append(json.loads((out/'complete.json').read_text()))
                continue
            torch.manual_seed(seed)
            head = torch.nn.Linear(4098, 8).state_dict()
            torch.manual_seed(seed)
            model = model_for(stage, gate)
            model.head.load_state_dict(head)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01)
            rng, best, curve = np.random.default_rng(seed), -np.inf, []
            tick = time.perf_counter()
            for epoch in range(1, 9):
                model.train()
                order, losses = rng.permutation(fit), []
                for start in range(0, len(order), 24):
                    ids = order[start:start+24]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model.forward_from(*old.batch(arrays, ids), start=stage)
                    loss = (F.cross_entropy(logits, torch.tensor(y[ids], device='cuda'), reduction='none')*torch.tensor(wt[ids], device='cuda')).mean()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                score = predict(model, arrays, val, stage)
                metrics, j = op.metrics(y[val], score), op.exact_j(y[val], score)
                curve.append(dict(epoch=epoch, loss=float(np.mean(losses)), **metrics))
                if j > best:
                    best, selected = j, epoch
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith('backbone.')}
                    torch.save(dict(state=state, epoch=epoch), out/'selected.pt')
                pd.DataFrame(curve).to_csv(out/'curve.csv', index=False)
                print('FIT', stage, gate, seed, epoch, 'J', round(float(j), 5), 'sec', round(time.perf_counter()-tick, 1), flush=True)
            receipt = dict(stage=stage, gate=gate, seed=seed, selected_epoch=selected, validation_j=float(best),
                           seconds=time.perf_counter()-tick, trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
            old.write(out/'complete.json', receipt)
            records.append(receipt)
            del model, optimizer
            torch.cuda.empty_cache()
    old.write(HERE/'frozen_choices.json', records)
    candidates = pd.DataFrame(records).query("gate != 'none'").groupby(['stage', 'gate']).validation_j.mean().sort_values(ascending=False)
    stage, gate = candidates.index[0]
    old.write(HERE/'source_selected.json', dict(stage=int(stage), gate=gate, validation_j=float(candidates.iloc[0])))


def restore(stage, gate, seed):
    model = model_for(stage, gate).eval()
    ck = torch.load(HERE/'training'/f's{stage}_{gate}_{seed}'/'selected.pt', map_location='cpu', weights_only=True)
    result = model.load_state_dict(ck['state'], strict=False)
    assert not result.unexpected_keys and all(k.startswith('backbone.') for k in result.missing_keys)
    return model


def evaluate():
    records = json.loads((HERE/'frozen_choices.json').read_text())
    assert len(records) == 15
    table, _ = old.load()
    ids = np.flatnonzero(table.role.eq('evaluation'))
    truth = pd.read_csv(old.PREVIOUS/'source_truth.csv').set_index('obs_id')
    y = truth.loc[table.iloc[ids].obs_id, 'class_id'].to_numpy(int)
    rows = []
    for r in records:
        stage, gate, seed = r['stage'], r['gate'], r['seed']
        _, arrays = load(stage)
        model = restore(stage, gate, seed)
        tick = time.perf_counter()
        score = predict(model, arrays, ids, stage)
        np.savez_compressed(HERE/'training'/f's{stage}_{gate}_{seed}'/'evaluation.npz', ids=ids, y=y, score=score)
        rows.append(dict(**r, **op.metrics(y, score), evaluation_seconds=time.perf_counter()-tick))
        print('EVAL', stage, gate, seed, rows[-1], flush=True)
        del model
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('all', 'prepare', 'train', 'evaluate'))
    args = parser.parse_args()
    setup()
    for name, func in (('prepare', prepare), ('train', train), ('evaluate', evaluate)):
        if args.action in ('all', name):
            func()
