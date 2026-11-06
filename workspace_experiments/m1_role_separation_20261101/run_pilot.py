"""Frozen-downstream M1 information/operator pilot; no formal model edits."""
from pathlib import Path
import sys
import time
import json
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
TASK = Path('E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2')
sys.path.insert(0, str(TASK))
import model_task as task
from sar_m123.pool_replacement import regional_mean, grid_occupancy

ARMS = ('additive', 'bounded_add', 'class_gate', 'mean_gate')


def dump(name, value):
    task.write(ROOT / name, value)


def anchor():
    model = task.Model('aggregate', seed=17)
    state = torch.load(TASK/'aggregate_17/last.pt', map_location='cpu', weights_only=True)
    model.restore(state['trainable_state'])
    return model.eval().requires_grad_(False)


@torch.no_grad()
def make_cache(model, a, rows, train, rid):
    dest = ROOT/'cache'
    dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        return {p.stem: np.load(p, mmap_mode='r') for p in dest.glob('*.npy')}
    prefix = np.load(task.PUBLIC/'cache/prefix.npy', mmap_mode='r')
    base = model.base
    block = base.backbone.model.layer3[-1]
    assert base.position == 'pre3x3' and block.downsample is None
    out = {}
    tick = time.perf_counter()
    for start in range(0, len(rows), 12):
        ids = np.arange(start, min(start+12, len(rows)))
        ix = [rows[i]['indices']['table_index'] for i in ids]
        identity = task.tensor(prefix, ix).flatten(0, 1)
        for layer in list(base.backbone.model.layer3.children())[:-1]:
            identity = layer(identity)
        x = block.relu(block.bn1(block.conv1(identity))).reshape(len(ids), 2, 256, 10, 10)
        v = task.tensor(a['valid'], ids)
        contexts, means, occupancies = [], [], []
        for p, op in enumerate(base.internal_m1):
            occ = grid_occupancy(v[:, p], (10, 10))
            _, d = op(x[:, p], occ, None, True)
            t = torch.stack([layer(d['tokens'][:, j]) for j, layer in enumerate(op.transforms)], 1)
            mixed = torch.einsum('bskhw,bskd->bdhw', d['write_weights'], t)
            uniform_token = d['tokens'][:, 0]
            mean_t = torch.stack([layer(uniform_token) for layer in op.transforms], 1)
            mean = mean_t.mean((1, 2))[..., None, None]*occ
            contexts.append(mixed)
            means.append(mean.expand_as(mixed))
            occupancies.append(occ)
        av = task.tensor(a['available'], ids)
        pair = task.legal(rows, ids, rid)[..., None] & av[:, None].bool() & task.tensor(a['available'], rid)[None].bool()
        inputs = [task.tensor(a[k], ix) for k, ix in [('desc', ids), ('desc', rid), ('mass', ids), ('mass', rid), ('write', ids)]]
        delta = model.matcher(*inputs, pair, torch.tensor([rows[i]['label'] for i in rid], device=task.DEVICE), arm='aggregate')[0]
        values = dict(x=x, identity=identity.reshape(len(ids), 2, 1024, 10, 10),
                      context=torch.stack(contexts, 1), mean_context=torch.stack(means, 1),
                      occupancy=torch.stack(occupancies, 1), delta=delta)
        for k, value in values.items():
            arr = value.cpu().numpy()
            if k not in out:
                out[k] = np.lib.format.open_memmap(dest/(k+'.npy'), mode='w+', dtype=arr.dtype,
                                                 shape=(len(rows),)+arr.shape[1:])
            out[k][ids] = arr
        if start % 120 == 0:
            print('CACHE', start+len(ids), len(rows), round(time.perf_counter()-tick, 1), flush=True)
    for arr in out.values():
        arr.flush()
    total = np.zeros((2, 256), np.float64)
    mass = np.zeros((2, 1), np.float64)
    for ids in np.array_split(train, 30):
        x = np.asarray(out['x'][ids], dtype=np.float64)
        occ = np.asarray(out['occupancy'][ids], dtype=np.float64)
        total += (x*x*occ).sum((0, 3, 4))
        mass += occ.sum((0, 3, 4))
    rms = np.sqrt(total/np.maximum(mass, 1e-12)).astype(np.float32)
    np.save(ROOT/'source_rms.npy', rms)
    task.write(dest/'complete.json', dict(n=len(rows), source=len(train), rms_source_only=True,
                                        seconds=time.perf_counter()-tick))
    return out


class Pilot(nn.Module):
    def __init__(self, model, arm):
        super().__init__()
        self.base = model.base
        self.arm = arm
        self.outputs = nn.ModuleList(copy.deepcopy(op.output) for op in self.base.internal_m1)
        for op in self.outputs:
            if arm != 'anchor':
                nn.init.zeros_(op.weight)
            op.requires_grad_(arm != 'anchor')
        self.register_buffer('gain', torch.stack([op.gamma.detach().clone() for op in self.base.internal_m1]))
        self.register_buffer('rms', torch.tensor(np.load(ROOT/'source_rms.npy'), device=task.DEVICE))
        self.eval()

    def forward(self, a, c, ids, mode=None):
        x = task.tensor(c['x'], ids)
        ctx = task.tensor(c['mean_context' if self.arm == 'mean_gate' else 'context'], ids)
        v = task.tensor(a['valid'], ids)
        slots = task.tensor(a['slots'], ids)
        av = task.tensor(a['available'], ids)
        results, ratios, updates = [], [], []
        for p, op in enumerate(self.outputs):
            raw = op(ctx[:, p])
            if self.arm in ('anchor', 'additive'):
                update = self.gain[p]*raw
            elif self.arm == 'bounded_add':
                update = self.gain[p]*self.rms[p, :, None, None]*raw.tanh()
            else:
                gate = raw.tanh()
                if mode == 'spatial_mean':
                    occ = task.tensor(c['occupancy'], ids)[:, p]
                    gate = (gate*occ).sum((2, 3), keepdim=True)/occ.sum((2, 3), keepdim=True).clamp_min(1e-8)
                    gate = gate*occ
                update = self.gain[p]*x[:, p]*gate
            if mode == 'off' or (mode == 'vv_only' and p == 1) or (mode == 'vh_only' and p == 0):
                update = update*0
            updates.append(update)
            ratios.append(update.flatten(1).norm(dim=1)/x[:, p].flatten(1).norm(dim=1).clamp_min(1e-8))
            reg, en = self.base.region_integrals(slots[:, p], slots.new_zeros(len(ids), 20, 3, 10, 10))
            z, _ = self.base.internal_m2[p](x[:, p]+update, reg, v[:, p], en)
            results.append(z)
        z = torch.stack(results, 1).flatten(0, 1)
        block = self.base.backbone.model.layer3[-1]
        z = block.relu(block.bn2(block.conv2(z)))
        z = block.bn3(block.conv3(z))
        z = block.relu(z+task.tensor(c['identity'], ids).flatten(0, 1)).reshape(len(ids), 2, 1024, 10, 10)
        updated = z+task.tensor(c['delta'], ids)
        deep = self.base.backbone.model.layer4(updated.flatten(0, 1))
        glob = regional_mean(deep, v.flatten(0, 1), v.flatten(0, 1))[0][:, 0].reshape(len(ids), 2, -1)
        pooled = []
        low = task.tensor(a['low'], ids)
        for p in range(2):
            reg, en = self.base.region_integrals(slots[:, p], slots.new_zeros(len(ids), 20, 3, 10, 10))
            ext = self.base.deep_pool[p](updated[:, p], reg, v[:, p], en)
            pooled.append((low[:, p]+self.base.deep_output[p](ext['pooled'])).flatten(1))
        features = torch.cat((glob, torch.stack(pooled, 1)), -1)*av[..., None]
        score = self.base.head(torch.cat((features.flatten(1), av), -1))
        return score, torch.stack(ratios, 1)


@torch.no_grad()
def predict(net, a, c, ids, mode=None):
    scores, ratios = [], []
    for start in range(0, len(ids), 24):
        s, r = net(a, c, ids[start:start+24], mode)
        scores.append(s.cpu().numpy())
        ratios.append(r.cpu().numpy())
    return dict(ids=ids, score=np.concatenate(scores), update_ratio=np.concatenate(ratios))


def main():
    task.current.setup(task.DEVICE)
    rows, train, hold, rid = task.records()
    a = task.arrays()
    model = anchor()
    c = make_cache(model, a, rows, train, rid)
    probe = np.r_[train[:3], hold[:3]]
    untouched = Pilot(model, 'anchor')
    actual = predict(untouched, a, c, probe)['score']
    expected = task.predict(model, a, rows, probe, rid)['score']
    parity = float(np.abs(actual-expected).max())
    assert parity < 1e-4, parity
    p = predict(untouched, a, c, hold)
    np.savez_compressed(ROOT/'anchor.npz', **p)
    y = np.array([r['label'] for r in rows])
    summary = [dict(arm='anchor', **task.metrics(y[hold], p['score']))]
    frame = pd.DataFrame([rows[i] for i in train])
    hard = set(json.loads((TASK/'source_difficulty.json').read_text(encoding='utf-8'))['hard'])
    weight = np.zeros(len(rows), np.float32)
    weight[train] = 1/(frame.groupby('label').mmsi.transform('nunique').to_numpy()*frame.groupby('mmsi').mmsi.transform('size').to_numpy())
    weight[train] *= [2. if i in hard else 1. for i in train]
    weight[train] /= weight[train].mean()
    weights = torch.tensor(weight, device=task.DEVICE)
    initial = None
    dump('parity.json', dict(max_score_error=parity, source_train=len(train), dev=len(hold), references=len(rid),
                            gain=untouched.gain.cpu().tolist(), anchor=summary[0]))
    print('PARITY', parity, summary[0], flush=True)
    for arm in ARMS:
        dest = ROOT/arm
        dest.mkdir(exist_ok=True)
        if (dest/'receipt.json').exists():
            receipt = json.loads((dest/'receipt.json').read_text())
            summary.append(dict(arm=arm, **receipt['final']))
            continue
        net = Pilot(model, arm)
        params = [p for p in net.parameters() if p.requires_grad]
        assert sum(p.numel() for p in params) == 2*256*32
        init = predict(net, a, c, probe)['score']
        if initial is None:
            initial = init
        assert np.max(abs(initial-init)) == 0
        optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=.01)
        rng = np.random.default_rng(17)
        curve, gradients = [], []
        tick = time.perf_counter()
        steps = 0
        for epoch in range(1, 5):
            order = rng.permutation(train)
            losses = []
            for start in range(0, len(order), 24):
                ids = order[start:start+24]
                optimizer.zero_grad(set_to_none=True)
                z, _ = net(a, c, ids)
                loss = (F.cross_entropy(z, torch.tensor(y[ids], device=task.DEVICE), reduction='none')*weights[ids]).mean()
                assert torch.isfinite(loss)
                loss.backward()
                if steps == 0:
                    gradients = [float(p.grad.norm()) for p in params]
                    assert all(np.isfinite(gradients)) and min(gradients) > 0
                torch.nn.utils.clip_grad_norm_(params, 5.)
                optimizer.step()
                steps += 1
                losses.append(float(loss.detach()))
            result = predict(net, a, c, hold)
            metric = task.metrics(y[hold], result['score'])
            curve.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), **metric))
            pd.DataFrame(curve).to_csv(dest/'curve.csv', index=False)
            print('EPOCH', arm, epoch, metric, round(time.perf_counter()-tick, 1), flush=True)
        torch.save(net.outputs.state_dict(), dest/'outputs.pt')
        for mode in (None, 'off', 'spatial_mean', 'vv_only', 'vh_only'):
            if mode == 'spatial_mean' and arm not in ('class_gate', 'mean_gate'):
                continue
            result = predict(net, a, c, hold, mode)
            np.savez_compressed(dest/(('final' if mode is None else mode)+'.npz'), **result)
        score = np.load(dest/'final.npz')['score']
        metric = task.metrics(y[hold], score)
        task.write(dest/'receipt.json', dict(arm=arm, steps=steps, seed=17, final=metric, trainable_parameters=16384,
                                           first_gradients=gradients, seconds=time.perf_counter()-tick,
                                           frozen_m2_m3_backbone_head=True, checkpoint='fixed epoch4'))
        summary.append(dict(arm=arm, **metric))
        pd.DataFrame(summary).to_csv(ROOT/'summary.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
