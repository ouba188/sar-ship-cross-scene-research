"""Live M1 design and bottleneck placement, with bounded downstream adaptation."""
from pathlib import Path
import sys
import copy
import json
import time
import argparse
import gc
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'m1_role_separation_20261101'
sys.path.insert(0, str(OLD))
import run_pilot as previous
task = previous.task
from sar_m123.pool_replacement import regional_mean

ARMS = ('none', 'pre_additive', 'post_additive', 'pre_class_gate',
        'post_class_gate', 'pre_mean_gate', 'post_mean_gate')


def caches():
    return {p.stem: np.load(p, mmap_mode='r') for p in (OLD/'cache').glob('*.npy')}


class Network(nn.Module):
    def __init__(self, arm, reader_state=None):
        super().__init__()
        carrier = previous.anchor()
        self.base = carrier.base
        self.arm = arm
        self.position = 'post' if arm.startswith('post_') else 'pre'
        self.design = arm.split('_', 1)[1] if '_' in arm else arm
        self.m1 = nn.ModuleList(copy.deepcopy(op) for op in self.base.internal_m1)
        if reader_state is not None and arm not in ('none', 'anchor'):
            with torch.no_grad():
                for p, op in enumerate(self.m1):
                    op.reader.weight.copy_(reader_state[self.position][p])
        for op in self.m1:
            if arm != 'anchor':
                nn.init.zeros_(op.output.weight)
            op.requires_grad_(arm not in ('none', 'anchor'))
            op.gamma.requires_grad_(False)
        self.base.backbone.model.layer3[-1].conv3.requires_grad_(arm != 'anchor')
        self.eval()

    def apply_m1(self, x, occupancy, available, mode=None):
        outputs, aux, ratios = [], [], []
        for p, op in enumerate(self.m1):
            z, d = op(x[:, p], occupancy[:, p], None, True)
            aux.append(d['aux_logits'])
            raw = d['update']/op.gamma
            if self.design == 'mean_gate' or mode == 'uniform_context':
                t = d['tokens'][:, 0]
                mixed = torch.stack([layer(t) for layer in op.transforms], 1).mean((1, 2))
                mixed = mixed[..., None, None]*occupancy[:, p]
                raw = op.output(mixed)
            if self.design in ('anchor', 'additive'):
                update = op.gamma*raw
            elif self.design == 'none':
                update = torch.zeros_like(x[:, p])
            else:
                gate = raw.tanh()
                if mode == 'spatial_mean':
                    occ = occupancy[:, p]
                    gate = (gate*occ).sum((2, 3), keepdim=True)/occ.sum((2, 3), keepdim=True).clamp_min(1e-8)
                    gate = gate*occ
                update = op.gamma*x[:, p]*gate
            if mode == 'off' or (mode == 'vv_only' and p == 1) or (mode == 'vh_only' and p == 0):
                update = update*0
            outputs.append(x[:, p]+update)
            ratios.append(update.flatten(1).norm(dim=1)/x[:, p].flatten(1).norm(dim=1).clamp_min(1e-8))
        return torch.stack(outputs, 1), torch.stack(aux, 1), torch.stack(ratios, 1)

    def forward(self, a, c, ids, mode=None):
        x = task.tensor(c['x'], ids)
        occupancy = task.tensor(c['occupancy'], ids)
        v = task.tensor(a['valid'], ids)
        slots = task.tensor(a['slots'], ids)
        av = task.tensor(a['available'], ids)
        if self.position == 'pre':
            x, aux, ratio = self.apply_m1(x, occupancy, av, mode)
        out = []
        for p in range(2):
            reg, en = self.base.region_integrals(slots[:, p], slots.new_zeros(len(ids), 20, 3, 10, 10))
            z, _ = self.base.internal_m2[p](x[:, p], reg, v[:, p], en)
            out.append(z)
        block = self.base.backbone.model.layer3[-1]
        z = torch.stack(out, 1).flatten(0, 1)
        z = block.relu(block.bn2(block.conv2(z))).reshape(len(ids), 2, 256, 10, 10)
        if self.position == 'post':
            z, aux, ratio = self.apply_m1(z, occupancy, av, mode)
        z = block.bn3(block.conv3(z.flatten(0, 1)))
        z = block.relu(z+task.tensor(c['identity'], ids).flatten(0, 1)).reshape(len(ids), 2, 1024, 10, 10)
        updated = z+task.tensor(c['delta'], ids)
        deep = self.base.backbone.model.layer4(updated.flatten(0, 1))
        glob = regional_mean(deep, v.flatten(0, 1), v.flatten(0, 1))[0][:, 0].reshape(len(ids), 2, -1)
        low = task.tensor(a['low'], ids)
        pooled = []
        for p in range(2):
            reg, en = self.base.region_integrals(slots[:, p], slots.new_zeros(len(ids), 20, 3, 10, 10))
            ext = self.base.deep_pool[p](updated[:, p], reg, v[:, p], en)
            pooled.append((low[:, p]+self.base.deep_output[p](ext['pooled'])).flatten(1))
        features = torch.cat((glob, torch.stack(pooled, 1)), -1)*av[..., None]
        score = self.base.head(torch.cat((features.flatten(1), av), -1))
        return score, aux, ratio

    def live_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def restore(self, state):
        params = dict(self.named_parameters())
        assert set(state) == {n for n, p in params.items() if p.requires_grad}
        with torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value)


@torch.no_grad()
def predict(model, a, c, ids, mode=None):
    scores, aux, ratio = [], [], []
    for start in range(0, len(ids), 24):
        z, u, r = model(a, c, ids[start:start+24], mode)
        scores.append(z.cpu().numpy())
        aux.append(u.cpu().numpy())
        ratio.append(r.cpu().numpy())
    return dict(ids=ids, score=np.concatenate(scores), auxiliary=np.concatenate(aux), update_ratio=np.concatenate(ratio))


def fit(arm, seed, rows, train, hold, a, c, reader_state=None):
    dest = ROOT/(arm+'_'+str(seed))
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'receipt.json').exists():
        return json.loads((dest/'receipt.json').read_text())
    torch.manual_seed(seed)
    net = Network(arm, reader_state)
    y = np.array([r['label'] for r in rows])
    params = {n: p for n, p in net.named_parameters() if p.requires_grad}
    allowed = lambda n: n.startswith('m1.') or n == 'base.backbone.model.layer3.5.conv3.weight'
    assert all(allowed(n) for n in params), list(params)
    groups = {}
    for n, p in params.items():
        lr = 1e-5 if n.startswith('base.') else (.001 if '.output.' in n else .0001)
        groups.setdefault(lr, []).append(p)
    optimizer = torch.optim.AdamW([dict(params=p, lr=lr) for lr, p in groups.items()], weight_decay=.01)
    difficulty = json.loads((task.ROOT/'source_difficulty.json').read_text(encoding='utf-8'))
    hard = set(difficulty['hard'])
    frame = pd.DataFrame([rows[i] for i in train])
    weights = np.zeros(len(rows), np.float32)
    weights[train] = 1/(frame.groupby('label').mmsi.transform('nunique').to_numpy()*frame.groupby('mmsi').mmsi.transform('size').to_numpy())
    weights[train] *= [2. if i in hard else 1. for i in train]
    weights[train] /= weights[train].mean()
    weights = torch.tensor(weights, device=task.DEVICE)
    initial = predict(net, a, c, hold)
    np.savez_compressed(dest/'initial.npz', **initial)
    source_probe = np.concatenate([train[y[train] == cls][:16] for cls in range(4)])
    np.savez_compressed(dest/'source_initial_probe.npz', **predict(net, a, c, source_probe))
    rng = np.random.default_rng(seed)
    curve, gradients, orders, clip_norms = [], [], [], []
    tick = time.perf_counter()
    steps = 0
    for epoch in range(1, 5):
        order = rng.permutation(train)
        orders.append(order.tolist())
        losses, ces, auxiliaries = [], [], []
        for start in range(0, len(order), 24):
            ids = order[start:start+24]
            labels = torch.tensor(y[ids], device=task.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            z, aux, _ = net(a, c, ids)
            ce = (F.cross_entropy(z, labels, reduction='none')*weights[ids]).mean()
            av = task.tensor(a['available'], ids)
            aux_ce = F.cross_entropy(aux.flatten(0, 1), labels[:, None].expand(-1, 2).flatten(), reduction='none').reshape(len(ids), 2)
            aux_loss = ((aux_ce*av).sum(1)/av.sum(1).clamp_min(1)*weights[ids]).mean()
            loss = ce if arm == 'none' else ce+.1*aux_loss
            assert torch.isfinite(loss)
            loss.backward()
            if steps in (0, 1):
                g = {n: float(p.grad.norm()) for n, p in params.items() if p.grad is not None}
                assert all(np.isfinite(list(g.values())))
                assert g['base.backbone.model.layer3.5.conv3.weight'] > 0
                if arm != 'none':
                    assert all(g['m1.'+str(p)+'.output.weight'] > 0 for p in range(2))
                    assert all(g['m1.'+str(p)+'.reader.weight'] > 0 for p in range(2))
                    if steps == 1:
                        assert all(g['m1.'+str(p)+'.value.weight'] > 0 for p in range(2))
                gradients.append(dict(step=steps, norms=g))
            clip_norms.append(float(torch.nn.utils.clip_grad_norm_(list(params.values()), 5.)))
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
            ces.append(float(ce.detach()))
            auxiliaries.append(float(aux_loss.detach()))
        p = predict(net, a, c, hold)
        metric = task.metrics(y[hold], p['score'])
        curve.append(dict(epoch=epoch, loss=float(np.mean(losses)), classification_ce=float(np.mean(ces)),
                          auxiliary_ce=float(np.mean(auxiliaries)), **metric))
        pd.DataFrame(curve).to_csv(dest/'curve.csv', index=False)
        print('EPOCH', arm, seed, epoch, metric, round(time.perf_counter()-tick, 1), flush=True)
    state = net.live_state()
    torch.save(state, dest/'last.pt')
    modes = [None] if arm == 'none' else [None, 'off', 'vv_only', 'vh_only']
    if 'class_gate' in arm:
        modes += ['uniform_context', 'spatial_mean']
    for mode in modes:
        p = predict(net, a, c, hold, mode)
        np.savez_compressed(dest/(('final' if mode is None else mode)+'.npz'), **p)
    source = predict(net, a, c, train)
    np.savez_compressed(dest/'source_final.npz', **source)
    original = predict(net, a, c, hold[:3])['score']
    fresh = Network(arm, reader_state)
    fresh.restore(torch.load(dest/'last.pt', map_location='cpu', weights_only=True))
    error = float(abs(original-predict(fresh, a, c, hold[:3])['score']).max())
    assert error == 0, error
    metric = task.metrics(y[hold], np.load(dest/'final.npz')['score'])
    receipt = dict(arm=arm, seed=seed, steps=steps, final=metric,
                   source=task.metrics(y[train], source['score']),
                   trainable_parameters=sum(p.numel() for p in params.values()),
                   gradient_checks=gradients, reload_error=error, seconds=time.perf_counter()-tick,
                   gradient_norm_max=max(clip_norms), clipped_steps=sum(n > 5 for n in clip_norms),
                   checkpoint_selection='fixed epoch4', source719=True, references32=True,
                   frozen_m2_m3_head_bn=True, initial=task.metrics(y[hold], initial['score']))
    receipt['source_calibrated_reader'] = reader_state is not None
    task.write(dest/'orders.json', orders)
    task.write(dest/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    task.current.setup(task.DEVICE)
    rows, train, hold, rid = task.records()
    a, c = task.arrays(), caches()
    probe = np.r_[train[:3], hold[:3]]
    if not (ROOT/'checks.json').exists():
        actual = predict(Network('anchor'), a, c, probe)['score']
        expected = task.predict(previous.anchor(), a, rows, probe, rid)['score']
        parity = float(abs(actual-expected).max())
        assert parity < 1e-4, parity
        baseline = predict(Network('none'), a, c, probe)['score']
        zero_errors = {}
        for arm in ARMS[1:]:
            z = predict(Network(arm), a, c, probe)['score']
            zero_errors[arm] = float(abs(z-baseline).max())
        assert max(zero_errors.values()) < 1e-5, zero_errors
        task.write(ROOT/'checks.json', dict(original_parity=parity, zero_start_errors=zero_errors))
        np.savez_compressed(ROOT/'anchor.npz', **predict(Network('anchor'), a, c, hold))
        print('CHECKS', parity, zero_errors, flush=True)
    for arm in args.arms:
        receipt = fit(arm, args.seed, rows, train, hold, a, c)
        print('DONE', arm, args.seed, receipt['final'], flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    results = []
    for p in ROOT.glob('*/receipt.json'):
        r = json.loads(p.read_text())
        results.append(dict(arm=r['arm'], seed=r['seed'], **r['final']))
    pd.DataFrame(results).to_csv(ROOT/'summary.csv', index=False)


if __name__ == '__main__':
    main()
