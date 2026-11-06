"""Bounded OSS2 M1 operator/position experiment, using an unchanged E17 carrier."""
from pathlib import Path
import argparse
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'m1_tuning_position_20261101'))
import tune as prior
from tune import torch, np, pd, e, task
from torch.nn import functional as F
from sar_m123.pool_replacement import grid_occupancy, regional_mean
sys.path.insert(0, str(ROOT))
from operators import RelationFeedback


@torch.no_grad()
def prepare_early(rows, a):
    dest = ROOT/'cache'
    dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        return {p.stem: np.load(p, mmap_mode='r') for p in dest.glob('*.npy')}
    net = e.previous.anchor().base
    backbone = net.backbone.model
    block = backbone.layer2[3]
    expected = np.load(task.PUBLIC/'cache/prefix.npy', mmap_mode='r')
    local_slots = np.load(task.PUBLIC/'cache/local_slot_masks.npy', mmap_mode='r')
    stores, errors = {}, []
    tick = time.perf_counter()
    for start in range(0, len(rows), 12):
        ids = np.arange(start, min(start+12, len(rows)))
        images = np.zeros((len(ids), 2, 1, 160, 160), np.float32)
        for j, idx in enumerate(ids):
            for p, view in enumerate(task.data.prepare(rows[idx])):
                if view is not None and view['valid_target']:
                    images[j, p, 0] = view['pixels'].astype(np.float32)/255
        z = net.prefix(torch.tensor(images, device=task.DEVICE), stop=1).flatten(0, 1)
        for layer in list(backbone.layer2.children())[:3]:
            z = layer(z)
        identity = z
        x = block.relu(block.bn1(block.conv1(z)))
        x = block.relu(block.bn2(block.conv2(x)))
        reconstructed = block.relu(block.bn3(block.conv3(x))+identity).reshape(len(ids), 2, 512, 20, 20)
        ix = [rows[i]['indices']['table_index'] for i in ids]
        errors.append(float(abs(reconstructed-task.tensor(expected, ix)).max()))
        values = dict(early_x=x.reshape(len(ids), 2, 128, 20, 20),
                      early_identity=identity.reshape(len(ids), 2, 512, 20, 20),
                      local_slots=np.array(local_slots[ix]))
        for key, value in values.items():
            arr = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            if key not in stores:
                stores[key] = np.lib.format.open_memmap(dest/(key+'.npy'), mode='w+', dtype=arr.dtype,
                                                       shape=(len(rows),)+arr.shape[1:])
            stores[key][ids] = arr
        if start % 120 == 0:
            print('EARLY_CACHE', start+len(ids), len(rows), max(errors), flush=True)
    for store in stores.values():
        store.flush()
    assert max(errors) < 1e-4, max(errors)
    task.write(dest/'complete.json', dict(n=len(rows), prefix_error=max(errors), seconds=time.perf_counter()-tick,
                                        source_and_development_only=True))
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return stores


class Network(e.Network):
    def __init__(self, position, kind, gain=.3):
        super().__init__('post_mean_gate')
        self.position, self.kind = position, kind
        self.base.requires_grad_(False)
        channels = 256 if position == 'late' else 128
        self.m1 = torch.nn.ModuleList(RelationFeedback(channels, kind, gain).to(task.DEVICE) for _ in range(2))
        layer = self.base.backbone.model.layer3[5] if position == 'late' else self.base.backbone.model.layer2[3]
        layer.conv3.requires_grad_(True)
        self.base.head.requires_grad_(True)
        self.trace = None
        self.eval()

    def apply(self, x, valid, available, mode):
        out, aux, ratios, traces = [], [], [], []
        for p, op in enumerate(self.m1):
            effective = 'off' if (mode == 'vv_only' and p == 1) or (mode == 'vh_only' and p == 0) else mode
            occ = grid_occupancy(valid[:, p], x.shape[-2:])*available[:, p, None, None, None]
            result, trace = op(x[:, p], occ, effective, details=True)
            z, u, ratio = result
            out.append(z)
            aux.append(u)
            ratios.append(ratio)
            if self.trace is not None:
                traces.append({k: t.detach().cpu().numpy() for k, t in trace.items()})
        if self.trace is not None:
            self.trace = traces
        return torch.stack(out, 1), torch.stack(aux, 1), torch.stack(ratios, 1)

    def forward(self, a, c, ids, mode=None):
        valid = task.tensor(a['valid'], ids)
        av = task.tensor(a['available'], ids)
        slots = task.tensor(a['slots'], ids)
        block = self.base.backbone.model.layer3[5]
        if self.position == 'early':
            early = self.base.backbone.model.layer2[3]
            x = task.tensor(c['early_x'], ids)
            x, aux, ratios = self.apply(x, valid, av, mode)
            x = early.relu(early.bn3(early.conv3(x.flatten(0, 1)))+
                           task.tensor(c['early_identity'], ids).flatten(0, 1))
            live2 = x.reshape(len(ids), 2, 512, 20, 20)
            low, ls = [], task.tensor(c['local_slots'], ids)
            for p in range(2):
                reg, en = self.base.region_integrals(ls[:, p], ls.new_zeros(len(ids), 20, 3, 20, 20))
                low.append(self.base.m2[p](live2[:, p], reg, valid[:, p], en)['pooled'])
            low = torch.stack(low, 1)
            # Frozen layers must still propagate gradients to the early M1.
            for layer in list(self.base.backbone.model.layer3.children())[:5]:
                x = layer(x)
            identity = x
            x = block.relu(block.bn1(block.conv1(x))).reshape(len(ids), 2, 256, 10, 10)
        else:
            x = task.tensor(c['x'], ids)
            identity = task.tensor(c['identity'], ids).flatten(0, 1)
            low = task.tensor(a['low'], ids)
        values = []
        for p in range(2):
            reg, en = self.base.region_integrals(slots[:, p], slots.new_zeros(len(ids), 20, 3, 10, 10))
            z, _ = self.base.internal_m2[p](x[:, p], reg, valid[:, p], en)
            values.append(z)
        x = torch.stack(values, 1).flatten(0, 1)
        x = block.relu(block.bn2(block.conv2(x))).reshape(len(ids), 2, 256, 10, 10)
        if self.position == 'late':
            x, aux, ratios = self.apply(x, valid, av, mode)
        x = block.relu(block.bn3(block.conv3(x.flatten(0, 1)))+identity).reshape(len(ids), 2, 1024, 10, 10)
        updated = x+task.tensor(c['delta'], ids)
        deep = self.base.backbone.model.layer4(updated.flatten(0, 1))
        glob = regional_mean(deep, valid.flatten(0, 1), valid.flatten(0, 1))[0][:, 0].reshape(len(ids), 2, -1)
        pooled = []
        for p in range(2):
            reg, en = self.base.region_integrals(slots[:, p], slots.new_zeros(len(ids), 20, 3, 10, 10))
            extra = self.base.deep_pool[p](updated[:, p], reg, valid[:, p], en)
            pooled.append((low[:, p]+self.base.deep_output[p](extra['pooled'])).flatten(1))
        features = torch.cat((glob, torch.stack(pooled, 1)), -1)*av[..., None]
        score = self.base.head(torch.cat((features.flatten(1), av), -1))
        return score, aux, ratios


def fit(position, kind, gain, seed, epochs, rows, train, hold, a, c, fixed_epoch=None):
    dest = ROOT/f'{position}_{kind}_g{int(gain*10)}_s{seed}'
    dest.mkdir(exist_ok=True)
    if (dest/'receipt.json').exists():
        receipt = json.loads((dest/'receipt.json').read_text())
        assert receipt['epochs'] == epochs and receipt['fixed_epoch'] == fixed_epoch
        return receipt
    torch.manual_seed(seed)
    net = Network(position, kind, gain)
    params = {n: p for n, p in net.named_parameters() if p.requires_grad}
    groups = {}
    for n, p in params.items():
        lr = 1e-5 if n.startswith('base.') else (.001 if '.output.' in n or '.relation_output.' in n else .0003)
        groups.setdefault(lr, []).append(p)
    opt = torch.optim.AdamW([dict(params=p, lr=lr) for lr, p in groups.items()], weight_decay=.01)
    weights = prior.weights(rows, train)
    y = np.array([r['label'] for r in rows])
    pred = e.predict(net, a, c, hold)
    np.savez_compressed(dest/'epoch0.npz', **pred)
    curve = [dict(epoch=0, **task.metrics(y[hold], pred['score']))]
    best = curve[0]
    torch.save(net.live_state(), dest/'best.pt')
    orders, gradients, norms = [], [], []
    rng = np.random.default_rng(seed)
    tick = time.perf_counter()
    for epoch in range(1, epochs+1):
        order = rng.permutation(train)
        orders.append(order.tolist())
        losses, aux_losses = [], []
        for start in range(0, len(order), 24):
            ids = order[start:start+24]
            labels = torch.tensor(y[ids], device=task.DEVICE)
            opt.zero_grad(set_to_none=True)
            score, aux, _ = net(a, c, ids)
            ce = (F.cross_entropy(score, labels, reduction='none')*weights[ids]).mean()
            av = task.tensor(a['available'], ids)
            ace = F.cross_entropy(aux.flatten(0, 1), labels[:, None].expand(-1, 2).flatten(), reduction='none').reshape(len(ids), 2)
            ace = ((ace*av).sum(1)/av.sum(1).clamp_min(1)*weights[ids]).mean()
            loss = ce+.02*ace
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss {dest.name} epoch{epoch}')
            loss.backward()
            if epoch == 1 and start in (0, 24):
                g = {n: float(p.grad.norm()) for n, p in params.items() if p.grad is not None}
                assert all(np.isfinite(list(g.values())))
                assert min(g[f'm1.{p}.output.weight'] for p in range(2)) > 0
                if kind != 'mean':
                    assert min(g[f'm1.{p}.relation_output.weight'] for p in range(2)) > 0
                gradients.append(dict(step=start//24, norms=g))
            norms.append(float(torch.nn.utils.clip_grad_norm_(list(params.values()), 5.)))
            opt.step()
            losses.append(float(ce.detach()))
            aux_losses.append(float(ace.detach()))
        pred = e.predict(net, a, c, hold)
        np.savez_compressed(dest/f'epoch{epoch}.npz', **pred)
        row = dict(epoch=epoch, ce=float(np.mean(losses)), auxiliary_ce=float(np.mean(aux_losses)),
                   vv_update=float(pred['update_ratio'][:, 0].mean()), vh_update=float(pred['update_ratio'][:, 1].mean()),
                   **task.metrics(y[hold], pred['score']))
        curve.append(row)
        if (fixed_epoch is None and prior.rank(row) > prior.rank(best)) or epoch == fixed_epoch:
            best = row
            torch.save(net.live_state(), dest/'best.pt')
        pd.DataFrame(curve).to_csv(dest/'curve.csv', index=False)
        print('EPOCH', dest.name, epoch, row['oa'], row['ba'], row['nll'], flush=True)
    torch.save(net.live_state(), dest/'last.pt')
    net.restore(torch.load(dest/'best.pt', map_location='cpu', weights_only=True))
    for mode in (None, 'off', 'relation_off', 'merged', 'spatial_mean', 'vv_only', 'vh_only', 'covariance_zero'):
        np.savez_compressed(dest/('selected.npz' if mode is None else mode+'.npz'), **e.predict(net, a, c, hold, mode))
    source = e.predict(net, a, c, train)
    np.savez_compressed(dest/'source.npz', **source)
    actual = np.load(dest/'selected.npz')['score']
    expected = np.load(dest/f'epoch{best["epoch"]}.npz')['score']
    reload_error = float(abs(actual-expected).max())
    assert reload_error == 0
    receipt = dict(position=position, kind=kind, gain=gain, seed=seed, epochs=epochs, fixed_epoch=fixed_epoch,
                   selected=best, last=curve[-1], steps=len(orders)*30, seconds=time.perf_counter()-tick,
                   source=task.metrics(y[train], source['score']), trainable=sum(p.numel() for p in params.values()),
                   reload_error=reload_error, gradients=gradients, clipped_steps=sum(v > 5 for v in norms),
                   max_gradient_norm=max(norms), selection='development BA/OA/NLL' if fixed_epoch is None else 'frozen seed17 epoch')
    task.write(dest/'receipt.json', receipt)
    task.write(dest/'orders.json', orders)
    print('DONE', dest.name, best, round(receipt['seconds'], 1), flush=True)
    del net, opt
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def summarize():
    summary = []
    for file in ROOT.glob('*/receipt.json'):
        r = json.loads(file.read_text())
        summary.append(dict(run=file.parent.name, position=r['position'], kind=r['kind'], gain=r['gain'], seed=r['seed'],
                            **r['selected'], last_oa=r['last']['oa'], last_ba=r['last']['ba'],
                            seconds=r['seconds'], trainable=r['trainable']))
    pd.DataFrame(summary).to_csv(ROOT/'summary.csv', index=False)


def checks(a, c, train, hold):
    if (ROOT/'checks.json').exists():
        return
    ids = np.r_[train[:2], hold[:2]]
    expected = e.predict(e.Network('none'), a, c, ids)['score']
    errors = {}
    for pos in ('late', 'early'):
        for kind in ('mean', 'second', 'class'):
            torch.manual_seed(17)
            net = Network(pos, kind)
            error = float(abs(e.predict(net, a, c, ids)['score']-expected).max())
            assert error < 1e-4, (pos, kind, error)
            errors[pos+'_'+kind] = error
            op = net.m1[0]
            x = torch.randn(2, op.value.in_channels, 1, 1, device=task.DEVICE).expand(-1, -1, 10, 10)
            _, d = op(x, torch.ones(2, 1, 10, 10, device=task.DEVICE), details=True)
            errors[pos+'_'+kind+'_constant_covariance'] = float(abs(d['covariance']).max())
            assert abs(d['covariance']).max() < 1e-5
            mask = torch.ones(2, 1, 10, 10, device=task.DEVICE)
            mask[:, :, :3] = 0
            x = torch.randn_like(x)
            xx = x.clone()
            xx[:, :, :3] = 100
            _, first = op(x, mask, details=True)
            _, second = op(xx, mask, details=True)
            assert torch.equal(first['covariance'], second['covariance'])
            assert torch.equal(first['mean'], second['mean'])
            del net
    task.write(ROOT/'checks.json', errors)
    print('CHECKS', errors, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--positions', nargs='+', default=['late', 'early'])
    parser.add_argument('--kinds', nargs='+', default=['mean', 'second', 'class'])
    parser.add_argument('--gains', nargs='+', type=float, default=[.1, .3])
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--fixed-epoch', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    task.current.setup(task.DEVICE)
    rows, train, hold, rid = task.records()
    a, c = task.arrays(), e.caches()
    c.update(prepare_early(rows, a))
    checks(a, c, train, hold)
    if args.prepare_only:
        return
    for position in args.positions:
        for gain in args.gains:
            for kind in args.kinds:
                fit(position, kind, gain, args.seed, 6, rows, train, hold, a, c, args.fixed_epoch)
                summarize()


if __name__ == '__main__':
    main()
