"""Native-pixel dense-suffix pilot, with matched continuation controls."""
from pathlib import Path
import argparse
import gc
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'staged_ablation_20261109'
sys.path.insert(0, str(SOURCE))
import staged as e
from sar_m123.tokens import REGIONS

torch, F = e.torch, e.F
ARMS = ('standard_current', 'dense_current', 'standard_mean', 'dense_mean')
SEEDS = (17, 29, 43)
STEPS, BATCH, MICRO = 96, 8, 4


def dense_suffix(bb):
    # torchvision replace_stride_with_dilation=[False, True, False].
    bb.layer3[0].conv2.stride = (1, 1)
    bb.layer3[0].downsample[0].stride = (1, 1)
    bb.layer3[0].stride = 1
    for block in list(bb.layer3.children())[1:] + list(bb.layer4.children()):
        block.conv2.dilation = (2, 2)
        block.conv2.padding = (2, 2)


def mean_message(matcher, q, r, qm, rm, write, legal, labels):
    """Same value encoder/writer; classwise area means, no learned comparison."""
    onehot = F.one_hot(labels, matcher.classes).to(q)
    updates = []
    for p in range(2):
        _, vq = matcher.encode(q[:, p], p)
        _, vr = matcher.encode(r[:, p], p)
        fields = []
        for region in REGIONS:
            ix = region.tolist()
            qa, ra = qm[:, p, ix], rm[:, p, ix]
            live = legal[:, :, p] & (ra.sum(-1) > 0)[None] & (qa.sum(-1) > 0)[:, None]
            area = ra / ra.sum(-1, keepdim=True).clamp_min(1e-20)
            refmean = torch.einsum('ri,rid->rd', area, vr[:, ix])
            weight = live[:, :, None] * onehot[None]
            count = weight.sum(1)
            proto = torch.einsum('brc,rd->bcd', weight.to(q), refmean) / count[..., None].clamp_min(1)
            active = (count > 0)[..., None] & (qa > 0)[:, None]
            evidence = -(vq[:, None, ix] - proto[:, :, None]).square().sum(-1) / .1
            evidence = evidence * active
            centered = (evidence - evidence.sum(1, keepdim=True) / active.sum(1, keepdim=True).clamp_min(1)) * active
            fields.append(centered.transpose(1, 2))
        field = torch.cat(fields, 1)
        mixed = torch.einsum('bshw,bsc->bchw', write[:, p], field)
        updates.append(.1 * matcher.output[p](mixed))
    return torch.stack(updates, 1)


class Network(e.Network):
    def __init__(self, arm, seed):
        super().__init__('full', seed, None)
        self.variant = arm
        self.dense = arm.startswith('dense')
        self.mean = arm.endswith('mean')
        self.grid = 20 if self.dense else 10
        if self.dense:
            dense_suffix(self.base.backbone.model)
        self.phase = 3

    def forward(self, a, c, rows, ids, rid, m3_on=True):
        valid, av = e.task.tensor(a['valid'], ids), e.task.tensor(a['available'], ids)
        local = e.task.tensor(c['local_slots'], ids)
        slots = local if self.dense else e.task.tensor(a['slots'], ids)
        writer = e.task.tensor(a['write20' if self.dense else 'write'], ids)
        x = e.task.tensor(c['early_x'], ids)
        values, responses = [], []
        for p in range(2):
            occ = e.old.grid_occupancy(valid[:, p], (20, 20)) * av[:, p, None, None, None]
            result, trace = self.m1[p](x[:, p], occ, details=True)
            values.append(result[0]); responses.append(trace['response'])
        x = torch.stack(values, 1)
        bb, n, g = self.base.backbone.model, len(ids), self.grid
        early = bb.layer2[3]
        z = early.relu(early.bn3(early.conv3(x.flatten(0, 1))) + e.task.tensor(c['early_identity'], ids).flatten(0, 1))
        live2 = z.reshape(n, 2, 512, 20, 20)
        low, masks = [], []
        for p in range(2):
            reg, en = self.base.region_integrals(local[:, p], local.new_zeros(n, 20, 3, 20, 20))
            low.append(self.base.m2[p](live2[:, p], reg, valid[:, p], en)['pooled'])
            masks.append(torch.minimum(reg, e.old.grid_occupancy(valid[:, p], (20, 20))))
        for block in list(bb.layer3.children())[:5]:
            z = block(z)
        block, identity = bb.layer3[5], z
        x = block.relu(block.bn1(block.conv1(z))).reshape(n, 2, 256, g, g)
        values = []
        for p in range(2):
            reg, en = self.base.region_integrals(slots[:, p], slots.new_zeros(n, 20, 3, g, g))
            values.append(self.base.internal_m2[p](x[:, p], reg, valid[:, p], en)[0])
        x = torch.stack(values, 1)
        x = block.relu(block.bn2(block.conv2(x.flatten(0, 1))))
        x = block.relu(block.bn3(block.conv3(x)) + identity).reshape(n, 2, 1024, g, g)
        if m3_on:
            pair = e.task.legal(rows, ids, rid)[..., None] & av[:, None].bool() & e.task.tensor(a['available'], rid)[None].bool()
            args = (e.task.tensor(a['desc'], ids), e.task.tensor(a['desc'], rid),
                    e.task.tensor(a['mass'], ids), e.task.tensor(a['mass'], rid), writer, pair,
                    torch.tensor([rows[i]['label'] for i in rid], device=e.task.DEVICE))
            delta = mean_message(self.matcher, *args) if self.mean else self.matcher(*args, arm='aggregate')[0]
            x = x + delta
        deep = bb.layer4(x.flatten(0, 1))
        glob = e.old.regional_mean(deep, valid.flatten(0, 1), valid.flatten(0, 1))[0][:, 0].reshape(n, 2, -1)
        pooled = []
        for p in range(2):
            reg, en = self.base.region_integrals(slots[:, p], slots.new_zeros(n, 20, 3, g, g))
            extra = self.base.deep_pool[p](x[:, p], reg, valid[:, p], en)['pooled']
            pooled.append((low[p] + self.base.deep_output[p](extra)).flatten(1))
        feat = torch.cat((glob, torch.stack(pooled, 1)), -1) * av[..., None]
        b = dict(features=torch.cat((feat.flatten(1), av), 1), x=live2,
                 mask=torch.stack(masks, 1), response=torch.stack(responses, 1), available=av,
                 valid=e.old.grid_occupancy(valid.flatten(0, 1), (20, 20)).reshape(n, 2, 1, 20, 20))
        return self.guide(b), None


def load(arm, seed):
    net = Network(arm, seed)
    state = torch.load(SOURCE / f'full_s{seed}_lr1/readout/final.pt', map_location='cpu', weights_only=True)
    net.load_state_dict(state, strict=True)
    del state
    return net.eval()


def configure(net):
    net.requires_grad_(False)
    bb = net.base.backbone.model
    for i in (1, 2, 3):
        getattr(bb.layer3[5], f'conv{i}').requires_grad_(True)
    for module in (net.base.internal_m2, net.base.deep_output, net.guide.value, net.guide.output, net.guide.head, net.matcher):
        module.requires_grad_(True)
    if net.mean:
        for name in ('key', 'key_norm', 'metric'):
            getattr(net.matcher, name).requires_grad_(False)
    net.eval()


@torch.no_grad()
def predict(net, a, c, rows, ids, rid):
    return np.concatenate([net(a, c, rows, ids[i:i+MICRO], rid)[0].cpu().numpy() for i in range(0, len(ids), MICRO)])


def smoke(rows, groups, a, c):
    ids = groups['validation'][:4]; rid = groups['references']
    low, high = load('standard_current', 17), load('dense_current', 17)
    with torch.no_grad():
        score = predict(low, a, c, rows, ids, rid)
        expected = np.load(SOURCE / 'full_s17_lr1/validation.npz')['score'][:4]
        replay = float(abs(score - expected).max())
        assert replay < 1e-4, replay
        # The dense backbone must contain the original lattice before regional operations.
        z = torch.randn(2, 512, 20, 20, device=e.task.DEVICE)
        zlow, zhigh = z.clone(), z.clone()
        for bl, bh in zip(list(low.base.backbone.model.layer3.children())[:5], list(high.base.backbone.model.layer3.children())[:5]):
            zlow, zhigh = bl(zlow), bh(zhigh)
        lattice = float((zlow - zhigh[..., ::2, ::2]).abs().max())
        assert torch.allclose(zlow, zhigh[..., ::2, ::2], atol=2e-4, rtol=2e-5), lattice
    del low, high, z, zlow, zhigh
    gc.collect(); torch.cuda.empty_cache()
    gradients = {}
    for arm in ARMS:
        net = load(arm, 17); configure(net)
        z = net(a, c, rows, ids[:2], rid)[0]
        F.cross_entropy(z, torch.tensor([rows[i]['label'] for i in ids[:2]], device=e.task.DEVICE)).backward()
        norms = {n: float(p.grad.norm()) for n, p in net.named_parameters() if p.requires_grad and p.grad is not None}
        assert np.isfinite(list(norms.values())).all()
        assert any(v > 0 for n, v in norms.items() if 'internal_m2' in n)
        assert any(v > 0 for n, v in norms.items() if 'matcher.output' in n)
        gradients[arm] = dict(grid=net.grid, gradients=norms, trainable=sum(p.numel() for p in net.parameters() if p.requires_grad))
        del net, z; gc.collect(); torch.cuda.empty_cache()
    e.task.write(ROOT / 'smoke.json', dict(replay_error=replay, lattice_error=lattice, variants=gradients))
    print('SMOKE_OK', replay, lattice, flush=True)


def fit(arm, seed, rows, groups, a, c):
    dest = ROOT / f'{arm}_s{seed}'; dest.mkdir(exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    net = load(arm, seed); configure(net)
    rid, train = groups['references'], groups['small']
    y = np.array([r['label'] for r in rows]); weight = e.weights(rows, train)
    orders = e.orders(train, seed+20261111, steps=STEPS, batch=BATCH, continuous=True)
    e.task.write(dest / 'orders.json', [x.tolist() for x in orders])
    params = {n:p for n,p in net.named_parameters() if p.requires_grad}
    buckets = {}
    for name, p in params.items():
        lr = 1e-5 if name.startswith('base.backbone.') else 3e-5 if name.startswith(('guide.head.', 'matcher.output.')) else 1e-4
        buckets.setdefault(lr, []).append(p)
    opt = torch.optim.AdamW([dict(params=p, lr=lr) for lr,p in buckets.items()], weight_decay=.01)
    metrics0 = {}
    for split in ('validation', 'evaluation'):
        z = predict(net, a, c, rows, groups[split], rid)
        np.savez_compressed(dest / f'initial_{split}.npz', ids=groups[split], score=z)
        metrics0[split] = e.task.metrics(y[groups[split]], z)
    tick = time.perf_counter(); curve = []
    torch.cuda.reset_peak_memory_stats()
    for step, batch in enumerate(orders, 1):
        opt.zero_grad(set_to_none=True); loss_sum = 0.
        for start in range(0, BATCH, MICRO):
            ids = batch[start:start+MICRO]
            score = net(a, c, rows, ids, rid)[0]
            labels = torch.tensor(y[ids], device=e.task.DEVICE)
            w = torch.tensor(weight[ids], device=e.task.DEVICE)
            loss = (F.cross_entropy(score, labels, reduction='none') * w).sum() / BATCH
            assert torch.isfinite(loss), (arm, seed, step)
            loss.backward(); loss_sum += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(list(params.values()), 5.)
        opt.step(); curve.append(dict(step=step, loss=loss_sum))
        if step % 24 == 0:
            e.prev.pd.DataFrame(curve).to_csv(dest / 'curve.csv', index=False)
            e.task.write(ROOT / 'status.json', dict(complete=False, arm=arm, seed=seed, step=step, steps=STEPS))
            print('STEP', arm, seed, step, round(loss_sum, 4), round(time.perf_counter()-tick, 1), flush=True)
    train_seconds = time.perf_counter()-tick
    torch.save(net.state_dict(), dest / 'final.pt')
    metrics = {}
    for split in ('validation', 'evaluation'):
        z = predict(net, a, c, rows, groups[split], rid)
        np.savez_compressed(dest / f'{split}.npz', ids=groups[split], score=z)
        metrics[split] = e.task.metrics(y[groups[split]], z)
    e.task.write(dest / 'complete.json', dict(arm=arm, seed=seed, steps=STEPS, examples=STEPS*BATCH,
        initial=metrics0, metrics=metrics, seconds=train_seconds, peak_gib=torch.cuda.max_memory_allocated()/1024**3,
        trainable=sum(p.numel() for p in params.values()), formal_model_changed=False))
    print('COMPLETE', arm, seed, metrics, flush=True)
    del net, opt, params, buckets, score, loss
    gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = parser.parse_args(); ROOT.mkdir(exist_ok=True)
    e.task.current.setup(e.task.DEVICE)
    rows, groups = e.prev.records(); a, c = e.prev.arrays()
    a['write'] = np.load(e.PREVIOUS / 'common_init/write.npy', mmap_mode='r')
    # Reuse native-geometry occupancy, not interpolation of the ten-grid writer.
    local = c['local_slots']
    a['write20'] = np.asarray(local) / np.maximum(np.asarray(local).sum(2, keepdims=True), 1)
    e.task.write(ROOT / 'protocol.json', dict(arms=ARMS, seeds=list(args.seeds), steps=STEPS, batch=BATCH, microbatch=MICRO,
        groups={k:len(v) for k,v in groups.items()}, input='unchanged native160; no resize/recrop',
        initial='same seed-specific full readout checkpoint for all four variants',
        change='layer3 stride 2 to 1; subsequent 3x3 dilation2; layer4 keeps stride2',
        grid='layer2 unchanged20; layer3 10->20; layer4 5->10',
        m3='same frozen layer1/layer2 comparison descriptors and 32 references; writer uses native grid occupancy',
        simple='area-mean per reference, equal legal references per class, unit squared value distance; no metric/bias/transport',
        optimization='96 paired joint continuation updates; source class/vessel weights, CE only, fixed BN/M1/prefix',
        intervention='inherited in starting weights; no new teacher selection or intervention pass',
        scope='small continuation pilot, not fresh end-to-end training or independent test; no model promotion'))
    smoke(rows, groups, a, c)
    if args.smoke_only:
        return
    for seed in args.seeds:
        for arm in ARMS:
            fit(arm, seed, rows, groups, a, c)
    e.task.write(ROOT / 'status.json', dict(complete=True, arms=ARMS, seeds=list(args.seeds)))


if __name__ == '__main__':
    main()
