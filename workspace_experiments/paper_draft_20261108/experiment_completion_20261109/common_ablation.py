"""Matched common-initialization architecture experiment; not checkpoint surgery."""
from pathlib import Path
import argparse
import json
import gc
import time
import numpy as np
import raw_current as raw

s, prev, task, torch = raw.s, raw.prev, raw.task, raw.torch
nn, F = torch.nn, torch.nn.functional
from sar_m123.current.spatial_m12 import SpatialM12ResNet50
from sar_m123.pool_replacement import regional_mean, grid_occupancy
from operators import RelationFeedback

ROOT = Path(__file__).resolve().parent/'common_init'
WEIGHTS = Path('C:/Users/25791/.cache/torch/hub/checkpoints')
ARMS = ('plain', 'full', 'no_m1', 'no_m2', 'no_m3')


def pretrained_path():
    # Identify the actual frozen prefix by direct tensors, not by its filename.
    anchor = prev.e.previous.anchor().base.backbone.model.state_dict()
    matches = []
    for path in WEIGHTS.glob('resnet50-*.pth'):
        state = torch.load(path, map_location='cpu', weights_only=True)
        keys = [k for k in anchor if k.startswith(('conv1.', 'bn1.', 'layer1.', 'layer2.'))]
        if all(torch.equal(anchor[k].cpu(), state[k]) for k in keys): matches.append(path)
    assert len(matches) == 1, matches
    return matches[0]


class Network(nn.Module):
    def __init__(self, arm, seed, weight_path):
        super().__init__()
        self.arm = arm
        self.use1 = arm not in ('plain', 'no_m1')
        self.use2 = arm not in ('plain', 'no_m2')
        self.use3 = arm not in ('plain', 'no_m3')
        torch.manual_seed(seed)
        self.base = SpatialM12ResNet50(weight_path, 'pre3x3', 'local', 4)
        self.base.internal_m1 = nn.ModuleList()
        self.base.m1 = nn.ModuleDict()
        self.base.m3 = nn.ModuleList()
        self.base.anchor_last = nn.Identity()
        self.base.head = nn.Identity()
        # Instantiate identical shared parameters before removing unused modules.
        self.m1 = nn.ModuleList(RelationFeedback(128, 'class', .3) for _ in range(2))
        self.guide = s.RegionalGuidance('class' if self.use1 else 'mean')
        self.guide.head.reset_parameters()
        self.matcher = task.MaskedCostLocalMatch(seed=seed, classes=4)
        if not self.use1: self.m1 = nn.ModuleList()
        if not self.use3: self.matcher = None
        if not self.use2:
            old = self.guide.head
            self.head = nn.Linear(4098, 4)
            with torch.no_grad():
                self.head.weight.copy_(torch.cat((old.weight[:, :2048], old.weight[:, 3584:5632], old.weight[:, -2:]), 1))
                self.head.bias.copy_(old.bias)
            self.guide = None
            self.base.internal_m2 = nn.ModuleList()
            self.base.m2 = nn.ModuleList()
            self.base.deep_pool = nn.ModuleList()
            self.base.deep_output = nn.ModuleList()
        self.to(task.DEVICE).eval()
        self.base.backbone.requires_grad_(False)
        # Same common backbone update budget in every arm, including Plain.
        self.base.backbone.model.layer2[3].conv3.requires_grad_(True)
        for i in (1, 2, 3): getattr(self.base.backbone.model.layer3[5], f'conv{i}').requires_grad_(True)
        for name, p in self.named_parameters():
            if '.scorer.' in name: p.requires_grad_(False)

    def forward(self, a, c, rows, ids, rid, m3_on=True):
        valid = task.tensor(a['valid'], ids)
        av = task.tensor(a['available'], ids)
        slots = task.tensor(a['slots'], ids)
        local = task.tensor(c['local_slots'], ids)
        x = task.tensor(c['early_x'], ids)
        aux, responses = [], []
        if self.use1:
            values = []
            for p in range(2):
                occ = grid_occupancy(valid[:, p], (20, 20))*av[:, p, None, None, None]
                result, trace = self.m1[p](x[:, p], occ, details=True)
                values.append(result[0]); aux.append(result[1]); responses.append(trace['response'])
            x = torch.stack(values, 1)
        else: responses = [x.new_zeros((len(ids), 4, 20, 20)) for _ in range(2)]
        bb = self.base.backbone.model
        early = bb.layer2[3]
        z = early.relu(early.bn3(early.conv3(x.flatten(0, 1)))+task.tensor(c['early_identity'], ids).flatten(0, 1))
        live2 = z.reshape(len(ids), 2, 512, 20, 20)
        low, masks = [], []
        if self.use2:
            for p in range(2):
                reg, en = self.base.region_integrals(local[:, p], local.new_zeros(len(ids), 20, 3, 20, 20))
                low.append(self.base.m2[p](live2[:, p], reg, valid[:, p], en)['pooled'])
                masks.append(torch.minimum(reg, grid_occupancy(valid[:, p], (20, 20))))
        for layer in list(bb.layer3.children())[:5]: z = layer(z)
        identity = z
        block = bb.layer3[5]
        x = block.relu(block.bn1(block.conv1(z))).reshape(len(ids), 2, 256, 10, 10)
        if self.use2:
            values = []
            for p in range(2):
                reg, en = self.base.region_integrals(slots[:, p], slots.new_zeros(len(ids), 20, 3, 10, 10))
                values.append(self.base.internal_m2[p](x[:, p], reg, valid[:, p], en)[0])
            x = torch.stack(values, 1)
        x = block.relu(block.bn2(block.conv2(x.flatten(0, 1))))
        x = block.relu(block.bn3(block.conv3(x))+identity).reshape(len(ids), 2, 1024, 10, 10)
        if self.use3 and m3_on:
            pair = task.legal(rows, ids, rid)[..., None] & av[:, None].bool() & task.tensor(a['available'], rid)[None].bool()
            delta, _, _, _ = self.matcher(task.tensor(a['desc'], ids), task.tensor(a['desc'], rid),
                task.tensor(a['mass'], ids), task.tensor(a['mass'], rid), task.tensor(a['write'], ids),
                pair, torch.tensor([rows[i]['label'] for i in rid], device=task.DEVICE), arm='aggregate')
            x = x+delta
        deep = bb.layer4(x.flatten(0, 1))
        glob = regional_mean(deep, valid.flatten(0, 1), valid.flatten(0, 1))[0][:, 0].reshape(len(ids), 2, -1)
        if self.use2:
            pooled = []
            for p in range(2):
                reg, en = self.base.region_integrals(slots[:, p], slots.new_zeros(len(ids), 20, 3, 10, 10))
                extra = self.base.deep_pool[p](x[:, p], reg, valid[:, p], en)['pooled']
                pooled.append((low[p]+self.base.deep_output[p](extra)).flatten(1))
            feat = torch.cat((glob, torch.stack(pooled, 1)), -1)*av[..., None]
            b = dict(features=torch.cat((feat.flatten(1), av), 1), x=live2,
                     mask=torch.stack(masks, 1), response=torch.stack(responses, 1), available=av,
                     valid=grid_occupancy(valid.flatten(0, 1), (20, 20)).reshape(len(ids), 2, 1, 20, 20))
            score = self.guide(b)
        else: score = self.head(torch.cat(((glob*av[..., None]).flatten(1), av), 1))
        return score, torch.stack(aux, 1) if aux else None


def load_arrays(rows):
    a, c = prev.arrays()
    # Writer geometry is deterministic and requires no class labels.
    path = ROOT/'write.npy'
    if not path.exists():
        out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(rows), 2, 20, 10, 10))
        for i, row in enumerate(rows):
            for p, view in enumerate(task.data.prepare(row)):
                good = view is not None and view['valid_target']
                out[i, p] = raw.build_masks(view['geometry'] if good else None,
                    view['footprint'] if good else np.zeros((160, 160)), 'coarse_fixed')['write']
        out.flush()
    a['write'] = np.load(path, mmap_mode='r')
    return a, c


@torch.no_grad()
def predict(net, a, c, rows, ids, rid):
    return np.concatenate([net(a, c, rows, ids[i:i+12], rid)[0].cpu().numpy() for i in range(0, len(ids), 12)])


def fit(arm, seed, steps, weight_path, rows, groups, a, c):
    dest = ROOT/f'{arm}_s{seed}_{steps}'; dest.mkdir(exist_ok=True)
    if (dest/'receipt.json').exists(): return
    net = Network(arm, seed, weight_path)
    params = {n: p for n, p in net.named_parameters() if p.requires_grad}
    buckets = {}
    for name, p in params.items():
        lr = 1e-5 if name.startswith('base.backbone.') else 3e-4
        buckets.setdefault(lr, []).append(p)
    opt = torch.optim.AdamW([dict(params=v, lr=k) for k, v in buckets.items()], weight_decay=.01)
    y = np.array([r['label'] for r in rows]); train = groups['small']; rid = groups['references']
    fr = prev.pd.DataFrame([rows[i] for i in train]); w = np.zeros(len(rows), np.float32)
    w[train] = 1/(fr.groupby('label').mmsi.transform('nunique').to_numpy()*fr.groupby('mmsi').mmsi.transform('size').to_numpy())
    w[train] /= w[train].mean()
    rng = np.random.default_rng(seed); stream = []
    while len(stream) < steps*12: stream.extend(rng.permutation(train).tolist())
    batches = np.array(stream[:steps*12]).reshape(steps, 12)
    task.write(dest/'orders.json', batches.tolist())
    curve = []; gradient = []; tick = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for step, ids in enumerate(batches, 1):
        opt.zero_grad(set_to_none=True)
        z, aux = net(a, c, rows, ids, rid)
        label = torch.tensor(y[ids], device=task.DEVICE); weight = torch.tensor(w[ids], device=task.DEVICE)
        ce = (F.cross_entropy(z, label, reduction='none')*weight).mean()
        ace = ce.new_zeros(())
        if aux is not None:
            av = task.tensor(a['available'], ids)
            terms = F.cross_entropy(aux.flatten(0, 1), label[:, None].expand(-1, 2).flatten(), reduction='none').reshape(len(ids), 2)
            ace = ((terms*av).sum(1)/av.sum(1).clamp_min(1)*weight).mean()
        loss = ce+.02*ace
        assert torch.isfinite(loss), (arm, step)
        loss.backward()
        if step in (1, 2):
            norms = {name: float(p.grad.norm()) for name, p in params.items() if p.grad is not None}
            assert all(np.isfinite(list(norms.values()))), norms
            gradient.append(dict(step=step, norms=norms))
        torch.nn.utils.clip_grad_norm_(list(params.values()), 5.)
        opt.step()
        curve.append(dict(step=step, loss=float(loss.detach()), ce=float(ce.detach()), aux=float(ace.detach())))
        if step%30 == 0 or step == steps:
            prev.pd.DataFrame(curve).to_csv(dest/'curve.csv', index=False)
            print('STEP', arm, seed, step, round(float(ce.detach()), 4), round(time.perf_counter()-tick, 1), flush=True)
    torch.save(net.state_dict(), dest/'final.pt')
    metrics = {}
    for split in ('small', 'validation', 'evaluation'):
        ids = groups[split]; z = predict(net, a, c, rows, ids, rid)
        np.savez_compressed(dest/f'{split}.npz', ids=ids, score=z)
        metrics[split] = task.metrics(y[ids], z)
    task.write(dest/'receipt.json', dict(arm=arm, seed=seed, steps=steps, batch=12, n_train=len(train),
        weight_path=str(weight_path), initialization='common ImageNet; no source-trained module weights',
        selected_step=steps, selection='fixed budget, no target score selection',
        trainable=sum(p.numel() for p in params.values()), parameters=sum(p.numel() for p in net.parameters()),
        seconds=time.perf_counter()-tick, peak_gib=torch.cuda.max_memory_allocated()/1024**3,
        metrics=metrics, gradients=gradient, intervention_training=False,
        scope='common initialization architecture pilot, not historical staged checkpoint replay'))
    print('DONE', arm, seed, metrics, flush=True)
    del net, opt; gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--steps', type=int, default=360)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43])
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); ROOT.mkdir(exist_ok=True)
    task.current.setup(task.DEVICE); rows, groups = prev.records()
    path = pretrained_path(); print('COMMON_PRETRAIN', path, flush=True)
    a, c = load_arrays(rows)
    task.write(ROOT/'protocol.json', dict(arms=args.arms, seeds=args.seeds, steps=args.steps,
        source=len(groups['small']), references=len(groups['references']), pretrain=str(path),
        initialization='ImageNet backbone plus fresh modules and head', query_labels='loss and reporting only',
        anchor='frozen ImageNet layer1/layer2 descriptors', reference_exclusion='same MMSI or same platform orbit',
        optimization='joint CE + 0.02 M1 auxiliary; fixed BN; four common conv weights trainable',
        scope='new unified-optimization pilot; does not replace current staged working version'))
    if args.prepare_only: return
    if args.smoke:
        report = {}
        ids = np.array([0, 138, 221, 478])
        for arm in ARMS:
            net = Network(arm, 17, path)
            z, aux = net(a, c, rows, ids, groups['references'])
            loss = F.cross_entropy(z, torch.tensor([rows[i]['label'] for i in ids], device=task.DEVICE))
            loss.backward()
            grad = {n: float(p.grad.norm()) for n, p in net.named_parameters() if p.requires_grad and p.grad is not None}
            assert all(np.isfinite(list(grad.values()))) and torch.isfinite(z).all()
            assert ('m1.0.output.weight' in grad) == net.use1
            assert ('base.internal_m2.0.transforms.0.2.weight' in grad) == net.use2
            assert ('matcher.output.0.weight' in grad) == net.use3
            report[arm] = dict(shape=list(z.shape), loss=float(loss.detach()), gradients=grad)
            del net; gc.collect(); torch.cuda.empty_cache()
        task.write(ROOT/'smoke.json', report)
        print('SMOKE_OK', list(report), flush=True)
        return
    for seed in args.seeds:
        for arm in args.arms: fit(arm, seed, args.steps, path, rows, groups, a, c)


if __name__ == '__main__': main()
