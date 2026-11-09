"""Raw-chip replay of the selected, fixed-anchor model. No fitting."""
from pathlib import Path
import sys
import json
import argparse
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
sys.path.insert(0, str(WORK/'m1_guided_m2_20261106'))
import study as s
torch, task, prev = s.torch, s.task, s.prev
from geometry import build_masks, pool_multiscale
from sar_m123 import tokens
from sar_m123.pool_replacement import grid_occupancy


class RawCurrent:
    def __init__(self):
        task.current.setup(task.DEVICE)
        self.rows, self.groups = prev.records()
        self.anchor = prev.e.previous.anchor()
        self.net = prev.Network('class')
        self.net.restore(torch.load(s.CHECKPOINT, map_location='cpu', weights_only=True))
        self.net.eval().requires_grad_(False)
        a, _ = prev.arrays()
        self.rid = self.groups['references']
        self.refdesc = task.tensor(a['desc'], self.rid)
        self.refmass = task.tensor(a['mass'], self.rid)
        self.refav = task.tensor(a['available'], self.rid).bool()
        self.reflabel = torch.tensor([self.rows[i]['label'] for i in self.rid], device=task.DEVICE)
        self.heads = {}
        for seed in (17, 29, 43):
            for arm in ('class', 'mean', 'content', 'no_m3_refit'):
                path = (ROOT.parent/'essential_experiments_20261109/no_m3_refit'/f'class_s{seed}/final.pt'
                        if arm == 'no_m3_refit' else s.ROOT/f'{arm}_s{seed}/final.pt')
                head = s.RegionalGuidance('class' if arm == 'no_m3_refit' else arm)
                head.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
                self.heads[arm, seed] = head.to(task.DEVICE).eval().requires_grad_(False)
        self.seen = {}
        def pool_hook(p):
            def hook(module, args):
                self.seen['x', p] = args[0]
                self.seen['mask', p] = torch.minimum(args[1], grid_occupancy(args[2], args[0].shape[-2:]))
            return hook
        def response_hook(p):
            def hook(module, args, out): self.seen['response', p] = out
            return hook
        for p in range(2):
            self.net.base.m2[p].register_forward_pre_hook(pool_hook(p))
            self.net.m1[p].reader.register_forward_hook(response_hook(p))
        self.net.base.head.register_forward_pre_hook(lambda module, args: self.seen.update(features=args[0]))

    @torch.no_grad()
    def inputs(self, rows):
        images = np.zeros((len(rows), 2, 1, 160, 160), np.float32)
        valid = np.zeros_like(images)
        av = np.zeros((len(rows), 2), np.float32)
        slots = np.zeros((len(rows), 2, 20, 10, 10), np.float32)
        local = np.zeros((len(rows), 2, 20, 20, 20), np.float32)
        geom = []
        for j, row in enumerate(rows):
            gs = []
            for p, view in enumerate(task.data.prepare(row)):
                present = view is not None and view['valid_target']
                if present:
                    images[j, p, 0] = view['pixels'].astype(np.float32)/255
                    valid[j, p, 0] = view['footprint']
                    av[j, p] = 1
                    native = tokens.slots(view['geometry'], view['footprint'])
                    for dest, size in ((slots, 10), (local, 20)):
                        dest[j, p] = grid_occupancy(torch.from_numpy(native).float()[None], (size, size))[0].numpy()
                gs.append(build_masks(view['geometry'] if present else None,
                                      view['footprint'] if present else np.zeros((160, 160)), 'coarse_fixed'))
            geom.append(gs)
        bb = self.anchor.base.backbone.model
        block = bb.layer2[3]
        f1 = self.anchor.base.prefix(torch.tensor(images, device=task.DEVICE), stop=1)
        z = f1.flatten(0, 1)
        for layer in list(bb.layer2.children())[:3]: z = layer(z)
        identity = z
        x = block.relu(block.bn1(block.conv1(z)))
        x = block.relu(block.bn2(block.conv2(x)))
        f2 = block.relu(block.bn3(block.conv3(x))+identity).reshape(len(rows), 2, 512, 20, 20)
        read = torch.tensor(np.stack([[g['read'] for g in gs] for gs in geom]), device=task.DEVICE)
        mass = torch.tensor(np.stack([[g['mass'] for g in gs] for gs in geom]), device=task.DEVICE)
        writer = torch.tensor(np.stack([[g['write'] for g in gs] for gs in geom]), device=task.DEVICE)
        desc = pool_multiscale(f1.flatten(0, 1), f2.flatten(0, 1), read.flatten(0, 1), 'coarse_fixed').reshape(len(rows), 2, 20, 768)
        legal = torch.tensor([[str(q['mmsi']) != str(self.rows[i]['mmsi']) and task.orbit(q) != task.orbit(self.rows[i])
                               for i in self.rid] for q in rows], device=task.DEVICE)
        pair = legal[..., None] & torch.tensor(av, device=task.DEVICE)[:, None].bool() & self.refav[None]
        delta = self.anchor.matcher(desc, self.refdesc, mass, self.refmass, writer, pair, self.reflabel, arm='aggregate')[0]
        a = dict(valid=valid, available=av, slots=slots, desc=desc.cpu().numpy(),
                 mass=mass.cpu().numpy(), write=writer.cpu().numpy())
        c = dict(early_x=x.reshape(len(rows), 2, 128, 20, 20).cpu().numpy(),
                 early_identity=identity.reshape(len(rows), 2, 512, 20, 20).cpu().numpy(),
                 local_slots=local, delta=delta.cpu().numpy())
        return a, c

    @torch.no_grad()
    def forward(self, rows, return_inputs=False):
        a, c = self.inputs(rows)
        ids = np.arange(len(rows))
        out = {}
        for off in (False, True):
            branch = dict(c, delta=np.zeros_like(c['delta'])) if off else c
            base = self.net(a, branch, ids)[0]
            b = {k: torch.stack([self.seen[k, p] for p in range(2)], 1) for k in ('x', 'mask', 'response')}
            b.update(features=self.seen['features'], available=task.tensor(a['available'], ids),
                     valid=grid_occupancy(task.tensor(a['valid'], ids).flatten(0, 1), (20, 20)).reshape(len(rows), 2, 1, 20, 20))
            if not off: out['upstream'] = base.cpu().numpy()
            for (arm, seed), head in self.heads.items():
                if (arm == 'no_m3_refit') != off: continue
                out[f'{arm}_s{seed}'] = head(b).cpu().numpy()
        return (out, a, c) if return_inputs else out


def replay():
    net = RawCurrent()
    ids = np.array([0, 138, 478, 719, 1141, 1518])
    out, a, c = net.forward([net.rows[i] for i in ids], True)
    olda, oldc = prev.arrays()
    errors = {k: float(np.abs(v-olda[k][ids]).max()) for k, v in a.items() if k in olda}
    errors.update({k: float(np.abs(v-oldc[k][ids]).max()) for k, v in c.items()})
    expected = net.net(olda, oldc, ids)[0].detach().cpu().numpy()
    errors['upstream_score'] = float(abs(expected-out['upstream']).max())
    for split in ('validation', 'evaluation'):
        for key in out:
            if key == 'upstream': continue
            arm, seed = key.rsplit('_s', 1)
            folder = (ROOT.parent/'essential_experiments_20261109/no_m3_refit'/f'class_s{seed}'
                      if arm == 'no_m3_refit' else s.ROOT/key)
            d = np.load(folder/f'{split}.npz')
            match = [(j, np.flatnonzero(d['ids'] == i)) for j, i in enumerate(ids)]
            vals = [float(abs(out[key][j]-d['score'][ix[0]]).max()) for j, ix in match if len(ix)]
            if vals: errors[f'{key}_{split}'] = max(vals)
    task.write(ROOT/'raw_replay.json', errors)
    assert max(errors.values()) < 1e-4, errors
    print(json.dumps(errors), flush=True)


def transfer(manifest):
    rows = task.data.read_manifest(manifest)
    net = RawCurrent()
    dest = ROOT/'oss1_transfer'; dest.mkdir(exist_ok=True)
    outputs = {}; tick = time.perf_counter()
    for start in range(0, len(rows), 8):
        result = net.forward(rows[start:start+8])
        for key, value in result.items(): outputs.setdefault(key, []).append(value)
        if start % 80 == 0: print('TRANSFER', start+len(value), len(rows), round(time.perf_counter()-tick, 1), flush=True)
    scores = {k: np.concatenate(v) for k, v in outputs.items()}
    np.savez_compressed(dest/'predictions.npz', ids=np.array([r['id'] for r in rows]), **scores)
    labels = np.array([r['label'] for r in rows])
    task.write(dest/'metrics.json', {k: task.metrics(labels, z) for k, z in scores.items()})
    task.write(dest/'receipt.json', dict(n=len(rows), seconds=time.perf_counter()-tick, training=False,
                                       references=32, reference_domain='OSS2 source', manifest=str(manifest)))
    print('DONE', len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    transfer(args.manifest) if args.manifest else replay()
