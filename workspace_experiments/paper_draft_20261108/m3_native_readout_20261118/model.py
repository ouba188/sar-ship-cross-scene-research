"""Native layer1 evidence and optional live M1 readout; unchanged M3 writer."""
from collections import OrderedDict
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = load('native_readout_previous', ROOT.parent/'pcea_class_local_20261117/model.py')
torch, np, nn = old.torch, old.np, old.nn
full = old.full
raw = old.old.t.raw
geometry = load('native_readout_geometry', ROOT.parents[1]/
                'm3_frequency_increment_20261028/resolution_matching_20261031/geometry.py')
START = ROOT.parent/'pcea_routed_correction_20261118/fits/routed_s17/best.pt'
ARMS = ('legacy', 'fine_anchor', 'fine_live')


class NativeReads:
    def __init__(self, rows):
        self.rows, self.cache = rows, OrderedDict()

    def one(self, index, code):
        key = (int(index), int(code))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        result = np.zeros((2, 20, 40, 40), np.float32)
        for p, view in enumerate(old.old.t.e.task.data.prepare(self.rows[index])):
            if view is None or not view['valid_target']:
                continue
            native = raw.tokens.slots(view['geometry'], view['footprint'])
            if code:
                native = old.old.t.ft.transform(native, code)
            result[p] = raw.tokens.cell_mass(native, 4)/geometry._cell_area(40)
        self.cache[key] = result
        if len(self.cache) > 256:
            self.cache.popitem(last=False)
        return result

    def batch(self, ids, code=0):
        return torch.as_tensor(np.stack([self.one(int(i), code) for i in ids]), device='cuda')


class Inputs(old.Inputs):
    def __init__(self):
        super().__init__()
        self.native_reads = NativeReads(self.rows)

    def data(self, ids, split=None, code=0):
        result = super().data(ids, split, code)
        result['read40'] = self.native_reads.batch(ids, code)
        return result


class Model(old.Model):
    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__('generic')
        self.load_state_dict(torch.load(START, map_location='cuda', weights_only=True), strict=True)
        self.start_checkpoint = START
        self.readout_arm = arm
        self.experiment_arm = arm
        self.eval()

    def extract(self, data, descriptors):
        if self.readout_arm == 'legacy' or not descriptors:
            return super().extract(data, descriptors)
        n = len(data['images'])
        bb = self.net.base.backbone.model
        f1 = self.net.base.prefix(data['images'], stop=1)
        z = f1.flatten(0, 1)
        for layer in list(bb.layer2.children())[:3]:
            z = layer(z)
        identity = z
        block = bb.layer2[3]
        x = block.relu(block.bn1(block.conv1(z)))
        x = block.relu(block.bn2(block.conv2(x)))
        if self.readout_arm == 'fine_live':
            values = []
            for p in range(2):
                occ = old.old.t.ft.staged.old.grid_occupancy(data['valid'][:, p], (20, 20))
                occ = occ*data['available'][:, p, None, None, None]
                result, _ = self.net.m1[p](x.reshape(n, 2, 128, 20, 20)[:, p], occ, details=True)
                values.append(result[0])
            live = torch.stack(values, 1).flatten(0, 1)
            f2 = block.relu(block.bn3(block.conv3(live))+identity)
        else:
            f2 = block.relu(self.anchor_bn(self.anchor_conv(x))+identity)
        # Only layer2 is interpolated. Layer1 retains its native stride-four grid.
        desc = raw.pool_multiscale(f1.flatten(0, 1), f2,
                                  data['read40'].flatten(0, 1), 'fine_fixed')
        a = {k: data[k] for k in ('valid', 'available', 'slots', 'mass', 'write')}
        a['desc'] = desc.reshape(n, 2, 20, 768)
        c = dict(early_x=x.reshape(n, 2, 128, 20, 20),
                 early_identity=identity.reshape(n, 2, 512, 20, 20),
                 local_slots=data['local_slots'])
        return a, c
