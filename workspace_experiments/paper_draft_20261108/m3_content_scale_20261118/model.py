"""Experimental content-driven M3 evidence granularity, chip to classification."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


native = load('content_scale_native', ROOT.parent/'m3_native_readout_20261118/model.py')
axis = load('content_scale_axis', ROOT.parent/'m3_axis_coupling_20261118/axis_operator.py')
# Older research imports use generic module names. Resolve these two by exact path.
selection = load('content_scale_selection', ROOT/'content_selection.py')
geo = load('content_scale_geometry', ROOT/'native_masks.py')
torch, nn, np, START = native.torch, native.nn, native.np, native.START


class MixedMatch(axis.AxisCoupledMatch):
    regions = selection.REGIONS

    def __init__(self, classes, geometry='axis'):
        super().__init__(classes=classes, geometry=geometry, coupling='paired')
        # Do not additionally change other-class context in this scale-selection test.
        self.mode = 'uniform'
        fine = axis.slot_positions()
        coarse = torch.stack([fine[list(ids)].mean(0) for ids in selection.GROUPS])
        self.positions = selection.pack(fine, coarse, 0)


class Inputs(native.old.Inputs):
    def __init__(self):
        super().__init__()
        self.mixed_reads = geo.MixedReads(native, selection.GROUPS)

    def data(self, ids, split=None, code=0):
        data = super().data(ids, split, code)
        data.update(self.mixed_reads.batch([self.rows[i] for i in ids], data['images'].device, code))
        return data


def prepare(rows, code=0):
    base = native.old.old.joint.full.cost.prepare(rows)
    if code:
        for key in ('images', 'valid'):
            value = torch.rot90(base[key], code % 4, (-2, -1))
            base[key] = value.flip(-1) if code >= 4 else value
    base.update(geo.MixedReads(native, selection.GROUPS).batch(rows, base['images'].device, code))
    return base


class Model(native.Model):
    def __init__(self, selection_mode='auto', geometry='axis'):
        super().__init__('fine_live')
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(20261118)
            self.selector = selection.ContentScaleSelector().to('cuda')
            matcher = MixedMatch(self.net.matcher.classes, geometry).to('cuda')
        loaded = matcher.load_state_dict(self.net.matcher.state_dict(), strict=False)
        assert loaded.missing_keys == ['position_strength_raw'] and not loaded.unexpected_keys
        self.net.matcher = matcher
        self.selection_mode = selection_mode
        self.scale_override = None
        self.scale_cost = None
        self.scale_trace = None
        self.eval()

    def extract(self, data, descriptors):
        # Keep readout available for the inherited finish(), including M3-off controls.
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
        values = []
        for p in range(2):
            occ = native.old.old.t.ft.staged.old.grid_occupancy(data['valid'][:, p], (20, 20))
            occ = occ*data['available'][:, p, None, None, None]
            result, _ = self.net.m1[p](x.reshape(n, 2, 128, 20, 20)[:, p], occ, details=True)
            values.append(result[0])
        f2 = block.relu(block.bn3(block.conv3(torch.stack(values, 1).flatten(0, 1)))+identity)
        # Each layer is pooled at its own grid; no resizing either feature map.
        first = geo.pool_native(f1.flatten(0, 1), data['read40'].flatten(0, 1))
        second = geo.pool_native(f2, data['read20'].flatten(0, 1))
        fine = torch.cat((first, second), -1).reshape(n, 2, 20, 768)
        mass = data['read40'].sum((-1, -2))*data['available'][..., None]
        chosen = self.selector(fine, mass, data['fine_write']*data['available'][..., None, None, None],
                               data['eligible'], self.selection_mode, self.scale_override)
        self.scale_cost = chosen.pop('scale_cost')
        self.scale_trace = {k: v.detach() for k, v in chosen.pop('scale_trace').items()}
        a = {k: data[k] for k in ('valid', 'available', 'slots')}
        a.update(chosen)
        c = dict(early_x=x.reshape(n, 2, 128, 20, 20),
                 early_identity=identity.reshape(n, 2, 512, 20, 20), local_slots=data['local_slots'])
        return a, c

    def reference(self, data):
        result = super().reference(data)
        result['scale_cost'] = self.scale_cost
        result['scale_trace'] = self.scale_trace
        return result

    def objective(self, data, rows, reference_data, reference_rows, weights=None, budget_weight=0.):
        """Live reference re-encoding; caller owns backward/optimizer/step schedule."""
        reference = self.reference(reference_data)
        z, auxiliary = self(data, rows, reference, reference_rows)
        labels = torch.tensor([r['label'] for r in rows], device=z.device)
        if weights is None:
            weights = torch.ones(len(rows), device=z.device)
        ce = (torch.nn.functional.cross_entropy(z, labels, reduction='none')*weights).mean()
        aux = torch.nn.functional.cross_entropy(auxiliary.flatten(0, 1),
            labels[:, None].expand(-1, 2).flatten(), reduction='none').reshape(len(rows), 2)
        available = data['available']
        aux = ((aux*available).sum(1)/available.sum(1).clamp_min(1)*weights).mean()
        budget = .5*(reference['scale_cost']+self.scale_cost)
        loss = ce+.02*aux+budget_weight*budget
        return loss, z, dict(ce=ce.detach(), auxiliary=aux.detach(), budget=budget.detach())

    def optimizer_groups(self, backbone_lr=1e-6, module_lr=1e-5, selector_lr=1e-4):
        groups = {'backbone': [], 'modules': [], 'selector': []}
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                key = ('selector' if name.startswith('selector.') else
                       'backbone' if name.startswith('net.base.backbone.') else 'modules')
                groups[key].append(parameter)
        rates = dict(backbone=backbone_lr, modules=module_lr, selector=selector_lr)
        return [dict(name=k, params=v, lr=rates[k]) for k, v in groups.items()]
