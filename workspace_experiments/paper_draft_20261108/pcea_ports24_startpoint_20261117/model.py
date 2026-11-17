"""Eight-class continuation of the trained, polarization-recovered 24-port Full."""
from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sp = load('ports24_sp_previous', ROOT.parent/'pcea_startpoint_20261117/model.py')
recovered = load('ports24_sp_recovered', ROOT.parent/'ports24_recovered_training_20261113/train.py')
joint = sp.old
t = recovered.t
torch, np, nn = joint.torch, joint.np, joint.nn
StartingPoint = sp.StartingPoint
full = SimpleNamespace(ROOT=recovered.ROOT, predict=t.ft.predict,
                       loss_for=t.ft.loss_for, metrics=t.metrics)


class Inputs(recovered.Inputs):
    def __init__(self):
        t.Inputs.__init__(self)
        self.original_rows = list(self.rows)
        self.restored = {}
        legal = np.unique(np.concatenate([self.groups[k] for k in ('small', 'validation', 'references')]))
        for i in legal:
            row = self.rows[i]
            if row.get('vv') is not None and row.get('vh') is not None:
                continue
            fixed, audit = recovered.recovery.recover(row)
            if audit['status'] == 'recovered':
                self.rows[i] = fixed
                self.restored[int(i)] = audit
        self.reference_rows = [self.rows[i] for i in self.groups['references']]
        assert len(self.restored) == 440
        self.prepare()


class Model(joint.Model):
    def __init__(self, seed, arm):
        assert arm in ('full', 'cross_mean')
        net = t.e.Network('full', seed, t.ft.r50.WEIGHTS, 8)
        joint.full.cost.ChipPath.__init__(self, net)
        self.arm, self.context, self.transport_graph = arm, None, None
        net.phase, net.fixed_delta = 3, None
        self.start_checkpoint = recovered.ROOT/f'selected_s{seed}/best.pt'
        self.load_state_dict(torch.load(self.start_checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.requires_grad_(False)
        net.base.backbone.model.requires_grad_(True)
        net.base.backbone.model.fc.requires_grad_(False)
        for module in (net.m1, net.base.internal_m2, net.base.deep_output,
                       net.matcher, net.guide.value, net.guide.output, net.guide.head):
            module.requires_grad_(True)
        if arm != 'full':
            torch.manual_seed(seed+20261116)
            self.ccrel = joint.CCREL(net.guide.value)
            self.exchange = joint.local.LocalExchange(net.guide.value, arm)
            # One regional mean and eight class-response slots, not the OSS2 five.
            self.exchange.slot = nn.Parameter(torch.zeros(9, 16))
            self.exchange.mean = torch.zeros(2, 3, 9, 32)
            self.exchange.scale = torch.ones(2, 3, 9, 32)
            for name in ('output', 'scale_output'):
                layers = nn.ModuleList(nn.Linear(144, 256, bias=False) for _ in range(2))
                for layer in layers:
                    nn.init.zeros_(layer.weight)
                setattr(self.exchange, name, layers)
            object.__setattr__(self.exchange, 'reader', self.ccrel)
            for p in range(2):
                def response(module, args, result, p=p):
                    if self.context is not None:
                        self.context['response'][p] = result[1]['response']
                def shallow(module, args, p=p):
                    occ = t.ft.staged.old.grid_occupancy(args[2], (20, 20))
                    self.context['shallow'][p] = args[0]
                    self.context['reader_mask'][p] = torch.minimum(args[1], occ)
                    self.context['reader_valid'][p] = occ
                def writer(module, args, p=p):
                    occ = t.ft.staged.old.grid_occupancy(args[2], (10, 10))
                    self.context['writer_mask'][p] = torch.minimum(args[1], occ)
                net.m1[p].register_forward_hook(response)
                net.base.m2[p].register_forward_pre_hook(shallow)
                net.base.internal_m2[p].register_forward_pre_hook(writer)
            net.base.backbone.model.layer3[5].conv2.register_forward_pre_hook(self._exchange)
        self.to(t.ft.DEVICE).eval()
