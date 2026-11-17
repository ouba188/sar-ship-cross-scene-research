"""Common pretrained initialization; live full-backbone/region/exchange learning."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'joint_full_local', ROOT.parent/'pcea_local_cross_20261117/model.py')
local = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = local
spec.loader.exec_module(local)
torch, nn, F, np, full = local.torch, local.nn, local.F, local.np, local.full
CCREL = local.old.old.CCREL
ARMS = ('full', 'cross_mean', 'cross_local')


class Model(full.LiveFull):
    def __init__(self, seed, arm):
        assert arm in ARMS
        net = full.staged.Network('full', seed, full.r50.WEIGHTS)
        full.cost.ChipPath.__init__(self, net)
        self.arm = arm
        self.net.phase, self.net.fixed_delta = 3, None
        self.context = None
        self.transport_graph = None
        self.requires_grad_(False)
        self.net.base.backbone.model.requires_grad_(True)
        self.net.base.backbone.model.fc.requires_grad_(False)
        for module in (net.m1, net.base.internal_m2, net.base.deep_output,
                       net.matcher, net.guide.value, net.guide.output, net.guide.head):
            module.requires_grad_(True)
        if arm != 'full':
            torch.manual_seed(seed+20261116)
            self.exchange = local.LocalExchange(net.guide.value, arm)
            self.ccrel = CCREL(net.guide.value)
            object.__setattr__(self.exchange, 'reader', self.ccrel)
            for p in range(2):
                def capture_response(module, args, result, p=p):
                    if self.context is not None:
                        self.context['response'][p] = result[1]['response']
                def capture_shallow(module, args, p=p):
                    c = self.context
                    occ = full.staged.old.grid_occupancy(args[2], (20, 20))
                    c['shallow'][p] = args[0]
                    c['reader_mask'][p] = torch.minimum(args[1], occ)
                    c['reader_valid'][p] = occ
                def capture_writer(module, args, p=p):
                    occ = full.staged.old.grid_occupancy(args[2], (10, 10))
                    self.context['writer_mask'][p] = torch.minimum(args[1], occ)
                net.m1[p].register_forward_hook(capture_response)
                net.base.m2[p].register_forward_pre_hook(capture_shallow)
                net.base.internal_m2[p].register_forward_pre_hook(capture_writer)
            net.base.backbone.model.layer3[5].conv2.register_forward_pre_hook(self._exchange)
        self.to('cuda').eval()

    def reference(self, data):
        # The same legal source images are re-encoded after every parameter update.
        a, c = self.extract(data, True)
        if self.arm != 'full':
            n = len(data['images'])
            values, responses, masks, occupancy = [], [], [], []
            for p in range(2):
                occ = full.staged.old.grid_occupancy(data['valid'][:, p], (20, 20))
                occ = occ*data['available'][:, p, None, None, None]
                result, trace = self.net.m1[p](c['early_x'][:, p], occ, details=True)
                values.append(result[0])
                responses.append(trace['response'])
                reg, _ = self.net.base.region_integrals(
                    c['local_slots'][:, p], occ.new_zeros(n, 20, 3, 20, 20))
                masks.append(torch.minimum(reg, occ))
                occupancy.append(occ)
            block = self.net.base.backbone.model.layer2[3]
            x = torch.stack(values, 1).flatten(0, 1)
            shallow = block.relu(block.bn3(block.conv3(x))+c['early_identity'].flatten(0, 1))
            mask = torch.stack(masks, 1)
            self.ccrel(shallow.reshape(n, 2, 512, 20, 20), None, mask,
                       torch.stack(responses, 1), torch.stack(occupancy, 1), None)
            self.exchange.source_raw = self.ccrel.raw
            self.exchange.source_weight = (data['available'].bool().all(1)[:, None] &
                                            (mask.sum((-2, -1)) > 0).all(1)).float()
            assert bool((self.exchange.source_weight.sum(0) > 0).all())
        return {k: a[k] for k in ('desc', 'mass', 'available')}

    def _exchange(self, module, args):
        c = self.context
        shallow, masks, valid = (torch.stack(c[k], 1) for k in
                                 ('shallow', 'reader_mask', 'reader_valid'))
        evidence, effective = self.ccrel(shallow, None, masks,
                                          torch.stack(c['response'], 1), valid, None)
        values = []
        for p in range(2):
            x = F.layer_norm(shallow[:, p].permute(0, 2, 3, 1), (512,)).permute(0, 3, 1, 2)
            values.append(self.ccrel.value[p](x))
        self.exchange.local_value = torch.stack(values, 1)
        x = args[0].reshape(-1, 2, 256, 10, 10)
        out = self.exchange(x, torch.stack(c['writer_mask'], 1), evidence,
                            effective, masks, c['available'])
        return (out.flatten(0, 1),)

    def finish(self, a, c, query_rows, reference, reference_rows, m3_on=True):
        self.context = dict(available=a['available'], **{k: [None, None] for k in
            ('response', 'shallow', 'reader_mask', 'reader_valid', 'writer_mask')})
        try:
            return super().finish(a, c, query_rows, reference, reference_rows, m3_on)
        finally:
            self.context = None


def calibrate(net, inputs, seed):
    return None
