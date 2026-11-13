"""Within-stage position controls with unchanged operators and readout endpoints."""
import torch
from torch import nn

POSITIONS = {
    'current': (3, 5, 5),
    'm1_early': (1, 5, 5),
    'm1_middle': (2, 5, 5),
    'm2_early': (3, 1, 5),
    'm2_middle': (3, 3, 5),
    'm3_early': (3, 5, 1),
    'm3_middle': (3, 5, 3),
}


def make_model(ft, seed, variant, classes=4):
    cost, raw, old = ft.cost, ft.cost.raw, ft.cost.e.old
    m1_at, m2_at, m3_at = POSITIONS[variant]

    class PlacementPath(cost.ChipPath):
        def __init__(self):
            net = cost.e.Network('full', seed, ft.r50.WEIGHTS, classes)
            super().__init__(net)
            net.phase, net.fixed_delta = 3, None
            self.requires_grad_(False)
            bb = net.base.backbone.model
            bb.requires_grad_(True)
            bb.fc.requires_grad_(False)
            for module in (net.m1, net.base.internal_m2, net.base.deep_output,
                           net.matcher, net.guide.value, net.guide.output, net.guide.head):
                module.requires_grad_(True)
            self.to(ft.DEVICE).eval()
            self.trace = []

        def train(self, mode=True):
            super().train(mode)
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            return self

        def extract(self, data, descriptors):
            n = len(data['images'])
            bb = self.net.base.backbone.model
            f1 = self.net.base.prefix(data['images'], stop=1)
            z = f1.flatten(0, 1)
            selected = None
            # Keep the unconditioned descriptor path identical at every position.
            for i, block in enumerate(bb.layer2):
                identity = z if block.downsample is None else block.downsample(z)
                x = block.relu(block.bn1(block.conv1(z)))
                x = block.relu(block.bn2(block.conv2(x)))
                if i == m1_at:
                    selected = (x, identity)
                if i == 3:
                    f2 = block.relu(self.anchor_bn(self.anchor_conv(x)) + identity)
                else:
                    z = block.relu(block.bn3(block.conv3(x)) + identity)
            a = {k: data[k] for k in ('valid', 'available', 'slots', 'mass', 'write')}
            if descriptors:
                a['desc'] = raw.pool_multiscale(
                    f1.flatten(0, 1), f2, data['read'].flatten(0, 1),
                    'coarse_fixed').reshape(n, 2, 20, 768)
            c = dict(early_x=selected[0].reshape(n, 2, 128, 20, 20),
                     early_identity=selected[1], local_slots=data['local_slots'])
            return a, c

        def finish(self, a, c, query_rows, reference, reference_rows, m3_on=True):
            net, n = self.net, len(query_rows)
            bb = net.base.backbone.model
            valid, av, slots = a['valid'], a['available'], a['slots']
            local = c['local_slots']
            x = c['early_x']
            values, aux, responses = [], [], []
            self.trace = []
            if net.phase >= 2:
                for p in range(2):
                    occ = old.grid_occupancy(valid[:, p], (20, 20))*av[:, p, None, None, None]
                    result, trace = net.m1[p](x[:, p], occ, details=True)
                    values.append(result[0])
                    aux.append(result[1])
                    responses.append(trace['response'])
                self.trace.append(('m1', m1_at, 'after_conv2_relu'))
                x = torch.stack(values, 1)
            else:
                responses = [x.new_zeros(n, classes, 20, 20) for _ in range(2)]
            x = x.flatten(0, 1)
            block = bb.layer2[m1_at]
            z = block.relu(block.bn3(block.conv3(x)) + c['early_identity'])
            for block in list(bb.layer2.children())[m1_at+1:]:
                z = block(z)
            live2 = z.reshape(n, 2, 512, 20, 20)
            low, masks = [], []
            for p in range(2):
                reg, en = net.base.region_integrals(local[:, p], local.new_zeros(n,20,3,20,20))
                low.append(net.base.m2[p](live2[:, p], reg, valid[:, p], en)['pooled'])
                masks.append(torch.minimum(reg, old.grid_occupancy(valid[:, p], (20,20))))
            delta = None
            if m3_on and net.phase >= 1:
                rows = query_rows+reference_rows
                ids, rid = ft.np.arange(n), ft.np.arange(n,n+len(reference_rows))
                pair = (ft.staged.task.legal(rows,ids,rid)[...,None]
                        & av[:,None].bool() & reference['available'][None].bool())
                labels = torch.tensor([r['label'] for r in reference_rows],device=ft.DEVICE)
                delta = net.matcher(a['desc'],reference['desc'],a['mass'],reference['mass'],
                                    a['write'],pair,labels,arm='aggregate')[0]
            for i, block in enumerate(bb.layer3):
                identity = z if block.downsample is None else block.downsample(z)
                x = block.relu(block.bn1(block.conv1(z)))
                if i == m2_at:
                    x = x.reshape(n,2,256,10,10)
                    values = []
                    for p in range(2):
                        reg,en = net.base.region_integrals(slots[:,p],slots.new_zeros(n,20,3,10,10))
                        values.append(net.base.internal_m2[p](x[:,p],reg,valid[:,p],en)[0])
                    x = torch.stack(values,1).flatten(0,1)
                    self.trace.append(('m2',i,'after_conv1_relu'))
                x = block.relu(block.bn2(block.conv2(x)))
                z = block.relu(block.bn3(block.conv3(x))+identity)
                if i == m3_at and delta is not None:
                    z = z+delta.flatten(0,1)
                    self.trace.append(('m3',i,'after_block_relu'))
            x = z.reshape(n,2,1024,10,10)
            deep = bb.layer4(z)
            glob = old.regional_mean(deep,valid.flatten(0,1),valid.flatten(0,1))[0][:,0].reshape(n,2,-1)
            pooled = []
            for p in range(2):
                reg,en = net.base.region_integrals(slots[:,p],slots.new_zeros(n,20,3,10,10))
                extra = net.base.deep_pool[p](x[:,p],reg,valid[:,p],en)['pooled']
                pooled.append((low[p]+net.base.deep_output[p](extra)).flatten(1))
            feat = torch.cat((glob,torch.stack(pooled,1)),-1)*av[...,None]
            b = dict(features=torch.cat((feat.flatten(1),av),1),x=live2,
                     mask=torch.stack(masks,1),response=torch.stack(responses,1),available=av,
                     valid=old.grid_occupancy(valid.flatten(0,1),(20,20)).reshape(n,2,1,20,20))
            net.readout_batch = b
            score = net.guide(b) if net.phase >= 3 else net.guide.head(b['features'])
            # Cached readout training explicitly opts in; live paths must not retain graphs.
            if not getattr(self, 'capture_readout', False):
                net.readout_batch = None
            return score, torch.stack(aux,1) if aux else None

    return PlacementPath()
