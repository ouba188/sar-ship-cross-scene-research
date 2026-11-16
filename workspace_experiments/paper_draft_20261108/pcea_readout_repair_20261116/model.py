"""Real shallow readout and source-anchored polarization interaction pilot."""
from pathlib import Path
import copy
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pcea_previous', ROOT.parent/'pcea_training_20261116/model.py')
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
full, torch, nn, F, np = previous.full, previous.torch, previous.nn, previous.F, previous.np
ARMS = ('coarse_raw', 'fine_raw', 'fine_std', 'fine_center_gate',
        'fine_center_product', 'fine_independent', 'bias_control')


class Exchange(nn.Module):
    def __init__(self, arm, reader):
        super().__init__()
        self.arm = arm
        self.size = 10 if arm == 'coarse_raw' else 20
        self.value = copy.deepcopy(reader).requires_grad_(False)
        self.project = nn.ModuleList(nn.Linear(32, 16, bias=False) for _ in range(2))
        self.region = nn.Parameter(torch.zeros(3, 16))
        self.slot = nn.Parameter(torch.zeros(5, 16))
        self.gate = nn.Sequential(nn.Linear(64, 16), nn.SiLU(), nn.Linear(16, 3))
        self.output = nn.ModuleList(nn.Linear(80, 256, bias=False) for _ in range(2))
        for layer in self.output:
            nn.init.zeros_(layer.weight)
        if arm == 'fine_independent':
            self.region.requires_grad_(False)
            self.slot.requires_grad_(False)
            self.gate.requires_grad_(False)
        self.register_buffer('mean', torch.zeros(2, 3, 5, 32))
        self.register_buffer('scale', torch.ones(2, 3, 5, 32))
        self.disabled = False
        self.no_product = False
        self.trace = None

    def read(self, shallow, masks, responses, valid):
        tokens, effective = [], []
        size = (self.size, self.size)
        for p in range(2):
            x = shallow[:, p]
            value = self.value[p](F.layer_norm(x.permute(0, 2, 3, 1), (512,)).permute(0, 3, 1, 2))
            mask = masks[:, p]
            response = responses[:, p]
            occupancy = valid[:, p]
            if self.size == 10:
                value = F.avg_pool2d(value, 2)
                mask = F.avg_pool2d(mask, 2)
                response = F.avg_pool2d(response, 2)
                occupancy = F.avg_pool2d(occupancy, 2)
            assert value.shape[-2:] == size
            value, mask, response, occupancy = (t.flatten(2) for t in (value, mask, response, occupancy))
            uniform = mask / mask.sum(-1, keepdim=True).clamp_min(1e-8)
            relative = (4*response-response.sum(1, keepdim=True))/3
            vw = occupancy/occupancy.sum(-1, keepdim=True).clamp_min(1e-8)
            centered = relative-(relative*vw).sum(-1, keepdim=True)
            score = 2*torch.tanh(centered/((centered.square()*vw).sum(-1, keepdim=True)+.01).sqrt())
            logits = score[:, None]+mask.clamp_min(1e-12).log()[:, :, None]
            weight = logits.masked_fill(mask[:, :, None] <= 0, -1e4).softmax(-1)
            weight = weight*(mask[:, :, None] > 0)
            weight = weight/weight.sum(-1, keepdim=True).clamp_min(1e-8)
            mean = torch.einsum('bru,bdu->brd', uniform, value)
            classes = torch.einsum('brku,bdu->brkd', weight, value)
            tokens.append(torch.cat((mean[:, :, None], classes), 2))
            effective.append(mask.sum(-1).square()/mask.square().sum(-1).clamp_min(1e-8))
        return torch.stack(tokens, 1), torch.stack(effective, 1)

    def fuse(self, u, v):
        b, r, k, d = u.shape
        er = self.region[None, :, None].expand(b, r, k, d)
        ek = self.slot[None, None].expand(b, r, k, d)
        gate = self.gate(torch.cat((u, v, er, ek), -1))
        alpha, beta = gate[..., :2].softmax(-1), gate[..., 2:].sigmoid()
        h = alpha[..., :1]*u+alpha[..., 1:]*v
        if self.arm != 'fine_center_gate' and not self.no_product:
            h = h+beta*u*v
        return h, alpha, beta

    def interaction(self, u, v):
        h, alpha, beta = self.fuse(u, v)
        if self.arm.startswith('fine_center'):
            z = torch.zeros_like(u)
            h = h-self.fuse(u, z)[0]-self.fuse(z, v)[0]+self.fuse(z, z)[0]
        return h, alpha, beta

    def forward(self, x, writer_masks, shallow, reader_masks, responses, available, valid):
        tokens, effective = self.read(shallow, reader_masks, responses, valid)
        standardized = self.arm not in ('coarse_raw', 'fine_raw')
        t = (tokens-self.mean)/self.scale if standardized else tokens
        u, v = (self.project[p](t[:, p]).tanh() for p in range(2))
        if self.arm == 'fine_independent':
            contents, alpha, beta = [u, v], None, None
        else:
            h, alpha, beta = self.interaction(u, v)
            contents = [h, h]
        active = available.bool().all(1)[:, None]
        active = active & (reader_masks.sum((-2, -1)) > 0).all(1)
        active = active & (writer_masks.sum((-2, -1)) > 0).all(1)
        writer = writer_masks/writer_masks.sum(2, keepdim=True).clamp_min(1)
        updates = []
        for p in range(2):
            mask = writer_masks[:, p].flatten(2)
            weights = mask/mask.sum(-1, keepdim=True).clamp_min(1e-8)
            region = torch.einsum('bru,bcu->brc', weights, x[:, p].flatten(2))
            scale = region.square().mean(-1, keepdim=True).sqrt().detach()
            delta = .1*scale*self.output[p](contents[p].flatten(2)).tanh()*active[..., None]
            updates.append(torch.einsum('brc,brhw->bchw', delta, writer[:, p]))
        delta = torch.stack(updates, 1)
        self.trace = dict(tokens=tokens.detach(), effective_cells=effective.detach(), active=active.detach(),
                          delta=delta.detach(), input_rms=x.detach().square().mean((2, 3, 4)).sqrt(),
                          update_rms=delta.detach().square().mean((2, 3, 4)).sqrt(),
                          alpha=None if alpha is None else alpha.detach(),
                          beta=None if beta is None else beta.detach())
        return x if self.disabled else x+delta


class Model(previous.Model):
    def __init__(self, seed, arm):
        assert arm in ARMS
        super().__init__(seed, 'pcea')
        self.arm = arm
        torch.manual_seed(seed+20261116)
        self.exchange = Exchange(arm, self.net.guide.value).to('cuda')
        for name, parameter in self.named_parameters():
            if not name.startswith('exchange.'):
                parameter.requires_grad_(False)
        for p in range(2):
            def capture_shallow(module, args, p=p):
                occupancy = full.staged.old.grid_occupancy(args[2], (20, 20))
                self.context.setdefault('shallow', [None, None])[p] = args[0]
                self.context.setdefault('reader_mask', [None, None])[p] = torch.minimum(args[1], occupancy)
                self.context.setdefault('reader_valid', [None, None])[p] = occupancy
            self.net.base.m2[p].register_forward_pre_hook(capture_shallow)
        if arm == 'bias_control':
            self.exchange.requires_grad_(False)
            self.class_bias = nn.Parameter(torch.zeros(4, device='cuda'))

    def forward(self, data, query_rows, reference, reference_rows, m3_on=True):
        z, aux = super().forward(data, query_rows, reference, reference_rows, m3_on=m3_on)
        if self.arm == 'bias_control' and not self.exchange.disabled:
            active = self.exchange.trace['active'].any(1)[:, None]
            z = z+active*(self.class_bias-self.class_bias.mean())
        return z, aux

    def _exchange(self, module, args):
        c = self.context
        x = args[0].reshape(-1, 2, 256, 10, 10)
        out = self.exchange(x, torch.stack(c['mask'], 1), torch.stack(c['shallow'], 1),
                            torch.stack(c['reader_mask'], 1), torch.stack(c['response'], 1),
                            c['available'], torch.stack(c['reader_valid'], 1))
        return (out.flatten(0, 1),)


@torch.no_grad()
def calibrate(net, inputs, seed):
    path = ROOT/f'source_statistics_s{seed}_g{net.exchange.size}.pt'
    if path.exists():
        stats = torch.load(path, map_location='cuda', weights_only=True)
    else:
        net.eval()
        ref = net.reference(inputs.data(inputs.groups['references'], 'small'))
        count = torch.zeros(3, device='cuda', dtype=torch.float64)
        total = torch.zeros_like(net.exchange.mean, dtype=torch.float64)
        square = torch.zeros_like(total)
        ids = inputs.groups['small']
        for start in range(0, len(ids), 12):
            part = ids[start:start+12]
            net(inputs.data(part, 'small'), [inputs.rows[i] for i in part], ref, inputs.reference_rows)
            trace = net.exchange.trace
            weight = torch.as_tensor(inputs.weights[part], device='cuda')[:, None]*trace['active']
            token = trace['tokens'].double()
            count += weight.sum(0)
            total += (token*weight[:, None, :, None, None]).sum(0)
            square += (token.square()*weight[:, None, :, None, None]).sum(0)
        assert bool((count > 0).all())
        mean = total/count[None, :, None, None]
        moment2 = square/count[None, :, None, None]
        std = (moment2-mean.square()).clamp_min(0).sqrt()
        floor = (.1*moment2.mean(-1, keepdim=True).sqrt()).clamp_min(.01)
        stats = dict(mean=mean.float(), scale=torch.maximum(std, floor).float(),
                     weighted_count=count.float())
        torch.save({k: v.cpu() for k, v in stats.items()}, path)
    net.exchange.mean.copy_(stats['mean'])
    net.exchange.scale.copy_(stats['scale'])
    return stats
