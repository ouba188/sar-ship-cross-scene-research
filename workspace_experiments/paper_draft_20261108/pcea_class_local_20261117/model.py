"""Class-response organization of local detail; unchanged PCEA writer."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('class_local_previous', ROOT.parent/'pcea_ports24_startpoint_20261117/model.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
torch, nn, np = old.torch, old.nn, old.np
F = old.joint.F
Inputs, full = old.Inputs, old.full
ARMS = ('full', 'generic', 'class_detail')


class CCREL(old.joint.CCREL):
    def forward(self, shallow, fine, masks, responses, valid, fine_valid):
        result = super().forward(shallow, fine, masks, responses, valid, fine_valid)
        gains = []
        for p in range(2):
            mask, response, occupancy = (x.flatten(2) for x in (masks[:, p], responses[:, p], valid[:, p]))
            uniform = mask/mask.sum(-1, keepdim=True).clamp_min(1e-8)
            k = response.shape[1]
            relative = (k*response-response.sum(1, keepdim=True))/(k-1)
            vw = occupancy/occupancy.sum(-1, keepdim=True).clamp_min(1e-8)
            centered = relative-(relative*vw).sum(-1, keepdim=True)
            score = 2*torch.tanh(centered/((centered.square()*vw).sum(-1, keepdim=True)+.01).sqrt())
            logits = score[:, None]+mask.clamp_min(1e-12).log()[:, :, None]
            weight = logits.masked_fill(mask[:, :, None] <= 0, -1e4).softmax(-1)
            weight = weight*(mask[:, :, None] > 0)
            weight = weight/weight.sum(-1, keepdim=True).clamp_min(1e-8)
            # Relative attention density: uniform attention gives exactly one.
            density = weight/uniform[:, :, None].clamp_min(1e-12)
            gain = 1+.5*torch.tanh(density.clamp_min(1e-12).log())
            gain = gain/(gain*uniform[:, :, None]).sum(-1, keepdim=True).clamp_min(1e-8)
            gain = torch.where(mask[:, :, None] > 0, gain, torch.ones_like(gain))
            gains.append(torch.cat((torch.ones_like(gain[:, :, :1]), gain), 2))
        self.detail_gain = torch.stack(gains, 1)
        self.detail_mass = masks.flatten(-2)/masks.flatten(-2).sum(-1, keepdim=True).clamp_min(1e-8)
        return result


class Exchange(old.joint.local.LocalExchange):
    def __init__(self, reader):
        super().__init__(reader, 'cross_mean')
        self.use_class_detail = False
        self.uniform_detail = False
        self.permute_detail = False
        self.detail_gain_trace = None
        for p in range(2):
            self.project[p].register_forward_pre_hook(self.weight_detail(p))

    def standardized(self, evidence):
        result = super().standardized(evidence)
        self.current_summary = result[0]
        return result

    def weight_detail(self, p):
        def hook(module, args):
            if not self.use_class_detail or self.uniform_detail:
                return args
            local = args[0]
            base = self.current_summary[:, p, :, :, None, :]
            gain = self.reader.detail_gain[:, p]
            if self.permute_detail:
                gain = torch.cat((gain[:, :, :1], gain[:, :, 1:].roll(1, 2)), 2)
            self.detail_gain_trace = self.reader.detail_gain.detach()
            return (base+gain[..., None]*(local-base),)
        return hook


class Model(old.Model):
    def __init__(self, arm):
        assert arm in ARMS
        # All continuation seeds share the same selected formal checkpoint.
        super().__init__(17, 'full' if arm == 'full' else 'cross_mean')
        self.experiment_arm = arm
        if arm == 'full':
            return
        reader = CCREL(self.net.guide.value).to('cuda')
        reader.load_state_dict(self.ccrel.state_dict(), strict=True)
        exchange = Exchange(self.net.guide.value).to('cuda')
        exchange.slot = nn.Parameter(torch.zeros(9, 16, device='cuda'))
        exchange.mean = torch.zeros(2, 3, 9, 32, device='cuda')
        exchange.scale = torch.ones(2, 3, 9, 32, device='cuda')
        for name in ('output', 'scale_output'):
            setattr(exchange, name, nn.ModuleList(nn.Linear(144, 256, bias=False, device='cuda') for _ in range(2)))
        exchange.load_state_dict(self.exchange.state_dict(), strict=True)
        self.ccrel, self.exchange = reader, exchange
        object.__setattr__(self.exchange, 'reader', self.ccrel)
        self.exchange.use_class_detail = arm == 'class_detail'
        self.eval()
