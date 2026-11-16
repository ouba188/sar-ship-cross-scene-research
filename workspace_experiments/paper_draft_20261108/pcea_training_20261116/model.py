"""Query-side PCEA at layer3[5].conv2 input; the established model stays unchanged."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pcea_full_training', ROOT.parent/'full_finetune_20261111/train.py')
full = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = full
spec.loader.exec_module(full)
sys.path.insert(0, str(full.ROOT))
torch, nn, F, np = full.torch, full.nn, full.F, full.np


class RegionalExchange(nn.Module):
    def __init__(self, arm, channels=256, dim=32, rank=16):
        super().__init__()
        self.arm = arm
        self.value = nn.ModuleList(nn.Conv2d(channels, dim, 1, bias=False) for _ in range(2))
        self.project = nn.ModuleList(nn.Linear(dim, rank, bias=False) for _ in range(2))
        self.region = nn.Parameter(torch.zeros(3, rank))
        self.slot = nn.Parameter(torch.zeros(5, rank))
        self.gate = nn.Sequential(nn.Linear(4*rank, rank), nn.SiLU(), nn.Linear(rank, 3))
        self.output = nn.ModuleList(nn.Linear(5*rank, channels, bias=False) for _ in range(2))
        for layer in self.output:
            nn.init.zeros_(layer.weight)
        if arm == 'independent':
            self.gate.requires_grad_(False)
            self.region.requires_grad_(False)
            self.slot.requires_grad_(False)
        self.trace = None
        self.disabled = False

    def forward(self, x, masks, responses, available, valid):
        b, _, channels, h, w = x.shape
        tokens, scales = [], []
        for p in range(2):
            mask = masks[:, p].flatten(2)
            uniform = mask / mask.sum(-1, keepdim=True).clamp_min(1e-8)
            xp = x[:, p]
            normalized = F.layer_norm(xp.permute(0,2,3,1), (channels,)).permute(0,3,1,2)
            value = self.value[p](normalized).flatten(2)
            response = F.adaptive_avg_pool2d(responses[:, p], (h,w)).flatten(2)
            contrast = (4*response-response.sum(1, keepdim=True))/3
            vw = valid[:, p].flatten(2)
            vw = vw/vw.sum(-1,keepdim=True).clamp_min(1e-8)
            centered = contrast-(contrast*vw).sum(-1,keepdim=True)
            score = 2*torch.tanh(centered/((centered.square()*vw).sum(-1,keepdim=True)+.01).sqrt())
            logits = score[:, None]+mask.clamp_min(1e-12).log()[:, :, None]
            weight = logits.masked_fill(mask[:, :, None]<=0, -1e4).softmax(-1)
            weight = weight*(mask[:, :, None]>0)
            weight = weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
            mean = torch.einsum('bru,bdu->brd', uniform, value)
            classes = torch.einsum('brku,bdu->brkd', weight, value)
            tokens.append(torch.cat((mean[:, :, None], classes), 2))
            regional = torch.einsum('bru,bcu->brc', uniform, xp.flatten(2))
            scales.append(regional.square().mean(-1,keepdim=True).sqrt().detach())
        u, v = (self.project[p](tokens[p]).tanh() for p in range(2))
        r, k, rank = u.shape[1:]
        if self.arm == 'independent':
            contents = [u, v]
            alpha = beta = None
        else:
            er = self.region[None,:,None].expand(b,r,k,rank)
            ek = self.slot[None,None].expand(b,r,k,rank)
            gates = self.gate(torch.cat((u,v,er,ek), -1))
            alpha, beta = gates[..., :2].softmax(-1), gates[..., 2:].sigmoid()
            fused = alpha[..., :1]*u + alpha[..., 1:]*v
            if self.arm == 'pcea':
                fused = fused+beta*(u*v)
            contents = [fused, fused]
        # Identical dual-valid eligibility for all three learned branch controls.
        active = available.bool().all(1)[:,None] & (masks.sum((-2,-1))>0).all(1)
        writer = masks/masks.sum(2,keepdim=True).clamp_min(1)
        delta = torch.stack([torch.einsum('brc,brhw->bchw',
            .1*scales[p]*self.output[p](contents[p].flatten(2)).tanh()*active[...,None],
            writer[:,p]) for p in range(2)],1)
        self.trace = dict(delta=delta.detach(), active=active.detach(),
                          masks=masks.detach(),
                          token_dispersion=torch.stack([((t[:,:,1:]-t[:,:,:1]).square().mean((2,3)).sqrt()/
                              t[:,:,0].square().mean(-1).sqrt().clamp_min(1e-8)).detach() for t in tokens],1),
                          input_rms=x.detach().square().mean((2,3,4)).sqrt(),
                          update_rms=delta.detach().square().mean((2,3,4)).sqrt(),
                          alpha=None if alpha is None else alpha.detach(),
                          beta=None if beta is None else beta.detach())
        return x if self.disabled else x+delta


class Model(full.LiveFull):
    def __init__(self, seed, arm):
        super().__init__(seed, 'full')
        checkpoint = full.ROOT/f'selected_s{seed}/best.pt'
        self.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True))
        self.arm = arm
        self.context = None
        if arm == 'baseline':
            return
        torch.manual_seed(seed+20261116)
        self.exchange = RegionalExchange(arm).to('cuda')
        # Hooks only intercept the query suffix; reference extraction stops at layer2.
        for p in range(2):
            def capture_response(module, args, result, p=p):
                self.context['response'][p] = result[1]['response']
            def capture_mask(module, args, p=p):
                occ = full.staged.old.grid_occupancy(args[2], (10,10))
                self.context['valid'][p] = occ
                self.context['mask'][p] = torch.minimum(args[1], occ)
            self.net.m1[p].register_forward_hook(capture_response)
            self.net.base.internal_m2[p].register_forward_pre_hook(capture_mask)
        self.net.base.backbone.model.layer3[5].conv2.register_forward_pre_hook(self._exchange)

    def _exchange(self, module, args):
        c = self.context
        x = args[0].reshape(-1,2,256,10,10)
        out = self.exchange(x, torch.stack(c['mask'],1), torch.stack(c['response'],1),
                            c['available'], torch.stack(c['valid'],1))
        return (out.flatten(0,1),)

    def finish(self, a, c, query_rows, reference, reference_rows, m3_on=True):
        self.context = dict(available=a['available'], mask=[None,None],
                            response=[None,None], valid=[None,None])
        try:
            return super().finish(a,c,query_rows,reference,reference_rows,m3_on)
        finally:
            self.context = None
