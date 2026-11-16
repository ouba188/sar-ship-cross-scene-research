"""Explicit trainable CCREL evidence -> PCEA, with a native layer1 residual."""
from pathlib import Path
import copy
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pcea_readout_previous', ROOT.parent/'pcea_readout_repair_20261116/model.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
full, torch, nn, F, np = old.full, old.torch, old.nn, old.F, old.np
ARMS = ('joint', 'joint_fine', 'joint_fine_gate', 'independent_fine')


class CCREL(nn.Module):
    """Existing class-contrast responses organize trainable regional evidence."""
    def __init__(self, reader, fine):
        super().__init__()
        self.value = copy.deepcopy(reader).requires_grad_(True)
        self.use_fine = fine
        self.no_fine = False
        if fine:
            self.fine_value = nn.ModuleList(nn.Conv2d(256, 32, 1, bias=False) for _ in range(2))
            for layer in self.fine_value:
                nn.init.zeros_(layer.weight)
        self.trace = None

    def forward(self, shallow, fine, masks, responses, valid, fine_valid):
        tokens, effective, residual_sizes = [], [], []
        for p in range(2):
            x = shallow[:, p]
            value = self.value[p](F.layer_norm(x.permute(0, 2, 3, 1), (512,)).permute(0, 3, 1, 2))
            residual = torch.zeros_like(value)
            if self.use_fine:
                # Native 40x40 content, normalized/projected BEFORE area alignment.
                low = fine[:, p]
                low = F.layer_norm(low.permute(0, 2, 3, 1), (256,)).permute(0, 3, 1, 2)
                detail = self.fine_value[p](low).tanh()*fine_valid[:, p]
                detail = F.avg_pool2d(detail, 2)
                scale = value.square().mean(1, keepdim=True).sqrt().detach()
                residual = .1*scale*detail
                if not self.no_fine:
                    value = value+residual
            residual_sizes.append(residual.detach().square().mean((1, 2, 3)).sqrt())
            value, mask, response, occupancy = (t.flatten(2) for t in
                (value, masks[:, p], responses[:, p], valid[:, p]))
            uniform = mask/mask.sum(-1, keepdim=True).clamp_min(1e-8)
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
        evidence = torch.stack(tokens, 1)
        self.trace = dict(fine_residual_rms=torch.stack(residual_sizes, 1))
        return evidence, torch.stack(effective, 1)


class PCEA(old.Exchange):
    def __init__(self, arm, reader):
        mode = 'fine_independent' if arm == 'independent_fine' else 'fine_std'
        super().__init__(mode, reader)
        del self.value
        self.gate_only = arm == 'joint_fine_gate'

    def fuse(self, u, v):
        # Preserve the paired product-off intervention without changing gate inputs.
        previous = self.no_product
        self.no_product = previous or self.gate_only
        try:
            return super().fuse(u, v)
        finally:
            self.no_product = previous

    def forward(self, x, writer_masks, evidence, effective, reader_masks, available):
        t = (evidence-self.mean)/self.scale
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
        self.trace = dict(tokens=evidence.detach(), effective_cells=effective.detach(), active=active.detach(),
            delta=delta.detach(), input_rms=x.detach().square().mean((2, 3, 4)).sqrt(),
            update_rms=delta.detach().square().mean((2, 3, 4)).sqrt(),
            alpha=None if alpha is None else alpha.detach(), beta=None if beta is None else beta.detach())
        return x if self.disabled else x+delta


class Model(old.Model):
    def __init__(self, seed, arm):
        assert arm in ARMS
        super().__init__(seed, 'fine_std')
        self.arm = arm
        # PCEA initialization is identical across arms, unaffected by extra residual parameters.
        torch.manual_seed(seed+20261116)
        self.exchange = PCEA(arm, self.net.guide.value).to('cuda')
        self.ccrel = CCREL(self.net.guide.value, arm != 'joint').to('cuda')
        self.fine_features = None
        def capture(module, args, result):
            self.fine_features = result.reshape(-1, 2, 256, 40, 40)
        self.net.base.backbone.model.layer1.register_forward_hook(capture)

    def forward(self, data, query_rows, reference, reference_rows, m3_on=True):
        self.fine_valid = full.staged.old.grid_occupancy(data['valid'].flatten(0, 1), (40, 40)).reshape(-1, 2, 1, 40, 40)
        try:
            return super().forward(data, query_rows, reference, reference_rows, m3_on=m3_on)
        finally:
            self.fine_features = None

    def _exchange(self, module, args):
        c = self.context
        x = args[0].reshape(-1, 2, 256, 10, 10)
        masks = torch.stack(c['reader_mask'], 1)
        evidence, effective = self.ccrel(torch.stack(c['shallow'], 1), self.fine_features,
            masks, torch.stack(c['response'], 1), torch.stack(c['reader_valid'], 1), self.fine_valid)
        out = self.exchange(x, torch.stack(c['mask'], 1), evidence, effective, masks, c['available'])
        return (out.flatten(0, 1),)


def calibrate(net, inputs, seed):
    # Initial reader is identical; retain fixed SOURCE-only units while the reader learns.
    return old.calibrate(net, inputs, seed)
