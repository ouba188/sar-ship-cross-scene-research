"""Paired head coordination and content-dependent regional modulation."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'coordinate_previous', ROOT.parent / 'pcea_coordinate_repair_20261117/model.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
full, torch, nn, F, np = old.full, old.torch, old.nn, old.F, old.np
ARMS = ('head_only', 'broadcast_head', 'affine_frozen', 'affine_head',
        'affine_regionmean', 'affine_independent')


class RegionalModulation(old.PCEA):
    def __init__(self, reader, arm):
        super().__init__('coordinated', reader)
        self.scale_output = nn.ModuleList(nn.Linear(80, 256, bias=False) for _ in range(2))
        for layer in self.scale_output:
            nn.init.zeros_(layer.weight)
        self.regional_mean = arm == 'affine_regionmean'
        self.independent = arm == 'affine_independent'
        self.no_scale = False

    def coefficients(self, t):
        if self.independent:
            # Both learned projections remain distinct; each output sees only its own input.
            pairs = [self.interaction(self.project[0](t[:, p]).tanh(),
                                      self.project[1](t[:, p]).tanh()) for p in range(2)]
            h = [pair[0] for pair in pairs]
            alpha = torch.stack([pair[1] for pair in pairs], 1)
            beta = torch.stack([pair[2] for pair in pairs], 1)
        else:
            u, v = (self.project[p](t[:, p]).tanh() for p in range(2))
            joint, alpha, beta = self.interaction(u, v)
            h = [joint, joint]
        a = [self.scale_output[p](h[p].flatten(2)) for p in range(2)]
        b = [self.output[p](h[p].flatten(2)) for p in range(2)]
        return a, b, alpha, beta

    def forward(self, x, writer_masks, evidence, effective, reader_masks, available):
        if self.source_raw is None:
            return super().forward(x, writer_masks, evidence, effective, reader_masks, available)
        t, source = self.standardized(evidence)
        a, b, alpha, beta = self.coefficients(t)
        if self.force_common:
            source_a, source_b, _, _ = self.coefficients(source)
            w = self.source_weight[..., None]
            a = [(q*w).sum(0)[None]/w.sum(0).clamp_min(1e-8) for q in source_a]
            b = [(q*w).sum(0)[None]/w.sum(0).clamp_min(1e-8) for q in source_b]
        active = available.bool().all(1)[:, None]
        active = active & (reader_masks.sum((-2, -1)) > 0).all(1)
        active = active & (writer_masks.sum((-2, -1)) > 0).all(1)
        writer = writer_masks/writer_masks.sum(2, keepdim=True).clamp_min(1)
        updates = []
        for p in range(2):
            mask = writer_masks[:, p].flatten(2)
            w = mask/mask.sum(-1, keepdim=True).clamp_min(1e-8)
            region = torch.einsum('bru,bcu->brc', w, x[:, p].flatten(2))
            scale = region.square().mean(-1, keepdim=True).sqrt().detach()
            local = F.layer_norm(x[:, p].permute(0, 2, 3, 1), (x.shape[2],))
            local = local.permute(0, 3, 1, 2).tanh()
            if self.regional_mean:
                local = torch.einsum('bru,bcu->brc', w, local.flatten(2))[..., None, None]
            else:
                local = local[:, None]
            multiplier = torch.zeros_like(a[p]) if self.no_scale else a[p]
            content = (multiplier[..., None, None]*local+b[p][..., None, None]).tanh()
            if self.force_zero:
                content = torch.zeros_like(content)
            # Identical 0.1 * regional RMS bound to the old broadcast writer.
            delta = .1*scale[..., None, None]*content*active[..., None, None, None]
            updates.append((delta*writer[:, p, :, None]).sum(1))
        delta = torch.stack(updates, 1)
        self.trace = dict(tokens=evidence.detach(), normalized=t.detach(),
            effective_cells=effective.detach(), active=active.detach(), delta=delta.detach(),
            input_rms=x.detach().square().mean((2, 3, 4)).sqrt(),
            update_rms=delta.detach().square().mean((2, 3, 4)).sqrt(),
            alpha=alpha.detach(), beta=beta.detach(),
            scale_coeff=torch.stack(a, 1).detach(), bias_coeff=torch.stack(b, 1).detach())
        return x if self.disabled else x+delta


class Model(old.Model):
    def __init__(self, seed, arm):
        assert arm in ARMS
        super().__init__(seed, 'coordinated')
        self.arm = arm
        if arm.startswith('affine_'):
            torch.manual_seed(seed+20261116)
            self.exchange = RegionalModulation(self.net.guide.value, arm).to('cuda')
        self.requires_grad_(False)
        if arm != 'head_only':
            self.ccrel.requires_grad_(True)
            self.exchange.requires_grad_(True)
        else:
            self.exchange.disabled = True
        if arm != 'affine_frozen':
            self.net.guide.head.requires_grad_(True)


def calibrate(net, inputs, seed):
    # Only readout projection/head/writer change; the frozen preprojection bank is valid.
    return old.calibrate(net, inputs, seed)
