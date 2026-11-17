"""Matched local-content and predicted-class competition factors for PCEA."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('resolution_previous', ROOT.parent/'pcea_conditional_modulation_20261117/model.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
torch, nn, F, np, full = old.torch, old.nn, old.F, old.np, old.full
ARMS = ('pooled', 'local', 'competition', 'both')


class Modulation(old.RegionalModulation):
    def __init__(self, reader, arm):
        super().__init__(reader, 'affine_frozen')
        self.use_local = arm in ('local', 'both')
        self.use_competition = arm in ('competition', 'both')
        self.local_value = None
        self.teacher_score = None
        self.no_detail = False
        self.no_competition = False
        self.shuffle_detail = False

    def routed_coefficients(self, h, p, score):
        # h: batch, region, slot, height, width, rank. Preserve the 5x16 output basis.
        if self.use_competition and not self.no_competition:
            selected = torch.zeros_like(score).scatter_(1, score.topk(2, dim=1).indices, 2.)
            route = torch.cat((torch.ones_like(selected[:, :1]), selected), 1)
            h = h*route[:, None, :, None, None, None]
        z = h.permute(0, 1, 3, 4, 2, 5).flatten(-2)
        a = self.scale_output[p](z).permute(0, 1, 4, 2, 3)
        b = self.output[p](z).permute(0, 1, 4, 2, 3)
        return a, b

    def local_joint(self, u, v):
        b, r, k, height, width, d = u.shape
        er = self.region[None, :, None, None, None].expand(b, r, k, height, width, d)
        ek = self.slot[None, None, :, None, None].expand(b, r, k, height, width, d)
        raw = self.gate(torch.cat((u, v, er, ek), -1))
        alpha, beta = raw[..., :2].softmax(-1), raw[..., 2:].sigmoid()
        h = alpha[..., :1]*u+alpha[..., 1:]*v
        if not self.no_product:
            h = h+beta*u*v
        return h

    def forward(self, x, writer_masks, evidence, effective, reader_masks, available):
        if self.source_raw is None:
            return super().forward(x, writer_masks, evidence, effective, reader_masks, available)
        if not self.use_local and not self.use_competition:
            return super().forward(x, writer_masks, evidence, effective, reader_masks, available)
        t, _ = self.standardized(evidence)
        active = available.bool().all(1)[:, None]
        active = active & (reader_masks.sum((-2, -1)) > 0).all(1)
        active = active & (writer_masks.sum((-2, -1)) > 0).all(1)
        writer = writer_masks/writer_masks.sum(2, keepdim=True).clamp_min(1)
        global_uv = [self.project[p](t[:, p]).tanh() for p in range(2)]
        updates, residual_sizes = [], []
        for p in range(2):
            detail = torch.zeros_like(self.local_value[:, p, None]).expand(-1, 3, -1, -1, -1)
            if self.use_local and not self.no_detail:
                center = evidence[:, p, :, 0, :, None, None]
                unit = self.scale[p, :, 0, :, None, None]
                detail = .5*((self.local_value[:, p, None]-center)/unit).tanh()
                if self.shuffle_detail:
                    flat = detail.flatten(-2)
                    order = torch.arange(flat.shape[-1], device=x.device).expand(flat.shape[:2]+(-1,)).clone()
                    for bi in range(len(x)):
                        for ri in range(3):
                            indices = torch.where(writer_masks[bi, p, ri].flatten() > 0)[0]
                            order[bi, ri, indices] = indices.flip(0)
                    detail = flat.gather(-1, order[:, :, None].expand_as(flat)).reshape_as(detail)
            residual_sizes.append(detail.detach().square().mean((1, 2, 3, 4)).sqrt())
            own = t[:, p, :, :, None, None, :]+detail.permute(0, 1, 3, 4, 2)[:, :, None]
            own = self.project[p](own).tanh()
            other = global_uv[1-p][:, :, :, None, None].expand_as(own)
            h = self.local_joint(own, other) if p == 0 else self.local_joint(other, own)
            a, b = self.routed_coefficients(h, p, self.teacher_score)
            mask = writer_masks[:, p].flatten(2)
            w = mask/mask.sum(-1, keepdim=True).clamp_min(1e-8)
            region = torch.einsum('bru,bcu->brc', w, x[:, p].flatten(2))
            scale = region.square().mean(-1, keepdim=True).sqrt().detach()
            local = F.layer_norm(x[:, p].permute(0, 2, 3, 1), (256,)).permute(0, 3, 1, 2).tanh()
            content = (a*local[:, None]+b).tanh()
            delta = .1*scale[..., None, None]*content*active[..., None, None, None]
            updates.append((delta*writer[:, p, :, None]).sum(1))
        delta = torch.stack(updates, 1)
        self.trace = dict(tokens=evidence.detach(), normalized=t.detach(), effective_cells=effective.detach(),
            active=active.detach(), delta=delta.detach(), detail_rms=torch.stack(residual_sizes, 1),
            input_rms=x.detach().square().mean((2, 3, 4)).sqrt(),
            update_rms=delta.detach().square().mean((2, 3, 4)).sqrt(),
            top2=self.teacher_score.topk(2, dim=1).indices.detach())
        return x if self.disabled else x+delta


class Model(old.Model):
    def __init__(self, seed, arm):
        assert arm in ARMS
        super().__init__(seed, 'affine_frozen')
        self.arm = arm
        torch.manual_seed(seed+20261116)
        self.exchange = Modulation(self.net.guide.value, arm).to('cuda')

    def forward(self, data, query_rows, reference, reference_rows, m3_on=True):
        self.exchange.teacher_score = data.get('teacher_score')
        return super().forward(data, query_rows, reference, reference_rows, m3_on=m3_on)

    def _exchange(self, module, args):
        c = self.context
        values = []
        for p in range(2):
            x = c['shallow'][p]
            x = F.layer_norm(x.permute(0, 2, 3, 1), (512,)).permute(0, 3, 1, 2)
            value = self.ccrel.value[p](x)
            occ = c['reader_valid'][p]
            # Occupancy-aware reduction does not invent pixels outside the native chip.
            value = F.avg_pool2d(value*occ, 2)/F.avg_pool2d(occ, 2).clamp_min(1e-8)
            values.append(value)
        self.exchange.local_value = torch.stack(values, 1)
        return super()._exchange(module, args)


calibrate = old.calibrate
