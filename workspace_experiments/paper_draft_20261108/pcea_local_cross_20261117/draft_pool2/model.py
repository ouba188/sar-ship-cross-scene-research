"""Local-content controls: independent, opposite mean, and opposite local attention."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'local_cross_previous', ROOT.parent/'pcea_conditional_modulation_20261117/model.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
torch, nn, F, np, full = old.torch, old.nn, old.F, old.np, old.full
ARMS = ('independent', 'cross_mean', 'cross_local')


def attend(query, value, mask, uniform=False):
    """Same-region content selection; mask is soft area, not physical point registration."""
    weight = mask / mask.sum(-1, keepdim=True).clamp_min(1e-8)
    if uniform:
        return (value*weight[..., None]).sum(-2, keepdim=True).expand_as(query)
    shape = query.shape
    q = F.normalize(query, dim=-1).reshape(-1, 1, shape[-2], shape[-1])
    k = F.normalize(value, dim=-1).reshape_as(q)
    v = value.reshape_as(q)
    bias = mask.clamp_min(1e-12).log().masked_fill(mask <= 0, -1e4)
    out = F.scaled_dot_product_attention(q, k, v,
        attn_mask=bias.reshape(-1, 1, 1, shape[-2]), dropout_p=0., scale=4.)
    active = (mask.sum(-1) > 0)[..., None, None]
    return out.reshape(shape)*active


class LocalExchange(old.RegionalModulation):
    def __init__(self, reader, arm):
        super().__init__(reader, 'affine_frozen')
        self.mode = arm
        self.local_value = None
        self.uniform_attention = False
        self.other_mean = False
        self.no_detail = False
        self.capture_attention = False

    def joint(self, u, v):
        b, r, k, n, d = u.shape
        er = self.region[None, :, None, None].expand(b, r, k, n, d)
        ek = self.slot[None, None, :, None].expand(b, r, k, n, d)
        raw = self.gate(torch.cat((u, v, er, ek), -1))
        alpha, beta = raw[..., :2].softmax(-1), raw[..., 2:].sigmoid()
        h = alpha[..., :1]*u + alpha[..., 1:]*v
        return h if self.no_product else h+beta*u*v

    def forward(self, x, writer_masks, evidence, effective, reader_masks, available):
        if self.source_raw is None:
            return super().forward(x, writer_masks, evidence, effective, reader_masks, available)
        t, _ = self.standardized(evidence)
        masks = reader_masks.flatten(-2)
        mass = masks/masks.sum(-1, keepdim=True).clamp_min(1e-8)
        uv, detail_rms = [], []
        for p in range(2):
            center = evidence[:, p, :, 0, :, None, None]
            unit = self.scale[p, :, 0, :, None, None]
            detail = .5*((self.local_value[:, p, None]-center)/unit).tanh()
            if self.no_detail:
                detail = torch.zeros_like(detail)
            detail_rms.append(detail.detach().square().mean((1, 2, 3, 4)).sqrt())
            local = t[:, p, :, :, None, :] + detail.flatten(-2).permute(0, 1, 3, 2)[:, :, None]
            uv.append(self.project[p](local).tanh())
        active = available.bool().all(1)[:, None]
        active = active & (reader_masks.sum((-2, -1)) > 0).all(1)
        active = active & (writer_masks.sum((-2, -1)) > 0).all(1)
        writer = writer_masks/writer_masks.sum(2, keepdim=True).clamp_min(1)
        updates, variation, attentions = [], [], []
        for p in range(2):
            own = uv[p]
            other_p = p if self.mode == 'independent' else 1-p
            other = uv[other_p]
            key_mask = masks[:, other_p, :, None].expand(-1, -1, 5, -1)
            if self.other_mean:
                other = (other*mass[:, other_p, :, None, :, None]).sum(-2, keepdim=True).expand_as(other)
            uniform = self.mode != 'cross_local' or self.uniform_attention
            context = attend(own, other, key_mask, uniform)
            own_mass = mass[:, p, :, None, :, None]
            context_mean = (context*own_mass).sum(-2, keepdim=True)
            variation.append(((context-context_mean).square()*own_mass).sum(-2).mean((1, 2, 3)).sqrt().detach())
            h = self.joint(own, context) if p == 0 else self.joint(context, own)
            batch, regions, slots, _, rank = h.shape
            # All arms interact on 20x20 content BEFORE the identical 10x10 writer reduction.
            h = h.permute(0, 1, 2, 4, 3).reshape(batch*regions*slots, rank, 20, 20)
            rm = reader_masks[:, p, :, None].expand(-1, -1, 5, -1, -1).reshape(-1, 1, 20, 20)
            h = F.avg_pool2d(h*rm, 2)/F.avg_pool2d(rm, 2).clamp_min(1e-8)
            h = h.reshape(batch, regions, slots, rank, 10, 10).permute(0, 1, 4, 5, 2, 3).flatten(-2)
            a = self.scale_output[p](h).permute(0, 1, 4, 2, 3)
            b = self.output[p](h).permute(0, 1, 4, 2, 3)
            mask = writer_masks[:, p].flatten(2)
            w = mask/mask.sum(-1, keepdim=True).clamp_min(1e-8)
            region = torch.einsum('bru,bcu->brc', w, x[:, p].flatten(2))
            scale = region.square().mean(-1, keepdim=True).sqrt().detach()
            local_x = F.layer_norm(x[:, p].permute(0, 2, 3, 1), (256,)).permute(0, 3, 1, 2).tanh()
            content = (a*local_x[:, None]+b).tanh()
            delta = .1*scale[..., None, None]*content*active[..., None, None, None]
            updates.append((delta*writer[:, p, :, None]).sum(1))
            if self.capture_attention:
                logits = 4*torch.matmul(F.normalize(own, dim=-1), F.normalize(other, dim=-1).transpose(-2, -1))
                logits = logits+key_mask.clamp_min(1e-12).log()[..., None, :]
                attention = logits.masked_fill(key_mask[..., None, :] <= 0, -1e4).softmax(-1)
                if uniform:
                    attention = (key_mask/key_mask.sum(-1, keepdim=True).clamp_min(1e-8))[..., None, :].expand_as(attention)
                attentions.append(attention.detach())
        delta = torch.stack(updates, 1)
        self.trace = dict(tokens=evidence.detach(), normalized=t.detach(), effective_cells=effective.detach(),
            active=active.detach(), delta=delta.detach(), detail_rms=torch.stack(detail_rms, 1),
            context_spatial_rms=torch.stack(variation, 1),
            input_rms=x.detach().square().mean((2, 3, 4)).sqrt(),
            update_rms=delta.detach().square().mean((2, 3, 4)).sqrt())
        if attentions:
            self.trace['attention'] = torch.stack(attentions, 1)
        return x if self.disabled else x+delta


class Model(old.Model):
    def __init__(self, seed, arm):
        assert arm in ARMS
        super().__init__(seed, 'affine_frozen')
        self.arm = arm
        torch.manual_seed(seed+20261116)
        self.exchange = LocalExchange(self.net.guide.value, arm).to('cuda')

    def _exchange(self, module, args):
        c = self.context
        values = []
        for p in range(2):
            x = c['shallow'][p]
            x = F.layer_norm(x.permute(0, 2, 3, 1), (512,)).permute(0, 3, 1, 2)
            values.append(self.ccrel.value[p](x))
        self.exchange.local_value = torch.stack(values, 1)
        return super()._exchange(module, args)


calibrate = old.calibrate
