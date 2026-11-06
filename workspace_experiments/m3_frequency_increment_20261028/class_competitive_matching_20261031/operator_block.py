"""Trainable class-conditioned matching; aggregate retains both real marginals."""
import torch
from torch import nn
from torch.nn import functional as F
from sar_m123.m3 import sinkhorn
from sar_m123.tokens import REGIONS


def weighted_cosine(q, r, w):
    scale = w.sqrt()[:, :, None]
    a = F.normalize(q[:, None] * scale, dim=-1)
    b = F.normalize(r[None] * scale, dim=-1)
    return torch.einsum('brik,brjk->brij', a, b)


class CompetitiveMatch(nn.Module):
    def __init__(self, seed=17, classes=8):
        super().__init__()
        self.classes = classes
        torch.manual_seed(seed)
        for name, size in (('shallow', 256), ('deep', 512), ('key', 64), ('value', 64)):
            setattr(self, name, nn.ModuleList(nn.Linear(size, 32, bias=False) for _ in range(2)))
            setattr(self, name + '_norm', nn.ModuleList(nn.LayerNorm(32) for _ in range(2)))
        self.metric = nn.Sequential(nn.Linear(96, 16), nn.GELU(), nn.Linear(16, 32))
        self.output = nn.ModuleList(nn.Conv2d(classes, 1024, 1, bias=False) for _ in range(2))
        for layer in self.output:
            nn.init.zeros_(layer.weight)

    def encode(self, x, p):
        a = self.shallow_norm[p](self.shallow[p](F.layer_norm(x[..., :256], (256,))))
        b = self.deep_norm[p](self.deep[p](F.layer_norm(x[..., 256:], (512,))))
        z = torch.cat((a, b), -1)
        k = self.key_norm[p](self.key[p](z))
        v = self.value_norm[p](self.value[p](z))
        return F.normalize(k, dim=-1), v

    def transport(self, cost, qa, ra, allowed):
        return sinkhorn(cost, qa[:, None], ra[None], steps=60) * allowed

    def metric_condition(self, mu, other, all_mu, arm):
        if arm == 'shared':
            mean = all_mu.expand_as(mu)
            return torch.cat((mean, mean, torch.zeros_like(mu)), -1)
        return torch.cat((mu, other, mu - other), -1)

    def cost_competition(self, competition, arm):
        return torch.zeros_like(competition) if arm == 'shared' else competition

    def write_message(self, centered, proto, active, pol):
        return centered.transpose(1, 2)

    def forward(self, q, r, qm, rm, write, legal, ref_labels, arm='competitive', details=False):
        if arm not in ('competitive', 'shared', 'aggregate'):
            raise ValueError(arm)
        if len(r) == 0:
            return (q.new_zeros((len(q), 2, 1024, 10, 10)), q.new_full((len(q), self.classes), -1e4),
                    torch.zeros((len(q), self.classes), dtype=torch.bool, device=q.device), [])
        labels = F.one_hot(ref_labels.long(), self.classes).to(q)
        updates, sums, counts, traces = [], [], [], []
        for p in range(2):
            kq, vq = self.encode(q[:, p], p)
            kr, vr = self.encode(r[:, p], p)
            fields, regions = [], []
            ds, dn = q.new_zeros((len(q), self.classes)), q.new_zeros((len(q), self.classes))
            for region in REGIONS:
                ix = region.tolist()
                qa, ra = qm[:, p, ix], rm[:, p, ix]
                live_ref = legal[:, :, p] & (ra.sum(-1) > 0)[None] & (qa.sum(-1) > 0)[:, None]
                # Query-specific legal support mass, including absent polarization/regions.
                mass = live_ref[..., None] * ra[None]
                total = torch.einsum('brj,rc->bc', mass, labels)
                available = total > 0
                mu = torch.einsum('brj,rc,rjd->bcd', mass, labels, kr[:, ix])
                mu = mu / total[..., None].clamp_min(1e-20)
                n = available.sum(-1, keepdim=True)
                all_mu = mu.sum(1, keepdim=True) / n[..., None].clamp_min(1)
                other_count = (n - available.long()).clamp_min(1)[..., None]
                other = (mu.sum(1, keepdim=True) - mu) / other_count
                condition = self.metric_condition(mu, other, all_mu, arm)
                w = 1 + .5 * self.metric(condition).tanh()
                sim = weighted_cosine(kq[:, ix], kr[:, ix], w[:, ref_labels])
                qa = qa / qa.sum(-1, keepdim=True).clamp_min(1e-20)
                ra = ra / ra.sum(-1, keepdim=True).clamp_min(1e-20)
                allowed = live_ref[..., None, None] & (qa > 0)[:, None, :, None] & (ra > 0)[None, :, None, :]
                best = torch.stack([sim.masked_fill(~(allowed & (ref_labels == cls)[None, :, None, None]),
                                                   -1e4).amax((1, 3)) for cls in range(self.classes)], 1)
                active = available[..., None] & (qa > 0)[:, None]
                best = best * active
                others = (best.sum(1, keepdim=True) - best) / other_count
                competition = (best - others) * active * (n > 1)[..., None]
                competition = self.cost_competition(competition, arm)
                cost = sim + competition[:, ref_labels, :, None]
                # sinkhorn pads only after this addition: dustbin logits stay exactly zero.
                original = self.transport(cost, qa, ra, allowed)
                row, col = original.sum(-1), original.sum(-2)
                score = (original * sim).sum(-1) / row.clamp_min(1e-20)
                live = row > 0
                beta = (score / .1).masked_fill(~live, -1e4).softmax(1) * live
                beta = beta / beta.sum(1, keepdim=True).clamp_min(1e-20)
                transport = original
                if arm == 'aggregate':
                    transport = row[..., None] * col[..., None, :] / row.sum(-1)[..., None, None].clamp_min(1e-20)
                moved = torch.einsum('brij,rjd->brid', transport, vr[:, ix])
                aligned = moved / row[..., None].clamp_min(1e-20)
                class_beta = beta[:, :, None] * labels[None, :, :, None]
                denom = class_beta.sum(1)
                class_beta = class_beta / denom[:, None].clamp_min(1e-20)
                proto = torch.einsum('brci,brid->bcid', class_beta, aligned)
                active = (denom > 0) & active
                distance = ((vq[:, None, ix] - proto).square() * w[:, :, None]).sum(-1)
                evidence = (-distance / .1) * active
                centered = (evidence - evidence.sum(1, keepdim=True) / active.sum(1, keepdim=True).clamp_min(1)) * active
                message = self.write_message(centered, proto, active, p)
                fields.append(message)
                aw = qa[:, None] * active
                valid_region = aw.sum(-1) > 0
                ds = ds + (distance * aw).sum(-1) / aw.sum(-1).clamp_min(1e-20)
                dn = dn + valid_region
                if details:
                    regions.append(dict(transport=transport, original_transport=original, beta=beta,
                                        value=vr[:, ix], query_value=vq[:, ix], metric=w, sim=sim,
                                        competition=competition, available=available, evidence=centered,
                                        message=message, prototype=proto))
            field = torch.cat(fields, 1)
            mixed = torch.einsum('bshw,bsc->bchw', write[:, p], field)
            updates.append(.1 * self.output[p](mixed))
            sums.append(ds); counts.append(dn)
            if details:
                traces.append(dict(regions=regions, mixed=mixed, message=field))
        count = torch.stack(counts).sum(0)
        local = (-torch.stack(sums).sum(0) / count.clamp_min(1) / .1).masked_fill(count <= 0, -1e4)
        return torch.stack(updates, 1), local, count > 0, traces
