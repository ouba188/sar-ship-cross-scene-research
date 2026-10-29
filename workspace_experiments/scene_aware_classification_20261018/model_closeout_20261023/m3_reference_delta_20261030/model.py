"""Protected M3 with a reference-specific residual writer."""
import torch
from torch import nn
from torch.nn import functional as F
from sar_m123 import tokens
from sar_m123.selective_matching import SelectiveReferenceMatch
from sar_m123.pool_replacement import grid_occupancy


class DeltaReferenceMatch(SelectiveReferenceMatch):
    def __init__(self, mode='delta'):
        super().__init__()
        self.mode = mode
        self.extra_output = nn.Conv2d(32, 1024, 1, bias=False)
        nn.init.zeros_(self.extra_output.weight)
        self.register_buffer('extra_scale', torch.ones(()))
        self.diagnostic = None

    def forward(self, x, anchor, local, slots, local_slots, valid, energy,
                local_energy, reference, pair_valid, return_details=False):
        original, detail = super().forward(x, anchor, local, slots, local_slots,
            valid, energy, local_energy, reference, pair_valid, True)
        with torch.no_grad():
            source = reference['features']
            read = torch.minimum(reference['slots'].clamp(0, 1),
                grid_occupancy(reference['valid'].to(source), source.shape[-2:]).clamp(0, 1))
            rm = read.sum((-1, -2))
            z = F.layer_norm(source.permute(0, 2, 3, 1), (source.shape[1],)).permute(0, 3, 1, 2)
            value = torch.einsum('kshw,kdhw->ksd', read, self.project(z))/rm.clamp_min(1e-8)[..., None]
            write_masks = torch.minimum(slots.clamp(0, 1),
                grid_occupancy(valid.to(x), x.shape[-2:]).clamp(0, 1))
            qm = write_masks.sum((-1, -2))
            full, common = [], []
            offset = 0
            for region, transport in zip(tokens.REGIONS, detail['transport']):
                ix = region.tolist()
                count = len(ix)
                weights = detail['reference_weights'][..., offset:offset+count]
                offset += count
                available = transport.sum(-1) > 0
                uniform = available.to(x)/available.sum(1, keepdim=True).clamp_min(1)
                qa = qm[:, None, ix]
                offered = qa/qa.sum(-1, keepdim=True).clamp_min(1e-20)
                moved = torch.einsum('bkij,kjd->bkid', transport, value[:, ix])
                moved = moved/offered[..., None].clamp_min(1e-20)
                full.append((weights[..., None]*moved).sum(1))
                common.append((uniform[..., None]*moved).sum(1))
            write = write_masks/write_masks.sum(1, keepdim=True).clamp_min(1)
            total = torch.einsum('bshw,bsd->bdhw', write, torch.cat(full, 1))
            mean = torch.einsum('bshw,bsd->bdhw', write, torch.cat(common, 1))
            delta = total-mean
            selected = delta if self.mode == 'delta' else total
            self.diagnostic = dict(total_rms=total.square().mean((1, 2, 3)).sqrt(),
                common_rms=mean.square().mean((1, 2, 3)).sqrt(),
                delta_rms=delta.square().mean((1, 2, 3)).sqrt(),
                selected_square=selected.square().sum(), selected_numel=selected.numel())
        correction = .1*self.extra_output(selected/self.extra_scale.clamp_min(1e-8))
        detail['extra_update'] = correction
        detail['update'] = detail['update']+correction
        return original+correction, detail


class LabelReferenceMatch(SelectiveReferenceMatch):
    """Write known reference-class evidence, never the query label."""
    def __init__(self, reference_labels, classes, live_compare=False):
        super().__init__()
        self.live_compare=live_compare
        self.register_buffer('reference_labels', torch.as_tensor(reference_labels,dtype=torch.long))
        self.register_buffer('extra_scale',torch.ones(()))
        self.extra_output = nn.Conv2d(classes,1024,1,bias=False)
        nn.init.zeros_(self.extra_output.weight)
        self.diagnostic = None

    def forward(self,x,anchor,local,slots,local_slots,valid,energy,local_energy,
                reference,pair_valid,return_details=False):
        original,detail = super().forward(x,anchor,local,slots,local_slots,valid,
            energy,local_energy,reference,pair_valid,True)
        with torch.no_grad():
            compare=detail
            if self.live_compare:
                live_reference=dict(reference,anchor=reference['features'])
                _,compare=super().forward(x,x,local,slots,local_slots,valid,
                    energy,local_energy,live_reference,pair_valid,True)
            masks = torch.minimum(slots.clamp(0,1),grid_occupancy(valid.to(x),x.shape[-2:]).clamp(0,1))
            qm = masks.sum((-1,-2))
            labels = F.one_hot(self.reference_labels,self.extra_output.in_channels).to(x)
            parts,priors = [],[]
            offset=0
            for region,transport in zip(tokens.REGIONS,compare['transport']):
                ix=region.tolist()
                count=len(ix)
                weights=compare['reference_weights'][...,offset:offset+count]
                offset+=count
                mass=transport.sum(-1)
                legal=mass>0
                qa=qm[:,None,ix]
                offered=qa/qa.sum(-1,keepdim=True).clamp_min(1e-20)
                effective=weights*mass/offered.clamp_min(1e-20)
                effective=effective/effective.sum(1,keepdim=True).clamp_min(1e-20)
                uniform=legal.to(x)/legal.sum(1,keepdim=True).clamp_min(1)
                parts.append(torch.einsum('bks,kc->bsc',effective,labels))
                priors.append(torch.einsum('bks,kc->bsc',uniform,labels))
            write=masks/masks.sum(1,keepdim=True).clamp_min(1)
            total=torch.einsum('bshw,bsc->bchw',write,torch.cat(parts,1))
            prior=torch.einsum('bshw,bsc->bchw',write,torch.cat(priors,1))
            evidence=total-prior
            self.diagnostic=dict(total_rms=total.square().mean((1,2,3)).sqrt(),
                common_rms=prior.square().mean((1,2,3)).sqrt(),
                delta_rms=evidence.square().mean((1,2,3)).sqrt(),
                selected_square=evidence.square().sum(),selected_numel=evidence.numel())
        correction=.1*self.extra_output(evidence/self.extra_scale.clamp_min(1e-8))
        detail['extra_update']=correction
        detail['class_reference_evidence']=evidence
        detail['class_reference_weights']=compare['reference_weights']
        detail['update']=detail['update']+correction
        return original+correction,detail


def attach(net, mode, reference_labels=None):
    device = next(net.parameters()).device
    for block in net.m3:
        for pol, old in enumerate(block):
            if mode.startswith('label'):
                new = LabelReferenceMatch(reference_labels,net.head.out_features,'live' in mode).to(device)
            else:
                new = DeltaReferenceMatch(mode).to(device)
            missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
            expected={'extra_output.weight','extra_scale'}
            if mode.startswith('label'):
                expected.add('reference_labels')
            assert set(missing) == expected and not unexpected
            block[pol] = new
    return net


def configure(net, arm):
    net.eval().requires_grad_(False)
    params = []
    for name, p in net.named_parameters():
        if not name.startswith('m3.'):
            continue
        active = '.reader' not in name if arm == 'control' else '.extra_output.' in name
        if active:
            p.requires_grad_(True)
            params.append(p)
    return params
