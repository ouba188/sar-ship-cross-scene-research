"""Current spatial M1/M2 and guided readout, without experiment imports.

M2/CCREL region order is BODY, AXIS_END, SIDE. Native 20-slot order is
BODY (0:8), SIDE (8:16), AXIS_END (16:20); region_integrals maps explicitly.
"""
from functools import lru_cache
import numpy as np  # Import before torch for the local Windows OpenMP runtime.
import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=16)
def _cell_weights(size, count, device, dtype):
    if size % count:
        raise ValueError('Native mask extent must be an integer multiple of feature grid.')
    stride=size//count
    edges=torch.cat((torch.tensor([-.5],device=device,dtype=dtype),
                     (torch.arange(count-1,device=device,dtype=dtype)+.5)*stride,
                     torch.tensor([size-.5],device=device,dtype=dtype)))
    pixels=torch.arange(size,device=device,dtype=dtype)
    overlap=(torch.minimum(edges[1:,None],pixels[None]+.5)-
             torch.maximum(edges[:-1,None],pixels[None]-.5)).clamp_min(0)
    return overlap/(edges[1:]-edges[:-1])[:,None]

def grid_occupancy(mask, shape):
    """Native-pixel area occupancy at stride/origin0 ResNet feature centers."""
    if mask.shape[-2:]==tuple(shape):return mask
    wy=_cell_weights(mask.shape[-2],shape[0],mask.device,mask.dtype)
    wx=_cell_weights(mask.shape[-1],shape[1],mask.device,mask.dtype)
    return torch.einsum('ih,bchw,jw->bcij',wy,mask,wx)

def regional_mean(x, masks, valid):
    valid = grid_occupancy(valid.to(x), x.shape[-2:]).clamp(0, 1)
    masks = torch.minimum(grid_occupancy(masks.to(x), x.shape[-2:]).clamp(0, 1), valid)
    mass = masks.sum((-2, -1))
    values = torch.einsum('brhw,bchw->brc', masks, x) / mass.clamp_min(1e-6)[..., None]
    return values, mass

def region_integrals(slots):
    """Read the existing 20 slots in M2's body/end/side ordering."""
    return torch.stack((slots[:, :8].sum(1), slots[:, 16:20].sum(1),
                        slots[:, 8:16].sum(1)), 1)


class RelationFeedback(nn.Module):
    def __init__(self, channels, kind, gain=.3, classes=4, hidden=32):
        super().__init__()
        self.kind, self.gain, self.classes = kind, gain, classes
        self.value = nn.Conv2d(channels, hidden, 1, bias=False)
        self.reader = nn.Conv2d(channels, classes, 1, bias=False)
        self.mean_transform = nn.Linear(hidden, hidden)
        self.transforms = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(classes))
        self.output = nn.Conv2d(hidden, channels, 1, bias=False)
        self.relation_output = nn.Conv2d(hidden, channels, 1, bias=False)
        self.auxiliary = nn.Linear((classes+1)*hidden, classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.relation_output.weight)
        nn.init.zeros_(self.auxiliary.weight)
        nn.init.zeros_(self.auxiliary.bias)

    def forward(self, x, occupancy, mode=None, details=False):
        b, _, h, w = x.shape
        mass = occupancy.flatten(2)
        weight = mass/mass.sum(-1, keepdim=True).clamp_min(1e-8)
        local = x.masked_fill(occupancy == 0, 0)
        value = self.value(local).flatten(2)
        response = self.reader(local).flatten(2)
        mean = (value*weight).sum(-1)
        mu = (response*weight).sum(-1)
        centered = response-mu[..., None]
        if self.kind == 'class':
            centered = self.classes/(self.classes-1)*(centered-centered.mean(1, keepdim=True))
        scaled = centered/((centered.square()*weight).sum(-1, keepdim=True)+.01).sqrt()
        covariance = torch.einsum('bdu,bku,bu->bkd', value-mean[..., None], scaled, weight[:, 0])
        if self.kind == 'mean':
            token = mean[:, None].expand(-1, self.classes, -1)
        else:
            token = covariance
        if mode == 'merged':
            token = token.mean(1, keepdim=True).expand_as(token)
        if mode == 'covariance_zero' and self.kind != 'mean':
            token = token*0
        transformed = torch.stack([layer(token[:, k]).tanh() for k, layer in enumerate(self.transforms)], 1)
        if self.kind == 'mean':
            relation = transformed.mean(1)[..., None].expand(-1, -1, h*w)
        else:
            relation = torch.einsum('bku,bkd->bdu', scaled.tanh(), transformed)/self.classes
        if mode == 'relation_off':
            relation = relation*0
        common = self.mean_transform(mean).tanh()[..., None, None]
        raw = self.output(common)+self.relation_output(relation.reshape(b, -1, h, w))
        gate = raw.tanh()*occupancy
        if mode == 'spatial_mean':
            gate = (gate*occupancy).sum((2, 3), keepdim=True)/occupancy.sum((2, 3), keepdim=True).clamp_min(1e-8)*occupancy
        update = self.gain*x*gate
        if mode == 'off':
            update = update*0
        auxiliary = self.auxiliary(torch.cat((mean, token.flatten(1)), 1))
        if self.kind in ('class', 'mean'):
            auxiliary = auxiliary+mu
        ratio = update.flatten(1).norm(dim=1)/x.flatten(1).norm(dim=1).clamp_min(1e-8)
        result = (x+update, auxiliary, ratio)
        if details:
            return result, dict(mean=mean, covariance=covariance, response=response.reshape(b, -1, h, w),
                                scaled=scaled.reshape(b, -1, h, w), gate=gate, update=update)
        return result

class SpatialEvidencePool(nn.Module):
    """Regional means without band descriptors, selectors or spectral gains."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x, masks, valid, energy=None, return_details=False):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError('Expected x with shape [B,channels,H,W].')
        occupancy = grid_occupancy(valid.to(x), x.shape[-2:]).clamp(0, 1)
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            local = x.to(dtype).masked_fill(occupancy == 0, 0)
            mean, mass = regional_mean(local, masks.to(dtype), valid.to(dtype))
        result = dict(mean=mean.to(x.dtype), pooled=mean.to(x.dtype), mass=mass)
        if return_details:
            result['valid'] = occupancy
        return result

class RegionPoolWriteback(nn.Module):
    """Pool three named regions and return each token only to its own footprint.

    This is context pooling/writeback, not spatial downsampling. Zero output
    projections preserve the pretrained path initially without blocking their
    first-step gradients. A zero scalar gain is deliberately not stacked on top.
    """
    def __init__(self, channels, hidden=32, strength=1.0):
        super().__init__()
        self.strength = strength
        self.pool = SpatialEvidencePool(channels)
        self.transforms = nn.ModuleList(nn.Sequential(nn.Linear(channels, hidden),
            nn.GELU(), nn.Linear(hidden, channels, bias=False)) for _ in range(3))
        for transform in self.transforms:
            nn.init.zeros_(transform[-1].weight)
        self.enabled = True

    def forward(self, x, masks, valid, energy=None):
        d = self.pool(x, masks, valid, energy, return_details=True)
        values = torch.stack([f(d['pooled'][:, r]) for r, f in enumerate(self.transforms)], 1)
        occupancy = grid_occupancy(valid.to(x), x.shape[-2:]).clamp(0, 1)
        weights = torch.minimum(masks.to(x).clamp(0, 1), occupancy)
        weights = weights/weights.sum(1, keepdim=True).clamp_min(1)
        update = self.strength*torch.einsum('brhw,brc->bchw', weights, values)
        d.update(write_weights=weights, update=update)
        return x+update if self.enabled else x, d

class RegionalGuidance(nn.Module):
    def __init__(self, arm, classes):
        super().__init__()
        self.arm = arm
        self.value = nn.ModuleList(nn.Conv2d(512,32,1,bias=False) for _ in range(2))
        self.output = nn.ModuleList(nn.Linear((classes+1)*32,512,bias=False) for _ in range(2))
        self.scorer = nn.ModuleList(nn.Conv2d(512,classes,1,bias=False) for _ in range(2))
        for layer in self.output: nn.init.zeros_(layer.weight)
        for layer in self.scorer:
            nn.init.normal_(layer.weight,std=.005)
            layer.requires_grad_(arm=='content')
        self.head = nn.Linear(7170,classes)

    def forward(self,b,mode=None,details=False):
        classes = self.head.out_features
        deltas=[];traces=[]
        for p in range(2):
            x=b['x'][:,p];mask=b['mask'][:,p].flatten(2)
            mass=mask.sum(-1,keepdim=True);uniform=mask/mass.clamp_min(1e-8)
            normalized=F.layer_norm(x.permute(0,2,3,1),(512,)).permute(0,3,1,2)
            v=self.value[p](normalized).flatten(2)
            response=b['response'][:,p].flatten(2)
            if self.arm=='class':
                relative=(classes*response-response.sum(1,keepdim=True))/(classes-1)
                valid=b['valid'][:,p].flatten(2)
                vw=valid/valid.sum(-1,keepdim=True).clamp_min(1e-8)
                centered=relative-(relative*vw).sum(-1,keepdim=True)
                score=2*torch.tanh(centered/((centered.square()*vw).sum(-1,keepdim=True)+.01).sqrt())
            elif self.arm=='content': score=2*torch.tanh(self.scorer[p](normalized).flatten(2))
            else: score=torch.zeros_like(response)
            logits=score[:,None]+mask.clamp_min(1e-12).log()[:,:,None]
            logits=logits.masked_fill(mask[:,:,None]<=0,-1e4)
            weights=logits.softmax(-1)*(mask[:,:,None]>0)
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
            if mode=='uniform': weights=uniform[:,:,None].expand(-1,-1,classes,-1)
            elif mode=='shuffle':
                weights=weights.clone()
                for i in range(len(x)):
                    for r in range(3):
                        ix=torch.where(mask[i,r]>0)[0]
                        weights[i,r,:,ix]=weights[i,r,:,ix.flip(0)]
            mean_v=torch.einsum('bru,bdu->brd',uniform,v)
            token=torch.einsum('brku,bdu->brkd',weights,v)
            descriptors=torch.cat((mean_v,token.flatten(2)),-1)
            mean_x=torch.einsum('bru,bdu->brd',uniform,x.flatten(2))
            scale=mean_x.square().mean(-1,keepdim=True).sqrt().detach()
            delta=.1*scale*self.output[p](descriptors).tanh()*(mass>0)
            delta=delta*b['available'][:,p,None,None]
            if mode=='off': delta=delta*0
            deltas.append(delta)
            if details: traces.append(dict(weights=weights,uniform=uniform,delta=delta,mean=mean_x))
        parts=[]
        for p in range(2):
            z=b['features'][:,p*3584:(p+1)*3584]
            parts.extend((z[:,:2048],z[:,2048:]+deltas[p].flatten(1)))
        score=self.head(torch.cat(parts+[b['features'][:,-2:]],1))
        return (score,traces) if details else score
