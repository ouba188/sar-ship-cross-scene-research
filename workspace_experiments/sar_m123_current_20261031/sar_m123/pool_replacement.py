"""Experimental native-grid M2 pooling replacements, separate from feedback."""
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F

from .backbone_regions import RegionResNet50, RegionFeedback


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


class LocalRegionPool(nn.Module):
    """Shared-channel 3x3/stride2 pooling, optionally guided by soft regions.

    The region prior describes feature-cell occupancy, not a receptive-field
    segmentation. A bounded learnable coefficient may favor or relax agreement.
    """
    def __init__(self, channels, mode='average', hidden=16):
        super().__init__()
        if mode not in ('average', 'maximum', 'blur', 'content', 'regional'):
            raise ValueError(mode)
        self.mode, self.channels = mode, channels
        self.scorer = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)
        self.beta = nn.Parameter(torch.zeros(()), requires_grad=mode == 'regional')
        self.scorer.requires_grad_(mode in ('content', 'regional'))

    @staticmethod
    def patches(x):
        b, c, h, w = x.shape
        return F.unfold(x, 3, padding=1, stride=2).reshape(b, c, 9, (h+1)//2, (w+1)//2)

    def weights(self, x, masks, valid):
        valid = grid_occupancy(valid.to(x), x.shape[-2:]).clamp(0, 1)
        occupancy = self.patches(valid)[:, 0]
        logits = torch.zeros_like(occupancy)
        if self.mode == 'blur':
            kernel = x.new_tensor([1, 2, 1, 2, 4, 2, 1, 2, 1]).reshape(1, 9, 1, 1)
            logits = kernel.log().expand_as(occupancy)
        if self.mode in ('content', 'regional'):
            z = F.layer_norm(x.permute(0, 2, 3, 1), (self.channels,)).permute(0, 3, 1, 2)
            logits = self.patches(self.scorer(z))[:, 0]
        if self.mode == 'regional':
            region = torch.minimum(grid_occupancy(masks.to(x), x.shape[-2:]).clamp(0, 1), valid)
            background = (valid-region.sum(1, keepdim=True)).clamp_min(0)
            membership = torch.cat((region, background), 1)
            membership = membership / membership.sum(1, keepdim=True).clamp_min(1e-6)
            neighbors = self.patches(membership)
            center = membership[:, :, ::2, ::2].unsqueeze(2)
            agreement = (neighbors*center).sum(1)
            logits = logits + 2*torch.tanh(self.beta)*agreement
        logits = logits.masked_fill(occupancy <= 0, -1e4)
        weights = torch.softmax(logits, dim=1)*occupancy
        return weights/weights.sum(1, keepdim=True).clamp_min(1e-8)

    def apply_weights(self, x, weights):
        patches = self.patches(x)
        if self.mode == 'maximum':
            values = patches.masked_fill(weights[:, None] <= 0, -torch.inf).max(2).values
            return torch.where(weights.sum(1, keepdim=True) > 0, values, torch.zeros_like(values))
        return (patches*weights[:, None]).sum(2)

    def forward(self, x, masks, valid, *, return_details=False):
        weights = self.weights(x, masks, valid)
        out = self.apply_weights(x, weights)
        return (out, dict(weights=weights)) if return_details else out


class PoolDownsampleBottleneck(nn.Module):
    """Replace stride2 convolution sampling, not the learned convolution kernel.

    Both branches use the same input-derived pooling field and grid. Functional
    stride1 convolutions leave the supplied pretrained block unmodified.
    """
    def __init__(self, block, pool):
        super().__init__()
        if block.conv2.stride != (2, 2) or block.downsample is None:
            raise ValueError('Expected a ResNet50 stage-entry stride2 bottleneck.')
        self.block, self.pool = block, pool

    @staticmethod
    def dense(conv, x):
        return F.conv2d(x, conv.weight, conv.bias, stride=1, padding=conv.padding,
                        dilation=conv.dilation, groups=conv.groups)

    def forward(self, x, masks, valid, *, return_details=False):
        b = self.block
        weights = self.pool.weights(x, masks, valid)
        value = b.relu(b.bn1(b.conv1(x)))
        value = self.pool.apply_weights(self.dense(b.conv2, value), weights)
        value = b.relu(b.bn2(value))
        value = b.bn3(b.conv3(value))
        identity = self.pool.apply_weights(self.dense(b.downsample[0], x), weights)
        identity = b.downsample[1](identity)
        out = b.relu(value + identity)
        return (out, dict(weights=weights)) if return_details else out


class PoolReplacementResNet50(RegionResNet50):
    """Standalone M2 comparison: feedback, readout replacement or downsampling."""
    def __init__(self, weight_path, mode='native', classes=8):
        super().__init__(weight_path, classes=classes, mode='none')
        allowed = ('native', 'feedback', 'readout4_region', 'readout4_merged',
                   'readout24_region', 'readout24_merged', 'down_average', 'down_maximum',
                   'down_blur', 'down_content', 'down_regional')
        if mode not in allowed:
            raise ValueError(mode)
        self.pool_mode = mode
        self.poolers = nn.ModuleList()
        if mode.startswith('down_'):
            self.poolers.extend(LocalRegionPool(512, mode.removeprefix('down_')) for _ in range(2))
        if mode == 'feedback':
            self.blocks = nn.ModuleList(RegionFeedback(512, gate_mode='fixed') for _ in range(2))
        if mode.startswith('readout4_'):
            self.head = nn.Linear(2*(2048+3*2048)+2, classes)
        elif mode.startswith('readout24_'):
            self.head = nn.Linear(2*(2048+3*512)+2, classes)

    def forward_from(self, prefix, masks, valid, available, start=2, return_details=False):
        if start != 2:
            raise ValueError('This bounded replacement compares only cached layer2 entry.')
        b, p = prefix.shape[:2]
        valid = valid*available[..., None, None, None].to(valid)
        masks = masks*available[..., None, None, None].to(masks)
        x = prefix
        trace = {}
        if self.pool_mode == 'feedback':
            result = [self.blocks[pol](x[:, pol], grid_occupancy(masks[:, pol],x.shape[-2:]),
                      grid_occupancy(valid[:, pol],x.shape[-2:]),return_details=True) for pol in range(p)]
            x = torch.stack([r[0] for r in result], 1)
            trace['feedback'] = [r[1] for r in result]
        shallow = x
        layer3 = self.backbone.model.layer3
        if self.pool_mode.startswith('down_'):
            outputs = [PoolDownsampleBottleneck(layer3[0], self.poolers[pol])(
                x[:, pol], masks[:, pol], valid[:, pol], return_details=True) for pol in range(p)]
            x = torch.stack([v[0] for v in outputs], 1)
            trace['pool_weights'] = torch.stack([v[1]['weights'] for v in outputs], 1)
            z = x.flatten(0, 1)
            for block in list(layer3.children())[1:]:
                z = block(z)
        else:
            z = layer3(x.flatten(0, 1))
        z = self.backbone.model.layer4(z)
        x = z.reshape(b, p, *z.shape[1:])
        global_token = regional_mean(z, valid.flatten(0, 1), valid.flatten(0, 1))[0][:, 0]
        values = global_token.reshape(b, p, -1)
        if self.pool_mode.startswith('readout'):
            source = shallow if self.pool_mode.startswith('readout24') else x
            region = masks.flatten(0, 1)
            if self.pool_mode.endswith('merged'):
                region = region.sum(1, keepdim=True).clamp(0, 1).expand(-1, 3, -1, -1)
            pooled, mass = regional_mean(source.flatten(0, 1), region, valid.flatten(0, 1))
            values = torch.cat((values, pooled.reshape(b, p, -1)), -1)
            trace['regional_tokens'] = pooled.reshape(b, p, 3, -1)
            trace['regional_mass'] = mass.reshape(b, p, 3)
        values = values*available[..., None].to(values)
        score = self.head(torch.cat((values.flatten(1), available.to(values)), 1))
        return (score, trace) if return_details else score

    def forward(self, images, masks, valid, available, *, return_details=False):
        return self.forward_from(self.prefix(images, stop=2), masks, valid, available,
                                 return_details=return_details)
