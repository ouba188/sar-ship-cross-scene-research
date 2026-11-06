"""Mean, low-rank second-order, and class-competitive context feedback."""
import torch
from torch import nn


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
