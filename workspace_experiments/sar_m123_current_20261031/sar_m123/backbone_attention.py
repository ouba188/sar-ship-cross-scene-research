"""Trainable class-response pooling and spatial feedback for internal M1."""

import torch
from torch import nn
from torch.nn import functional as F


class ClassResponseFeedback(nn.Module):
    """Read mean/high/low class tokens and write them back to local features.

    Train ``details['aux_logits']`` with source class labels to give the reader
    semantic class axes. Missing images use an all-zero valid mask; omit them
    from the auxiliary loss. Each backbone insertion owns a separate instance.

    Details: weights/write_weights [B,3,K,H,W], responses [B,3,K],
    tokens [B,3,K,D], aux_logits [B,K], update [B,C,H,W]. Branch order is
    mean, high, low. Responses are mu, qp, qm in m12.attention notation;
    its packed representation is [mu, (qp-qm)/2, (qp+qm)/2]. The reported
    update includes gamma and is exactly the residual added to x.
    """

    def __init__(self, channels, classes=8, hidden=32, temperature=1.):
        super().__init__()
        if temperature <= 0:
            raise ValueError('Feedback temperature must be positive.')
        self.channels, self.classes = channels, classes
        self.temperature = float(temperature)
        self.reader = nn.Conv2d(channels, classes, 1, bias=False)
        self.value = nn.Conv2d(channels, hidden, 1, bias=False)
        self.transforms = nn.ModuleList([
            nn.Linear(hidden, hidden, bias=False) for _ in range(3)])
        self.output = nn.Conv2d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.gamma = nn.Parameter(torch.tensor(.1))

    def forward(self, x, valid, return_details=False):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError('Expected x with shape [B,channels,H,W].')
        if valid.ndim != 4 or valid.shape[:2] != (x.shape[0], 1):
            raise ValueError('Expected valid with shape [B,1,h,w].')
        b, _, h, w = x.shape
        # Accumulate occupancy and softmax in float32 under mixed precision.
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        occupancy = F.interpolate(valid.to(device=x.device, dtype=dtype),
                                  size=(h, w), mode='area').clamp(0, 1)
        mass = occupancy.flatten(2)
        total = mass.sum(-1, keepdim=True)
        norm = mass / torch.where(total > 0, total, torch.ones_like(total))
        local = x.masked_fill(occupancy == 0, 0)
        response = self.reader(local).flatten(2).to(dtype)
        mu = (response * norm).sum(-1)
        centered = response - mu[..., None]
        centered = centered - centered.mean(1, keepdim=True)
        centered = centered - (centered * norm).sum(-1, keepdim=True)

        # Empty rows get finite softmax inputs, then zero mass and gradients.
        present = norm > 0
        log_norm = torch.where(present, norm, torch.ones_like(norm)).log()
        logits = torch.stack((log_norm + centered / self.temperature,
                              log_norm - centered / self.temperature), 1)
        logits = logits.masked_fill(~present[:, None], -torch.inf)
        logits = torch.where((total > 0)[:, None], logits, torch.zeros_like(logits))
        sides = logits.softmax(-1) * present[:, None]
        weights = torch.cat((norm[:, None].expand(-1, 1, self.classes, -1), sides), 1)
        responses = torch.cat((mu[:, None], (sides * centered[:, None]).sum(-1)), 1)

        value = self.value(local).flatten(2)
        tokens = torch.einsum('bsku,bdu->bskd', weights.to(value.dtype), value)
        transformed = torch.stack([layer(tokens[:, i])
                                   for i, layer in enumerate(self.transforms)], 1)
        # Normalize across class AND branch at each position, avoiding 1/HW
        # attenuation. Fractional occupancy is applied once at the write site.
        write_total = weights.sum((1, 2), keepdim=True)
        write_weights = weights / torch.where(write_total > 0, write_total,
                                              torch.ones_like(write_total))
        write_weights = write_weights * mass[:, None]
        mixed = torch.einsum('bsku,bskd->bdu', write_weights.to(transformed.dtype),
                             transformed).reshape(b, -1, h, w)
        update = self.gamma * self.output(mixed)
        out = x + update
        if return_details:
            return out, dict(aux_logits=mu, weights=weights.reshape(b, 3, self.classes, h, w),
                             write_weights=write_weights.reshape(b, 3, self.classes, h, w),
                             responses=responses, tokens=tokens,
                             class_response=response.reshape(b, self.classes, h, w),
                             centered_response=centered.reshape(b, self.classes, h, w),
                             valid=occupancy, update=update)
        return out
