"""Band-preserving regional readout and spectral class-response feedback."""

import torch
from torch import nn

from .backbone_attention import ClassResponseFeedback
from .pool_replacement import grid_occupancy


def _group_sizes(channels, groups):
    if channels <= 0 or groups <= 0:
        raise ValueError('channels and groups must be positive.')
    groups = min(channels, groups)
    size, extra = divmod(channels, groups)
    return [size + (i < extra) for i in range(groups)]


def _evidence(x, masks, valid, energy):
    """Accumulate preintegrated masses once, keeping empty bands at the mean."""
    b, _, h, w = x.shape
    if masks.ndim != 4 or masks.shape[0] != b or masks.shape[-2:] != (h, w):
        raise ValueError('masks must be aligned [B,R,H,W].')
    if valid.ndim != 4 or valid.shape[:2] != (b, 1):
        raise ValueError('valid must have shape [B,1,h,w].')
    if energy.shape != (b, masks.shape[1], 3, h, w):
        raise ValueError('energy must be preintegrated [B,R,3,H,W].')
    dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        occupancy = grid_occupancy(valid.to(device=x.device, dtype=dtype), (h, w)).clamp(0, 1)
        region = torch.minimum(masks.to(device=x.device, dtype=dtype).clamp(0, 1), occupancy)
        local = x.to(dtype).masked_fill(occupancy == 0, 0)
        mass = region.sum((-2, -1))
        denominator = torch.where(mass > 0, mass, torch.ones_like(mass))
        mean = torch.einsum('brhw,bchw->brc', region, local) / denominator[..., None]
        # Slot occupancy is already inside energy: do not multiply it twice.
        energy = energy.to(device=x.device, dtype=dtype).clamp_min(0)
        energy = energy.masked_fill(region[:, :, None] == 0, 0)
        energy_mass = energy.sum((-2, -1))
        energy_denominator = torch.where(energy_mass > 0, energy_mass, torch.ones_like(energy_mass))
        bands = torch.einsum('brfhw,bchw->brfc', energy, local) / energy_denominator[..., None]
        bands = torch.where(energy_mass[..., None] > 0, bands, mean[:, :, None])
    return mean, bands, mass, energy_mass, occupancy


class EvidencePool(nn.Module):
    """Read fixed regional means plus three separately retained Gabor bands.

    Inputs: x [B,C,H,W], aligned masks [B,R,H,W], valid [B,1,h,w],
    and energy [B,R,3,H,W]. Energy is the native slot times nonnegative
    original Gabor log-energy, already integrated onto the feature grid.
    Native valid masks use origin-zero ``grid_occupancy``. A missing
    polarization is a batch row with zero valid; its readouts are zero.

    Always returns a dict: mean/pooled [B,R,C], bands [B,R,3,C], mass
    [B,R], weights [B,R,G,4]. Branch order is mean, band0, band1, band2.
    Empty-energy bands fall back to mean; empty regions have zero weights.
    G=min(groups,C) contiguous groups split C as evenly as possible.

    Trainable: group branch_bias, zero-initialized gain, and (if adaptive)
    descriptor_to_logits. Seven image-only descriptors are log1p of the
    three regional energy densities, three energy fractions, and regional
    occupancy. No labels, reference similarities, or frequency-agreement
    penalties enter this selector. adaptive=False uses only static biases.

    pooled = mean + tanh(gain_g) * sum_j softmax(logits_g)[j]
    * (branch_j - mean), independently for each channel group. Thus initial
    pooled is exactly mean; gain learns on the first downstream step and
    the selector learns once gain opens. The mean remains a fixed anchor.
    return_details adds descriptor, energy_mass, gain and valid to the dict.
    """

    def __init__(self, channels, groups=8, adaptive=True):
        super().__init__()
        self.channels, self.adaptive = channels, adaptive
        self.group_sizes = _group_sizes(channels, groups)
        self.groups = len(self.group_sizes)
        self.branch_bias = nn.Parameter(torch.zeros(self.groups, 4))
        self.gain = nn.Parameter(torch.zeros(self.groups))
        self.descriptor_to_logits = nn.Linear(7, self.groups * 4, bias=False) if adaptive else None
        if adaptive:
            nn.init.normal_(self.descriptor_to_logits.weight, std=.02)

    def forward(self, x, masks, valid, energy, return_details=False):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError('Expected x with shape [B,channels,H,W].')
        mean, bands, mass, energy_mass, occupancy = _evidence(x, masks, valid, energy)
        denominator = torch.where(mass > 0, mass, torch.ones_like(mass))
        total = energy_mass.sum(-1, keepdim=True)
        fractions = energy_mass / torch.where(total > 0, total, torch.ones_like(total))
        descriptor = torch.cat(((energy_mass / denominator[..., None]).log1p(),
                                fractions, (mass / (x.shape[-2] * x.shape[-1]))[..., None]), -1)
        b, r = mass.shape
        logits = self.branch_bias.expand(b, r, -1, -1)
        if self.adaptive:
            logits = logits + self.descriptor_to_logits(
                descriptor.to(self.descriptor_to_logits.weight.dtype)).reshape(b, r, self.groups, 4)
        weights = logits.to(mean.dtype).softmax(-1) * (mass > 0)[..., None, None]
        differences = torch.cat((torch.zeros_like(mean[:, :, None]), bands - mean[:, :, None]), 2)
        updates = []
        for group, values in enumerate(differences.split(self.group_sizes, dim=-1)):
            updates.append((weights[:, :, group, :, None] * values).sum(2)
                           * self.gain[group].tanh())
        pooled = mean + torch.cat(updates, -1)
        result = dict(mean=mean.to(x.dtype), bands=bands.to(x.dtype), pooled=pooled.to(x.dtype),
                      mass=mass, weights=weights)
        if return_details:
            result.update(descriptor=descriptor, energy_mass=energy_mass,
                          gain=self.gain.tanh(), valid=occupancy)
        return result


class SpectralClassResponseFeedback(ClassResponseFeedback):
    """Retain mean/high/low class attention and select its residual channels.

    x [B,C,H,W], valid [B,1,h,w]; optional energy_field [B,3,H,W]
    contains original nonnegative Gabor log-energy grid means, without a
    region-slot multiplier. Multiply these field values by valid occupancy
    once to form global band descriptors. Absent/zero energy falls back to
    the valid mean. Bands stay separate in every channel-group MLP input:
    [mean channels, band0 channels, band1 channels, band2 channels].
    This is frequency-informed channel selection, not a DCT implementation.

    Inherited reader/value/transforms/output/gamma are trainable. Additional
    channel_selectors map each contiguous group's four descriptors through
    Linear/GELU/Linear to 2*sigmoid weights in (0,2). No frequency agreement
    is assigned a favorable or unfavorable sign. The inherited zero output
    projection ensures out == x initially, while allowing output gradients
    immediately and selector/value gradients after the first output update.

    Train aux_logits [B,classes] against source labels, excluding missing
    polarizations. The unchanged mean branch supplies these logits. Returns
    out, or (out, details): all old attention evidence remains at its original
    keys; attention_update is the old residual, channel_weights [B,C,1,1]
    modulates it, and update is exactly the final residual added to x.
    Additional mean [B,C], bands [B,3,C], energy_mass [B,3] expose the
    descriptors. Missing rows have zero evidence/update and preserve x.
    """

    def __init__(self, channels, classes=8, hidden=32, groups=8):
        super().__init__(channels, classes=classes, hidden=hidden)
        self.group_sizes = _group_sizes(channels, groups)
        self.groups = len(self.group_sizes)
        self.channel_selectors = nn.ModuleList([
            nn.Sequential(nn.Linear(4 * size, hidden), nn.GELU(), nn.Linear(hidden, size))
            for size in self.group_sizes])

    def forward(self, x, valid, energy_field=None, return_details=False):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError('Expected x with shape [B,channels,H,W].')
        b, _, h, w = x.shape
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        occupancy = grid_occupancy(valid.to(device=x.device, dtype=dtype), (h, w)).clamp(0, 1)
        if energy_field is None:
            energy_field = x.new_zeros(b, 3, h, w)
        if energy_field.shape != (b, 3, h, w):
            raise ValueError('energy_field must be aligned [B,3,H,W].')
        energy = energy_field.to(device=x.device, dtype=dtype).clamp_min(0)
        energy = energy.masked_fill(occupancy == 0, 0) * occupancy
        mean, bands, mass, energy_mass, _ = _evidence(
            x, occupancy, occupancy, energy[:, None])
        descriptors = torch.cat((mean[:, 0, None], bands[:, 0]), 1)
        selected = []
        for values, selector in zip(descriptors.split(self.group_sizes, dim=-1), self.channel_selectors):
            logits = selector(values.flatten(1).to(selector[0].weight.dtype))
            selected.append(2 * logits.sigmoid())
        channel_weights = torch.cat(selected, -1).to(x.dtype)[:, :, None, None]
        channel_weights = channel_weights * (mass > 0)[:, :, None, None]
        _, details = super().forward(x, occupancy, return_details=True)
        attention_update = details['update']
        update = attention_update * channel_weights
        out = x + update
        if return_details:
            details.update(attention_update=attention_update, update=update,
                           channel_weights=channel_weights, mean=mean[:, 0],
                           bands=bands[:, 0], energy_mass=energy_mass[:, 0])
            return out, details
        return out
