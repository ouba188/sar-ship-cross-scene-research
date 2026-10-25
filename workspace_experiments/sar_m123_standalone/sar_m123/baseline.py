"""Native axial, five-point, and separated-neighborhood baseline paths."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def axis_regions(view):
    m = view['masks']
    axis = view['ship_axis']
    place = view['placement']
    yy, xx = np.indices((160, 160))
    cx = axis['center'][0]+place['pad_left']-place['crop_x'] if axis else 0.
    cy = axis['center'][1]+place['pad_top']-place['crop_y'] if axis else 0.
    partitions = []
    for offset in (0., np.pi/2):
        if axis is None:
            partitions.append(np.zeros((3, 160, 160), np.float32))
            continue
        angle = axis['angle']+offset
        length = axis['length_px'] if offset == 0 else axis['width_px']
        c = (xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle)
        partitions.append(np.stack((m[0]*(c < -length/6), m[0]*(abs(c) <= length/6),
                                    m[0]*(c > length/6))).astype(np.float32))
    shape = [axis['length_px'], axis['width_px']] if axis else [0., 0.]
    # Scrambling preserved counts in the original control; no randomness is needed here.
    for j, slots in enumerate((m, partitions[0], partitions[1], partitions[0])):
        count = slots.sum((1, 2))
        shape.extend(count.tolist() if j == 0 else
                     [float(count[1]), float((count[0]+count[2])/2), float(abs(count[0]-count[2]))])
    return np.concatenate((m, partitions[0])), np.asarray(shape, np.float32)


def sampled(features, masks, angles, radii):
    b, c, h, w = features.shape
    along = torch.stack((angles.cos(), angles.sin()), -1)*radii[:, :1, None]
    across = torch.stack((-angles.sin(), angles.cos()), -1)*radii[:, 1:, None]
    offsets = torch.stack((torch.zeros_like(along), along, -along, across, -across), -2)
    y, x = torch.meshgrid(torch.arange(h, device=features.device),
                          torch.arange(w, device=features.device), indexing='ij')
    centers = torch.stack((x, y), -1).to(features.dtype)
    grid = centers[None, :, :, None, None]+offsets[:, None, None]
    grid = 2*(grid+.5)/features.new_tensor((w, h))-1
    t = angles.shape[1]
    grid = grid[:, None].expand(-1, 3, -1, -1, -1, -1, -1).reshape(b*3, h, w*t*5, 2)
    source = torch.cat((features[:, None]*masks[:, :, None], masks[:, :, None]), 2)
    values = F.grid_sample(source.reshape(b*3, c+1, h, w), grid,
                           mode='bilinear', padding_mode='zeros', align_corners=False)
    values = values.reshape(b, 3, c+1, h, w, t, 5)
    return values[:, :, :c], values[:, :, c]


def describe(numerator, mass):
    nm, mm = numerator[:, 1:].sum(1, keepdim=True), mass[:, 1:].sum(1, keepdim=True)
    numerator = torch.cat((numerator[:, :1], nm, nm), 1)
    mass = torch.cat((mass[:, :1], mm, mm), 1)
    v = numerator/mass[:, :, None].clamp_min(1e-8)
    center = mass[..., 0]
    wa, wc = center*torch.minimum(mass[..., 1], mass[..., 2]), center*torch.minimum(mass[..., 3], mass[..., 4])
    values = []
    for stat, weight in zip((v[..., 0], (v[..., 1]+v[..., 2])/2,
                            (v[..., 3]+v[..., 4])/2, abs(v[..., 1]-v[..., 2]),
                            abs(v[..., 3]-v[..., 4])), (center, wa, wc, wa, wc)):
        pooled = (stat*weight[:, :, None]).sum((3, 4))/weight.sum((2, 3))[:, :, None].clamp_min(1e-8)
        values.append(pooled.permute(0, 1, 3, 2))
    return torch.stack(values, -2)


@torch.inference_mode()
def extract(prepared, maps, backbone):
    pooled = np.zeros((2, 6, 64), np.float32)
    shapes = np.zeros((2, 14), np.float32)
    five = np.zeros((2, 3, 2, 5, 64), np.float32)
    centered, levels, usable = np.zeros((2, 128), np.float32), np.zeros(2, np.float32), np.zeros(2, bool)
    device = backbone.device
    for p, view in enumerate(prepared):
        if view is None:
            continue
        mask, shapes[p] = axis_regions(view)
        m = torch.tensor(mask[None], device=device)
        original = torch.tensor(view['masks'][None], device=device)
        axis = view['ship_axis']
        radius = [np.clip(axis['length_px']/32, .75, 3), np.clip(axis['width_px']/32, .5, 2)] if axis else [1., 1.]
        radii = torch.tensor([radius], dtype=torch.float32, device=device)
        ship_angle = torch.tensor(axis['angle'] if axis else np.pi/4, dtype=torch.float32, device=device)
        radar_angle = torch.tensor(view['radar_axis'] or 0., dtype=torch.float32, device=device)
        pp, ff = [], []
        for k in range(4):
            features = F.avg_pool2d(maps[p]['stem'][k:k+1], 2)
            weight = F.avg_pool2d(m.rot90(k, (-2, -1)), 4)
            pp.append(torch.einsum('bchw,brhw->brc', features, weight)/
                      weight.sum((2, 3)).clamp_min(1e-8)[:, :, None])
            ship = ship_angle-k*np.pi/2 if axis else ship_angle
            radar = radar_angle-k*np.pi/2 if view['radar_axis'] is not None else radar_angle
            angles = torch.stack((ship, radar))[None]
            num, mass = sampled(features, F.avg_pool2d(original.rot90(k, (-2, -1)), 4), angles, radii)
            ff.append(describe(num, mass))
        pooled[p] = torch.stack(pp).mean(0)[0].cpu().numpy()
        five[p] = torch.stack(ff).mean(0)[0].cpu().numpy()
        usable[p] = bool(view['masks'][0].any() and view['masks'][2].any())
        if usable[p]:
            x = torch.tensor(view['pixels'].copy(), device=device).float()[None, None]/255
            region = original[:, 2:3]
            mu = (x*region).sum((2, 3), keepdim=True)/region.sum((2, 3), keepdim=True).clamp_min(1)
            encoded = .5+.5*(x-mu)*region
            result = 0
            for k in range(4):
                f = F.avg_pool2d(backbone.stem(encoded.rot90(k, (-2, -1))), 2)
                weight = F.avg_pool2d(region.rot90(k, (-2, -1)), 4)
                avg = (f*weight).sum((2, 3))/weight.sum((2, 3)).clamp_min(1e-8)
                maximum = f.masked_fill(weight <= 0, -torch.inf).amax((2, 3))
                maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
                result = result+torch.cat((avg, maximum), 1)/4
            centered[p] = result[0].cpu().numpy()
            levels[p] = float(mu[0, 0, 0, 0])
    base = np.concatenate((pooled[:, :3].reshape(-1), shapes.reshape(-1)))
    slots = pooled[:, 3:6]
    axis = np.concatenate((slots[:, 1], (slots[:, 0]+slots[:, 2])/2,
                           abs(slots[:, 0]-slots[:, 2])), -1).reshape(-1)
    return dict(axis=np.concatenate((base, axis)), five=five.reshape(60, 64),
                centered=centered, means=levels, usable=usable)


class Five(nn.Module):
    def __init__(self):
        super().__init__()
        self.content, self.output = nn.Linear(64, 8), nn.Linear(480, 8)

    def forward(self, x):
        return self.output(F.gelu(self.content(x)).flatten(1))


class Neighborhood(nn.Module):
    def __init__(self):
        super().__init__()
        self.content, self.output = nn.Linear(260, 32), nn.Linear(32, 8)

    def forward(self, x):
        return self.output(F.gelu(self.content(x)))


class Calibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))

    def forward(self, x):
        return self.net(x)


class Baseline:
    def __init__(self, assets, seed):
        root = Path(assets)/'upstream/models'
        self.ridge = dict(np.load(root/'axis_ridge.npz'))
        self.neighborhood_scaler = dict(np.load(root/'neighborhood_scaler.npz'))
        self.heads, self.checkpoints = {}, {}
        for name, model in (('five', Five()), ('neighborhood', Neighborhood()), ('calibration', Calibration())):
            ck = torch.load(root/f'{name}_seed{seed}.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(ck['state'])
            self.heads[name] = model.eval().requires_grad_(False)
            self.checkpoints[name] = ck

    @torch.inference_mode()
    def __call__(self, features):
        z = self.ridge
        raw = features['axis'][None]
        ridge = ((raw-z['mean'])/z['std']/np.sqrt(raw.shape[1]))@z['coef'].T+z['intercept']
        ck = self.checkpoints['five']
        x = ((features['five'][None]-ck['mu'])/ck['scale']).astype(np.float32)
        a = ridge+self.heads['five'](torch.tensor(x)).numpy().astype(float)
        z = self.neighborhood_scaler
        usable = features['usable']
        f = ((features['centered']-z['mean'])/z['std']*usable[:, None]).astype(np.float32)
        level = ((features['means']-z['level_mean'])/z['level_std']*usable).astype(np.float32)
        x = np.concatenate((f.reshape(-1), level, usable.astype(np.float32)))[None]
        u = a+self.heads['neighborhood'](torch.tensor(x)).numpy().astype(float)
        ck = self.checkpoints['calibration']
        x = ((u-u.mean(1, keepdims=True)-ck['mu'])/ck['scale']).astype(np.float32)
        b = u+self.heads['calibration'](torch.tensor(x)).numpy().astype(float)
        return b[0], dict(axis=ridge[0], A=a[0], U=u[0])
