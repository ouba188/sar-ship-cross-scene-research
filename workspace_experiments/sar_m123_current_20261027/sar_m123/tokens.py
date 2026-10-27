"""Frozen C4 local readout: 8 body, 8 side and 4 axis-end tokens."""
import numpy as np
import torch

from . import frequency

REGIONS = (np.arange(8), np.arange(8, 16), np.arange(16, 20))
YY, XX = np.mgrid[:160, :160]


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def soft_regions(g):
    """Original soft OBB regions, order body/end/side; not input T/O/R masks."""
    if g is None:
        return np.zeros((3, 160, 160), np.float64)
    cx, cy, length, width, angle = g[:5]
    u = (XX-cx)*np.cos(angle)+(YY-cy)*np.sin(angle)
    v = -(XX-cx)*np.sin(angle)+(YY-cy)*np.cos(angle)

    def rectangle(l, w):
        return np.clip(l/2+.5-np.abs(u), 0, 1)*np.clip(w/2+.5-np.abs(v), 0, 1)

    body = rectangle(length, width)
    return np.stack((body, np.maximum(rectangle(1.5*length, width)-body, 0),
                     np.maximum(rectangle(length, 2*width)-body, 0)))


def slots(g, footprint):
    if g is None:
        return np.zeros((20, 160, 160), np.float64)
    cx, cy, length, width, angle = g[:5]
    u = ((XX-cx)*np.cos(angle)+(YY-cy)*np.sin(angle))/length
    v = (-(XX-cx)*np.sin(angle)+(YY-cy)*np.cos(angle))/width
    region = soft_regions(g)*footprint
    ix = np.clip(np.floor((u+.5)*4).astype(int), 0, 3)
    iy, end = (v >= 0).astype(int), (u >= 0).astype(int)
    masks = [region[0]*(ix == a)*(iy == b) for a in range(4) for b in range(2)]
    masks += [region[2]*(ix == a)*(iy == b) for a in range(4) for b in range(2)]
    masks += [region[1]*(end == a)*(iy == b) for a in range(2) for b in range(2)]
    return np.stack(masks)


def cell_mass(mask, stride=4):
    """Pixel-area integral over true center-Voronoi cells (origin zero)."""
    edges = np.r_[-.5, np.arange(stride/2, 160-stride/2, stride), 159.5]
    points = edges+.5
    idx = np.minimum(np.floor(points).astype(int), 159)
    frac = points-idx

    def integral(a):
        prefix = np.concatenate([np.zeros((*a.shape[:-1], 1)), np.cumsum(a, axis=-1)], -1)
        return np.diff(prefix[..., idx]+a[..., idx]*frac, axis=-1)

    return integral(integral(mask).swapaxes(-1, -2)).swapaxes(-1, -2)


def grid_moments(features, weights, energy_weights):
    mass = weights.sum(-1)
    den = np.maximum(mass, 1e-20)
    mean = (weights@features)/den[:, None]
    energy_mean = energy_weights.sum(-1)/den[:, None]
    cross = (energy_weights@features)/den[:, None, None]
    joint = cross-energy_mean[..., None]*mean[:, None]
    mean[mass <= 0] = 0
    joint[mass <= 0] = 0
    return joint, mean


@torch.inference_mode()
def extract(prepared, maps, projection):
    """Extract one chip pair, using caller-owned frozen R50 maps.

    prepared/maps are VV,VH sequences. Each present map has C4 first,
    layer1=(4,256,40,40), optionally stem=(4,64,80,80). Geometry coordinates
    refer to the 160 canvas. Projection is fixed mean256/components32x256.
    No labels, observation IDs, query cache or fitting is used here.
    """
    result = dict(layer1=np.zeros((2, 20, 256), np.float32),
                  stem=np.zeros((2, 20, 64), np.float32),
                  mass=np.zeros((2, 20), np.float32), valid=np.zeros(2, bool),
                  joint=np.zeros((2, 20, 96), np.float32),
                  mean=np.zeros((2, 20, 32), np.float32),
                  e_image=np.zeros((2, 20, 24), np.float32),
                  coverage=np.zeros((2, 20, 3), np.float32))
    for p, item in enumerate(prepared):
        if item is None or item['geometry'] is None:
            continue
        mask = slots(item['geometry'], item['footprint'])
        result['mass'][p] = mask.sum((-1, -2))
        result['valid'][p] = True
        spectral = frequency.extract(item['raw'], mask, item['placement'])
        result['e_image'][p] = spectral['e_image']
        result['coverage'][p] = spectral['coverage']
        joint, mean = np.zeros((20, 3, 32)), np.zeros((20, 32))
        layer = maps[p]['layer1']
        device = layer.device
        center = torch.as_tensor(projection['mean'], device=device)[None, :]
        components = torch.as_tensor(projection['components'].T.copy(), device=device)
        for k in range(4):
            rotated = np.rot90(mask, k, (-2, -1))
            weights = cell_mass(rotated, 4).reshape(20, -1)
            z = layer[k].flatten(1)
            norm = torch.tensor(weights/np.maximum(weights.sum(-1, keepdims=True), 1e-20),
                                dtype=torch.float32, device=device)
            result['layer1'][p] += (norm@z.T).cpu().numpy()/4
            if 'stem' in maps[p]:
                ws = cell_mass(rotated, 2).reshape(20, -1)
                ns = torch.tensor(ws/np.maximum(ws.sum(-1, keepdims=True), 1e-20),
                                  dtype=torch.float32, device=device)
                result['stem'][p] += (ns@maps[p]['stem'][k].flatten(1).T).cpu().numpy()/4
            projected = ((z.T-center)@components).cpu().numpy().astype(np.float64)
            energy = np.rot90(spectral['energy_field'], k, (-2, -1))
            ew = cell_mass(rotated[:, None]*energy[None], 4).reshape(20, 3, -1)
            jj, mm = grid_moments(projected, weights, ew)
            joint += jj/4
            mean += mm/4
        result['joint'][p] = joint.reshape(20, 96)
        result['mean'][p] = mean
    projected = (result['layer1']-projection['mean'])@projection['components'].T
    result['x'] = unit(projected).astype(np.float32)
    result['x'][result['mass'] <= 0] = 0
    return result


def body_descriptor(x, mass):
    body = (x[..., :8, :]*mass[..., :8, None]).sum(-2)
    body /= np.maximum(mass[..., :8].sum(-1)[..., None], 1e-8)
    return unit(body).reshape(*x.shape[:-3], -1)


def region_pool(x, mass):
    return np.stack([(x[:, :, r]*mass[:, :, r, None]).sum(2)
                     / np.maximum(mass[:, :, r].sum(2)[..., None], 1e-8)
                     for r in REGIONS], 2)
