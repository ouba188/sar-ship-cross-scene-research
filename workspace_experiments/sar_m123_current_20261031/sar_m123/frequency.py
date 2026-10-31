"""Native-pixel Gabor statistics used by the selected M3, not SLC frequency.

Filtering precedes crop/pad. Joint moments average directions before log;
gate energy averages the direction-wise log of pooled energies instead.
Those two descriptors are intentionally not interchangeable.
"""
from functools import lru_cache

import numpy as np
from scipy.signal import fftconvolve

WAVELENGTHS = (4., 8., 16.)
ANGLES = np.arange(4) * np.pi / 4
RADII = np.ceil(3 * np.asarray(WAVELENGTHS) / 2).astype(int)


@lru_cache(maxsize=12)
def kernel(wavelength, theta):
    sigma = wavelength / 2
    radius = int(np.ceil(3 * sigma))
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    envelope = np.exp(-(xx*xx + yy*yy) / (2*sigma*sigma))
    carrier = np.exp(2j*np.pi*(xx*np.cos(theta) + yy*np.sin(theta))/wavelength)
    dc = (envelope*carrier).sum() / envelope.sum()
    result = envelope * (carrier-dc)
    return result / np.sqrt(np.sum(np.abs(result)**2))


def response(raw, psi):
    radius = psi.shape[0] // 2
    filtered = fftconvolve(np.pad(raw, radius, mode='reflect'), psi, mode='same')
    return filtered[radius:radius+raw.shape[0], radius:radius+raw.shape[1]]


def place(field, placement):
    x, y = placement['crop_x'], placement['crop_y']
    w, h = placement['crop_width'], placement['crop_height']
    dx, dy = placement['pad_left'], placement['pad_top']
    canvas = np.zeros((*field.shape[:-2], 160, 160), dtype=field.dtype)
    canvas[..., dy:dy+h, dx:dx+w] = field[..., y:y+h, x:x+w]
    return canvas


def mask_coordinates(masks, placement, shape):
    owner, cy, cx = np.nonzero(masks)
    weights = masks[owner, cy, cx]
    yy = cy-placement['pad_top']+placement['crop_y']
    xx = cx-placement['pad_left']+placement['crop_x']
    if not ((yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])).all():
        raise ValueError('Token mask includes pixels outside the original chip footprint')
    mass = np.bincount(owner, weights=weights, minlength=20)
    return owner, yy, xx, weights, mass


def pool(field, coordinates):
    owner, yy, xx, weights, mass = coordinates
    total = np.bincount(owner, weights=weights*field[yy, xx], minlength=20)
    return np.divide(total, mass, out=np.zeros(20), where=mass > 0)


def extract(raw, masks, placement):
    """Return native energy fields, image-frame descriptors and boundary coverage.

    raw is the full original uint8 grayscale chip, before 160-pixel crop/pad.
    The ship-frame experimental descriptor is not used by selected pair_joint.
    """
    raw = np.asarray(raw, dtype=np.float64) / 255
    coordinates = mask_coordinates(masks, placement, raw.shape)
    _, yy, xx, _, mass = coordinates
    energy = np.zeros((20, 12))
    fields = []
    coverage = np.zeros((20, 3))
    for b, wavelength in enumerate(WAVELENGTHS):
        directional = []
        for a, theta in enumerate(ANGLES):
            power = np.abs(response(raw, kernel(wavelength, float(theta))))**2
            directional.append(power)
            energy[:, 4*b+a] = pool(power, coordinates)
        fields.append(np.log1p(np.mean(directional, axis=0)/1e-4))
        radius = RADII[b]
        interior = ((xx >= radius) & (xx < raw.shape[1]-radius)
                    & (yy >= radius) & (yy < raw.shape[0]-radius))
        owner, _, _, weights, _ = coordinates
        total = np.bincount(owner, weights=weights*interior, minlength=20)
        coverage[:, b] = np.divide(total, mass, out=np.zeros(20), where=mass > 0)
    descriptor = np.concatenate((np.log1p(energy/1e-4),
                                energy/(energy.sum(-1, keepdims=True)+1e-8)), -1)
    descriptor[mass <= 0] = 0
    return dict(energy_field=place(np.stack(fields), placement),
                e_image=descriptor.astype(np.float32),
                coverage=np.clip(coverage, 0, 1).astype(np.float32))
