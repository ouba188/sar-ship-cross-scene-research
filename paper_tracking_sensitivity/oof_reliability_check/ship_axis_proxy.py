"""Label-free central bright-component axis; a proxy, not a true vessel heading."""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing, label
from pyproj import Transformer, Geod

GEOD = Geod(ellps='WGS84')


def axial_difference(a, b):
    return np.abs((np.asarray(a)-np.asarray(b)+90.) % 180.-90.)


def measure_axis(image, center=None):
    image = np.asarray(image, dtype=float)
    h, w = image.shape
    cx, cy = center if center is not None else ((w-1)/2., (h-1)/2.)
    yy, xx = np.mgrid[:h, :w]
    roi = (np.abs(xx-cx) <= w/4.) & (np.abs(yy-cy) <= h/4.)
    smooth = gaussian_filter(image, .8)
    median, upper, peak = np.quantile(smooth[roi], [.5, .75, .99])
    # A quantile alone selects negligible long tails on nearly black water.
    threshold = max(upper, median+.15*(peak-median))
    mask = binary_closing((smooth > threshold) & roi, structure=np.ones((3, 3))) & roi
    components, count = label(mask)
    candidates = []
    for k in range(1, count+1):
        y, x = np.nonzero(components == k)
        if len(x) >= 8:
            candidates.append((float(np.min((x-cx)**2+(y-cy)**2)), -len(x), k))
    result = dict(status='no_component', accepted=False, angle_pixel_deg=np.nan,
                  elongation=np.nan, centroid_offset=np.nan, pixels=0, cx=cx, cy=cy)
    if not candidates:
        return result
    _, _, k = min(candidates)
    y, x = np.nonzero(components == k)
    weights = np.maximum(smooth[y, x]-threshold, 1e-6)
    weights /= weights.sum()
    mx, my = np.sum(weights*x), np.sum(weights*y)
    xy = np.column_stack((x-mx, y-my))
    values, vectors = np.linalg.eigh(xy.T @ (weights[:, None]*xy))
    angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1])) % 180.
    elongation = np.sqrt(values[-1]/max(values[0], 1e-6))
    offset = float(np.hypot(mx-cx, my-cy))
    boundary = np.any((np.abs(x-cx) >= w/4.-1) | (np.abs(y-cy) >= h/4.-1))
    accepted = elongation >= 2. and offset <= min(w, h)*.1 and not boundary
    result.update(status='accepted' if accepted else 'weak_or_cluttered', accepted=bool(accepted),
                  angle_pixel_deg=float(angle), elongation=float(elongation), centroid_offset=offset,
                  pixels=len(x), cx=float(mx), cy=float(my), touches_roi_boundary=bool(boundary))
    return result


def world_axis(geo, px, py, angle_pixel_deg):
    gt = np.asarray(geo['transform'], dtype=float)
    origin = np.array([gt[0]+px*gt[1]+py*gt[2], gt[3]+px*gt[4]+py*gt[5]])
    vector = np.array([[gt[1], gt[2]], [gt[4], gt[5]]]) @ np.array(
        [np.cos(np.deg2rad(angle_pixel_deg)), np.sin(np.deg2rad(angle_pixel_deg))])
    vector = vector/np.linalg.norm(vector)*100.
    transformer = Transformer.from_crs(int(geo['epsg']), 4326, always_xy=True)
    lon, lat = transformer.transform(*origin)
    lon2, lat2 = transformer.transform(*(origin+vector))
    azimuth, _, _ = GEOD.inv(lon, lat, lon2, lat2)
    return dict(longitude=float(lon), latitude=float(lat), axis_true_north_deg=float(azimuth % 180.))


def combine_axes(a, b):
    difference = float(axial_difference(a, b))
    z = np.exp(2j*np.deg2rad(a))+np.exp(2j*np.deg2rad(b))
    return float((np.rad2deg(np.angle(z))/2.) % 180.), difference
