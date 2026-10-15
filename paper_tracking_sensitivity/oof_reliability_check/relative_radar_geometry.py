"""Exact-product grid ground-range azimuth approximation, NOT beam azimuth.

Angles are clockwise from true north, in degrees. A same-line increasing-pixel
chord is expressed in the target's local WGS84 east/north frame. This estimates
the GRD range axis; it does not recover the satellite-to-target beam vector.

Source evidence: GDAL's frmts/safe/safedataset.cpp maps annotation pixel/line
to GCP pixel/line and longitude/latitude to GCP X/Y:
https://github.com/OSGeo/gdal/blob/master/frmts/safe/safedataset.cpp
Independently read on 2026-10-09: Sentinel-1 Product Specification,
S1-RS-MDA-52-7441, 3.16.1, PDF pp. 185-186, Tables 6-90/91/92: GRD grid
rows are spread in azimuth, points across range, line/pixel are zero-based
image indices and geographic values refer to pixel centers. This supports
the fixed-line increasing-pixel range-axis interpretation (not exact beam).
https://sentiwiki.copernicus.eu/__attachments/a_19b24d9a39965738b9030586e95d8cae0fe3a26d0c571e5761848cef8ac2f9b4/S1-RS-MDA-52-7441%20-%20Sentinel-1%20Product%20Specification%202025%20-%203.16.1.pdf
The header projection, when available in the tail, is checked as well.
No orbit-heading constants or representative SAFE annotations are used.
"""

import json
from http.client import IncompleteRead
import math
from pathlib import Path
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

from pyproj import Geod


BASE = 'https://sentinel-s1-l1c.s3.amazonaws.com/'
TAIL_BYTES = 900_000
MAX_NEAREST_GRID_DISTANCE_M = 20_000.
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / 'relative_observation_geometry_20261009/radar_grid'
GEOD = Geod(ellps='WGS84')
FIELDS = ('line', 'pixel', 'latitude', 'longitude', 'incidenceAngle')
METHOD = 'grid_ground_range_azimuth_approximation'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _product_url(product):
    match = re.fullmatch(
        r'(S1[ABC])_(IW|EW|SM)_GRD[HFM]_(1S[DS][VH])_'
        r'(\d{8}T\d{6})_(\d{8}T\d{6})_\d{6}_[0-9A-F]{6}_[0-9A-F]{4}', product)
    if not match:
        raise ValueError('Expected a Sentinel-1 GRD product identifier (without .SAFE)')
    mission, mode, pol, start, stop = match.groups()
    date = start[:8]
    # Keep the existing fetch_s1_grid.py AWS layout (including its DV folder).
    channel = 'vv' if pol.endswith('V') else 'hh'
    key = (f'GRD/{date[:4]}/{int(date[4:6])}/{int(date[6:8])}/{mode}/DV/'
           f'{product}/annotation/{mode.lower()}-{channel}.xml')
    return BASE + key, (mission, start, stop)


def _block(data, tag):
    start = re.search(rb'<' + tag.encode() + rb'(?:\s[^>]*)?>', data)
    end_tag = b'</' + tag.encode() + b'>'
    end = data.find(end_tag, start.end()) if start else -1
    if start is None or end < 0:
        raise ValueError(f'Complete {tag} block unavailable in bounded tail')
    return ET.fromstring(data[start.start():end + len(end_tag)])


def _validate(points):
    if not points:
        raise ValueError('empty_grid')
    indices, coordinates = set(), set()
    for point in points:
        if not isinstance(point, dict) or any(k not in point for k in FIELDS):
            raise ValueError('missing_grid_fields')
        if any(not isinstance(point[k], (int, float)) or not math.isfinite(point[k]) for k in FIELDS):
            raise ValueError('nonnumeric_grid_fields')
        if any(point[k] < 0 or int(point[k]) != point[k] for k in ('line', 'pixel')):
            raise ValueError('invalid_line_or_pixel')
        if not (-180 <= point['longitude'] <= 180 and -90 <= point['latitude'] <= 90
                and 0 <= point['incidenceAngle'] <= 90):
            raise ValueError('invalid_geographic_coordinate_or_incidence')
        index = point['line'], point['pixel']
        coordinate = point['longitude'], point['latitude']
        if index in indices:
            raise ValueError('ambiguous_line_pixel')
        if coordinate in coordinates:
            raise ValueError('ambiguous_geographic_coordinates')
        indices.add(index)
        coordinates.add(coordinate)


def _parse_tail(data, identity):
    grid = _block(data, 'geolocationGrid')
    listing = grid.find('geolocationGridPointList')
    if listing is None:
        raise ValueError('missing_grid_point_list')
    points = []
    for element in listing.findall('geolocationGridPoint'):
        point = {}
        for field in FIELDS:
            value = element.findtext(field)
            if value is None:
                raise ValueError(f'missing_grid_field:{field}')
            point[field] = int(value) if field in ('line', 'pixel') else float(value)
        points.append(point)
    if 'count' in listing.attrib and int(listing.attrib['count']) != len(points):
        raise ValueError('grid_point_count_mismatch')
    _validate(points)
    flags = []
    if b'<adsHeader' in data:
        header = _block(data, 'adsHeader')
        mission, start, stop = identity
        expected = {'missionId': mission, 'productType': 'GRD',
                    'startTime': start, 'stopTime': stop}
        for key, value in expected.items():
            actual = header.findtext(key)
            if actual is None:
                raise ValueError(f'missing_header_field:{key}')
            if key.endswith('Time'):
                actual = actual[:19].replace('-', '').replace(':', '')
            if actual != value:
                raise ValueError(f'product_header_mismatch:{key}')
        flags.append('product_header_checked')
    else:
        flags.append('product_header_unavailable_in_tail')
    projection = re.search(rb'<projection>([^<]+)</projection>', data)
    if projection and projection.group(1).strip() != b'Ground Range':
        raise ValueError('not_ground_range_projection')
    return points, flags


def fetch_grid(product, cache_dir=DEFAULT_CACHE):
    """Return {status, product, points, quality_flags, ...}; cache successes only.

    Points are numeric dicts with line/pixel (integers), latitude/longitude
    (degrees), incidenceAngle (degrees). One direct urllib suffix request, at
    most 900000 body bytes, 30-second timeout; no full-download fallback.
    Status is ok, unavailable, or invalid_annotation. Bad product IDs raise
    ValueError. Successful JSON includes source_url, bytes_read, content_range.
    """
    url, identity = _product_url(product)
    destination = Path(cache_dir) / f'{product}.json'
    if destination.exists():
        cached = json.loads(destination.read_text(encoding='utf-8'))
        if cached.get('product') != product or cached.get('status') != 'ok':
            raise ValueError('cache_product_or_status_mismatch')
        _validate(cached['points'])
        return cached
    result = dict(product=product, source_url=url, points=[], quality_flags=[])
    request = urllib.request.Request(url, headers={'Range': f'bytes=-{TAIL_BYTES}'})
    try:
        with OPENER.open(request, timeout=30) as response:
            result['http_status'] = response.status
            result['content_range'] = response.headers.get('Content-Range')
            length = response.headers.get('Content-Length')
            if response.status == 200 and (length is None or int(length) > TAIL_BYTES):
                return dict(result, status='unavailable', error='server_ignored_bounded_range')
            data = response.read(TAIL_BYTES)
            result['bytes_read'] = len(data)
    except (urllib.error.URLError, TimeoutError, OSError, IncompleteRead) as error:
        return dict(result, status='unavailable', error=str(error))
    try:
        points, flags = _parse_tail(data, identity)
    except (ValueError, ET.ParseError) as error:
        return dict(result, status='invalid_annotation', error=str(error))
    result.update(status='ok', points=points, quality_flags=flags)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def _inside_footprint(rows):
    """Test the target (local origin) against the ordered grid perimeter."""
    boundary = (rows[0] + [row[-1] for row in rows[1:]]
                + list(reversed(rows[-1][:-1]))
                + [row[0] for row in reversed(rows[1:-1])])
    inside = False
    for a, b in zip(boundary, boundary[1:] + boundary[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        if length2:
            t = min(1., max(0., -(a[0] * dx + a[1] * dy) / length2))
            if math.hypot(a[0] + t * dx, a[1] + t * dy) <= 1e-6:
                return True
        if (a[1] > 0) != (b[1] > 0):
            crossing_x = a[0] - a[1] * dx / dy
            if crossing_x > 0:
                inside = not inside
    return inside


def range_azimuth(grid, longitude, latitude):
    """Estimate increasing-pixel azimuth at (longitude, latitude), in degrees.

    Accept a fetch_grid result or its points list. Return status, azimuth_deg
    ([0,360)), axial_azimuth_deg ([0,180)), nearest_grid_distance_m, method,
    quality_flags, and chosen line/pixel_pair/segment_length_m. Invalid grids
    return null angles; invalid query coordinates raise ValueError. Axial
    angles discard direction and can be compared with an unoriented ship axis.

    Choose the nearest grid node's row, then its closest adjacent segment in
    the target's geodesic east/north plane. No interpolation across rows.
    Reject targets outside the grid perimeter or more than 20 km from a node
    with status outside_grid and null angles. At least two complete rows are
    required to bound the footprint. The polygon is itself a grid approximation;
    flags are diagnostics, not a calibrated angular uncertainty.
    """
    if not (math.isfinite(longitude) and math.isfinite(latitude)
            and -180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise ValueError('Query must be finite longitude, latitude in degrees')
    result = dict(status='invalid_grid', azimuth_deg=None, axial_azimuth_deg=None,
                  nearest_grid_distance_m=None, method=METHOD,
                  quality_flags=['grid_ground_range_approximation_not_exact_beam'])
    if isinstance(grid, dict):
        if grid.get('status') != 'ok':
            result['quality_flags'].append('grid_unavailable')
            return result
        result['quality_flags'].extend(grid.get('quality_flags', []))
        points = grid['points']
    else:
        points = grid
    try:
        _validate(points)
    except ValueError as error:
        result['quality_flags'].append(str(error))
        return result
    local = []
    for point in points:
        bearing, _, distance = GEOD.inv(longitude, latitude, point['longitude'], point['latitude'])
        angle = math.radians(bearing)
        local.append((distance * math.sin(angle), distance * math.cos(angle), distance, point))
    nearest = min(local, key=lambda node: node[2])
    result['nearest_grid_distance_m'] = nearest[2]
    line = nearest[3]['line']
    row = sorted((node for node in local if node[3]['line'] == line), key=lambda node: node[3]['pixel'])
    if len(row) < 2:
        result['quality_flags'].append('nearest_row_has_no_pixel_neighbor')
        return result
    lines = sorted({p['line'] for p in points})
    rows = [sorted((node for node in local if node[3]['line'] == value),
                   key=lambda node: node[3]['pixel']) for value in lines]
    if len(rows) < 2:
        result['quality_flags'].append('insufficient_rows_for_footprint')
        return result
    if any([node[3]['pixel'] for node in other] != [node[3]['pixel'] for node in rows[0]]
           for other in rows[1:]):
        result['quality_flags'].append('incomplete_grid_rows')
        return result
    if nearest[2] > MAX_NEAREST_GRID_DISTANCE_M:
        result.update(status='outside_grid')
        result['quality_flags'].append('nearest_grid_distance_exceeds_20km')
        return result
    if not _inside_footprint(rows):
        result.update(status='outside_grid')
        result['quality_flags'].append('outside_grid_footprint')
        return result
    segments = []
    for a, b in zip(row, row[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        if length2 == 0:
            continue
        t = -(a[0] * dx + a[1] * dy) / length2
        bounded_t = min(1., max(0., t))
        distance2 = (a[0] + bounded_t * dx)**2 + (a[1] + bounded_t * dy)**2
        segments.append((distance2, a, b, dx, dy, t, math.sqrt(length2)))
    if not segments:
        result['quality_flags'].append('degenerate_row')
        return result
    _, a, b, dx, dy, t, length = min(segments, key=lambda item: item[0])
    azimuth = math.degrees(math.atan2(dx, dy)) % 360
    if azimuth == 360.:
        azimuth = 0.
    if line in (min(lines), max(lines)):
        result['quality_flags'].append('nearest_row_on_grid_boundary')
    if t < 0 or t > 1:
        result['quality_flags'].append('range_segment_extrapolation')
    result.update(status='ok', azimuth_deg=azimuth, axial_azimuth_deg=azimuth % 180,
                  line=line, pixel_pair=[a[3]['pixel'], b[3]['pixel']], segment_length_m=length)
    return result
