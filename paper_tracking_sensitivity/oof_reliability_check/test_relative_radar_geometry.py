import io
import json
import math
from http.client import IncompleteRead
from urllib.error import URLError

import pytest

import relative_radar_geometry as radar


PRODUCT = 'S1A_IW_GRDH_1SDV_20250101T142509_20250101T142534_057252_070AF6_8BC8'


def point(line, pixel, lon, lat):
    return dict(line=line, pixel=pixel, longitude=lon, latitude=lat, incidenceAngle=35.)


def with_neighbor_rows(points):
    dx = points[-1]['longitude'] - points[0]['longitude']
    dy = points[-1]['latitude'] - points[0]['latitude']
    return [dict(p, line=line, longitude=p['longitude']-dy*offset,
                 latitude=p['latitude']+dx*offset)
            for line, offset in [(0, -.5), (10, 0.), (20, .5)] for p in points]


def grid_xml(points):
    body = ''.join('<geolocationGridPoint>' + ''.join(f'<{k}>{v}</{k}>' for k, v in p.items())
                   + '</geolocationGridPoint>' for p in points)
    return (f'<geolocationGrid><geolocationGridPointList count="{len(points)}">{body}'
            '</geolocationGridPointList></geolocationGrid>').encode()


@pytest.mark.parametrize('points,expected', [
    ([point(0, 0, 0, -.01), point(0, 10, 0, .01)], 0.),
    ([point(0, 0, -.01, 0), point(0, 10, .01, 0)], 90.),
])
def test_true_north_and_east(points, expected):
    result = radar.range_azimuth(with_neighbor_rows(points), 0., 0.)
    assert result['status'] == 'ok'
    assert result['azimuth_deg'] == pytest.approx(expected, abs=1e-8)
    assert 1000 < result['nearest_grid_distance_m'] < 1200


def test_reversing_axis_preserves_axial_angle_in_target_frame():
    # High latitude and oblique direction expose endpoint-bearing reversal errors.
    points = with_neighbor_rows([point(0, 0, 10., 70.), point(0, 100, 10.4, 70.1)])
    normal = radar.range_azimuth(points, 10.2, 70.05)
    reverse = radar.range_azimuth([dict(p, pixel=100-p['pixel']) for p in points], 10.2, 70.05)
    assert (reverse['azimuth_deg'] - normal['azimuth_deg']) % 360 == pytest.approx(180.)
    assert reverse['axial_azimuth_deg'] == pytest.approx(normal['axial_azimuth_deg'])


def test_uses_same_row_even_when_other_row_point_is_closer_than_neighbor():
    points = [point(0, 0, -.01, 0), point(0, 10, .01, 0),
              point(10, 0, -.01, .001), point(10, 10, .01, .001)]
    result = radar.range_azimuth(list(reversed(points)), -.009, 0.)
    assert result['line'] == 0
    assert result['pixel_pair'] == [0, 10]
    assert result['azimuth_deg'] == pytest.approx(90.)


@pytest.mark.parametrize('points,flag', [
    ([], 'empty_grid'),
    ([[0., 1., 35.]], 'missing_grid_fields'),
    ([dict(pixel=0, latitude=0., longitude=0., incidenceAngle=35.)], 'missing_grid_fields'),
    ([point(0, 0, 0, 0), point(0, 0, .01, 0)], 'ambiguous_line_pixel'),
    ([point(0, 0, 0, 0), point(0, 10, 0, 0)], 'ambiguous_geographic_coordinates'),
    ([point(0, 0, 0, 0), point(10, 10, .01, 0)], 'nearest_row_has_no_pixel_neighbor'),
    ([point(0, 0, 0, math.nan)], 'nonnumeric_grid_fields'),
    ([point(0, 0, 0, 100)], 'invalid_geographic_coordinate_or_incidence'),
])
def test_bad_grid_never_produces_angle(points, flag):
    result = radar.range_azimuth(points, 0., 0.)
    assert result['status'] == 'invalid_grid'
    assert result['azimuth_deg'] is None
    assert flag in result['quality_flags']


def test_lon_lat_order_and_query_validation():
    points = with_neighbor_rows([point(0, 0, 120., 30.), point(0, 10, 120.01, 30.)])
    assert radar.range_azimuth(points, 120., 30.)['status'] == 'ok'
    with pytest.raises(ValueError):
        radar.range_azimuth(points, 30., 120.)


def test_distant_query_is_flagged():
    result = radar.range_azimuth(with_neighbor_rows([point(0, 0, 0, 0), point(0, 10, .01, 0)]), 1., 1.)
    assert result['status'] == 'outside_grid'
    assert result['azimuth_deg'] is None
    assert 'nearest_grid_distance_exceeds_20km' in result['quality_flags']


def test_close_but_outside_footprint_rejected():
    points = with_neighbor_rows([point(0, 0, 0, 0), point(0, 10, .01, 0)])
    result = radar.range_azimuth(points, .011, 0.)
    assert result['nearest_grid_distance_m'] < 200
    assert result['status'] == 'outside_grid'
    assert result['azimuth_deg'] is None
    assert 'outside_grid_footprint' in result['quality_flags']


def test_sparse_interior_over_20km_rejected():
    points = with_neighbor_rows([point(0, 0, -1, 0), point(0, 10, 1, 0)])
    result = radar.range_azimuth(points, 0., 0.)
    assert result['azimuth_deg'] is None
    assert 'nearest_grid_distance_exceeds_20km' in result['quality_flags']


def test_missing_row_points_and_single_row_rejected():
    row = [point(0, 0, 0, 0), point(0, 10, .01, 0)]
    assert 'insufficient_rows_for_footprint' in radar.range_azimuth(row, 0., 0.)['quality_flags']
    result = radar.range_azimuth(with_neighbor_rows(row)[:-1], .005, 0.)
    assert result['azimuth_deg'] is None
    assert 'incomplete_grid_rows' in result['quality_flags']


class Response(io.BytesIO):
    status = 206

    def __init__(self, data):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data)), 'Content-Range': f'bytes 0-{len(data)-1}/{len(data)}'}

    def read(self, size=-1):
        assert 0 <= size <= radar.TAIL_BYTES
        return super().read(size)


def test_bounded_fetch_structured_tail_and_separate_cache(monkeypatch, tmp_path):
    points = [point(0, 0, 120.123456789, 30.), point(0, 10, 120.2, 30.)]
    body = b'broken XML prefix></other>' + grid_xml(points) + b'</product>'
    calls = []

    def open_request(request, timeout):
        calls.append(request.full_url)
        assert PRODUCT in request.full_url
        assert request.get_header('Range') == 'bytes=-900000'
        assert timeout == 30
        return Response(body)

    monkeypatch.setattr(radar.OPENER, 'open', open_request)
    first = radar.fetch_grid(PRODUCT, tmp_path)
    assert first['status'] == 'ok'
    assert first['points'] == points
    assert 'product_header_unavailable_in_tail' in first['quality_flags']
    assert json.loads((tmp_path / f'{PRODUCT}.json').read_text()) == first
    assert radar.fetch_grid(PRODUCT, tmp_path) == first
    assert len(calls) == 1


@pytest.mark.parametrize('body', [b'', b'<geolocationGrid><geolocationGridPointList>',
    grid_xml([point(0, 0, 0., 0.)]).replace(b'count="1"', b'count="2"'),
    grid_xml([point(0, 0, 0., 0.)]).replace(b'<line>0</line>', b'')])
def test_incomplete_or_malformed_grid_not_cached(monkeypatch, tmp_path, body):
    monkeypatch.setattr(radar.OPENER, 'open', lambda *a, **kw: Response(body))
    assert radar.fetch_grid(PRODUCT, tmp_path)['status'] == 'invalid_annotation'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('error', [URLError('unavailable'), IncompleteRead(b'partial', 100)])
def test_unavailable_network_not_cached(monkeypatch, tmp_path, error):
    def unavailable(*args, **kwargs):
        raise error
    monkeypatch.setattr(radar.OPENER, 'open', unavailable)
    fetched = radar.fetch_grid(PRODUCT, tmp_path)
    assert fetched['status'] == 'unavailable'
    assert radar.range_azimuth(fetched, 0., 0.)['azimuth_deg'] is None
    assert not list(tmp_path.iterdir())


def test_ignored_range_does_not_download_whole_annotation(monkeypatch, tmp_path):
    response = Response(b'')
    response.status = 200
    response.headers['Content-Length'] = str(radar.TAIL_BYTES + 1)
    monkeypatch.setattr(radar.OPENER, 'open', lambda *a, **kw: response)
    assert radar.fetch_grid(PRODUCT, tmp_path)['error'] == 'server_ignored_bounded_range'


def test_mismatched_product_header_rejected(monkeypatch, tmp_path):
    header = (b'<adsHeader><missionId>S1C</missionId><productType>GRD</productType>'
              b'<startTime>2025-01-01T14:25:09.123</startTime>'
              b'<stopTime>2025-01-01T14:25:34.123</stopTime></adsHeader>')
    monkeypatch.setattr(radar.OPENER, 'open', lambda *a, **kw: Response(header + grid_xml([point(0, 0, 0, 0)])))
    result = radar.fetch_grid(PRODUCT, tmp_path)
    assert result['error'] == 'product_header_mismatch:missionId'
