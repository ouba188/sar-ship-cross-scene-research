"""Read existing AIS/crop metadata and produce one independent row per instance."""
import csv
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PureWindowsPath

sys.dont_write_bytecode = True
import pandas as pd
from pyproj import CRS, Transformer

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ART = ROOT / 'visual_reliable_baseline/artifacts'
MANIFEST = ART / 'eight_class_adaptive_20260916/manifest.csv'
AUDIT = ART / 'real_k_asset_audit_20260917'
SYNC = Path('E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata/local_knowledge_sync_v1')
ARCHIVE = Path('E:/Hermes/paper_tracking/port_ais_2024_2025_from_zip')
PRODUCT_RE = re.compile(r'S1[ABC]_IW_GRDH_1SDV_\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6}_[0-9A-F]{4}')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def dump_csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def js(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'))


def number(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def norm(x):
    return str(PureWindowsPath(str(x))).lower()


def name(x):
    return PureWindowsPath(str(x)).name


def mmsi(x):
    return str(x or '').removesuffix('.0')


def pair(length, width):
    a, b = number(length), number(width)
    return (a, b) if a is not None and b is not None and a >= b > 0 else None


def epoch(x):
    try:
        d = datetime.fromisoformat(x.replace('Z', '+00:00'))
        return d.timestamp() if d.tzinfo else None
    except (ValueError, TypeError):
        return None


def load_links(rows):
    original = {(r['instance_id'], r['polarization'].lower()): r for r in
                read_csv(AUDIT / 'fdrive_sources/original_sources_private.csv')}
    bridge = {r['instance_id']: r for r in read_csv(ROOT / 'geometry_source_recovery_20261016/geometry_input_manifest.csv')}
    returned = defaultdict(list)
    for p in sorted((SYNC.parent).glob('server*/manifest.csv')):
        for r in read_csv(p):
            returned[(norm(r['source_json']), r['obj_uid'])].append((str(p), r))
    views = []
    for i, row in enumerate(rows):
        for pol in ('vv', 'vh'):
            v = dict(global_row=i, instance_id=row['instance_id'], product=row['product'],
                     port=row['port'], mmsi=row['mmsi'], pol=pol,
                     image_path=row['image_relpath_' + pol], source_link='current_manifest')
            for k in ('source_json', 'source_tiff', 'obj_uid', 'source_object_index',
                      'target_center_x', 'target_center_y', 'crop_left', 'crop_top', 'crop_side'):
                v[k] = row.get(k + '_' + pol, '')
            old = original.get((row['instance_id'], pol))
            if old:
                assert old['product'] == row['product'] and mmsi(old['mmsi']) == row['mmsi']
                assert name(old['context_output_png']) == name(v['image_path'])
                for axis in ('x', 'y'):
                    assert abs(float(old['target_center_' + axis]) - float(v['target_center_' + axis])) < 1e-6
                for k in ('source_json', 'source_tiff', 'obj_uid', 'source_object_index',
                          'crop_left', 'crop_top', 'crop_side'):
                    if not v[k]:
                        v[k] = old[k]
                v['source_link'] = 'exact_original_crop_manifest'
                v['source_link_evidence'] = str(AUDIT / 'fdrive_sources/original_sources_private.csv')
                v['original_imo'] = old.get('imo', '')
                v['crop_position_method'] = old.get('position_method', '')
                v['crop_uncertainty_m'] = number(old.get('uncertainty_m'))
            repaired = bridge[row['instance_id']]
            assert repaired['product'] == row['product'] and mmsi(repaired['mmsi']) == row['mmsi']
            for k in ('source_json', 'source_tiff', 'obj_uid', 'source_object_index',
                      'crop_left', 'crop_top', 'target_center_x', 'target_center_y'):
                if not v[k]:
                    v[k] = repaired.get(k + '_' + pol, '')
            matches = returned.get((norm(v['source_json']), v['obj_uid']), [])
            if matches:
                signatures = {(r['product'],r['mmsi'],r['crop_left'],r['crop_top'],r['crop_side']) for _,r in matches}
                assert len(signatures)==1, (row['instance_id'],pol,'conflicting_returned_crop')
                p, r = matches[0]
                assert r['product']==row['product'] and mmsi(r['mmsi'])==row['mmsi']
                assert name(r['context_output_png'])==name(v['image_path'])
                for k in ('crop_left','crop_top','crop_side','target_center_x','target_center_y'):
                    if not v[k]:
                        v[k]=r[k]
                    else:
                        assert abs(float(v[k])-float(r[k]))<1e-6, (row['instance_id'],pol,k)
                v.update(source_link='exact_returned_crop_manifest',source_link_evidence=p,
                         original_imo=r.get('imo',''),crop_position_method=r.get('position_method',''),
                         crop_uncertainty_m=number(r.get('uncertainty_m')))
            views.append(v)
    return views


def file_maps():
    jsons, ais = defaultdict(set), defaultdict(set)
    for f in sorted((SYNC / 'private').glob('server*_file_manifest.csv')):
        for r in read_csv(f):
            if r.get('status') not in ('downloaded', 'reused_local'):
                continue
            p = Path(r['local_path'])
            if not p.is_file():
                continue
            if r['kind'] == 'source_json':
                jsons[norm(r['remote_path'])].add(str(p))
            elif r['kind'] == 'ais':
                match = PRODUCT_RE.search(p.name)
                if match:
                    ais[match.group()].add(str(p))
    for r in read_csv(AUDIT / 'detection_recovery/catalog_inventory_private.csv'):
        if r['local_json'] and Path(r['local_json']).is_file():
            jsons[norm(r['source_json'])].add(r['local_json'])
    for f in (AUDIT / 'ais_file_inventory.json', AUDIT / 'ais_reexport/export_file_inventory.json'):
        for p, paths in json.loads(f.read_text(encoding='utf-8')).items():
            ais[p].update(x for x in paths if Path(x).is_file())
    for r in read_csv(ARCHIVE / 'dedup_tables/ais_txt_file_manifest.csv'):
        p = ARCHIVE / r['relative_path']
        if p.is_file() and p.stat().st_size:
            ais[r['product']].add(str(p))
    return jsons, ais


def object_records(views, paths):
    byname = defaultdict(set)
    for remote, locals_ in paths.items():
        byname[name(remote)].update(locals_)
    cache = {}
    keys = Counter()
    for n, v in enumerate(views):
        if n % 12000 == 0:
            print('JSON_VIEWS', n, '/', len(views), flush=True)
        remote = v['source_json']
        candidates = sorted(paths.get(norm(remote), set()))
        method = 'exact_source_path'
        if not candidates and remote and Path(remote).is_file():
            candidates = [remote]
        if not candidates:
            candidates = sorted(byname.get(name(remote).lower(), set()))
            method = 'same_product_pol_tile_and_exact_uid'
        found = []
        for p in candidates:
            if p not in cache:
                data = json.loads(Path(p).read_text(encoding='utf-8-sig'))
                index = defaultdict(list)
                for j, obj in enumerate(data.get('objects', [])):
                    index[str(obj.get('obj_uid') or obj.get('source_obj_uid') or '')].append((j, obj))
                    keys.update(obj.get('meta', {}).keys())
                cache[p] = (data.get('image', ''), index)
            image, index = cache[p]
            matches = index.get(v['obj_uid'], [])
            if not v['obj_uid'] or len(matches) != 1 or name(image) != name(v['source_tiff']):
                continue
            j, obj = matches[0]
            meta = obj.get('meta', {})
            if mmsi(meta.get('mmsi')) != v['mmsi'] or v['product'] not in image:
                continue
            found.append((p, j, obj, meta))
        v['object_status'] = 'no_exact_uid_json'
        if not found:
            continue
        signatures = {(x[3].get('length_m'), x[3].get('beam_m'), x[3].get('ais_time_utc')) for x in found}
        if len(signatures) > 1:
            v['object_status'] = 'conflicting_json_copies'
            v['object_conflict_sources'] = js([x[0] for x in found])
            continue
        p, j, obj, meta = found[0]
        v.update(object_status='exact_uid_product_mmsi', object_local_json=p, json_link_method=method,
                 json_object_index=j, source_index_agrees=(number(v['source_object_index']) == j),
                 object_length_m=number(meta.get('length_m')), object_width_m=number(meta.get('beam_m')),
                 object_attribute_source=meta.get('source', ''), object_imo=str(meta.get('imo', '')),
                 ais_time_utc=meta.get('ais_time_utc', ''), ais_epoch=epoch(meta.get('ais_time_utc', '')),
                 position_method=meta.get('ais_position_method', ''),
                 position_uncertainty_m=number(meta.get('ais_position_uncertainty_m')),
                 geometry_confidence=meta.get('geometry_confidence', ''),
                 redetect_status=meta.get('redetect_status', ''))
        coordinates = []
        for d in (obj, meta):
            for x, y in (('lon', 'lat'), ('longitude', 'latitude'), ('ais_lon', 'ais_lat'), ('ais_longitude', 'ais_latitude')):
                a, b = number(d.get(x)), number(d.get(y))
                if a is not None and b is not None and -180 <= a <= 180 and -90 <= b <= 90:
                    coordinates.append((a, b))
        v['stored_coordinates_json'] = js(sorted(set(coordinates)))
    (OUT / 'object_metadata_keys.json').write_text(json.dumps(dict(keys), indent=2), encoding='utf-8')
    return len(cache)


def headers():
    sys.path.insert(0, str(ROOT / 'geometry_source_recovery_20261016'))
    from recover_source_bridge import load_headers
    found = load_headers()
    for p, h in json.loads((ROOT / 'geometry_source_recovery_20261016/source_georefs.json').read_text(encoding='utf-8')).items():
        found.append(dict(h, path=p))
    bypath, byname = defaultdict(list), defaultdict(list)
    for h in found:
        bypath[norm(h['path'])].append(h)
        byname[name(h['path']).lower()].append(h)
    return bypath, byname


def unique_header(candidates):
    if not candidates:
        return None, 'missing_exact_source_header'
    a = candidates[0]
    for b in candidates[1:]:
        if (a['epsg'], a['width'], a['height']) != (b['epsg'], b['width'], b['height']) or any(
                abs(x-y) > .01 for x, y in zip(a['transform'], b['transform'])):
            return None, 'conflicting_source_headers'
    return dict(a), 'ok'


@lru_cache(None)
def projection(epsg):
    c = CRS.from_epsg(epsg)
    assert c.is_projected and all(a.unit_name == 'metre' for a in c.axis_info)
    return Transformer.from_crs(c, 4326, always_xy=True)


def geo_records(views):
    bypath, byname = headers()
    for n, v in enumerate(views):
        if n % 12000 == 0:
            print('GEO_VIEWS', n, '/', len(views), flush=True)
        source = v['source_tiff']
        cand = bypath.get(norm(source), [])
        method = 'exact_source_path_header'
        if not cand:
            cand = [h for h in byname.get(name(source).lower(), []) if v['port'].lower() in norm(h['path'])]
            method = 'unique_product_pol_tile_port_header'
        h, status = unique_header(cand)
        if h is None and status != 'conflicting_source_headers' and 'int8_grid_4x4_o256' in source:
            scene_name = v['product'] + '_' + v['pol'].upper() + '_UTM_8bit.tif'
            scene, _ = unique_header([h for h in byname.get(scene_name.lower(), []) if v['port'].lower() in norm(h['path'])])
            match = re.search(r'__r(\d+)_c(\d+)\.tif$', name(source))
            if scene and match:
                r, c = map(int, match.groups())
                assert 1 <= r <= 4 and 1 <= c <= 4
                x0, y0 = max(0, int(scene['width']*(c-1)/4)-128), max(0, int(scene['height']*(r-1)/4)-128)
                x1, y1 = min(scene['width'], int(scene['width']*c/4)+128), min(scene['height'], int(scene['height']*r/4)+128)
                g = scene['transform']
                h = dict(scene, width=x1-x0, height=y1-y0,
                         transform=[g[0]+x0*g[1]+y0*g[2], g[1], g[2], g[3]+x0*g[4]+y0*g[5], g[4], g[5]])
                method = 'original_scene_header_documented_4x4_overlap256'
        v['ground_scale_status'] = status
        if h is None:
            continue
        x, y = number(v['target_center_x']), number(v['target_center_y'])
        if x is not None and y is not None and not (0 <= x < h['width'] and 0 <= y < h['height']):
            v['ground_scale_status'] = 'crop_center_outside_source'
            continue
        g = h['transform']
        crs = int(h['epsg'])
        tr = projection(crs)
        px, py = math.hypot(g[1], g[4]), math.hypot(g[2], g[5])
        left, top, side = (number(v[k]) for k in ('crop_left', 'crop_top', 'crop_side'))
        v.update(source_pixel_x_m=px, source_pixel_y_m=py, epsg=crs,
                 georef_source=h.get('header_source', ''), georef_header_path=h['path'],
                 georef_link_method=method, source_transform_json=js(g),
                 ground_scale_status='source_header_verified')
        if x is not None and y is not None:
            lon, lat = tr.transform(g[0]+x*g[1]+y*g[2], g[3]+x*g[4]+y*g[5])
            v.update(target_center_lon=lon, target_center_lat=lat)
        if left is not None and top is not None:
            v['chip_transform_json'] = js([g[0]+left*g[1]+top*g[2],g[1],g[2],g[3]+left*g[4]+top*g[5],g[4],g[5]])
        with Path(v['image_path']).open('rb') as f:
            head = f.read(24)
        assert head[:8] == b'\x89PNG\r\n\x1a\n', v['image_path']
        w, ht = struct.unpack('>II', head[16:24])
        v.update(png_width=w, png_height=ht)
        if side == w == ht:
            v.update(pixel_x_m=px, pixel_y_m=py, ground_scale_status='valid_native_png_original_crop_grid',
                     affine_xx=g[1],affine_xy=g[2],affine_yx=g[4],affine_yy=g[5])
            if 'chip_transform_json' not in v:
                v['ground_scale_status']='valid_native_png_scale_position_missing'
        else:
            v['ground_scale_status'] = 'source_header_valid_png_crop_shape_mismatch'


def raw_records(rows, views, sources):
    wanted = defaultdict(set)
    needed = set()
    for r in rows:
        wanted[r['product']].add(r['mmsi'])
    for v in views:
        if v.get('ais_epoch') is not None:
            needed.add((v['product'], v['mmsi'], v['ais_epoch']))
    exact = defaultdict(list)
    product_sizes, vessel_sizes = defaultdict(dict), defaultdict(dict)
    product_imos, vessel_imos = defaultdict(set), defaultdict(set)
    receipts = []
    for n, (product, ships) in enumerate(wanted.items()):
        candidates = sources.get(product, set())
        if not candidates:
            receipts.append(dict(product=product, status='no_local_raw_file'))
            continue
        # Use one existing export per product; preserve alternatives as provenance.
        sync = [p for p in candidates if str(SYNC).lower() in str(Path(p)).lower()]
        path = max(sync or candidates, key=lambda p: (Path(p).stat().st_size, p))
        matched = 0
        fields_bad = 0
        with Path(path).open(encoding='utf-8-sig', newline='') as f:
            for line, fields in enumerate(csv.reader(f), 1):
                if len(fields) != 18:
                    fields_bad += 1
                    continue
                ship = mmsi(fields[1].strip())
                if ship not in ships:
                    continue
                t, lon, lat = number(fields[0]), number(fields[6]), number(fields[7])
                if t is None or lon is None or lat is None or not (-180 <= lon <= 180 and -90 <= lat <= 90):
                    continue
                matched += 1
                rec = dict(product=product, mmsi=ship, epoch=t, lon=lon, lat=lat,
                           length_m=number(fields[14]), width_m=number(fields[15]),
                           raw_length=fields[14], raw_width=fields[15], imo=fields[12].strip(),
                           source_txt=path, source_line=line, export_source=fields[17],
                           raw_motion=tuple(fields[j] for j in (4,8,9)))
                key = (product, ship, t)
                if key in needed:
                    exact[key].append(rec)
                dims = pair(fields[14], fields[15])
                if dims:
                    product_sizes[(product,ship)].setdefault(dims, rec)
                    vessel_sizes[ship].setdefault(dims, rec)
                if rec['imo'] not in ('', '0'):
                    product_imos[(product,ship)].add(rec['imo'])
                    vessel_imos[ship].add(rec['imo'])
        receipts.append(dict(product=product,status='read',path=path,bytes=Path(path).stat().st_size,
                             cohort_records=matched,bad_field_rows=fields_bad,
                             other_copies_json=js(sorted(candidates-{path}))))
        if n % 50 == 0:
            print('RAW_PRODUCTS', n, '/', len(wanted), flush=True)
    dump_csv('read_ais_sources.csv', receipts)
    evidence=[]
    for (product,ship), variants in product_sizes.items():
        for dims, rec in variants.items():
            evidence.append({k:v for k,v in rec.items() if k!='raw_motion'})
    dump_csv('dimension_evidence.csv',evidence)
    return exact, product_sizes, vessel_sizes, product_imos, vessel_imos


def resolve_raw(candidates, coords):
    if not candidates:
        return [], 'no_exact_timestamp_record'
    good = [r for r in candidates if all(abs(r['lon']-x)<=1e-6 and abs(r['lat']-y)<=1e-6 for x,y in coords)]
    if not good:
        return [], 'stored_ais_coordinate_mismatch'
    variants = {(r['lon'], r['lat'], r['raw_motion']) for r in good}
    if len(variants) != 1:
        return [], 'conflicting_exact_position_or_motion'
    return good, 'exact_product_mmsi_timestamp' + ('_coordinate' if coords else '_no_stored_coordinate')


def assemble(rows, views, raw):
    exact, ps, vs, pi, vi = raw
    byrow = defaultdict(list)
    for v in views:
        byrow[v['global_row']].append(v)
        dims = pair(v.get('object_length_m'), v.get('object_width_m'))
        if dims and v.get('object_attribute_source') == 'raw_ais_review_only':
            rec = dict(product=v['product'],mmsi=v['mmsi'],epoch=v.get('ais_epoch'),
                       source_json=v.get('object_local_json'),obj_uid=v['obj_uid'],
                       length_m=dims[0],width_m=dims[1],imo=v.get('object_imo',''))
            vs[v['mmsi']].setdefault(dims,rec)
        if v.get('object_imo') not in (None,'','0'):
            vi[v['mmsi']].add(v['object_imo'])
    output = []
    for i, row in enumerate(rows):
        vv = byrow[i]
        out = {k:row[k] for k in ('instance_id','mmsi','product','port','class_name','data_origin')}
        out.update(global_row=i, ais_length_m=None,ais_width_m=None,ais_size_valid=False,
                   ais_size_scope='missing',ais_size_source='',ais_record_match_quality='no_object_time_anchor',
                   ais_dimension_components_status='not_preserved_in_local_export',
                   ais_A_m=None,ais_B_m=None,ais_C_m=None,ais_D_m=None,
                   static_update_time_utc='',ground_scale_semantics='projected_ground_grid_sampling_not_effective_resolution')
        good = [v for v in vv if v['object_status']=='exact_uid_product_mmsi']
        times = {v['ais_epoch'] for v in good if v.get('ais_epoch') is not None}
        dims = {pair(v.get('object_length_m'),v.get('object_width_m')) for v in good
                if v.get('object_attribute_source')=='raw_ais_review_only'} - {None}
        out['object_dimension_variants_json'] = js(sorted(dims))
        out['identity_link_quality'] = 'both_views_exact_uid' if len(good)==2 else 'one_view_exact_uid' if good else 'manifest_identity_only'
        out['object_view_time_conflict'] = len(times)>1
        out['position_methods_json'] = js(sorted({v.get('position_method','') for v in good}))
        out['position_uncertainty_m_max'] = max((v['position_uncertainty_m'] for v in good if v.get('position_uncertainty_m') is not None),default=None)
        out['position_link_quality'] = 'metadata_association_not_independent_detection_validation' if good else 'observation_position_unverified'
        out['object_sources_json'] = js([dict(pol=v['pol'],path=v['object_local_json'],uid=v['obj_uid'],index=v['json_object_index']) for v in good])
        expected_imos={v.get('original_imo') for v in vv}|{v.get('object_imo') for v in good}
        expected_imos-={None,'','0'}
        product_imos=pi.get((row['product'],row['mmsi']),set())
        out['expected_imo_variants_json']=js(sorted(expected_imos))
        out['same_product_imo_variants_json']=js(sorted(product_imos))
        out['same_product_imo_conflict']=len(product_imos)>1 or bool(expected_imos and product_imos and not expected_imos.intersection(product_imos))
        chosen_dims = None
        hard_conflict = len(dims)>1 or len(times)>1 or len(expected_imos)>1 or any(v['object_status']=='conflicting_json_copies' for v in vv)
        if len(times)==1:
            t = next(iter(times))
            out['ais_epoch'] = t
            out['ais_time_utc'] = datetime.fromtimestamp(t,timezone.utc).isoformat()
            mid = sum(datetime.strptime(s,'%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc).timestamp() for s in row['product'].split('_')[4:6])/2
            out['ais_minus_sar_midpoint_s'] = t-mid
            coords = sorted({tuple(c) for v in good for c in json.loads(v.get('stored_coordinates_json','[]'))})
            matched, quality = resolve_raw(exact.get((row['product'],row['mmsi'],t),[]),coords)
            out['ais_record_match_quality'] = quality
            out['ais_exact_candidate_count'] = len(exact.get((row['product'],row['mmsi'],t),[]))
            raw_dims = {pair(r['length_m'],r['width_m']) for r in matched}-{None}
            exact_imos={r['imo'] for r in matched}-{None,'','0'}
            hard_conflict |= len(exact_imos)>1 or bool(expected_imos and exact_imos and not expected_imos.intersection(exact_imos))
            out['exact_raw_dimension_variants_json'] = js(sorted(raw_dims))
            hard_conflict |= len(raw_dims)>1 or bool(raw_dims and dims and raw_dims!=dims)
            out['exact_object_dimension_disagreement'] = bool(raw_dims and dims and raw_dims!=dims)
            if len(raw_dims)==1 and not hard_conflict:
                chosen_dims=next(iter(raw_dims))
                rec=next(r for r in matched if pair(r['length_m'],r['width_m'])==chosen_dims)
                out.update(ais_size_scope='exact_observation_record',ais_size_source=rec['source_txt'],
                           ais_source_line=rec['source_line'],ais_raw_lon=rec['lon'],ais_raw_lat=rec['lat'])
        if chosen_dims is None and len(dims)==1 and not hard_conflict:
            chosen_dims=next(iter(dims))
            out.update(ais_size_scope='exact_uid_object_metadata',ais_size_source=out['object_sources_json'])
        product_dims = ps.get((row['product'],row['mmsi']),{})
        vessel_dims = vs.get(row['mmsi'],{})
        out.update(same_product_static_variants=len(product_dims),
                   same_product_static_variants_json=js(sorted(product_dims)),
                   vessel_static_variants=len(vessel_dims),vessel_static_variants_json=js(sorted(vessel_dims)),
                   vessel_static_conflict=len(vessel_dims)>1,
                   vessel_imo_variants_json=js(sorted(vi.get(row['mmsi'],set()))))
        if chosen_dims is None and not hard_conflict:
            if len(product_dims)==1 and not out['same_product_imo_conflict']:
                chosen_dims=next(iter(product_dims))
                rec=product_dims[chosen_dims]
                out.update(ais_size_scope='same_product_static_consensus_not_exact_observation',
                           ais_size_source=rec['source_txt'],ais_source_line=rec['source_line'],
                           static_evidence_epoch=rec['epoch'])
            elif not product_dims and len(vessel_dims)==1 and len(vi.get(row['mmsi'],set())|expected_imos)<=1:
                chosen_dims=next(iter(vessel_dims))
                rec=vessel_dims[chosen_dims]
                out.update(ais_size_scope='cross_observation_mmsi_static_consensus_not_position_match',
                           ais_size_source=rec.get('source_txt',rec.get('source_json','')),
                           ais_source_line=rec.get('source_line'),static_evidence_epoch=rec.get('epoch'),
                           static_evidence_product=rec['product'])
        if hard_conflict:
            out['ais_size_scope']='conflicting_exact_evidence'
        if chosen_dims:
            out.update(ais_length_m=chosen_dims[0],ais_width_m=chosen_dims[1],ais_size_valid=True)
        statuses={'exact_observation_record':'valid_exact_record','exact_uid_object_metadata':'valid_uid_metadata',
                  'same_product_static_consensus_not_exact_observation':'valid_same_product_static',
                  'cross_observation_mmsi_static_consensus_not_position_match':'valid_cross_observation_static',
                  'conflicting_exact_evidence':'conflicting_exact_evidence','missing':'missing_or_conflicting_static'}
        out['ais_size_status']=statuses[out['ais_size_scope']]
        out['size_validity_semantics']='positive_length_ge_width;provenance_association_not_surveyed_truth'
        for v in vv:
            pol=v['pol']
            for k in ('source_link','source_link_evidence','source_json','source_tiff','obj_uid','object_status','object_local_json',
                      'json_link_method','source_index_agrees','object_length_m','object_width_m',
                      'position_method','position_uncertainty_m','geometry_confidence','redetect_status',
                      'ground_scale_status','pixel_x_m','pixel_y_m','source_pixel_x_m','source_pixel_y_m',
                      'epsg','georef_source','georef_header_path','georef_link_method','chip_transform_json',
                      'source_transform_json','target_center_lon','target_center_lat','png_width','png_height',
                      'affine_xx','affine_xy','affine_yx','affine_yy','crop_position_method','crop_uncertainty_m'):
                out[k+'_'+pol]=v.get(k)
        out['pixel_x_m']=out['pixel_x_m_vv']
        out['pixel_y_m']=out['pixel_y_m_vv']
        out['pixel_reference_pol']='vv'
        out['mmsi_valid']=bool(re.fullmatch(r'[0-9]{9}',row['mmsi']))
        if not out['mmsi_valid']:
            out['identity_link_quality']='missing_mmsi'
        out['ais_position_match_verified']=out['ais_record_match_quality'].endswith('_timestamp_coordinate')
        out['transform_json_order']='GDAL_x0_xx_xy_y0_yx_yy'
        scale_vv=out['pixel_x_m_vv'] is not None
        scale_vh=out['pixel_x_m_vh'] is not None
        for pol in ('vv','vh'):
            out['scale_valid_'+pol]=out['pixel_x_m_'+pol] is not None
            out['chip_position_valid_'+pol]=bool(out['chip_transform_json_'+pol])
        out['georef_status']='both_valid' if scale_vv and scale_vh else 'vv_only' if scale_vv else 'vh_only' if scale_vh else 'unavailable'
        out['ais_size_and_vv_scale_valid']=bool(out['ais_size_valid'] and out['pixel_x_m_vv'] is not None)
        output.append(out)
    return output


def main():
    rows=read_csv(MANIFEST)
    assert len(rows)==38091 and len({r['instance_id'] for r in rows})==len(rows)
    for r in rows:
        r['mmsi']=mmsi(r['mmsi'])
    views=load_links(rows)
    maps, sources=file_maps()
    count=object_records(views,maps)
    geo_records(views)
    dump_csv('view_metadata.csv',views)
    print('VIEW_COVERAGE',dict(Counter(v['object_status'] for v in views)),flush=True)
    raw=raw_records(rows,views,sources)
    out=assemble(rows,views,raw)
    dump_csv('ais_metadata_sidecar.csv',out)
    frame=pd.DataFrame(out)
    summary=dict(rows=len(out),ports=frame.port.nunique(),products=frame['product'].nunique(),
                 vessels=frame.loc[frame.mmsi_valid,'mmsi'].nunique(),
                 missing_mmsi_rows=int((~frame.mmsi_valid).sum()),json_files_read=count,
                 ais_size_scope=frame.ais_size_scope.value_counts().to_dict(),
                 ais_record_match_quality=frame.ais_record_match_quality.value_counts().to_dict(),
                 valid_ais_size=int(frame.ais_size_valid.sum()),
                 vessel_static_conflict_rows=int(frame.vessel_static_conflict.sum()),
                 valid_vv_scale=int(frame.pixel_x_m_vv.notna().sum()),
                 valid_vh_scale=int(frame.pixel_x_m_vh.notna().sum()),
                 valid_both_scale=int((frame.pixel_x_m_vv.notna()&frame.pixel_x_m_vh.notna()).sum()),
                 size_and_vv_scale=int(frame.ais_size_and_vv_scale_valid.sum()),
                 scale_status_vv=frame.ground_scale_status_vv.value_counts().to_dict(),
                 scale_status_vh=frame.ground_scale_status_vh.value_counts().to_dict())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    grouped=frame.groupby('port').agg(rows=('instance_id','size'),ais_size=('ais_size_valid','sum'),
                                    vv_scale=('pixel_x_m_vv','count'),vh_scale=('pixel_x_m_vh','count'),
                                    size_and_vv_scale=('ais_size_and_vv_scale_valid','sum'))
    grouped.to_csv(OUT/'coverage_by_port.csv',encoding='utf-8-sig')
    frame[frame.vessel_static_conflict|frame.ais_size_scope.eq('conflicting_exact_evidence')].to_csv(OUT/'dimension_conflicts.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
