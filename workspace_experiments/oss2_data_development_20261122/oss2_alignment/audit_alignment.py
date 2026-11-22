"""Read-only OSS2 source audit. Outputs contain local sample IDs; do not publish."""
from pathlib import Path
import csv
import json
import re
import sys
from collections import Counter
import numpy as np
import tifffile
from PIL import Image
from lxml import etree as ET

ROOT = Path(__file__).resolve().parent
MANIFEST = Path('E:/临时会话/m1_coverage_20261101/manifest.json')
SOURCE = Path('E:/临时会话/oss2_port_prototypes_20261121/source_observations_local.csv')
PNGROOT = Path('D:/Documents/project_root/project_data/public_opensarship2/opensarship2/chips')
COORD = re.compile(r'_x(-?\d+)_y(-?\d+)_(vv|vh)\.tif$', re.I)


def quantizer(x, y):
    """Find an exact positive clipped affine quantizer, without moving any pixel.

    Each observed output bin constrains b between y-.5-a*min(x)
    and y+.5-a*max(x). Saturation removes the outer bound. Pairwise
    comparisons give a feasible interval for a; exact replay is the final test.
    """
    x, y = x.astype(np.float64).ravel(), y.ravel()
    if not np.isfinite(x).all():
        return dict(exact=False, monotone=False, slope=None, offset=None)
    idx = np.argsort(x, kind='stable')
    dx, dy = np.diff(x[idx]), np.diff(y[idx].astype(np.int16))
    mono = bool(np.all(dy >= 0) and np.all(dy[dx == 0] == 0))
    lx, lv, ux, uv = [], [], [], []
    for value in np.unique(y):
        xx = x[y == value]
        if value > 0:
            lx.append(xx.min()); lv.append(float(value)-.5)
        if value < 255:
            ux.append(xx.max()); uv.append(float(value)+.5)
    if not lx or not ux:
        return dict(exact=False, monotone=mono, slope=None, offset=None)
    lx, lv, ux, uv = map(np.asarray, (lx, lv, ux, uv))
    delta = lx[:, None]-ux[None, :]
    rhs = lv[:, None]-uv[None, :]
    if np.any((delta == 0) & (rhs > 0)):
        return dict(exact=False, monotone=mono, slope=None, offset=None)
    lo = max(0., float(np.max(rhs[delta > 0]/delta[delta > 0], initial=0.)))
    hi = float(np.min(rhs[delta < 0]/delta[delta < 0], initial=np.inf))
    if not np.isfinite(hi) or hi < lo or hi <= 0:
        return dict(exact=False, monotone=mono, slope=None, offset=None)
    for a in [(lo+hi)/2, lo, hi]:
        lower, upper = np.max(lv-a*lx), np.min(uv-a*ux)
        for b in [(lower+upper)/2, lower+1e-9, upper-1e-9]:
            replay = np.clip(np.floor(a*x+b+.5), 0, 255).astype(np.uint8)
            if np.array_equal(replay, y):
                return dict(exact=True, monotone=mono, slope=float(a), offset=float(b))
    return dict(exact=False, monotone=mono, slope=None, offset=None)


def attributes(node):
    return {n.get('name'): n.get('value') for n in node.findall('attrib')}


def dump_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def main():
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        source = {r['sample_id']: r for r in csv.DictReader(f)}
    allrows = json.loads(MANIFEST.read_text(encoding='utf8'))
    rows = [r for r in allrows if r['id'] in source]
    assert len(rows) == len(source) == 719
    assert all(r['role'] in ('fit_train', 'utility_train') for r in rows)
    products, result = {}, []
    grid_fields = ['swath', 'first_line_time', 'last_line_time', 'line_time_interval',
                   'num_output_lines', 'num_samples_per_line', 'sample_type']
    for ordinal, r in enumerate(rows):
        meta = source[r['id']]
        folder = Path(r['source_paths']['vv']).parent.parent
        if r['scene_id'] not in products:
            root = ET.parse(meta['metadata_xml_path']).getroot()
            abstract = root.find('Abstracted_Metadata'); attrs = attributes(abstract)
            bands = {attributes(n).get('polarization'): attributes(n) for n in abstract if n.tag.startswith('Band_')}
            parser = ET.XMLParser(recover=True, encoding='utf8')
            shiproot = ET.parse(str(folder/'Ship.xml'), parser).getroot()
            ships = {}
            for ship in shiproot.findall('ship'):
                sar = ship.find('SARShipInformation')
                xy = (int(sar.findtext('Center_x')), int(sar.findtext('Center_y')))
                ships.setdefault(xy, []).append(ship)
            prod = dict(product=r['scene_id'], metadata_xml=meta['metadata_xml_path'],
                        product_id_exact=attrs.get('PRODUCT') == r['scene_id'].removesuffix('(1)'),
                        both_bands=all(p in bands for p in ['VV', 'VH']),
                        bands_same_grid=all(bands.get('VV', {}).get(k) == bands.get('VH', {}).get(k)
                                            and bands.get('VV', {}).get(k) is not None for k in grid_fields),
                        shared_tie_point_grids=root.find('tie-point-grids') is not None,
                        coregistered_stack=attrs.get('coregistered_stack', ''),
                        grid_values=json.dumps({k: [bands.get(p, {}).get(k) for p in ['VV', 'VH']] for k in grid_fields}),
                        annotation_vv=bands.get('VV', {}).get('annotation', ''),
                        annotation_vh=bands.get('VH', {}).get('annotation', ''),
                        xml_parser_messages=len(parser.error_log))
            products[r['scene_id']] = (prod, ships)
        prod, ships = products[r['scene_id']]
        paths = {p: Path(r['source_paths'][p]) for p in ['vv', 'vh']}
        coords = {p: COORD.search(paths[p].name) for p in paths}
        xy = tuple(int(coords['vv'].group(i)) for i in [1, 2])
        matches = ships.get(xy, [])
        basename = paths['vv'].name.removeprefix('Visual_').replace('_vv.tif', '.tif')
        rawpath = folder/'Patch'/basename
        calpath = folder/'Patch_Cal'/basename
        raw = tifffile.imread(rawpath)
        with tifffile.TiffFile(rawpath) as tf:
            t = tf.pages[0]
            joint = len(tf.pages) == 1 and t.samplesperpixel == 2 and t.planarconfig == 1
            tags = {tag.name: str(tag.value) for tag in t.tags.values()}
        images = {p: tifffile.imread(paths[p]) for p in paths}
        out = dict(sample_id=r['id'], audit_ordinal=ordinal, role=r['role'], label=r['label'],
                   class_name=r['class_name'], product=r['scene_id'],
                   product_metadata_valid=prod['product_id_exact'] and prod['both_bands'],
                   product_band_grid_equal=prod['bands_same_grid'],
                   same_filename_center=all(tuple(int(coords[p].group(i)) for i in [1, 2]) == xy for p in paths),
                   same_product_folder=paths['vv'].parent == paths['vh'].parent,
                   xml_center_unique=len(matches) == 1,
                   xml_identity_match=len(matches) == 1 and matches[0].findtext('AISShipInformation/MMSI') == r['mmsi'],
                   raw_joint_pixel_interleaved=joint, raw_shape=str(raw.shape),
                   raw_all_finite=bool(np.isfinite(raw).all()), raw_min=float(np.nanmin(raw)),
                   raw_zero_count=int(np.sum(raw == 0)), raw_path=str(rawpath),
                   raw_nodata_tag=tags.get('GDAL_NODATA', ''), raw_mask_ifd=False,
                   raw_geotransform_present=any(k in tags for k in ['ModelPixelScaleTag','ModelTransformationTag','ModelTiepointTag']),
                   raw_orientation=tags.get('Orientation','absent'),
                   native_shape_equal=images['vv'].shape == images['vh'].shape == raw.shape[:2],
                   vv_path=str(paths['vv']), vh_path=str(paths['vh']),
                   metadata_xml=prod['metadata_xml'], ship_xml=str(folder/'Ship.xml'),
                   raw_crop_window_record_available=False, actual_registration_error_known=False,
                   valid_mask_available=False, valid_mask_status='unknown; raster footprint is not valid-data support')
        for pol, ch in [('vv', 1), ('vh', 0)]:
            q = quantizer(raw[...,ch], images[pol])
            for k, v in q.items(): out[f'{pol}_raw_to_uint8_{k}'] = v
            out[f'{pol}_joint_channel'] = ch
            pngpath = PNGROOT/pol/(r['id']+'.png')
            png = np.asarray(Image.open(pngpath))
            out[f'{pol}_png_equal_uint8'] = bool(np.array_equal(png, images[pol]))
            out[f'{pol}_png_path'] = str(pngpath)
            out[f'{pol}_detector_available'] = r[pol] is not None
            out[f'{pol}_zero_border_fraction'] = float(np.mean(np.concatenate([images[pol][0], images[pol][-1], images[pol][:,0], images[pol][:,-1]]) == 0))
        out['same_pixel_index_derivation_verified'] = bool(all(out[k] for k in [
            'product_metadata_valid', 'product_band_grid_equal', 'same_filename_center', 'same_product_folder',
            'xml_center_unique', 'xml_identity_match', 'raw_joint_pixel_interleaved', 'raw_all_finite', 'native_shape_equal',
            'vv_raw_to_uint8_exact', 'vh_raw_to_uint8_exact', 'vv_png_equal_uint8', 'vh_png_equal_uint8']))
        # Only coordinate-index agreement is established; physical residual registration is not measured.
        out['alignment_valid_recommendation'] = 'true_shared_pixel_grid' if out['same_pixel_index_derivation_verified'] else 'unknown'
        out['polarizations_coregistered_recommendation'] = 'true_shared_product_grid_not_measured_error' if out['same_pixel_index_derivation_verified'] else 'unknown'
        result.append(out)
    dump_csv('row_evidence_local.csv', result)
    dump_csv('product_evidence_local.csv', [p[0] for p in products.values()])
    boolkeys = [k for k,v in result[0].items() if isinstance(v,bool)]
    summary = dict(scope='Existing 719 source observations: fit_train + utility_train only; no heldout images',
                   source_rows=len(result), source_products=len(products), counts={k:sum(bool(r[k]) for r in result) for k in boolkeys},
                   exact_vv=sum(r['vv_raw_to_uint8_exact'] for r in result), exact_vh=sum(r['vh_raw_to_uint8_exact'] for r in result),
                   raw_nodata_tags=Counter(r['raw_nodata_tag'] for r in result),
                   raw_min_range=[min(r['raw_min'] for r in result),max(r['raw_min'] for r in result)])
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()
