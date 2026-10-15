"""Read-only dataset provenance audit; outputs are confined to this directory."""

from pathlib import Path
from itertools import combinations
import json
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
from lxml import etree as ET

OUT = Path(__file__).resolve().parent
WORK = OUT.parent.parent
ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
SOURCE = WORK / 'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916'
NATIVE_ROOTS = {
    'oss1': Path('D:/Documents/project_root/OpenSARShip_1'),
    'oss2': Path('D:/Documents/project_root/project_data/OpenSARShip_2'),
}
NATIVE2 = Path('D:/Documents/project_root/project_data/public_opensarship2/opensarship2/metadata')


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save(name, frame):
    pd.DataFrame(frame).to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def present(series):
    return ~series.astype(str).str.strip().str.lower().isin(['', 'na', 'nan', 'none', '0', '-1'])


def parse_product(frame):
    p = frame['product']
    for key, pattern in {
        'platform': r'^(S1[A-Z])_', 'mode': r'^S1[A-Z]_([^_]+)_',
        'product_type': r'_(GRDH|SLC)_', 'start': r'_(\d{8}T\d{6})_',
        'orbit': r'_\d{8}T\d{6}_\d{8}T\d{6}_(\d{6})_',
        'datatake': r'_\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_([0-9A-F]{6})_',
    }.items():
        frame[key] = p.str.extract(pattern, expand=False).fillna('')
    frame['acquisition_start_key'] = frame.platform + '|' + frame.start
    frame['datatake_key'] = frame.platform + '|' + frame.orbit + '|' + frame.datatake
    return frame


XML_FIELDS = {
    'xml_mmsi': 'AISShipInformation/MMSI', 'ais_epoch': 'AISShipInformation/Time',
    'ais_lon': 'AISShipInformation/Longitude', 'ais_lat': 'AISShipInformation/Latitude',
    'ais_length': 'AISShipInformation/Length', 'ais_width': 'AISShipInformation/Width',
    'ais_type_xml': 'AISShipInformation/Ship_Type',
    'sar_time': 'SARShipInformation/Time_UTC', 'sar_lon': 'SARShipInformation/Center_longitude',
    'sar_lat': 'SARShipInformation/Center_latitude', 'incidence_xml': 'SARShipInformation/Incidence',
    'manual_length': 'SARShipInformation/Manual_Length',
    'head_x': 'SARShipInformation/Head_x', 'head_y': 'SARShipInformation/Head_y',
    'tail_x': 'SARShipInformation/Tail_x', 'tail_y': 'SARShipInformation/Tail_y',
    'marine_type_xml': 'MarineTrafficInformation/Elaborated_type',
    'noise_xml': 'ChipNoise/Interference', 'multi_ship_xml': 'ChipNoise/Multi_ship',
    'port_noise_xml': 'ChipNoise/Port', 'match_distance_m': 'MatchInformation/Distance_error',
}


def xml_inventory(name):
    records, files = [], []
    for path in sorted(NATIVE_ROOTS[name].glob('*/Ship.xml')):
        parser = ET.XMLParser(recover=True, encoding='utf-8')
        raw = path.read_bytes()
        decoded = raw.decode('utf-8', errors='replace')
        tree = ET.fromstring(decoded.encode('utf-8'), parser)
        ships = tree.findall('.//ship')
        files.append(dict(dataset=name, product=path.parent.name, xml_rows=len(ships),
                          utf8_replacement_characters=decoded.count('\ufffd'),
                          replacement_tags='|'.join(sorted({e.tag for e in tree.iter()
                                                          if '\ufffd' in (e.text or '')})),
                          parser_messages=len(parser.error_log),
                          messages=' | '.join(str(e.message) for e in parser.error_log)[:600]))
        for ordinal, ship in enumerate(ships):
            row = dict(product=path.parent.name, xml_ordinal=ordinal,
                       center_x=ship.findtext('SARShipInformation/Center_x', '').strip(),
                       center_y=ship.findtext('SARShipInformation/Center_y', '').strip())
            row.update({k: ship.findtext(v, '').strip() for k, v in XML_FIELDS.items()})
            records.append(row)
    frame = pd.DataFrame(records)
    if frame.duplicated(['product', 'center_x', 'center_y']).any():
        raise ValueError('Ambiguous XML center join: ' + name)
    return frame, files


def public(name, xml, native=False):
    folder = ROOT / f'opensarship{name[-1]}_transfer_20261007'
    frame = read((folder / 'native_label_mapping.csv') if native and name == 'oss1'
                 else (folder / 'label_mapping.csv') if native
                 else (folder / 'target_manifest.csv'))
    frame['row_id'] = np.arange(len(frame))
    frame['product'] = frame.scene_id
    frame['identity'] = frame.mmsi_or_ship_id.where(present(frame.mmsi_or_ship_id), '')
    if name == 'oss2':
        xy = frame.source_path_vv.str.extract(r'_x(-?\d+)_y(-?\d+)_vv\.tif$')
        frame['center_x'], frame['center_y'] = xy[0], xy[1]
    frame = frame.merge(xml, on=['product', 'center_x', 'center_y'], how='left',
                        validate='one_to_one', indicator=True)
    if not frame['_merge'].eq('both').all():
        raise ValueError('Unmatched manifest/XML rows: ' + name)
    if not frame.identity.eq(frame.xml_mmsi.where(present(frame.xml_mmsi), '')).all():
        raise ValueError('Manifest/XML MMSI mismatch: ' + name)
    frame = parse_product(frame.drop(columns='_merge'))
    frame['dataset'] = name + ('_native' if native else '')
    frame['marine_type_present'] = present(frame.marine_type_xml)
    frame['ais_datetime'] = pd.to_datetime(pd.to_numeric(frame.ais_epoch, errors='coerce'), unit='s', utc=True)
    sar = pd.to_datetime(frame.sar_time, format='%Y/%m/%d.%H:%M:%S', errors='coerce', utc=True)
    frame['sar_datetime'] = sar
    frame['ais_minus_sar_seconds'] = (frame.ais_datetime - sar).dt.total_seconds()
    frame['geo_key'] = pd.to_numeric(frame.sar_lon, errors='coerce').round(6).astype(str) + '|' + pd.to_numeric(frame.sar_lat, errors='coerce').round(6).astype(str)
    return frame


def summary(frame, name):
    return dict(dataset=name, rows=len(frame), products=frame['product'].nunique(),
                acquisition_start_keys=frame.acquisition_start_key.nunique(), datatakes=frame.datatake_key.nunique(),
                identities=frame.loc[present(frame.identity), 'identity'].nunique(),
                missing_identity=int((~present(frame.identity)).sum()),
                date_min=frame.start.min(), date_max=frame.start.max(),
                classes=frame.loc[frame.class_name.ne(''), 'class_name'].nunique() if 'class_name' in frame else None)


def overlap(a, b, name_a, name_b, key):
    aa = a.loc[present(a.identity)] if 'identity' in key else a
    bb = b.loc[present(b.identity)] if 'identity' in key else b
    aa, bb = aa.dropna(subset=key), bb.dropna(subset=key)
    sa = set(aa[key].itertuples(index=False, name=None))
    sb = set(bb[key].itertuples(index=False, name=None))
    common = sa & sb
    return dict(dataset_a=name_a, dataset_b=name_b, key='+'.join(key), shared_keys=len(common),
                a_rows=sum(v in common for v in aa[key].itertuples(index=False, name=None)),
                b_rows=sum(v in common for v in bb[key].itertuples(index=False, name=None)))


def main():
    xmls, inventories = {}, []
    for name in NATIVE_ROOTS:
        xmls[name], records = xml_inventory(name)
        inventories.extend(records)
    save('xml_inventory', inventories)
    frames = {name: public(name, xmls[name]) for name in NATIVE_ROOTS}
    natives = {name + '_native': public(name, xmls[name], True) for name in NATIVE_ROOTS}
    source = read(SOURCE / 'manifest.csv')
    source['sample_id'] = source.instance_id
    source['identity'] = source.mmsi.where(present(source.mmsi), '')
    source['row_id'] = np.arange(len(source))
    source = parse_product(source)
    frames['source24'] = source
    frames['oss2_grdh'] = frames['oss2'].loc[frames['oss2'].product_type.eq('GRDH')].copy()
    allframes = {**frames, **natives}
    save('cohort_summary', [summary(f, n) for n, f in allframes.items()])
    dimensions, fields, nums = [], [], []
    for name, frame in allframes.items():
        for keys in [['class_name'], ['platform', 'mode', 'product_type'], ['class_name', 'product_type'],
                     ['class_name', 'label_field'], ['marine_type_present'], ['noise_xml'],
                     ['class_name', 'noise_xml'], ['data_origin', 'class_name'], ['label_mapping_reason']]:
            if all(k in frame for k in keys):
                for values, count in frame.groupby(keys, dropna=False).size().items():
                    if not isinstance(values, tuple):
                        values = (values,)
                    dimensions.append(dict(dataset=name, dimension='+'.join(keys),
                                           value='|'.join(map(str, values)), rows=int(count)))
        for field in frame:
            fields.append(dict(dataset=name, field=field, rows=len(frame),
                               nonempty=int((~frame[field].astype(str).str.lower().isin(['', 'na', 'nan', 'none', 'nat'])).sum())))
        for field in ['ais_minus_sar_seconds', 'incidence_xml', 'ais_length', 'ais_width', 'manual_length', 'match_distance_m']:
            if field not in frame:
                continue
            v = pd.to_numeric(frame[field], errors='coerce').dropna()
            nums.append(dict(dataset=name, field=field, valid=len(v), zero=int(v.eq(0).sum()),
                             min=v.min(), median=v.median(), p95=v.quantile(.95), max=v.max(),
                             abs_gt_300=int(v.abs().gt(300).sum()) if field == 'ais_minus_sar_seconds' else None))
    save('cohort_dimensions', dimensions)
    save('field_availability', fields)
    save('numeric_metadata', nums)
    overlaps = []
    for a, b in [('oss1', 'oss2'), ('oss1', 'oss2_grdh'), ('source24', 'oss1'), ('source24', 'oss2'),
                 ('oss1_native', 'oss2_native')]:
        for keys in [['identity'], ['product'], ['acquisition_start_key'], ['datatake_key'],
                     ['identity', 'datatake_key'], ['identity', 'sar_datetime'],
                     ['identity', 'datatake_key', 'geo_key']]:
            if all(k in allframes[a] and k in allframes[b] for k in keys):
                overlaps.append(overlap(allframes[a], allframes[b], a, b, keys))
    save('overlap_summary', overlaps)
    pair_summaries = []
    for a, b in [('oss1', 'oss2'), ('oss1_native', 'oss2_native')]:
        cols = ['sample_id', 'product', 'identity', 'datatake_key', 'class_name', 'sar_datetime',
                'center_x', 'center_y', 'sar_lon', 'sar_lat', 'geo_key', 'product_type', 'ais_epoch',
                'manual_length', 'ais_length', 'ais_width', 'marine_type_xml']
        aa, bb = allframes[a], allframes[b]
        pairs = aa[[c for c in cols if c in aa]].merge(bb[[c for c in cols if c in bb]],
                    on=['identity', 'datatake_key'], suffixes=('_a', '_b'))
        pairs['same_geo'] = pairs.geo_key_a.eq(pairs.geo_key_b)
        pairs['same_sar_time'] = pairs.sar_datetime_a.eq(pairs.sar_datetime_b)
        pairs['same_ais_message_time'] = pairs.ais_epoch_a.eq(pairs.ais_epoch_b)
        pairs['same_class'] = pairs.class_name_a.eq(pairs.class_name_b)
        coords = [np.radians(pd.to_numeric(pairs[c], errors='coerce').to_numpy())
                  for c in ['sar_lon_a', 'sar_lat_a', 'sar_lon_b', 'sar_lat_b']]
        lon_a, lat_a, lon_b, lat_b = coords
        h = np.sin((lat_b-lat_a)/2)**2 + np.cos(lat_a)*np.cos(lat_b)*np.sin((lon_b-lon_a)/2)**2
        pairs['sar_geo_distance_m'] = 6371000 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        for field in ['manual_length', 'ais_length', 'ais_width']:
            pairs[field + '_delta_b_minus_a'] = pd.to_numeric(pairs[field + '_b'], errors='coerce') - pd.to_numeric(pairs[field + '_a'], errors='coerce')
        for scope, subset in [('all_datatake_pairs', pairs), ('same_sar_time', pairs.loc[pairs.same_sar_time])]:
            result = dict(dataset_a=a, dataset_b=b, scope=scope, pairs=len(subset),
                          same_class=int(subset.same_class.sum()),
                          same_ais_message=int(subset.same_ais_message_time.sum()),
                          exact_geo=int(subset.same_geo.sum()),
                          geo_median_m=subset.sar_geo_distance_m.median(),
                          geo_max_m=subset.sar_geo_distance_m.max())
            for field in ['manual_length', 'ais_length', 'ais_width']:
                delta = subset[field + '_delta_b_minus_a'].abs()
                result.update({field + '_equal': int(delta.eq(0).sum()),
                               field + '_absdiff_median': delta.median(),
                               field + '_absdiff_max': delta.max()})
            pair_summaries.append(result)
        save(a + '_' + b + '_acquisition_pairs', pairs)
    save('coobservation_summary', pair_summaries)
    save('public_metadata', pd.concat([frames['oss1'], frames['oss2']], ignore_index=True))
    purges = []
    common = np.arange(len(source))
    for name in ['oss1', 'oss2']:
        rows = np.load(ROOT / f'opensarship{name[-1]}_transfer_20261007/source_split.npz')['source_rows']
        common = np.intersect1d(common, rows)
        for scope, target in [('fine', frames[name]), ('native', natives[name + '_native'])]:
            same = set(target.loc[present(target.identity), 'identity'])
            purges.append(dict(dataset=name, target_scope=scope, source_rows=len(rows),
                               full_source_overlap=int(source.identity.isin(same).sum()),
                               retained_source_overlap=int(source.iloc[rows].identity.isin(same).sum())))
    for fold in range(5):
        rows = np.load(WORK / f'ports_public_data_study_20261012/fold_{fold}/fits/source_rows.npy')
        for name in ['oss1', 'oss2']:
            purges.append(dict(dataset=name, target_scope=f'fold_{fold}', source_rows=len(rows),
                               full_source_overlap=int(source.identity.isin(set(frames[name].identity)).sum()),
                               retained_source_overlap=int(source.iloc[rows].identity.isin(set(frames[name].identity)).sum())))
    save('source_purge', purges)
    duplicates = []
    for name, frame in frames.items():
        for keys in [['identity'], ['identity', 'product'], ['identity', 'datatake_key'],
                     ['identity', 'product', 'tile_base']]:
            if not all(k in frame for k in keys):
                continue
            valid = frame.loc[present(frame.identity)]
            groups = valid.groupby(keys).agg(rows=('sample_id', 'size'), classes=('class_name', 'nunique')).reset_index()
            d = groups.loc[groups.rows.gt(1)]
            duplicates.append(dict(dataset=name, key='+'.join(keys), groups=len(d),
                                   rows=int(d.rows.sum()), excess_rows=int((d.rows-1).sum()),
                                   conflicting_label_groups=int(d.classes.gt(1).sum())))
            if name == 'source24' and len(keys) == 3:
                save('source_local_repeated_keys', d)
            if name == 'oss2' and keys == ['identity', 'datatake_key']:
                repeat_rows = valid.merge(d[keys], on=keys)
                save('oss2_within_datatake_repeated_rows', repeat_rows)
                conflicts = groups.loc[groups.classes.gt(1), keys]
                save('oss2_label_conflict_rows', valid.merge(conflicts, on=keys))
    save('within_cohort_repeats', duplicates)
    support = []
    for name in ['oss1', 'oss2', 'oss2_grdh']:
        f = frames[name].set_index('sample_id', drop=False)
        with np.load(WORK / f'public_oss_comparison_20261012/targets/shared_frozen/{name}/predictions.npz') as z:
            f = f.loc[z['ids']].reset_index(drop=True)
            for draw, indices in enumerate(z['support']):
                sf, qf = f.iloc[indices], f.loc[z['eval_masks'][draw]]
                for keys in [['identity'], ['product'], ['identity', 'datatake_key']]:
                    row = overlap(sf, qf, name + '_support', name + '_query', keys)
                    row.update(dataset=name, draw=draw)
                    support.append(row)
    save('support_query_overlap', support)
    probes = []
    for name in ['oss1', 'oss2', 'source24']:
        frame = frames[name]
        groups = ['product_type', 'class_name'] + (['data_origin'] if name == 'source24' else [])
        selected = frame.groupby(groups, sort=True).head(2)
        for row in selected.itertuples():
            for pol in ['vv', 'vh']:
                path = getattr(row, ('image_relpath_' if name == 'source24' else 'chip_path_') + pol)
                with Image.open(path) as im:
                    a = np.asarray(im)
                    record = dict(dataset=name, sample_id=row.sample_id, class_name=row.class_name,
                                  pol=pol, path=path, mode=im.mode, dtype=str(a.dtype),
                                  width=im.width, height=im.height, pixel_min=int(a.min()), pixel_max=int(a.max()))
                if name == 'oss2':
                    src = getattr(row, 'source_path_' + pol)
                    with Image.open(src) as im:
                        original = np.asarray(im)
                    record.update(tiff_path=src, tiff_png_equal=bool(np.array_equal(a, original)))
                probes.append(record)
    save('image_probes', probes)
    save('image_probe_summary', pd.DataFrame(probes).groupby(['dataset', 'mode', 'dtype']).size().reset_index(name='files'))
    write('COMPLETE.json', dict(status='complete', xml_files=len(inventories),
                               xml_rows={n: len(f) for n, f in xmls.items()},
                               cohorts={n: len(f) for n, f in allframes.items()},
                               shared_source_rows=len(common), image_probe_files=len(probes),
                               no_image_hashes=True, model_fits=0, gpu_runs=0,
                               xml_join='product + native x/y; unique and MMSI checked',
                               acquisition_caution='datatake is a shared observation opportunity, not identical pixels'))
    print(pd.read_csv(OUT / 'cohort_summary.csv').to_string(index=False))
    print(pd.read_csv(OUT / 'overlap_summary.csv').to_string(index=False))


if __name__ == '__main__':
    main()
