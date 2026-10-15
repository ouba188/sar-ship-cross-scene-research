"""Read exact object metadata and a bounded radar-header sample; no guessed headings."""

import collections
import json
from pathlib import Path
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

sys.path.insert(0, 'E:/临时会话/visual_reliable_baseline')
sys.path.insert(0, 'E:/Hermes/scripts')
from build_returned_local_knowledge_v1 import load_file_maps
from fetch_s1_incidence import key_for, BASE

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT/'relative_observation_inventory_20261009'
MANIFEST = Path('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv')
SYNC = Path('E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata/local_knowledge_sync_v1')


def radar_sample(frame):
    cache = json.loads((ROOT.parent/'s1_product_incidence.json').read_text(encoding='utf-8'))
    products = frame[['product']].drop_duplicates().sort_values('product')
    products['platform'] = products['product'].str[:3]
    products['pass_name'] = products['product'].map(lambda p: cache.get(p, {}).get('pass', 'missing'))
    # Two examples per available platform/pass group, at most twelve small headers.
    sample = products.groupby(['platform', 'pass_name'], sort=True).head(2).head(12)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    records = []
    for product in sample['product']:
        url = BASE+key_for(product)
        record = dict(product=product, url=url)
        try:
            request = urllib.request.Request(url, headers={'Range': 'bytes=0-80000'})
            with opener.open(request, timeout=20) as response:
                data = response.read(80001)
                record['http_status'] = response.status
            parser = ET.XMLPullParser(['end'])
            parser.feed(data)
            tags = {'platformHeading', 'pass', 'projection', 'missionId', 'startTime', 'stopTime'}
            for _, element in parser.read_events():
                name = element.tag.split('}')[-1]
                if name in tags and name not in record:
                    record[name] = element.text
            record['status'] = 'heading_present' if 'platformHeading' in record else 'heading_absent_in_prefix'
            record['bytes_read'] = len(data)
        except Exception as error:
            record['status'] = 'fetch_failed'
            record['error'] = str(error)
        records.append(record)
        print(json.dumps(record), flush=True)
    (OUT/'radar_header_samples.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(MANIFEST, low_memory=False).fillna('')
    json_map, _, georef = load_file_maps(SYNC)
    records = []
    for pol in ['vv', 'vh']:
        for remote, rows in frame.groupby('source_json_'+pol, sort=False):
            if remote not in json_map:
                continue
            payload = json.loads(json_map[remote].read_text(encoding='utf-8'))
            objects = payload.get('objects', [])
            uids = collections.defaultdict(list)
            for i, obj in enumerate(objects):
                uid = str(obj.get('obj_uid') or obj.get('source_obj_uid') or '')
                if uid:
                    uids[uid].append(i)
            for row in rows.to_dict('records'):
                uid = str(row['obj_uid_'+pol])
                indices = uids.get(uid, [])
                item = dict(instance_id=row['instance_id'], product=row['product'], port=row['port'],
                            polarization=pol, status='ambiguous_or_missing_uid')
                if len(indices) != 1:
                    records.append(item)
                    continue
                obj = objects[indices[0]]
                box = obj.get('cxcywh_angle') or []
                meta = obj.get('meta') or {}
                item.update(status='uid_matched', source_index_agrees=indices[0] == int(row['source_object_index_'+pol]),
                    georef_available=row['source_tiff_'+pol] in georef,
                    angle_policy=meta.get('redetect_angle_policy', ''),
                    box_angle=float(box[4]) if len(box) >= 5 else np.nan,
                    motion_key_names='|'.join(k for k in meta if any(t in k.lower() for t in ['heading', 'course', 'cog', 'sog', 'speed'])),
                    length_metadata_present=bool(meta.get('length_m')), beam_metadata_present=bool(meta.get('beam_m')))
                records.append(item)
    table = pd.DataFrame(records)
    table.to_csv(OUT/'object_direction_inventory.csv', index=False)
    matched = table[table.status.eq('uid_matched')]
    summary = dict(manifest_rows=len(frame), manifest_products=frame['product'].nunique(),
        exact_uid_matched_views=len(matched), exact_uid_matched_instances=matched.instance_id.nunique(),
        index_agrees_views=int(matched.source_index_agrees.sum()),
        georef_available_views=int(matched.georef_available.sum()),
        angle_policy_counts=matched.angle_policy.value_counts(dropna=False).to_dict(),
        box_angle_counts=matched.box_angle.value_counts(dropna=False).to_dict(),
        motion_key_names=matched.motion_key_names.value_counts(dropna=False).to_dict(),
        length_metadata_views=int(matched.length_metadata_present.sum()), beam_metadata_views=int(matched.beam_metadata_present.sum()),
        conclusion='Axis-aligned crop boxes must not be treated as ship-body headings. This inventory does not recover original OBB angles.',
        relative_angle_calculated=False, classifier_inputs_added=False,
        radar_sample_scope='Bounded exact-product prefix sample; not full heading coverage or local look-azimuth recovery',
        ais_schema='Raw TXT motion column meanings unverified; dedup lacks named heading/COG/SOG fields',
        next='Recover original rotated boxes or estimate image-body axes with explicit georeference; verify radar azimuth convention before combining.')
    headers = radar_sample(frame)
    summary['radar_sample_products'] = len(headers)
    summary['radar_sample_heading_present'] = sum(r['status'] == 'heading_present' for r in headers)
    (OUT/'inventory.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
