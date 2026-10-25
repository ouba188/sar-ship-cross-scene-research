"""OSS2 native inputs and frozen OBB association; writes metadata beside this file."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
LEGACY = WORK / 'scene_aware_classification_20261018/oss2_distribution_transfer_20261023'
DETECTOR = Path('E:/Docms/SARlabelimg/550k.pt')
CLASS_MAP = {0: 0, 2: 1, 3: 2, 4: 3}
sys.path.insert(0, str(WORK / 'sar_m123_standalone'))
from sar_m123.input import prepare_view


def save_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def detect(table, device, batch_size):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    path = HERE / 'detection_views.json'
    records = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    done = {(r['sample_id'], r['pol']) for r in records}
    jobs = [(r, pol) for r in table.to_dict('records') for pol in ('vv', 'vh')
            if (r['sample_id'], pol) not in done]
    if not jobs:
        return records
    model = YOLO(str(DETECTOR), task='obb')
    started = time.perf_counter()
    for first in range(0, len(jobs), batch_size):
        batch = jobs[first:first + batch_size]
        images, sizes = [], []
        for row, pol in batch:
            with Image.open(row['chip_path_' + pol]) as im:
                raw = np.asarray(im.convert('L'))
            h, w = raw.shape
            side = max(1024, int(np.ceil(max(h, w) / 32) * 32))
            ox, oy = (side-w)//2, (side-h)//2
            canvas = np.zeros((side, side, 3), np.uint8)
            canvas[oy:oy+h, ox:ox+w] = raw[:, :, None]
            images.append(canvas)
            sizes.append((h, w, ox, oy))
        outputs = model.predict(images, imgsz=1024, conf=.08, iou=.3,
                                device=device, verbose=False, save=False)
        for (row, pol), result, (h, w, ox, oy) in zip(batch, outputs, sizes):
            raw = result.obb.xywhr.cpu().numpy().copy()
            confidence = result.obb.conf.cpu().numpy()
            raw[:, :2] -= [ox, oy]
            long, short = raw[:, 2:4].max(1), raw[:, 2:4].min(1)
            accepted = np.linalg.norm(raw[:, :2]-[(w-1)/2, (h-1)/2], axis=1) <= max(8., .3*min(h, w))
            accepted &= (raw[:, 0] >= 0) & (raw[:, 0] < w) & (raw[:, 1] >= 0) & (raw[:, 1] < h)
            accepted &= (long >= 4) & (short >= 1.5) & (long <= 1.5*max(h, w))
            candidates = np.flatnonzero(accepted)
            rec = dict(obs_id=int(row['obs_id']), sample_id=row['sample_id'], pol=pol,
                       native_h=h, native_w=w, detected=bool(len(candidates)),
                       detector_path=str(DETECTOR), detection_available=True,
                       raw_detection_count=len(raw), accepted_detection_count=len(candidates))
            if len(candidates):
                j = int(candidates[np.argmax(confidence[candidates])])
                cx, cy, bw, bh, angle = raw[j]
                rec.update(cx=float(cx), cy=float(cy), length=float(max(bw, bh)),
                           width=float(min(bw, bh)), ship_axis=float((angle+(np.pi/2 if bh > bw else 0)) % np.pi),
                           confidence=float(confidence[j]), raw_xywhr=raw[j].tolist(),
                           axis_valid=bool(max(bw, bh)/min(bw, bh) >= 1.2))
            else:
                rec.update(cx=None, cy=None, length=None, width=None, ship_axis=None,
                           confidence=0., raw_xywhr=None, axis_valid=False)
            records.append(rec)
        if first % (batch_size*20) == 0 or first+batch_size >= len(jobs):
            save_json('detection_views.json', records)
            print('OBB', min(first+batch_size, len(jobs)), '/', len(jobs),
                  'seconds', round(time.perf_counter()-started, 1), flush=True)
    return records


def view_spec(view):
    detections, axis = [], None
    if view['detected'] and view['confidence'] >= .25:
        cx, cy, length, width, angle = view['raw_xywhr']
        uv = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [length/2, width/2]
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        detections = [dict(corners=(uv @ rotation.T + [cx, cy]).tolist(), confidence=view['confidence'])]
        if view['axis_valid']:
            axis = dict(center=[view['cx'], view['cy']], length_px=view['length'],
                        width_px=view['width'], angle=view['ship_axis'])
    return dict(image=view['path'], detections=detections, target_index=0 if detections else None,
                ship_axis=axis, radar_axis_rad=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detect', action='store_true')
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    full = pd.read_csv(LEGACY / 'cohort.csv', dtype={'mmsi': str, 'mmsi_or_ship_id': str})
    full = full.rename(columns={'row_index': 'legacy_row_index', 'class_id': 'original_class_id'})
    full['obs_id'] = full.manifest_position.astype(int)
    full['pass_id'] = full.pass_group
    full['overpass'] = full.pass_group
    full['class_id'] = full.original_class_id.map(CLASS_MAP).astype('Int64')
    table = full[full.original_class_id.isin(CLASS_MAP)].copy()
    table['class_id'] = table.class_id.astype(int)
    table['y'] = table.class_id
    assert table.sample_id.is_unique and table.obs_id.is_unique
    assert table.mmsi.str.fullmatch(r'[0-9]{9}').all()
    assert not table.mmsi.eq('000000000').any()
    full[~full.original_class_id.isin(CLASS_MAP)].assign(
        exclusion_reason='outside_prespecified_four_classes').to_csv(HERE / 'class_scope_exclusions.csv', index=False)
    if args.detect:
        detections = detect(table, args.device, args.batch_size)
    else:
        path = HERE / 'detection_views.json'
        detections = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    lookup = {(d['sample_id'], d['pol']): d for d in detections}
    assert len(lookup) == len(detections)
    native = pd.read_csv(LEGACY / 'native_geometry.csv').set_index(['sample_id', 'pol'])
    views, sample_stats = [], []
    for row in table.to_dict('records'):
        stat = dict(sample_id=row['sample_id'], all_views_prepare_ok=True, prepare_error='',
                    any_obb=False, both_obb=True, detection_complete=True)
        for pol in ('vv', 'vh'):
            path = Path(row['chip_path_' + pol])
            v = dict(obs_id=row['obs_id'], sample_id=row['sample_id'], pol=pol,
                     path=str(path), exists=path.is_file(), detection_available=False,
                     detected=False, confidence=0., axis_valid=False, raw_xywhr=None,
                     cx=None, cy=None, length=None, width=None, ship_axis=None)
            if v['exists']:
                with Image.open(path) as im:
                    v.update(actual_w=im.width, actual_h=im.height, mode=im.mode,
                             native_w=im.width, native_h=im.height)
                assert (v['native_w'], v['native_h']) == (row['chip_width_px'], row['chip_height_px'])
            else:
                raise FileNotFoundError(path)
            d = lookup.get((row['sample_id'], pol))
            if d is not None:
                assert d['obs_id'] == row['obs_id']
                assert (d['native_w'], d['native_h']) == (v['native_w'], v['native_h'])
                v.update(d)
            stat['detection_complete'] &= v['detection_available']
            v['accepted_target'] = bool(v['detected'] and v['confidence'] >= .25)
            v['native_crop160'] = max(v['native_w'], v['native_h']) > 160
            v['prepare_ok'], v['prepare_error'] = False, ''
            if v['detection_available']:
                try:
                    prepared = prepare_view(view_spec(v))
                    v['prepare_ok'] = True
                    v.update(prepared['placement'])
                    v['valid_target'] = bool(prepared['valid_target'])
                    v['retained_native_fraction'] = float(prepared['footprint'].sum()/(v['native_w']*v['native_h']))
                    n = native.loc[(row['sample_id'], pol)]
                    left, top = v['crop_x'], v['crop_y']
                    right, bottom = left+v['crop_width'], top+v['crop_height']
                    v['estimated_xml_head_tail_inside_actual_crop'] = bool(
                        np.isfinite([n.head_x, n.tail_x, n.head_y, n.tail_y]).all()
                        and min(n.head_x, n.tail_x) >= left and max(n.head_x, n.tail_x) < right
                        and min(n.head_y, n.tail_y) >= top and max(n.head_y, n.tail_y) < bottom)
                except ValueError as error:
                    v['prepare_error'] = str(error)
                    stat['all_views_prepare_ok'] = False
                    stat['prepare_error'] += pol + ': ' + str(error) + '; '
            else:
                stat['all_views_prepare_ok'] = False
                stat['prepare_error'] += pol + ': detection unavailable; '
            stat['obb_' + pol] = v['accepted_target']
            stat['valid_' + pol] = bool(v.get('valid_target', False))
            stat['axis_' + pol] = bool(v['accepted_target'] and v['axis_valid'])
            stat['any_obb'] |= v['accepted_target']
            stat['both_obb'] &= v['accepted_target']
            v['raw_xywhr'] = json.dumps(v['raw_xywhr'])
            views.append(v)
        stat['runnable'] = bool(stat['detection_complete'] and stat['all_views_prepare_ok'] and
                                (stat['valid_vv'] or stat['valid_vh']))
        stat['exclusion_reason'] = ('detection_pending' if not stat['detection_complete'] else
                                    'prepare_protocol_blocker' if not stat['all_views_prepare_ok'] else
                                    'no_obb_either_polarization_conf025' if not stat['any_obb'] else '')
        sample_stats.append(stat)
    table = table.merge(pd.DataFrame(sample_stats), on='sample_id', validate='one_to_one')
    table['input_scale'], table['canvas_size'] = 1., 160
    table.to_csv(HERE / 'data_inventory.csv', index=False)
    view_table = pd.DataFrame(views)
    view_table.to_csv(HERE / 'view_inventory.csv', index=False)
    view_table[view_table.native_crop160].to_csv(HERE / 'crop160_impact.csv', index=False)
    table[table.exclusion_reason.ne('')].to_csv(HERE / 'inventory_exclusions.csv', index=False)
    pd.DataFrame(dict(sample_id=table.sample_id, radar_axis_valid=False, radar_axis_rad=np.nan,
                      radar_axis_source='not_reliably_established_for_OSS2_native_chip')).to_csv(HERE / 'geometry.csv', index=False)
    coverage = table.groupby(['class_id', 'original_class_id', 'class_name']).agg(
        observations=('sample_id', 'size'), vessels=('mmsi', 'nunique'), passes=('pass_group', 'nunique'),
        any_obb=('any_obb', 'sum'), both_obb=('both_obb', 'sum'), runnable=('runnable', 'sum'))
    coverage.to_csv(HERE / 'class_coverage.csv')
    receipt = dict(status='ready' if table.detection_complete.all() and table.all_views_prepare_ok.all() else 'blocked',
        observations=len(table), vessels=table.mmsi.nunique(), passes=table.pass_group.nunique(),
        runnable=int(table.runnable.sum()), any_obb=int(table.any_obb.sum()), both_obb=int(table.both_obb.sum()),
        exclusions=table.exclusion_reason.value_counts().to_dict(),
        native_larger_than160_observations=int(view_table[view_table.native_crop160].sample_id.nunique()),
        original_eight_class_larger_than160=int(full.chip_width_px.gt(160).sum()),
        detection_records=len(detections), detector=str(DETECTOR), detector_pretraining_provenance='not audited',
        detector_policy='identical to OSS1 detect_axis.py: native centered pad1024 conf.08 iou.3; centered plausible box, max confidence; final conf.25',
        radar_valid=0, class_map=CLASS_MAP, obs_id_definition='manifest_position in full 3745-row target_manifest; NOT GRDH compact row_index',
        full_prepare_view_checks=len(view_table), training_run=False, git_used=False)
    save_json('PREPARATION_RECEIPT.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    print(coverage.to_string(), flush=True)


if __name__ == '__main__':
    main()
