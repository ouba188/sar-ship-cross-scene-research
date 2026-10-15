"""Measure paired image axes only where exact object/georeference joins exist."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, 'E:/临时会话/visual_reliable_baseline')
from build_returned_local_knowledge_v1 import load_file_maps
from ship_axis_proxy import measure_axis, world_axis, combine_axes
from sar_joint_completion import write_json

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT/'relative_observation_geometry_20261009'
SOURCE = Path('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
SYNC = Path('E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata/local_knowledge_sync_v1')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'AXES_COMPLETE.json').exists():
        raise ValueError('Preserve completed axes')
    write_json(OUT/'AXIS_PROTOCOL.json', dict(axis='central bright-component second-moment proxy',
        resized=128, gaussian_sigma=.8, center_window_fraction=.5,
        threshold='max(q75,q50+.15*(q99-q50)) within center window',
        minimum_pixels=8, minimum_elongation=2., maximum_center_offset_fraction=.1,
        reject_window_boundary=True, maximum_vv_vh_axial_disagreement_deg=15.,
        target_labels_used=False, parameter_search=False, zero_crop_box_angle_used=False,
        angle='undirected modulo180; georeferenced endpoints transformed to WGS84 geodesic bearing',
        not_claimed='not true heading, bow/stern, or guaranteed object segmentation'))
    frame = pd.read_csv(SOURCE/'manifest.csv', low_memory=False)
    inventory = pd.read_csv(ROOT/'relative_observation_inventory_20261009/object_direction_inventory.csv')
    supported = set(zip(inventory.loc[inventory.status.eq('uid_matched'), 'instance_id'],
                        inventory.loc[inventory.status.eq('uid_matched'), 'polarization']))
    _, _, georefs = load_file_maps(SYNC)
    records, pairs = [], []
    for row in frame.to_dict('records'):
        views = {}
        for pol in ('vv', 'vh'):
            if (row['instance_id'], pol) not in supported:
                continue
            geo = georefs.get(row['source_tiff_'+pol])
            if geo is None:
                continue
            with Image.open(row['image_relpath_'+pol]) as image:
                w, h = image.size
                array = np.asarray(image.convert('L').resize((128, 128), Image.Resampling.BILINEAR))
            cx, cy = row['target_chip_center_x_'+pol], row['target_chip_center_y_'+pol]
            center = ((cx+.5)*128/w-.5, (cy+.5)*128/h-.5) if np.isfinite([cx, cy]).all() else None
            value = measure_axis(array, center)
            value.update(instance_id=row['instance_id'], product=row['product'], port=row['port'], polarization=pol)
            if np.isfinite(value['angle_pixel_deg']):
                # Undo non-square image resize before projecting the image axis.
                rad = np.deg2rad(value['angle_pixel_deg'])
                native_angle = np.rad2deg(np.arctan2(np.sin(rad)*h/128, np.cos(rad)*w/128))
                value.update(world_axis(geo, row['target_center_x_'+pol], row['target_center_y_'+pol], native_angle))
            records.append(value)
            views[pol] = value
        if not views:
            continue
        pair = dict(instance_id=row['instance_id'], product=row['product'], port=row['port'],
                    accepted=False, status='missing_or_weak_view', body_axis_deg=np.nan, disagreement_deg=np.nan)
        if len(views) == 2 and all(v['accepted'] for v in views.values()):
            angle, difference = combine_axes(views['vv']['axis_true_north_deg'], views['vh']['axis_true_north_deg'])
            accepted = difference <= 15.
            pair.update(accepted=accepted, status='accepted' if accepted else 'polarization_disagreement',
                        body_axis_deg=angle, disagreement_deg=difference,
                        longitude=views['vv']['longitude'], latitude=views['vv']['latitude'])
        pairs.append(pair)
        if len(pairs) % 2000 == 0:
            print(json.dumps(dict(instances=len(pairs), accepted=sum(p['accepted'] for p in pairs))), flush=True)
    views, pairs = pd.DataFrame(records), pd.DataFrame(pairs)
    views.to_csv(OUT/'view_axes.csv', index=False)
    pairs.to_csv(OUT/'paired_axes.csv', index=False)
    coverage = frame[['instance_id', 'port', 'class_id']].merge(pairs[['instance_id', 'accepted']],
        how='left', on='instance_id', validate='one_to_one').fillna({'accepted': False})
    coverage.groupby(['port', 'class_id']).accepted.agg(['sum', 'count']).to_csv(OUT/'axis_coverage.csv')
    write_json(OUT/'AXES_COMPLETE.json', dict(instances=len(pairs), accepted=int(pairs.accepted.sum()),
        accepted_products=pairs.loc[pairs.accepted, 'product'].nunique(),
        accepted_ports=pairs.loc[pairs.accepted, 'port'].nunique(),
        pair_status=pairs.status.value_counts().to_dict(), view_status=views.status.value_counts().to_dict()))
    print((OUT/'AXES_COMPLETE.json').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    main()
