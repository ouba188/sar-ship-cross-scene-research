"""Original-label provenance and fresh native-pixel detector geometry diagnostic.

This is not a replay of the missing full-tile detector context. No source labels,
images or historical features are modified.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
import pandas as pd
from PIL import Image
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sea_ship_redetect_preview import axis_aligned_box

OUT = ROOT / 'geometry_recovery_effect_20261016/visual_diagnosis/geometry_trace'
DATA = Path('E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915')
WEIGHT = ROOT / 'v5_server_toolkit_20260912/models/550k.pt'


def bounds(detection):
    cx, cy, w, h, _ = axis_aligned_box(detection)
    return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])


def outside(image, box):
    yy, xx = np.indices(image.shape)
    x0, y0, x1, y1 = box
    inside = ((xx+.5 >= x0) & (xx+.5 < x1) &
              (yy+.5 >= y0) & (yy+.5 < y1))
    bright = image >= 200
    return int((bright & ~inside).sum()), int(bright.sum())


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / 'geometry_source_recovery_20261016/geometry_input_manifest.csv',
                           low_memory=False)
    source = pd.read_csv(DATA / 'metadata/server2/manifest.csv', low_memory=False).set_index('obj_uid')
    jobs = []
    canvases = []
    for grow in (24846, 35475):
        row = manifest.iloc[grow]
        for pol in ('vv', 'vh'):
            crop = source.loc[row['obj_uid_'+pol]]
            name = Path(crop.source_json).name
            files = list((DATA / 'metadata/local_knowledge_sync_v1/source_json/server2').glob('*_'+name))
            assert len(files) == 1
            label = json.loads(files[0].read_text(encoding='utf-8-sig'))
            obj = label['objects'][int(crop.source_object_index)]
            assert obj['obj_uid'] == crop.name
            points = np.asarray(obj['pts']) - [crop.crop_left, crop.crop_top]
            historical = np.array([crop.left-crop.crop_left, crop.top-crop.crop_top,
                                   crop.right-crop.crop_left, crop.bottom-crop.crop_top])
            rounded = np.concatenate((np.floor(points.min(0)), np.ceil(points.max(0))))
            np.testing.assert_array_equal(historical, rounded)
            with Image.open(row['image_relpath_'+pol]) as im:
                im_array = np.array(im.convert('L'))
            assert im_array.shape == (64, 64)
            # Preserve native pixel size. Unknown surrounding pixels are zero padding.
            canvas = np.zeros((1024, 1024, 3), np.uint8)
            canvas[480:544, 480:544] = im_array[:, :, None]
            canvases.append(canvas)
            jobs.append((grow, pol, crop, obj, historical, im_array, files[0]))
    model = YOLO(str(WEIGHT), task='obb')
    results = model.predict(source=canvases, imgsz=1024, conf=.08, iou=.30,
                            device='cpu', verbose=False, save=False)
    records = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, job, result in zip(axes.flat, jobs, results):
        grow, pol, crop, obj, historical, image, source_json = job
        raw = result.obb.xywhr.detach().cpu().numpy()
        conf = result.obb.conf.detach().cpu().numpy()
        keep = np.flatnonzero(np.linalg.norm(raw[:, :2]-512, axis=1) <= 16)
        if not len(keep):
            raise RuntimeError(f'No central detection for {grow} {pol}; do not fabricate a box')
        pick = int(keep[np.argmax(conf[keep])])
        cx, cy, w, h, angle = map(float, raw[pick])
        detection = dict(cx=cx-480, cy=cy-480, width=w, height=h,
                         angle=math.degrees(angle))
        correct = bounds(detection)
        wrong = bounds(dict(detection, width=max(w, h), height=min(w, h)))
        corners = result.obb.xyxyxyxy[pick].detach().cpu().numpy()-480
        np.testing.assert_allclose((correct[2:]-correct[:2]),
                                   1.25*np.ptp(corners, axis=0), atol=1e-4, rtol=0)
        old_out, total = outside(image, historical)
        new_out, _ = outside(image, correct)
        wrong_out, _ = outside(image, wrong)
        record = dict(global_row=grow, pol=pol, mmsi=str(crop.mmsi), obj_uid=crop.name,
                      source_json=str(source_json), source_object_index=int(crop.source_object_index),
                      original_angle_policy=obj['meta'].get('redetect_angle_policy'),
                      original_box=historical.tolist(), original_float_box=obj['cxcywh_angle'],
                      fresh_xywhr_native=[cx-480, cy-480, w, h, angle],
                      fresh_confidence=float(conf[pick]), fresh_width_lt_height=bool(w<h),
                      fresh_legacy_adapter_box=wrong.tolist(), fresh_correct_box=correct.tolist(),
                      display_bright_total=total, original_bright_outside=old_out,
                      fresh_legacy_bright_outside=wrong_out, fresh_correct_bright_outside=new_out,
                      historical_raw_obb_available=False, full_tile_replay=False)
        records.append(record)
        ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        for b, color, label, style in ((historical, 'cyan', 'Historical AABB', '-'),
                                      (wrong, 'orange', 'Fresh OBB, old adapter', '--'),
                                      (correct, 'lime', 'Fresh OBB, fixed adapter', '-')):
            ax.add_patch(Rectangle(b[:2], b[2]-b[0], b[3]-b[1], fill=False,
                                   edgecolor=color, linewidth=1.3, linestyle=style, label=label))
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor='magenta',
                             linewidth=.8, label='Fresh detector corners'))
        ax.set_title(f'{crop.mmsi} {pol.upper()} | native 64x64\n'
                     f'fresh w={w:.2f}, h={h:.2f}, angle={math.degrees(angle):.1f} deg', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axes.flat[0].legend(loc='lower left', fontsize=7)
    fig.suptitle('Geometry diagnosis only: native chip on zero-padded canvas, not original scene replay', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT / 'original_vs_detector_geometry.png', dpi=160)
    plt.close(fig)
    (OUT / 'trace.json').write_text(json.dumps(dict(model=str(WEIGHT),
        canvas_size=1024, native_offset=480, selected_by='highest confidence within 16 px of center',
        cases=records), indent=2), encoding='utf-8')
    pd.DataFrame(records).to_csv(OUT / 'trace.csv', index=False)
    print(pd.DataFrame(records)[['global_row','pol','fresh_width_lt_height',
                                'original_bright_outside','fresh_legacy_bright_outside',
                                'fresh_correct_bright_outside']].to_string(index=False))


if __name__ == '__main__':
    main()
