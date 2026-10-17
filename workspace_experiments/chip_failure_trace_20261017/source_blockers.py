"""Show native source chips whose local auxiliary supervision is blocked."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAIRS = [(834, 1044, 'Fishing: only same-class support is locally invalid'),
         (283, 833, 'Offshore supply: only same-class support is locally invalid'),
         (88, 955, 'Tug: query and both available supports lack accepted axes')]


def main():
    geometry = pd.read_csv(ROOT/'oss1_joint_geometry_20261017/geometry.csv',
                           dtype={'mmsi': str}).set_index('global_row')
    detections = pd.read_csv(ROOT/'oss1_joint_geometry_20261017/detections.csv').set_index('global_row')
    views = pd.DataFrame(json.loads((ROOT/'oss1_joint_geometry_20261017/detection_views.json').read_text()))
    ids = sorted({13, 88, 283, 833, 834, 955, 959, 1016, 1044})
    records = []
    for index in ids:
        g, d = geometry.loc[index], detections.loc[index]
        record = dict(global_row=index, class_name=g.class_name, mmsi=g.mmsi,
            acquisition=g.acquisition, axis_valid=d.axis_valid,
            confidence=d.confidence, native_h=d.native_h, native_w=d.native_w,
            chip_path_vv=g.chip_path_vv, chip_path_vh=g.chip_path_vh,
            raw_patch_path=g.raw_patch_path, calibrated_patch_path=g.calibrated_patch_path,
            ship_xml_path=g.ship_xml_path)
        raw = tifffile.imread(g.raw_patch_path)
        calibrated = tifffile.imread(g.calibrated_patch_path)
        assert raw.shape == calibrated.shape == (d.native_h, d.native_w, 2)
        record.update(raw_finite=bool(np.isfinite(raw).all()), raw_min=float(raw.min()),
            raw_max=float(raw.max()), calibrated_finite=bool(np.isfinite(calibrated).all()))
        for pol in ['vv', 'vh']:
            image = tifffile.imread(g['chip_path_'+pol])
            view = views[views.global_row.eq(index)&views.pol.eq(pol)].iloc[0]
            record[pol+'_detected'] = bool(view.detected)
            record[pol+'_axis_valid'] = bool(view.axis_valid)
            record[pol+'_pixel_max'] = int(image.max())
            record[pol+'_pixel_std'] = float(image.std())
            record[pol+'_finite'] = bool(np.isfinite(image).all())
        records.append(record)
    pd.DataFrame(records).to_csv(HERE/'source_blocked_chips.csv', index=False)
    fig, axes = plt.subplots(3, 4, figsize=(12, 8.5))
    for row, (query, support, title) in enumerate(PAIRS):
        for col, (index, pol) in enumerate([(query,'vv'),(query,'vh'),(support,'vv'),(support,'vh')]):
            g, d = geometry.loc[index], detections.loc[index]
            image = tifffile.imread(g['chip_path_'+pol])
            ax = axes[row,col]
            ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_title(f"{'Source query' if col<2 else 'Same-class support'} {index} | {pol.upper()}\n"
                         f"{image.shape[1]}x{image.shape[0]} | accepted axis: {bool(d.axis_valid)}", fontsize=9)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(title.split(':', 1)[0], fontsize=9)
    fig.suptitle('Source supervision blockers | native uint8, fixed 0-255 display\n'
                 'No accepted detection is not proof of an empty chip or a wrong label', fontsize=11)
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(HERE/'figures/source_blocked_chips.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
