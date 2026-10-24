"""Locate input origin of the coarse-neighbor perturbation before clipping."""
import json
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
import run


def coarse(x):
    return gaussian_filter(x, 4, mode='reflect')-gaussian_filter(x, 12, mode='reflect')


def main():
    table = pd.read_csv(run.UP/'table.csv', dtype={'mmsi': str})
    cohort = pd.read_csv(run.HERE/'cohort.csv', dtype={'mmsi': str})
    snapshots = {int(s['obs_id']): s for s in json.loads((run.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    rows = []
    for _, row in cohort.iterrows():
        i = int(row.table_index)
        for p in range(2):
            pix, fp, g, placement, bands, regions = run.load_view(table, snapshots, i, p)
            with Image.open(table.iloc[i]['image_'+('vv','vh')[p]]) as im:
                raw = np.array(im.convert('L'), dtype=np.float32)/255
            native_g = g.copy()
            native_g[:2] += [placement['crop_x']-placement['pad_left'], placement['crop_y']-placement['pad_top']]
            body = run.geo.soft_regions(native_g, shape=raw.shape)[0]
            all_delta = -.25*run.window(run.place(bands[2], placement), regions[1])
            body_delta = -.25*run.window(run.place(coarse(raw*body), placement), regions[1])
            outside_delta = -.25*run.window(run.place(coarse(raw*(1-body)), placement), regions[1])
            error = float(abs(all_delta-body_delta-outside_delta).max())
            assert error < 2e-7, error
            denominator = max(float((all_delta*all_delta).sum()), 1e-20)
            projection = float((body_delta*all_delta).sum()/denominator)
            rows.append(dict(obs_id=int(row.obs_id), class_id=int(row.class_id), main_sample=bool(row.main_sample), pol=p,
                body_signed_projection=projection, outside_signed_projection=float((outside_delta*all_delta).sum()/denominator),
                body_norm_ratio=float(np.linalg.norm(body_delta)/np.sqrt(denominator)),
                outside_norm_ratio=float(np.linalg.norm(outside_delta)/np.sqrt(denominator)),
                closure_error=error))
    f = pd.DataFrame(rows); f.to_csv(run.HERE/'coarse_origins.csv', index=False)
    summaries = []
    for p, part in f[f.main_sample].groupby('pol'):
        summaries.append(dict(pol=int(p), n=len(part), median_body_projection=part.body_signed_projection.median(),
            q10_body_projection=part.body_signed_projection.quantile(.1), q90_body_projection=part.body_signed_projection.quantile(.9),
            fraction_body_projection_gt_half=(part.body_signed_projection>.5).mean(),
            median_outside_norm_ratio=part.outside_norm_ratio.median(), max_closure=part.closure_error.max()))
    s = pd.DataFrame(summaries); s.to_csv(run.HERE/'coarse_origins_summary.csv', index=False)
    print(s.to_string(index=False))


if __name__ == '__main__':
    main()
