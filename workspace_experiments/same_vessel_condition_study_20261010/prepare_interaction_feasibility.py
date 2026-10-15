"""Source-only AIS structure/motion and cached native-grid viewing proxies."""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE/'interaction_feasibility_20261012'
GEOMETRY = Path('E:/Hermes/paper_tracking/sensitivity/oof_reliability_check')
sys.path.insert(0, str(GEOMETRY))
from relative_radar_geometry import DEFAULT_CACHE, range_azimuth
from incidence_transport import matched_pairs


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'data.npz').exists():
        raise ValueError('Preserve completed interaction data')
    meta = pd.read_csv(BASE/'observations.csv')
    motion = pd.read_csv(BASE/'motion_knowledge_20261011/motion_knowledge.csv').set_index('obs_id')
    with np.load(BASE/'motion_knowledge_20261011/attributes.npz') as f:
        arrays = {k: f[k] for k in f.files}
    with np.load(BASE/'knowledge_train_only_20261011/attributes.npz') as f:
        dims = {k: f[k] for k in f.files}
    n = len(meta)
    values = np.full((n, 7), np.nan)
    # logL, logW, sin2relative_heading, cos2relative_heading, incidence/10,
    # log1pSOG, signed ground-range-axis speed proxy (not beam radial speed).
    rows, grids, coverage = [], {}, []
    for role in ('fit', 'held'):
        ids = arrays[role+'_obs_ids']
        np.testing.assert_array_equal(ids, dims[role+'_obs_ids'])
        dm = dims[role+'_valid']
        values[ids[dm], :2] = np.log(dims[role+'_values'][dm])
        for i in ids:
            r = motion.loc[i]; original = meta.iloc[i]
            assert int(r.mmsi) == int(original.mmsi) and r['product'] == original['product']
            entry = dict(obs_id=int(i), role=role, grid_status='no_exact_motion',
                         heading=r.true_heading_deg, sog=r.sog_knots, range_azimuth=np.nan)
            if r.match_status == 'matched':
                product = r['product']
                path = DEFAULT_CACHE/f'{product}.json'
                if product not in grids:
                    grids[product] = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
                if grids[product] is None:
                    entry['grid_status'] = 'not_in_existing_cache'
                else:
                    grid = grids[product]
                    assert grid['product'] == product
                    local = range_azimuth(grid, float(r.raw_lon), float(r.raw_lat))
                    entry['grid_status'] = local['status']
                    if local['status'] == 'ok':
                        azimuth = float(local['azimuth_deg'])
                        entry.update(range_azimuth=azimuth, nearest_grid_m=local['nearest_grid_distance_m'])
                        if np.isfinite(r.true_heading_deg):
                            relative = np.deg2rad(2*(r.true_heading_deg-azimuth))
                            values[i, 2:4] = [np.sin(relative), np.cos(relative)]
                        if np.isfinite(r.cog_deg) and np.isfinite(r.sog_knots):
                            values[i, 6] = r.sog_knots*np.cos(np.deg2rad(r.cog_deg-azimuth))
                if np.isfinite(r.sog_knots) and 0 <= r.sog_knots < 102.2:
                    values[i, 5] = np.log1p(r.sog_knots)
                values[i, 4] = original.incidence_deg/10.
            rows.append(entry)
        eligible = ids[np.isfinite(values[ids, :6]).all(1)]
        pairs = matched_pairs(meta, eligible)
        arrays[role+'_eligible'] = eligible
        arrays[role+'_pairs'] = pairs
        coverage.append(dict(role=role, original=len(ids), eligible=len(eligible),
            ships=int(meta.iloc[eligible].mmsi.nunique()), pairs=len(pairs),
            pair_ships=int(meta.iloc[pairs[:, 0]].mmsi.nunique()),
            pair_classes=sorted(map(int, meta.iloc[pairs[:, 0]].class_id.unique()))))
    np.savez_compressed(OUT/'data.npz', values=values,
        feature_names=np.array(['log_length', 'log_beam', 'sin2body_range', 'cos2body_range',
                                'incidence_over10', 'log1p_sog', 'range_speed_proxy']),
        **{k: v for k, v in arrays.items() if k.endswith(('_obs_ids', '_eligible', '_pairs'))})
    pd.DataFrame(rows).to_csv(OUT/'geometry_join.csv', index=False)
    receipt = dict(coverage=coverage, cached_products=sum(v is not None for v in grids.values()),
        missing_products=sum(v is None for v in grids.values()), downloads=0,
        geometry='Native exact-product local GRD increasing-pixel ground-range axis, not exact beam direction.',
        body='AIS reported true heading, not COG, no image-derived axis.',
        structure='AIS reported length/beam proxies only, not hull shape or scattering-center geometry.',
        motion='SOG translation proxy only; no measured roll/pitch/yaw angular velocity or acceleration.',
        pairing='Same vessel/port/platform/known pass, different acquisitions; same historical source split.')
    (OUT/'DATA_COVERAGE.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
