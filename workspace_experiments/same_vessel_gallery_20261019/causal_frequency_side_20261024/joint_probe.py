"""Read native chips; persist bounded joint space/scale/direction diagnostics."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
GALLERY = ROOT.parent
REGIONS = ('body', 'end', 'side')
PILOT = ((26869, 26870), (8369, 8370), (25622, 25623), (30780, 30781))


def gradients(im, sigma):
    return tuple(gaussian_filter(im, sigma, order=o, mode='reflect', truncate=3.) * sigma
                 for o in ((0, 1), (1, 0)))


def tensor(gx, gy, mask):
    return np.array([[np.mean(gx[mask] ** 2), np.mean(gx[mask] * gy[mask])],
                     [np.mean(gx[mask] * gy[mask]), np.mean(gy[mask] ** 2)]])


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def share(t, angle):
    n = np.array([np.cos(angle), np.sin(angle)])
    return float(n @ t @ n / np.trace(t))


def checks():
    a = np.random.default_rng(24).normal(size=(31, 31))
    full = np.ones_like(a, dtype=bool)
    t = tensor(*gradients(a, 2), full)
    tr = tensor(*gradients(np.rot90(a), 2), full)
    r = rotation(-np.pi / 2)
    zero = max(abs(g).max() for g in gradients(np.ones_like(a), 2))
    err = abs(tr - r @ t @ r.T).max()
    assert zero < 1e-12 and err < 1e-12, (zero, err)
    return dict(constant_derivative_error=float(zero), rotation_tensor_error=float(err))


def load():
    data = json.loads((GALLERY / 'gallery_data.js').read_text(encoding='utf-8')
                      .split('=', 1)[1].strip().removesuffix(';'))['observations']
    frame = pd.read_csv(ROOT / 'round05_margin/diagnostic_input.csv', dtype={'mmsi': str})
    frame['timestamp'] = pd.to_datetime(frame.time)
    assert not frame.duplicated(['obs_id', 'pol']).any()
    return data, frame.set_index(['obs_id', 'pol'], drop=False)


def image_data(oid, pol, data, frame):
    row = frame.loc[(oid, pol)]
    p = data[str(oid)]['images'][pol]['priors']
    a = np.array(p['scale']['transform'])[[1, 2, 4, 5]].reshape(2, 2)
    # These native chips have an isotropic 10 m grid; do not silently resize others.
    if not np.allclose(a.T @ a, np.eye(2) * 100, atol=1e-5):
        raise ValueError(f'Unsupported native grid: {oid}/{pol}')
    im = np.asarray(Image.open(row.image_path).convert('L'), dtype=float) / 255.
    return im, p, row


def masks_for(im, sh, length, width):
    yy, xx = np.indices(im.shape)
    dx, dy = xx + .5 - sh['center'][0], yy + .5 - sh['center'][1]
    c, s = np.cos(sh['angle']), np.sin(sh['angle'])
    u, v = dx * c + dy * s, -dx * s + dy * c
    return {
        'body': (abs(u) <= length / 2) & (abs(v) <= width / 2),
        'end': (abs(u) > length / 2) & (abs(u) <= length / 2 + 8) & (abs(v) <= width / 2 + 2),
        'side': (abs(u) <= length / 2) & (abs(v) > width / 2) & (abs(v) <= width / 2 + 8),
    }


def measure(old, new, pol, data, frame):
    items = [image_data(i, pol, data, frame) for i in (old, new)]
    mean_l = np.mean([p['ship_axis']['length_px'] for _, p, _ in items])
    mean_w = np.mean([p['ship_axis']['width_px'] for _, p, _ in items])
    rows, tensors = [], {}
    for mode in ('own_box', 'paired_size'):
        for oid, (im, p, record) in zip((old, new), items):
            sh = p['ship_axis']
            length, width = (mean_l, mean_w) if mode == 'paired_size' else (sh['length_px'], sh['width_px'])
            masks = masks_for(im, sh, length, width)
            yy, xx = np.indices(im.shape)
            for scale in (10., 20., 40.):
                sigma = scale / 10.; pad = int(np.ceil(3 * sigma))
                valid = (xx >= pad) & (xx < im.shape[1] - pad) & (yy >= pad) & (yy < im.shape[0] - pad)
                gx, gy = gradients(im, sigma)
                for region, original in masks.items():
                    m = original & valid
                    coverage = float(m.sum() / original.sum()) if original.any() else 0.
                    if not m.any():
                        continue
                    t = tensor(gx, gy, m)
                    if np.trace(t) <= 1e-16:
                        continue
                    tensors[(mode, oid, region, scale)] = t
                    rows.append(dict(query_id=new, history_id=old, obs_id=oid, pol=pol,
                        mode=mode, region=region, scale_m=scale, n_pixels=int(m.sum()), coverage=coverage,
                        energy=float(np.trace(t)), ship_share=share(t, sh['angle']),
                        radar_share=share(t, p['radar_axis']), image_path=record.image_path))
    x = pd.DataFrame(rows)
    out = []
    p0, p1 = items[0][1], items[1][1]
    s0, s1 = p0['ship_axis']['angle'], p1['ship_axis']['angle']
    r0, r1 = items[0][2], items[1][2]
    turn = rotation(s1 - s0)
    for mode in ('own_box', 'paired_size'):
        for region in REGIONS:
            z = x[(x['mode'] == mode) & (x.region == region) & (x.scale_m.isin([10., 20.]))]
            if len(z) != 4:
                continue
            z = z.set_index(['obs_id', 'scale_m'])
            ratio0 = z.loc[(old, 10.), 'energy'] / z.loc[(old, 20.), 'energy']
            ratio1 = z.loc[(new, 10.), 'energy'] / z.loc[(new, 20.), 'energy']
            t0, t1 = [tensors[(mode, i, region, 10.)] for i in (old, new)]
            rot = turn @ t0 @ turn.T
            raw_dist = np.linalg.norm(t1 / np.trace(t1) - t0 / np.trace(t0))
            residual = np.linalg.norm(t1 / np.trace(t1) - rot / np.trace(rot))
            a0, a1 = r0.margin_interaction, r1.margin_interaction
            state = ('C' if a0 > 0 else 'W') + ('C' if a1 > 0 else 'W')
            out.append(dict(query_id=new, history_id=old, mmsi=r1.mmsi, pol=pol, mode=mode,
                region=region, class_id=int(r1.class_id), port=r1.port, state=state,
                query_pass=r1.overpass, history_pass=r0.overpass, eligible=bool(z.coverage.min() >= .9),
                min_coverage=z.coverage.min(), ratio_old=ratio0, ratio_new=ratio1,
                delta_log_ratio=float(np.log(ratio1 / ratio0)), tensor_distance=float(raw_dist),
                rotation_residual=float(residual), rotation_improves=bool(residual < raw_dist),
                radar_old=share(t0, p0['radar_axis']), radar_new=share(t1, p1['radar_axis']),
                radar_rotation_only=share(rot, p1['radar_axis']),
                abs_ship_turn_deg=float(abs(np.degrees(np.arctan2(np.sin(2*(s1-s0)), np.cos(2*(s1-s0)))))/2),
                abs_relative_turn_deg=float(abs(np.degrees(np.arctan2(
                    np.sin(2*((s1-p1['radar_axis'])-(s0-p0['radar_axis']))),
                    np.cos(2*((s1-p1['radar_axis'])-(s0-p0['radar_axis']))))))/2),
                delta_incidence=float(r1.incidence-r0.incidence),
                delta_saturation=float(r1.target_saturated-r0.target_saturated),
                margin_old=float(a0), margin_new=float(a1), margin_delta=float(a1-a0),
                image_margin_delta=float(r1.margin_image-r0.margin_image)))
    return rows, out


def summaries(pairs, out):
    valid = pairs[pairs.eligible].copy()
    summary = valid.groupby(['pol', 'mode', 'region']).agg(
        n=('query_id', 'size'), ships=('mmsi', 'nunique'),
        query_passes=('query_pass', 'nunique'), median_raw_tensor=('tensor_distance', 'median'),
        median_rotation_residual=('rotation_residual', 'median'),
        rotation_improves_fraction=('rotation_improves', 'mean'),
        median_abs_delta_log_ratio=('delta_log_ratio', lambda s: s.abs().median()))
    summary.to_csv(out / 'summary.csv')
    valid['abs_incidence_change'] = valid.delta_incidence.abs()
    valid['abs_saturation_change'] = valid.delta_saturation.abs()
    associations = []
    for (pol, region), g in valid[valid['mode'].eq('paired_size')].groupby(['pol', 'region']):
        comparisons = [('rotation_residual', k) for k in ('abs_ship_turn_deg', 'abs_relative_turn_deg',
                       'abs_incidence_change', 'abs_saturation_change')]
        comparisons += [('delta_log_ratio', 'margin_delta'), ('delta_log_ratio', 'image_margin_delta')]
        for a, b in comparisons:
            ok = g[[a, b]].dropna()
            rho = float(spearmanr(ok[a], ok[b]).statistic) if ok[a].nunique()>1 and ok[b].nunique()>1 else np.nan
            # Class-centered ranks check composition, not causal adjustment.
            ranks = g[[a, b]].rank()
            ranks = ranks - ranks.groupby(g.class_id).transform('mean')
            within = float(ranks[a].corr(ranks[b]))
            associations.append(dict(pol=pol, region=region, x=a, y=b, n=len(ok),
                spearman=rho, class_centered_rank_correlation=within))
    pd.DataFrame(associations).to_csv(out / 'associations.csv', index=False)
    by_class = valid[valid['mode'].eq('paired_size')].groupby(['pol', 'region', 'class_id']).agg(
        n=('query_id', 'size'), median_rotation_residual=('rotation_residual', 'median'),
        median_delta_log_ratio=('delta_log_ratio', 'median'), median_margin_delta=('margin_delta', 'median'))
    by_class.to_csv(out / 'class_summary.csv')
    a = valid[valid['mode'].eq('own_box')].set_index(['query_id', 'pol', 'region'])
    b = valid[valid['mode'].eq('paired_size')].set_index(['query_id', 'pol', 'region'])
    both = a[['delta_log_ratio']].join(b[['delta_log_ratio']], lsuffix='_own', rsuffix='_paired', how='inner')
    both['same_change_sign'] = np.sign(both.delta_log_ratio_own) == np.sign(both.delta_log_ratio_paired)
    both.to_csv(out / 'region_size_sensitivity.csv')
    return summary


def panels(pairs, data, frame, out, pilot=False):
    os.environ['MPLCONFIGDIR'] = str(out / '.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    candidates = pairs[pairs.eligible & pairs['mode'].eq('paired_size') & pairs.region.eq('body') & pairs.pol.eq('vh')]
    selected = []
    if pilot:
        selected = list(candidates.itertuples(index=False))
    else:
        for _, g in candidates.groupby('state'):
            g = g.assign(abs_change=g.margin_delta.abs()).sort_values(['abs_change', 'query_id'])
            selected.append(next(g.iloc[[len(g)//2]].itertuples(index=False)))
    pd.DataFrame([r._asdict() for r in selected]).to_csv(out/'review_selection.csv', index=False)
    for r in selected:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
        for row, pol in enumerate(('vv', 'vh')):
            items = [image_data(i, pol, data, frame) for i in (r.history_id, r.query_id)]
            maps = [sum(g*g for g in gradients(im, 1)) for im, _, _ in items]
            vmax = max(float(np.quantile(z, .99)) for z in maps)
            extent_lim = max(max(im.shape)*5 for im, _, _ in items)
            for col, (oid, (im, p, record), response) in enumerate(zip((r.history_id, r.query_id), items, maps)):
                sh = p['ship_axis']; cx, cy = sh['center']; h, w = im.shape
                extent = (-cx*10, (w-cx)*10, (h-cy)*10, -cy*10)
                for offset, z in ((0, im), (2, response)):
                    ax=axes[row, col+offset]
                    ax.imshow(z, cmap='gray' if offset==0 else 'magma', vmin=0,
                              vmax=1 if offset==0 else vmax, extent=extent, interpolation='nearest')
                    corners=np.asarray(p['detections'][p['target_index']]['corners'])
                    corners=(corners-np.array([cx,cy]))*10
                    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor='lime', linewidth=.7))
                    for angle,color in ((sh['angle'],'cyan'),(p['radar_axis'],'yellow')):
                        u=80*np.array([np.cos(angle),np.sin(angle)])
                        ax.plot([-u[0],u[0]],[-u[1],u[1]],color=color,lw=.8)
                    ax.set_xlim(-extent_lim,extent_lim);ax.set_ylim(extent_lim,-extent_lim)
                    ax.set_title(f'{oid} {pol.upper()} '+('PNG' if offset==0 else 'scale 10m response'))
                    ax.set_xlabel('native metres; no resampling')
        fig.suptitle(f'{r.state}: {r.history_id} -> {r.query_id}; interaction margin {r.margin_old:+.3f} -> {r.margin_new:+.3f}\n'
                     'Green: detected OBB; cyan: ship axis; yellow: radar axis. Diagnostic, no new prediction.')
        fig.savefig(out/f'case_{r.query_id}.png', dpi=125)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round', choices=('pilot','coverage'), required=True)
    args=parser.parse_args();pilot=args.round=='pilot'
    out=ROOT/('round06_joint_pilot' if pilot else 'round07_joint_coverage');out.mkdir(exist_ok=True)
    data,frame=load();verification=checks()
    if pilot:
        requests=[(a,b,pol) for a,b in PILOT for pol in ('vv','vh')]
    else:
        prev=pd.read_csv(ROOT/'round05_margin/predictions.csv')
        prev=prev[prev.classifier.eq('interaction') & prev.arm.eq('history')]
        requests=[(int(r.latest_history_id),int(r.obs_id),r.pol) for r in prev.itertuples()]
    all_rows,all_pairs,excluded=[],[],[]
    for old,new,pol in requests:
        r0,r1=frame.loc[(old,pol)],frame.loc[(new,pol)]
        reason=''
        if 31237 in (old,new):reason='confirmed_bad_observation'
        elif r0.mmsi!=r1.mmsi:reason='identity_mismatch'
        elif r0.timestamp>=r1.timestamp:reason='not_earlier'
        elif r0.overpass==r1.overpass:reason='same_overpass'
        if reason:
            excluded.append(dict(history_id=old,query_id=new,pol=pol,reason=reason));continue
        try:rows,pairs=measure(old,new,pol,data,frame)
        except (ValueError,KeyError,FileNotFoundError) as e:
            excluded.append(dict(history_id=old,query_id=new,pol=pol,reason=str(e)));continue
        all_rows.extend(rows);all_pairs.extend(pairs)
    measurements=pd.DataFrame(all_rows);pairs=pd.DataFrame(all_pairs)
    measurements.to_csv(out/'measurements.csv',index=False);pairs.to_csv(out/'pairs.csv',index=False)
    pd.DataFrame(excluded,columns=['history_id','query_id','pol','reason']).to_csv(out/'excluded.csv',index=False)
    summary=summaries(pairs,out);panels(pairs,data,frame,out,pilot)
    if pilot:
        q=pairs[pairs['mode'].eq('paired_size') & pairs.query_id.eq(8370) & pairs.pol.eq('vv') & pairs.region.eq('end')].iloc[0]
        verification['pilot_expected_ratio_error']=abs(q.ratio_new-.7548)
        assert verification['pilot_expected_ratio_error']<.0001
    manifest=dict(round=args.round,requested_pair_views=len(requests),excluded=excluded,
        retained_pair_views=len(pairs[['query_id','pol']].drop_duplicates()),
        retained_ships=pairs.mmsi.nunique(),unique_observation_pairs=len(pairs[['history_id','query_id']].drop_duplicates()),
        query_passes=pairs.query_pass.nunique(),history_passes=pairs.history_pass.nunique(),
        scales_m=[10,20,40],readout_coverage_threshold=.9,checks=verification,
        no_classifier_training=True,physical_causal_effect_estimated=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False));print(summary.to_string())


if __name__=='__main__':
    main()
