"""Matched-mask/amplitude controls and exact two-branch linear attribution."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
import time
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import public_oss1_probe as p

q = p.q
OUT = p.ROOT/'round24_polarization_attribution'
POLS = ('vv', 'vh')
REGIONS = ('body', 'end', 'side')
BANDS = ('fine', 'middle')


def cosine(a, b):
    return float(np.dot(a, b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-15))


def make_changes(images, views):
    own = [q.regions(im, view) for im, view in zip(images, views)]
    common = {r: own[0][r] & own[1][r] for r in REGIONS}
    assert min(mask.sum() for mask in common.values()) >= 8
    bands = [q.bands(im)[0] for im in images]
    proposals = {}
    for mode in ('own', 'common'):
        for i, im in enumerate(images):
            masks = own[i] if mode == 'own' else common
            for region, mask in masks.items():
                taper = np.minimum(distance_transform_edt(mask)/2, 1)*mask
                for band in BANDS:
                    delta = np.clip(im-.5*taper*bands[i][band], 0, 1)-im
                    proposals[mode, i, region, band] = delta, taper, mask
    target = .5*min(np.linalg.norm(v[0]) for v in proposals.values())
    assert target > 1e-8
    changes = []
    for (mode, i, region, band), (delta, taper, mask) in proposals.items():
        im = images[i]
        for dose in (.5, 1.):
            altered = im + dose*target*delta/np.linalg.norm(delta)
            change = altered-im
            basis = taper*((1-im) if change.sum() >= 0 else im)
            factor = change.sum()/basis.sum() if basis.sum() else 0.
            level = im+factor*basis
            assert abs(factor) <= 1
            assert np.max(abs(change[~mask])) == 0
            assert abs(np.linalg.norm(change)-dose*target) < 1e-10
            assert abs((level-im).sum()-change.sum()) < 1e-10
            assert altered.min() >= 0 and altered.max() <= 1
            assert level.min() >= 0 and level.max() <= 1
            meta = dict(mode=mode, pol=POLS[i], region=region, band=band, dose=dose,
                        l2=float(np.linalg.norm(change)), pixels=int(mask.sum()),
                        mean_delta=float(change.mean()), level_l2=float(np.linalg.norm(level-im)))
            changes.append((i, altered, level, meta))
    return changes, own, common


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    baseline = pd.read_csv(p.OUT/'baseline.csv', dtype={'mmsi': str}).set_index('obs_id', drop=False)
    old_cases = pd.read_csv(p.OUT/'case_records.csv')
    ids = sorted(old_cases.obs_id.unique())
    raw = np.load(p.OUT/'native_features.npy')
    heads = [dict(np.load(p.OUT/f'head_f{k}.npz')) for k in range(3)]
    _, dets = p.load()
    stem = p.stem_model()
    records, decompositions, channel_rows, coverage = [], [], [], []
    errors = dict(feature_replay=0., score_replay=0., linear_effect=0., decomposition=0.)
    with q.d.torch.inference_mode():
        for oid in ids:
            row = baseline.loc[oid]
            head = heads[int(row.fold)]
            base = q.d.score(raw[oid], head)
            errors['score_replay'] = max(errors['score_replay'], float(abs(base-row[[f'score_{k}' for k in range(8)]].to_numpy(float)).max()))
            images = [p.image(row, pol) for pol in POLS]
            assert images[0].shape == images[1].shape
            assert max(images[0].shape) <= 160
            views = [p.make_view(dets[oid, pol]) for pol in POLS]
            changes, own, common = make_changes(images, views)
            features = {}
            y, rival = int(row.class_id), int(row.rival)
            effective = (head['coef'][y]-head['coef'][rival])/head['std']/np.sqrt(raw.shape[1])
            w = effective[:256].reshape(2, 128)
            for i, pol in enumerate(POLS):
                for region in REGIONS:
                    coverage.append(dict(obs_id=int(oid), pol=pol, region=region,
                        own_pixels=int(own[i][region].sum()), common_pixels=int(common[region].sum())))
                entries = [c for c in changes if c[0] == i]
                inputs = [images[i]]+[im for _, altered, level, meta in entries for im in (altered, level)]
                desc = q.d.ip.describe(stem, *q.d.prepare_batch(inputs, views[i]), inputs=('full',))['full_global'].numpy().astype(float)
                sl = slice(i*128, (i+1)*128)
                error = float(abs(desc[0]-raw[oid, sl]).max())
                errors['feature_replay'] = max(errors['feature_replay'], error)
                assert error < 2e-5
                modified = np.tile(raw[oid], (len(desc)-1, 1))
                modified[:, sl] = desc[1:]
                scores = q.d.score(modified, head)
                for n, (_, altered, level, meta) in enumerate(entries):
                    a, b = scores[2*n:2*n+2]
                    df = desc[1+2*n]-raw[oid, sl]
                    dl = desc[2+2*n]-raw[oid, sl]
                    delta, ld = float(a[y]-a[rival]-row.baseline_margin), float(b[y]-b[rival]-row.baseline_margin)
                    errors['linear_effect'] = max(errors['linear_effect'], abs(delta-float(w[i]@df)), abs(ld-float(w[i]@dl)))
                    key = meta['mode'], meta['region'], meta['band'], meta['dose'], i
                    features[key] = (df, dl)
                    records.append(dict(obs_id=int(oid), mmsi=row.mmsi, fold=int(row.fold), class_id=y,
                        class_name=row.class_name, rival=rival, base_pred=int(row.base_pred), baseline_margin=row.baseline_margin,
                        **meta, pred=int(a.argmax()), level_pred=int(b.argmax()), delta_fixed=delta,
                        level_delta=ld, residual=delta-ld, true_score_change=float(a[y]-base[y]),
                        rival_score_change=float(a[rival]-base[rival]),
                        **{f'score_{k}': float(v) for k, v in enumerate(a)},
                        **{f'level_score_{k}': float(v) for k, v in enumerate(b)}))
            for mode in ('own', 'common'):
                for region in REGIONS:
                    for band in BANDS:
                        for dose in (.5, 1.):
                            for effect in ('raw', 'mean_adjusted'):
                                dfs = [features[mode, region, band, dose, i] for i in (0, 1)]
                                df = np.stack([a if effect == 'raw' else a-b for a, b in dfs])
                                matrix = df@w.T
                                fc = .5*(w[0]+w[1])*(df[0]-df[1])
                                wc = .5*(w[0]-w[1])*(df[0]+df[1])
                                gap = float(matrix[0, 0]-matrix[1, 1])
                                error = abs(fc.sum()+wc.sum()-gap)
                                errors['decomposition'] = max(errors['decomposition'], error)
                                info = dict(obs_id=int(oid), mode=mode, region=region, band=band,
                                            dose=dose, effect=effect)
                                decompositions.append(dict(**info, vv_change_vv_head=matrix[0, 0],
                                    vv_change_vh_head=matrix[0, 1], vh_change_vv_head=matrix[1, 0],
                                    vh_change_vh_head=matrix[1, 1], vv_minus_vh=gap,
                                    feature_term=float(fc.sum()), weight_term=float(wc.sum()),
                                    feature_change_cosine=cosine(df[0], df[1]), weight_cosine=cosine(w[0], w[1]),
                                    vv_df_norm=float(np.linalg.norm(df[0])), vh_df_norm=float(np.linalg.norm(df[1]))))
                                if mode == 'common' and dose == 1. and effect == 'raw':
                                    for channel in range(128):
                                        channel_rows.append(dict(**info, coordinate=channel,
                                            readout='mean' if channel < 64 else 'maximum', stem_channel=channel%64,
                                            vv_df=df[0, channel], vh_df=df[1, channel],
                                            vv_weight=w[0, channel], vh_weight=w[1, channel],
                                            feature_term=fc[channel], weight_term=wc[channel]))
            print('completed case', oid, 'seconds', round(time.perf_counter()-started, 2), flush=True)
    assert errors['score_replay'] < 1e-12
    assert errors['linear_effect'] < 1e-12 and errors['decomposition'] < 1e-12
    table = pd.DataFrame(records)
    dec = pd.DataFrame(decompositions)
    table.to_csv(OUT/'interventions.csv', index=False)
    dec.to_csv(OUT/'decomposition.csv', index=False)
    pd.DataFrame(channel_rows).to_csv(OUT/'coordinate_contributions.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT/'region_coverage.csv', index=False)
    old = pd.read_csv(p.OUT/'interventions.csv.gz')
    old = old[old.obs_id.isin(ids)]
    keys = ['obs_id', 'pol', 'region', 'band', 'dose']
    compare = old[keys+['delta_fixed', 'pred', 'l2']].rename(columns={c:'old_'+c for c in ['delta_fixed','pred','l2']})
    for mode in ('own', 'common'):
        sub = table[table['mode'].eq(mode)][keys+['delta_fixed', 'pred', 'l2']]
        compare = compare.merge(sub.rename(columns={c:mode+'_'+c for c in ['delta_fixed','pred','l2']}), on=keys, validate='one_to_one')
    compare.to_csv(OUT/'matched_comparison.csv', index=False)
    high = dec[dec['mode'].eq('common') & dec.dose.eq(1.) & dec.effect.eq('raw')]
    high.to_csv(OUT/'high_dose_summary.csv', index=False)
    (OUT/'run.json').write_text(json.dumps(dict(status='complete', cases=[int(v) for v in ids],
        interventions=len(table), decomposition_rows=len(dec), errors=errors,
        seconds=time.perf_counter()-started, cpu_threads=2), indent=2), encoding='utf-8')
    panels(baseline, dets, table, old_cases)


def panels(baseline, dets, table, old_cases):
    os.environ['MPLCONFIGDIR'] = str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for oid, selected in old_cases.groupby('obs_id', sort=True):
        row = baseline.loc[oid]
        region, band = selected.iloc[0][['region', 'band']]
        images = [p.image(row, pol) for pol in POLS]
        views = [p.make_view(dets[oid, pol]) for pol in POLS]
        changes, own, common = make_changes(images, views)
        limits = max(np.max(abs(a-images[i])) for i,a,l,m in changes if m['mode']=='common' and m['region']==region and m['band']==band and m['dose']==1.)
        fig, axes = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True)
        for i, pol in enumerate(POLS):
            _, altered, level, meta = next(v for v in changes if v[0]==i and v[3]['mode']=='common' and v[3]['region']==region and v[3]['band']==band and v[3]['dose']==1.)
            effect = table[(table.obs_id==oid) & table.pol.eq(pol) & table['mode'].eq('common') & table.region.eq(region) & table.band.eq(band) & table.dose.eq(1.)].iloc[0]
            for col, (name, im) in enumerate([('native: own cyan / common lime', images[i]), ('actual pixel change', altered-images[i]), ('matched intervention', altered)]):
                axes[i,col].imshow(im, cmap='RdBu_r' if col==1 else 'gray',
                                   vmin=-limits if col==1 else 0, vmax=limits if col==1 else 1, interpolation='nearest')
                axes[i,col].contour(common[region], levels=[.5], colors='lime', linewidths=.7)
                if col==0:
                    axes[i,col].contour(own[i][region], levels=[.5], colors='cyan', linewidths=.5)
                axes[i,col].set_title(f'{pol} {name}', fontsize=8)
            axes[i,2].set_xlabel(f'pred {effect.base_pred}->{effect.pred}; margin {effect.delta_fixed:+.6f}; L2 {effect.l2:.6f}')
        fig.suptitle(f'OSS1 {oid} {row.class_name}: shared {region}/{band}\nSame pixels and L2; same display scale for VV/VH change maps')
        fig.savefig(OUT/f'case_{oid}_matched.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
