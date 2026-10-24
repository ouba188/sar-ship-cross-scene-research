"""Public-data replication of frozen local spatial-frequency interventions."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeClassifier
import classification_joint_probe as q

ROOT = q.ROOT
OUT = ROOT / 'round23_public_oss1'
MANIFEST = Path('E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/target_manifest.csv')
DETECTIONS = Path('E:/临时会话/oss1_joint_geometry_20261017/detection_views.json')
METADATA = Path('E:/临时会话/all_factor_probe_20261017/metadata/oss1_factor_metadata.csv')
KEYS = ['pol', 'region', 'band', 'dose']


def make_view(det):
    angle = det['ship_axis']
    c, s = np.cos(angle), np.sin(angle)
    offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    offsets = offsets * [det['length']/2, det['width']/2]
    corners = offsets @ np.array([[c, s], [-s, c]]) + [det['cx'], det['cy']]
    return dict(h=det['native_h'], w=det['native_w'], priors=dict(
        target_index=0,
        detections=[dict(shape_usable=True, confidence=det['confidence'], corners=corners.tolist())],
        ship_axis=dict(center=[det['cx'], det['cy']], angle=angle,
                       length_px=det['length'], width_px=det['width'])))


def overpass(product):
    fields = product.split('_')
    return '_'.join([fields[0], fields[6], fields[7]])


def load():
    frame = pd.read_csv(MANIFEST, dtype={'mmsi_or_ship_id': str})
    meta = pd.read_csv(METADATA)
    assert np.array_equal(frame.sample_id.to_numpy(), meta.sample_id.to_numpy())
    frame['obs_id'] = np.arange(len(frame))
    frame['mmsi'] = frame.mmsi_or_ship_id
    frame['overpass'] = frame.scene_id.map(overpass)
    for col in ['product_range_spacing_m', 'product_azimuth_spacing_m']:
        frame[col] = meta[col].to_numpy()
    detections = json.loads(DETECTIONS.read_text(encoding='utf-8'))
    views = {(v['global_row'], v['pol']): v for v in detections}
    return frame, views


def image(row, pol):
    return np.asarray(q.d.ip.Image.open(row['chip_path_'+pol]).convert('L'), dtype=float)/255


def stem_model():
    q.d.torch.set_num_threads(2)
    model = q.d.resnet50(weights=None)
    model.load_state_dict(q.d.torch.load(q.d.ip.WEIGHTS, map_location='cpu', weights_only=True))
    return q.d.torch.nn.Sequential(model.conv1, model.bn1, model.relu).eval()


def prepare():
    OUT.mkdir(exist_ok=True)
    frame, dets = load()
    coverage, ids = [], []
    for _, row in frame.iterrows():
        ok = True
        for pol in ('vv', 'vh'):
            det = dets[row.obs_id, pol]
            reason = ''
            stats = {}
            if not det['axis_valid'] or det['confidence'] < .25:
                reason = 'missing_or_low_confidence_axis'
            elif not Path(row['chip_path_'+pol]).is_file():
                reason = 'missing_chip'
            else:
                view = make_view(det)
                im = image(row, pol)
                assert im.shape == (view['h'], view['w'])
                masks = q.d.ip.masks_for(view)
                try:
                    _, _, place = q.d.ip.crop_pad(im, masks, 160)
                    regions = q.regions(im, view)
                    retained = np.zeros_like(im, dtype=bool)
                    y, x = place['crop_y'], place['crop_x']
                    retained[y:y+place['crop_height'], x:x+place['crop_width']] = True
                    stats = {name+'_pixels': int(mask.sum()) for name, mask in regions.items()}
                    if min(stats.values()) < 8:
                        reason = 'insufficient_region'
                    elif any(np.any(mask & ~retained) for mask in regions.values()):
                        reason = 'region_truncated_by_canvas'
                    stats.update(place)
                except ValueError:
                    reason = 'target_exceeds_canvas'
            coverage.append(dict(obs_id=int(row.obs_id), pol=pol, reason=reason,
                                 confidence=det['confidence'], **stats))
            ok &= not bool(reason)
        if ok:
            ids.append(row.obs_id)
    pd.DataFrame(coverage).to_csv(OUT/'view_coverage.csv', index=False)
    cohort = frame.copy().reset_index(drop=True)
    cohort['intervention_eligible'] = cohort.obs_id.isin(ids)
    cohort['fold'] = -1
    splits = []
    for fold, (train, test) in enumerate(GroupKFold(3).split(cohort, groups=cohort.overpass)):
        cohort.loc[test, 'fold'] = fold
        test_ships = set(cohort.iloc[test].mmsi)
        train = np.array([i for i in train if cohort.iloc[i].mmsi not in test_ships])
        assert set(cohort.iloc[train].class_id) == set(range(8)), (fold, 'source_missing_class')
        assert not set(cohort.iloc[train].overpass) & set(cohort.iloc[test].overpass)
        assert not set(cohort.iloc[train].mmsi) & test_ships
        for name, ix in [('train', train), ('test', test)]:
            splits += [dict(fold=fold, partition=name, obs_id=int(cohort.iloc[i].obs_id)) for i in ix]
    cohort.to_csv(OUT/'cohort.csv', index=False)
    pd.DataFrame(splits).to_csv(OUT/'splits.csv', index=False)
    cohort.groupby(['class_id', 'class_name']).agg(n=('obs_id', 'size'), ships=('mmsi', 'nunique'),
        passes=('overpass', 'nunique'), intervention_n=('intervention_eligible', 'sum')).to_csv(OUT/'class_coverage.csv')
    initial_panels(cohort[cohort.intervention_eligible], dets)
    print('prepared', len(cohort), 'observations', cohort.mmsi.nunique(), 'ships',
          cohort.overpass.nunique(), 'passes', flush=True)


def initial_panels(cohort, dets):
    os.environ['MPLCONFIGDIR'] = str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples = cohort.groupby('class_id', sort=True).head(1)
    fig, axes = plt.subplots(len(examples), 2, figsize=(7, 20), constrained_layout=True)
    for r, (_, row) in enumerate(examples.iterrows()):
        for p, pol in enumerate(('vv', 'vh')):
            im = image(row, pol)
            ax = axes[r, p]
            ax.imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            for region, color in [('body', 'lime'), ('end', 'cyan'), ('side', 'orange')]:
                ax.contour(q.regions(im, make_view(dets[row.obs_id, pol]))[region],
                           levels=[.5], colors=color, linewidths=.6)
            ax.set_title(f'{row.obs_id} {row.class_name} {pol} {im.shape}', fontsize=8)
            ax.set_aspect('equal')
    fig.savefig(OUT/'initial_native_regions.png', dpi=140)
    plt.close(fig)


def infer():
    started = time.perf_counter()
    cohort = pd.read_csv(OUT/'cohort.csv', dtype={'mmsi': str})
    split = pd.read_csv(OUT/'splits.csv')
    _, dets = load()
    stem = stem_model()
    raw = np.zeros((len(cohort), 260), np.float64)
    with q.d.torch.inference_mode():
        for start in range(0, len(cohort), 16):
            batches = [[], [], []]
            addresses = []
            for i in range(start, min(start+16, len(cohort))):
                row = cohort.iloc[i]
                for p, pol in enumerate(('vv', 'vh')):
                    view = make_view(dets[row.obs_id, pol])
                    batch = q.d.prepare_batch([image(row, pol)], view)
                    for k, tensor in enumerate(batch):
                        batches[k].append(tensor)
                    addresses.append((i, p))
                    sh = view['priors']['ship_axis']
                    det = dets[row.obs_id, pol]
                    if det['axis_valid'] and det['confidence'] >= .25:
                        raw[i, 256+2*p:258+2*p] = [sh['length_px'], sh['width_px']]
            desc = q.d.ip.describe(stem, *[q.d.torch.cat(b) for b in batches], inputs=('full',))['full_global'].numpy()
            for (i, p), value in zip(addresses, desc):
                raw[i, p*128:(p+1)*128] = value
    np.save(OUT/'native_features.npy', raw)
    heads = []
    scores = np.zeros((len(cohort), 8))
    fit_info = []
    for fold in range(3):
        train_ids = split[(split.fold == fold) & split.partition.eq('train')].obs_id
        train = np.flatnonzero(cohort.obs_id.isin(train_ids))
        test = np.flatnonzero(cohort.fold.eq(fold))
        mean, std = raw[train].mean(0), np.maximum(raw[train].std(0), 1e-6)
        x = (raw-mean)/std/np.sqrt(raw.shape[1])
        y = cohort.iloc[train].class_id.to_numpy()
        weights = len(y)/(8*np.bincount(y)[y])
        model = RidgeClassifier(alpha=10., solver='cholesky').fit(x[train], y, sample_weight=weights)
        assert np.array_equal(model.classes_, np.arange(8))
        head = dict(mean=mean, std=std, coef=model.coef_, intercept=model.intercept_)
        heads.append(head)
        np.savez(OUT/f'head_f{fold}.npz', **head)
        scores[test] = q.d.score(raw[test], head)
        fit_info.append(dict(fold=fold, source_n=len(train), query_n=len(test),
                             source_classes=np.bincount(y, minlength=8).tolist()))
    baseline = cohort.copy()
    baseline['base_pred'] = scores.argmax(1)
    wrong = scores.copy()
    wrong[np.arange(len(cohort)), cohort.class_id] = -np.inf
    baseline['rival'] = wrong.argmax(1)
    baseline['baseline_margin'] = scores[np.arange(len(cohort)), cohort.class_id]-wrong.max(1)
    for k in range(8):
        baseline[f'score_{k}'] = scores[:, k]
    baseline.to_csv(OUT/'baseline.csv', index=False)
    print('features and frozen heads ready', round(time.perf_counter()-started, 1), flush=True)
    records, skipped = [], []
    replay_error = 0.
    with q.d.torch.inference_mode():
        for i, row in baseline.iterrows():
            if not row.intervention_eligible:
                continue
            head = heads[row.fold]
            base, y, rival = scores[i], int(row.class_id), int(row.rival)
            info = dict(obs_id=row.obs_id, mmsi=row.mmsi, class_id=y, class_name=row.class_name,
                        overpass=row.overpass, fold=row.fold, base_pred=row.base_pred,
                        rival=rival, baseline_margin=row.baseline_margin)
            for p, pol in enumerate(('vv', 'vh')):
                view = make_view(dets[row.obs_id, pol])
                im = image(row, pol)
                items = q.perturb(im, view)
                if not items:
                    skipped.append(dict(obs_id=row.obs_id, pol=pol, reason='zero_matched_change'))
                    continue
                images = [im]+[x for altered, level, meta in items for x in (altered, level)]
                desc = q.d.ip.describe(stem, *q.d.prepare_batch(images, view), inputs=('full',))['full_global'].numpy()
                sl = slice(p*128, (p+1)*128)
                error = float(abs(desc[0]-raw[i, sl]).max())
                replay_error = max(replay_error, error)
                assert error < 2e-5, (row.obs_id, pol, error)
                changed = np.tile(raw[i], (len(desc)-1, 1))
                changed[:, sl] = desc[1:]
                ss = q.d.score(changed, head)
                for n, (_, _, meta) in enumerate(items):
                    altered, control = ss[2*n:2*n+2]
                    pred, lp = int(altered.argmax()), int(control.argmax())
                    delta = float(altered[y]-altered[rival]-row.baseline_margin)
                    ld = float(control[y]-control[rival]-row.baseline_margin)
                    records.append(dict(**info, pol=pol, **meta, pred=pred, level_pred=lp,
                        delta_fixed=delta, level_delta=ld, residual=delta-ld,
                        true_score_change=float(altered[y]-base[y]), rival_score_change=float(altered[rival]-base[rival]),
                        rescued=row.base_pred != y and pred == y, harmed=row.base_pred == y and pred != y,
                        **{f'score_{k}': float(v) for k, v in enumerate(altered)},
                        **{f'level_score_{k}': float(v) for k, v in enumerate(control)}))
            if i % 100 == 0 or i == len(baseline)-1:
                print('intervened', i+1, '/', len(baseline), 'seconds', round(time.perf_counter()-started, 1), flush=True)
    pd.DataFrame(records).to_csv(OUT/'interventions.csv.gz', index=False, compression='gzip')
    pd.DataFrame(skipped, columns=['obs_id', 'pol', 'reason']).to_csv(OUT/'skipped.csv', index=False)
    (OUT/'run.json').write_text(json.dumps(dict(status='complete', n=len(baseline),
        fits=fit_info, interventions=len(records), replay_error=replay_error,
        seconds=time.perf_counter()-started, cpu_threads=2), indent=2), encoding='utf-8')


def interval(values, rng):
    values = np.asarray(values)
    boot = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def metrics(g, pred):
    correct = g[pred].to_numpy() == g.class_id.to_numpy()
    return dict(oa=float(correct.mean()), ba=float(pd.Series(correct).groupby(g.class_id.to_numpy()).mean().mean()))


def analyze():
    baseline = pd.read_csv(OUT/'baseline.csv', dtype={'mmsi': str})
    table = pd.read_csv(OUT/'interventions.csv.gz', dtype={'mmsi': str})
    high = table[table.dose.eq(1.)]
    rng = np.random.default_rng(20261024)
    summaries = []
    for key, g in high.groupby(KEYS):
        record = dict(zip(KEYS, key), n=len(g), ships=g.mmsi.nunique())
        for effect in ['delta_fixed', 'residual']:
            for grouping in ['mmsi', 'overpass']:
                vals = g.groupby(grouping)[effect].mean()
                avg, lo, hi = interval(vals, rng)
                prefix = effect+'_'+grouping
                record.update({prefix+'_mean': avg, prefix+'_lo': lo, prefix+'_hi': hi,
                               prefix+'_positive': int((vals>0).sum()), prefix+'_n': len(vals)})
        record.update(rescued=int(g.rescued.sum()), harmed=int(g.harmed.sum()), **metrics(g, 'pred'))
        for k, v in metrics(g, 'level_pred').items():
            record['control_'+k] = v
        summaries.append(record)
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv', index=False)
    by_ship_class = high.groupby(KEYS+['class_id', 'class_name', 'mmsi'])[['delta_fixed', 'residual']].mean().reset_index()
    by_class = by_ship_class.groupby(KEYS+['class_id', 'class_name']).agg(
        ships=('mmsi', 'nunique'), delta_fixed=('delta_fixed', 'mean'), residual=('residual', 'mean')).reset_index()
    by_class.to_csv(OUT/'by_class.csv', index=False)
    by_class.groupby(KEYS)[['delta_fixed', 'residual']].mean().to_csv(OUT/'class_equal_summary.csv')
    high.groupby(KEYS+['overpass'])[['delta_fixed', 'residual']].agg(['size', 'mean']).to_csv(OUT/'by_pass.csv')
    baseline.groupby('fold').apply(lambda x: pd.Series(metrics(x, 'base_pred')), include_groups=False).to_csv(OUT/'baseline_folds.csv')
    baseline.assign(correct=baseline.base_pred.eq(baseline.class_id)).groupby(['class_id', 'class_name']).agg(
        n=('obs_id', 'size'), accuracy=('correct', 'mean')).to_csv(OUT/'baseline_classes.csv')
    contrasts = []
    for effect in ['delta_fixed', 'residual']:
        wide = high.pivot(index=['obs_id', 'mmsi', 'overpass', 'pol'], columns=['region', 'band'], values=effect)
        for pol in ['vv', 'vh']:
            w = wide.xs(pol, level='pol')
            for region in ['body', 'end', 'side']:
                diff = (w[region, 'fine']-w[region, 'middle']).rename('value').reset_index()
                avg, lo, hi = interval(diff.groupby('mmsi').value.mean(), rng)
                contrasts.append(dict(pol=pol, contrast=region+'_fine_minus_middle', effect=effect, mean=avg, lo=lo, hi=hi))
            diff = ((w['body', 'fine']-w['body', 'middle'])-(w['side', 'fine']-w['side', 'middle'])).rename('value').reset_index()
            avg, lo, hi = interval(diff.groupby('mmsi').value.mean(), rng)
            contrasts.append(dict(pol=pol, contrast='body_vs_side_interaction', effect=effect, mean=avg, lo=lo, hi=hi))
    pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv', index=False)
    dose = table.pivot(index=['obs_id', 'pol', 'region', 'band'], columns='dose', values='delta_fixed')
    eligible = baseline[baseline.intervention_eligible]
    summary = dict(**metrics(baseline, 'base_pred'), eligible_metrics=metrics(eligible, 'base_pred'),
        eligible_n=len(eligible), eligible_ships=eligible.mmsi.nunique(),
        n=len(baseline), ships=baseline.mmsi.nunique(),
        passes=baseline.overpass.nunique(), operations=len(table),
        dose_same_sign=float((dose[.5]*dose[1.]>0).mean()),
        dose_larger_magnitude=float((abs(dose[1.])>=abs(dose[.5])).mean()),
        energy_decreased=float((table.band_energy_after<table.band_energy_before).mean()),
        rescued_unique=high.loc[high.rescued, 'obs_id'].nunique(),
        harmed_unique=high.loc[high.harmed, 'obs_id'].nunique())
    (OUT/'analysis.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    high[high.rescued|high.harmed].to_csv(OUT/'flips.csv', index=False)
    panels(baseline, high)
    print(json.dumps(summary, indent=2), flush=True)


def panels(baseline, table):
    os.environ['MPLCONFIGDIR'] = str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _, dets = load()
    rescued = set(table.loc[table.rescued, 'obs_id'])
    harmed = set(table.loc[table.harmed, 'obs_id'])
    rows = []
    for kind in ['rescued', 'harmed', 'stays_correct', 'stays_wrong', 'chemical_side', 'container_side']:
        if kind == 'rescued':
            candidates = table[table.rescued]
        elif kind == 'harmed':
            candidates = table[table.harmed]
        elif kind in ['chemical_side', 'container_side']:
            cid = 3 if kind == 'chemical_side' else 4
            candidates = table[table.class_id.eq(cid) & table.pol.eq('vh') & table.region.eq('side') & table.band.eq('middle')]
        else:
            candidates = table[~table.obs_id.isin(rescued|harmed)]
            candidates = candidates[(candidates.base_pred == candidates.class_id) == (kind == 'stays_correct')]
        if candidates.empty:
            continue
        selected = candidates.sort_values(['obs_id', 'pol', 'region', 'band']).iloc[0]
        oid = int(selected.obs_id)
        row = baseline[baseline.obs_id.eq(oid)].iloc[0]
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
        for r, pol in enumerate(('vv', 'vh')):
            im = image(row, pol)
            view = make_view(dets[oid, pol])
            altered, level, meta = next(v for v in q.perturb(im, view) if v[2]['region'] == selected.region
                and v[2]['band'] == selected.band and v[2]['dose'] == 1)
            component = q.bands(im)[0][selected.band]
            limit = max(abs(component).max(), 1e-8)
            effect = table[(table.obs_id == oid) & (table.pol == pol) & (table.region == selected.region) & (table.band == selected.band)].iloc[0]
            rows.append(dict(case=kind, **effect.to_dict()))
            for c, (label, data) in enumerate([('native', im), ('component', component), ('attenuated', altered), ('mean control', level)]):
                axes[r,c].imshow(data, cmap='RdBu_r' if c == 1 else 'gray',
                                 vmin=-limit if c == 1 else 0, vmax=limit if c == 1 else 1, interpolation='nearest')
                axes[r,c].contour(q.regions(im, view)[selected.region], levels=[.5], colors='lime', linewidths=.7)
                axes[r,c].set_title(f'{pol} {label}', fontsize=9)
                axes[r,c].set_aspect('equal')
            axes[r, 2].set_xlabel(f'pred {effect.base_pred}->{effect.pred}; margin delta {effect.delta_fixed:+.5f}')
        fig.suptitle(f'OSS1 row {oid} {row.class_name}: {kind}, {selected.region}/{selected.band}\nNative pixels, fixed 0-1 display; component has separate symmetric scale')
        fig.savefig(OUT/f'case_{kind}_{oid}.png', dpi=150)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'case_records.csv', index=False)


def panels_only():
    baseline = pd.read_csv(OUT/'baseline.csv', dtype={'mmsi': str})
    table = pd.read_csv(OUT/'interventions.csv.gz', dtype={'mmsi': str})
    panels(baseline, table[table.dose.eq(1.)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'infer', 'analyze', 'panels_only'])
    args = parser.parse_args()
    globals()[args.stage]()
