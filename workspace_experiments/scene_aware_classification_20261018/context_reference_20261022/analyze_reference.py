"""Independent cached analysis and native case review; no model fitting."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from osgeo import gdal
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent/'context_expansion_20261022'
sys.path.insert(0, str(HERE/'moments'))
from extract_moments import all_obb_background
from run_reference import ARMS, reference_parameters, feature_change

SEED = 20261022
REPLICATES = 2000
CLASSES = ['bulk_carrier', 'fishing_vessel', 'general_cargo']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def evaluate(y, pred):
    return dict(OA=float(np.mean(y == pred)*100), BA=float(np.mean([np.mean(pred[y == c] == c) for c in range(3)])*100))


def margin(scores, truth):
    rivals = scores.copy()
    rivals[np.arange(len(truth)), truth] = -np.inf
    return scores[np.arange(len(truth)), truth]-rivals.max(1)


def kind(y, old, new):
    return np.select([(old != y) & (new == y), (old == y) & (new != y), (old == y) & (new == y)],
                     ['rescued', 'harmed', 'CC'], default='WW')


def summarize_score_direction(predictions):
    q = predictions.copy()
    for c in range(3):
        q[f'delta_s{c}'] = q[f's{c}']-q[f'b{c}']
    q['general_minus_bulk_change'] = q.delta_s2-q.delta_s0
    q.groupby(['arm', 'class_id', 'eligible'])[['delta_s0', 'delta_s1', 'delta_s2', 'general_minus_bulk_change']].agg(['mean', 'median']).to_csv(HERE/'score_direction_summary.csv')


def summarize_acquisitions(p):
    acq = p.groupby(['arm', 'pass_group']).agg(n=('instance_id', 'size'), context=('eligible', 'sum'), correct=('correct', 'sum'), base_correct=('base_correct', 'sum'), net=('net', 'sum'),
        rescued=('kind', lambda s: int(s.eq('rescued').sum())), harmed=('kind', lambda s: int(s.eq('harmed').sum()))).reset_index()
    acq['OA_delta_pp'] = acq.net/acq.n*100
    coverage = []
    for (arm, acquisition), q in p.groupby(['arm', 'pass_group']):
        by_class = q.groupby('class_id')[['base_correct', 'correct']].mean()
        coverage.append(dict(arm=arm, pass_group=acquisition, present_classes=len(by_class),
                             class_counts=json.dumps(q.class_id.value_counts().sort_index().to_dict()),
                             baseline_BA_present=float(by_class.base_correct.mean()*100), BA_present=float(by_class.correct.mean()*100),
                             BA_delta_present_pp=float((by_class.correct-by_class.base_correct).mean()*100)))
    acq = acq.merge(pd.DataFrame(coverage), on=['arm', 'pass_group'], validate='one_to_one')
    acq.to_csv(HERE/'acquisition_effects.csv', index=False)
    return acq


def clustered_ci(t, difference, key):
    y = t.class_id.to_numpy()
    groups = list(t.groupby(key, sort=True).indices.values())
    counts = np.array([[np.sum(y[ix] == c) for c in range(3)] for ix in groups])
    sums = np.array([[difference[ix][y[ix] == c].sum() for c in range(3)] for ix in groups])
    rng = np.random.default_rng(SEED)
    oa, ba = [], []
    for start in range(0, REPLICATES, 100):
        draws = rng.integers(0, len(groups), size=(min(100, REPLICATES-start), len(groups)))
        num, den = sums[draws].sum(1), counts[draws].sum(1)
        assert (den > 0).all()
        oa.extend(num.sum(1)/den.sum(1)*100)
        ba.extend((num/den).mean(1)*100)
    return dict(blocks=len(groups), replicates=REPLICATES, seed=SEED,
                OA_delta=float(difference.mean()*100), BA_delta=float(np.mean([difference[y == c].mean() for c in range(3)])*100),
                OA_low=float(np.quantile(oa, .025)), OA_high=float(np.quantile(oa, .975)),
                BA_low=float(np.quantile(ba, .025)), BA_high=float(np.quantile(ba, .975)))


def reconstruct_references(mom, mask, replacement, ids):
    active = mom['neighbor_valid'] & mask[..., None]
    count = active.sum(1)
    external = tuple((mom['neighbor_'+name]*active[..., None]).sum(1)/np.maximum(count[..., None], 1) for name in ('mean', 'second'))
    eligible = (count > 0).all(1) & mom['target_valid'].all(1)
    lookup = dict(zip(ids, range(len(ids))))
    donor = np.arange(len(ids))
    for row in replacement.itertuples():
        donor[lookup[row.instance_id]] = lookup[row.donor]
    refs = dict(own_matched=(mom['target_mean'].astype(float), mom['target_second'].astype(float)),
                external=external, same_scene_replacement=tuple(x[donor] for x in external))
    return refs, eligible, donor


def choose_cases(p):
    q = p[p.arm.eq('external') & p.eligible].copy()
    chosen = []
    for outcome, priorities in [('rescued', [2, 1]), ('harmed', [0, 1]), ('CC', [0]), ('WW', [2])]:
        available = q[q.kind.eq(outcome)]
        for c in priorities:
            subset = available[available.class_id.eq(c) & ~available.instance_id.isin(chosen)].sort_values('instance_id')
            if len(subset):
                chosen.append(subset.iloc[0].instance_id)
        if not any(q[q.instance_id.isin(chosen)].kind.eq(outcome)):
            chosen.append(available.sort_values('instance_id').iloc[0].instance_id)
    if 'pair_0069192' not in chosen:
        chosen.append('pair_0069192')
    return chosen


def render_cases(t, edges, p, transforms, chosen, replacement):
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetCacheMax(64*1024**2)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    catalogs = pd.read_csv(HERE.parent/'data_image_review/ports24/fullscene_candidate_catalogs.csv')
    points_cache = {}
    t = t.set_index('instance_id')
    outputs = []
    for iid in chosen:
        row = t.loc[iid]
        prod = row['product']
        if prod not in points_cache:
            path = catalogs[catalogs['product'].eq(prod) & catalogs.pol.eq('vv')].iloc[0].audit_path
            det = pd.read_csv(path, usecols=['source_pts', 'conf', 'pixel_status'])
            det = det[det.conf.ge(.25) & det.pixel_status.eq('valid')]
            pts = np.array([json.loads(v) for v in det.source_pts])
            points_cache[prod] = pts, (pts.min(1), pts.max(1))
        points, bounds = points_cache[prod]
        nodes = [('target', row.target_detection_id, [json.loads(row[f'target_window_{pol}']) for pol in ('vv', 'vh')], 0.)]
        actual = edges[edges.instance_id.eq(iid)].sort_values('slot')
        nodes.extend((f'neighbor slot{e.slot}', e.detection_id, [json.loads(e.read_window_vv), json.loads(e.read_window_vh)], e.distance_m) for e in actual.itertuples())
        datasets = [gdal.Open(row[f'raster_{pol}']) for pol in ('vv', 'vh')]
        width = max(max(win[2] for win in node[2]) for node in nodes)
        height = max(max(win[3] for win in node[2]) for node in nodes)
        zoom = 2 if max(width, height) <= 160 else 1
        cellw, rowh = max(300, width*zoom+16), height*zoom+90
        top = 152
        panel = Image.new('RGB', (4*cellw, top+len(nodes)*rowh+30), 'white')
        ink = ImageDraw.Draw(panel)
        scores = p[p.instance_id.eq(iid)].set_index('arm')
        external = scores.loc['external']
        ink.text((8, 5), f'{iid} {row.port} truth={row.class_name} external={external.kind} N={len(actual)}', font=font, fill='black')
        for k, arm in enumerate(ARMS):
            a = scores.loc[arm]
            ink.text((8, 27+k*20), f'{arm}: {a.base_prediction}->{a.prediction}; true margin {a.base_margin:+.4f} -> {a.margin:+.4f}', font=font, fill='black')
        tr = transforms[(transforms.instance_id == iid) & (transforms.arm == 'external')].iloc[0]
        ink.text((8, 111), f'External feature change norm/original={tr.relative_feature_change:.3f}; cosine(old,new)={tr.feature_cosine:.4f}', font=font, fill='black')
        ink.text((8, 132), 'VV raw | VH raw | VV actual BG mask | VH actual BG mask; cyan=BG, red=excluded. Nearest display only.', font=font, fill='black')
        for ni, (role, detection, windows, distance) in enumerate(nodes):
            y = top+ni*rowh
            for pi, (pol, ds, win) in enumerate(zip(('vv', 'vh'), datasets, windows)):
                if ni == 0:
                    image = np.array(Image.open(EXP/'readout/native_targets'/f'{iid}_{pol}.png'))
                else:
                    image = ds.ReadAsArray(*win)
                bg, boxes = all_obb_background(points, bounds, win)
                rgb = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
                overlay = rgb.copy()
                overlay[bg] = .55*overlay[bg]+.45*np.array([20, 220, 255])
                overlay[~bg] = .45*overlay[~bg]+.55*np.array([255, 40, 40])
                for j, a in [(pi, rgb), (pi+2, overlay)]:
                    x = j*cellw+8
                    ink.text((x, y), f'{role} {detection} {pol.upper()} {distance:.0f}m', font=font, fill='black')
                    ink.text((x, y+19), f'window={win}', font=font, fill='black')
                    ink.text((x, y+38), f'BG={int(bg.sum())}; boxes={len(boxes)}', font=font, fill='black')
                    tile = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((image.shape[1]*zoom, image.shape[0]*zoom), Image.Resampling.NEAREST)
                    panel.paste(tile, (x, y+60))
        path = HERE/f'reference_case_{iid}.png'
        panel.save(path)
        datasets.clear()
        outputs.append(dict(instance_id=iid, kind=external.kind, class_name=row.class_name, panel=str(path), actual_neighbors=len(actual),
                            selection='posthoc outcome/class strata, first ID; plus prior shoreline case 0069192; descriptive only, no tuning'))
    pd.DataFrame(outputs).to_csv(HERE/'review_panels.csv', index=False)


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    t = pd.read_csv(EXP/'readout/targets.csv', dtype={'mmsi': str}, low_memory=False)
    p = pd.read_csv(HERE/'predictions.csv')
    edges = pd.read_csv(EXP/'readout/neighbors.csv')
    strengths = pd.read_csv(HERE/'learned_strength.csv')
    replacement = pd.read_csv(HERE/'replacement_map.csv')
    with np.load(EXP/'readout/features.npz') as f:
        h, mask, ids = f['target'].astype(float), f['mask'], f['instance_ids']
    with np.load(HERE/'moments/moments.npz') as f:
        mom = {k: f[k] for k in f.files}
    assert np.array_equal(ids, t.instance_id.to_numpy(str)) and np.array_equal(ids, mom['instance_ids'])
    refs, eligible, donors = reconstruct_references(mom, mask, replacement, ids)
    y = t.class_id.to_numpy()
    p = p.merge(t[['instance_id', 'mmsi', 'port', 'pass_group', 'class_name', 'neighbors']], on='instance_id', validate='many_to_one')
    p['base_correct'], p['correct'] = p.base_prediction.eq(p.class_id), p.prediction.eq(p.class_id)
    p['net'] = p.correct.astype(int)-p.base_correct.astype(int)
    p['kind'] = kind(p.class_id.to_numpy(), p.base_prediction.to_numpy(), p.prediction.to_numpy())
    p['base_margin'] = margin(p[['b0', 'b1', 'b2']].to_numpy(), p.class_id.to_numpy())
    p['margin'] = margin(p[['s0', 's1', 's2']].to_numpy(), p.class_id.to_numpy())
    p['margin_change'] = p.margin-p.base_margin
    summarize_score_direction(p)
    baseline = p[p.arm.eq('external')].set_index('instance_id').loc[ids]
    bp = baseline.base_prediction.to_numpy()
    baseline_stats = evaluate(y, bp)
    pd.DataFrame([baseline_stats]).to_csv(HERE/'reference_baseline_metrics.csv', index=False)
    class_rows = p.groupby(['arm', 'class_id', 'class_name']).agg(n=('instance_id', 'size'), correct=('correct', 'sum'), base_correct=('base_correct', 'sum'),
        net=('net', 'sum'), rescued=('kind', lambda s: int(s.eq('rescued').sum())), harmed=('kind', lambda s: int(s.eq('harmed').sum()))).reset_index()
    class_rows['recall_delta_pp'] = 100*class_rows.net/class_rows.n
    class_rows.to_csv(HERE/'class_effects.csv', index=False)
    acq = summarize_acquisitions(p)
    p.groupby(['arm', 'port']).agg(n=('instance_id', 'size'), net=('net', 'sum'), context=('eligible', 'sum')).reset_index().to_csv(HERE/'port_effects.csv', index=False)
    p[p.prediction.ne(p.base_prediction)].to_csv(HERE/'prediction_flips.csv', index=False)
    predictions = {arm: p[p.arm.eq(arm)].set_index('instance_id').loc[ids].prediction.to_numpy() for arm in ARMS}
    interval_rows = []
    for arm in ARMS:
        for block in ('pass_group', 'mmsi'):
            interval_rows.append(dict(variant='trained', arm=arm, reference='baseline', scope='all', block=block,
                **clustered_ci(t, (predictions[arm] == y).astype(int)-(bp == y), block)))
    for other in ('own_matched', 'same_scene_replacement', 'constant_external'):
        for block in ('pass_group', 'mmsi'):
            interval_rows.append(dict(variant='trained', arm='external', reference=other, scope='all', block=block,
                **clustered_ci(t, (predictions['external'] == y).astype(int)-(predictions[other] == y), block)))
    chosen = choose_cases(p)
    chosen_indices = set(np.flatnonzero(np.isin(ids, chosen)))
    transform_rows, attribution_rows, fixed_rows, group_ablation, fold_checks, channel_rows = [], [], [], [], [], []
    strength_index = strengths.set_index(['fold', 'arm', 'group'])
    for fold in range(6):
        with np.load(HERE/f'fold{fold}_cache.npz') as f:
            data = {k: f[k] for k in f.files}
        train, test = data['train'], data['test']
        assert np.array_equal(data['eligible'], eligible)
        mu0, var0, const = reference_parameters(mom, refs, train, eligible)
        external_alpha = strength_index.loc[(fold, 'external')].sort_index().alpha.to_numpy()
        max_replay, max_inactive = 0., 0.
        for arm in ARMS:
            alpha = strength_index.loc[(fold, arm)].sort_index().alpha.to_numpy()
            mu, second = const if arm == 'constant_external' else tuple(x[test] for x in refs[arm])
            raw = feature_change(h[test], mu, second, mu0, var0, eligible[test])
            channel_alpha = np.repeat(alpha.reshape(2, 8), 8, axis=1)
            aa = np.repeat(channel_alpha[:, None, :], 2, axis=1).reshape(256)
            delta_h = raw*aa
            new_h = h[test]+delta_h
            group_score = data[arm][test]*alpha[None, :, None]
            score = data['base'][test]+group_score.sum(1)
            reference = p[p.fold.eq(fold) & p.arm.eq(arm)].set_index('instance_id').loc[ids[test]]
            error = float(np.abs(score-reference[['s0', 's1', 's2']].to_numpy()).max())
            assert error < 1e-10
            max_replay = max(max_replay, error)
            if (~eligible[test]).any():
                max_inactive = max(max_inactive, float(np.abs(delta_h[~eligible[test]]).max()), float(np.abs(group_score[~eligible[test]]).max()))
            score_fixed = data['base'][test]+np.einsum('ngc,g->nc', data[arm][test], external_alpha)
            pred_fixed = score_fixed.argmax(1)
            for j, i in enumerate(test):
                fixed_rows.append(dict(fold=fold, arm=arm, instance_id=ids[i], class_id=int(y[i]), prediction=int(pred_fixed[j]),
                    base_prediction=int(bp[i]), **{f's{c}':float(score_fixed[j, c]) for c in range(3)}))
            var = np.maximum(second-mu**2, 0)
            active = var0 > 1e-8
            scale = np.sqrt(var0/np.maximum(var, np.maximum(.01*var0, 1e-12))).clip(.25, 4.)
            scale = np.where(active, scale, 1.)
            shift = mu0-scale*mu
            scale = np.broadcast_to(scale, (len(test), 2, 64))
            shift = np.broadcast_to(shift, (len(test), 2, 64))
            shaped = h[test].reshape(-1, 2, 2, 64)
            scaling = (scale[:, :, None, :]-1)*shaped
            shifting = np.broadcast_to(shift[:, :, None, :], shaped.shape).copy()
            scaling *= channel_alpha[None, :, None, :]*active[None, :, None, :]
            shifting *= channel_alpha[None, :, None, :]*active[None, :, None, :]
            scaling[~eligible[test]], shifting[~eligible[test]] = 0, 0
            assert np.allclose((scaling+shifting).reshape(-1, 256), delta_h, atol=1e-10)
            feature_norm = np.linalg.norm(h[test], axis=1)
            relative = np.linalg.norm(delta_h, axis=1)/np.maximum(feature_norm, 1e-12)
            cos = np.sum(h[test]*new_h, axis=1)/np.maximum(feature_norm*np.linalg.norm(new_h, axis=1), 1e-12)
            base_margin = margin(data['base'][test], y[test])
            score_margin = margin(score, y[test])
            outcome = kind(y[test], bp[test], score.argmax(1))
            for j, i in enumerate(test):
                transform_rows.append(dict(instance_id=ids[i], fold=fold, arm=arm, eligible=bool(eligible[i]), class_id=int(y[i]), kind=outcome[j],
                    relative_feature_change=float(relative[j]), feature_cosine=float(cos[j]),
                    mean_delta_norm=float(np.linalg.norm(delta_h[j].reshape(2, 2, 64)[:, 0])), max_delta_norm=float(np.linalg.norm(delta_h[j].reshape(2, 2, 64)[:, 1])),
                    scaling_part_norm=float(np.linalg.norm(scaling[j])), shifting_part_norm=float(np.linalg.norm(shifting[j])),
                    base_margin=float(base_margin[j]), margin=float(score_margin[j]), margin_change=float(score_margin[j]-base_margin[j]),
                    gain_at_low=int((scale[j] == .25).sum()), gain_at_high=int((scale[j] == 4).sum())))
                if i in chosen_indices:
                    for pi, polarization in enumerate(('VV', 'VH')):
                        for pool in range(2):
                            for channel in range(64):
                                flat = pi*128+pool*64+channel
                                channel_rows.append(dict(instance_id=ids[i], fold=fold, arm=arm, polarization=polarization,
                                    pool=('mean', 'max')[pool], channel=channel, group=pi*8+channel//8,
                                    alpha=float(channel_alpha[pi, channel]), original=float(h[i, flat]),
                                    corrected=float(new_h[j, flat]), delta=float(delta_h[j, flat]),
                                    candidate_gain=float(scale[j, pi, channel]), candidate_shift=float(shift[j, pi, channel]),
                                    applied_scaling_part=float(scaling[j, pi, pool, channel]), applied_shifting_part=float(shifting[j, pi, pool, channel])))
                    rival = int(np.argmax(np.where(np.arange(3) == y[i], -np.inf, data['base'][i])))
                    for group in range(16):
                        removed = score[j]-group_score[j, group]
                        attribution_rows.append(dict(instance_id=ids[i], fold=fold, arm=arm, group=group, polarization=('VV', 'VH')[group//8],
                            channel_start=(group%8)*8, channel_end=(group%8+1)*8-1, alpha=float(alpha[group]), truth=int(y[i]), base_rival=rival,
                            score_delta_true=float(group_score[j, group, y[i]]), score_delta_rival=float(group_score[j, group, rival]),
                            true_vs_base_rival_delta=float(group_score[j, group, y[i]]-group_score[j, group, rival]),
                            deleted_group_prediction=int(removed.argmax()), deleted_group_margin=float(margin(removed[None], y[i:i+1])[0])))
            if arm == 'external':
                for group in range(16):
                    removed = score-group_score[:, group]
                    pred = removed.argmax(1)
                    correct, original_correct = pred == y[test], score.argmax(1) == y[test]
                    group_ablation.append(dict(fold=fold, group=group, alpha=float(alpha[group]), n=len(test),
                        rescued_vs_fitted=int((correct & ~original_correct).sum()), harmed_vs_fitted=int((~correct & original_correct).sum()),
                        net_vs_fitted=int(correct.sum()-original_correct.sum())))
        assert max_inactive == 0
        fold_checks.append(dict(fold=fold, score_cache_replay_error=max_replay, inactive_change_max=max_inactive))
    transforms = pd.DataFrame(transform_rows)
    transforms.to_csv(HERE/'feature_transform_diagnostics.csv', index=False)
    pd.DataFrame(attribution_rows).to_csv(HERE/'case_group_attribution.csv', index=False)
    pd.DataFrame(channel_rows).to_csv(HERE/'case_feature_channels.csv', index=False)
    pd.DataFrame(group_ablation).to_csv(HERE/'external_group_deletion.csv', index=False)
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(HERE/'fixed_external_strength_predictions.csv', index=False)
    fixed_metrics = []
    for arm in ARMS:
        aligned = fixed[fixed.arm.eq(arm)].set_index('instance_id').loc[ids]
        pred = aligned.prediction.to_numpy()
        for scope, selection in [('all', np.ones(len(t), bool)), ('eligible', eligible)]:
            fixed_metrics.append(dict(arm=arm, scope=scope, n=int(selection.sum()), **evaluate(y[selection], pred[selection]),
                rescued=int(((bp[selection] != y[selection]) & (pred[selection] == y[selection])).sum()),
                harmed=int(((bp[selection] == y[selection]) & (pred[selection] != y[selection])).sum())))
        if arm != 'external':
            for block in ('pass_group', 'mmsi'):
                interval_rows.append(dict(variant='fixed_external_strength', arm='external', reference=arm, scope='all', block=block,
                    **clustered_ci(t, (predictions['external'] == y).astype(int)-(pred == y), block)))
    pd.DataFrame(fixed_metrics).to_csv(HERE/'fixed_external_strength_metrics.csv', index=False)
    pd.DataFrame(interval_rows).to_csv(HERE/'paired_intervals.csv', index=False)
    strengths.groupby(['arm', 'group', 'polarization']).alpha.agg(['mean', 'min', 'max', lambda s: int(s.gt(0).sum())]).reset_index().rename(columns={'<lambda_0>':'active_folds'}).to_csv(HERE/'group_strength_summary.csv', index=False)
    transforms[transforms.eligible].groupby(['arm', 'class_id', 'kind']).agg(n=('instance_id', 'size'), median_feature_change=('relative_feature_change', 'median'),
        median_cosine=('feature_cosine', 'median'), median_margin_change=('margin_change', 'median')).reset_index().to_csv(HERE/'transform_outcome_summary.csv', index=False)
    p[p.instance_id.isin(chosen)].to_csv(HERE/'review_cases.csv', index=False)
    # Check actual replacement footprint against recipient OBB AABB, as specified by the protocol.
    target_lookup = t.set_index('instance_id')
    edge_lookup = {i: g for i, g in edges.groupby('instance_id')}
    checked_windows = 0
    for row in replacement[replacement.changed].itertuples():
        recipient = target_lookup.loc[row.instance_id]
        assert recipient['product'] == target_lookup.loc[row.donor, 'product']
        points = np.array(json.loads(recipient.target_points))
        lo, hi = points.min(0), points.max(0)
        for edge in edge_lookup[row.donor].itertuples():
            for pol in ('vv', 'vh'):
                x, yy, w, hh = json.loads(getattr(edge, f'read_window_{pol}'))
                assert not (x < hi[0] and x+w > lo[0] and yy < hi[1] and yy+hh > lo[1])
                checked_windows += 1
    replacement_scopes = []
    for name, select in [('all_eligible', replacement), ('changed', replacement[replacement.changed]),
                         ('changed_disjoint', replacement[replacement.changed & replacement.shared_objects.eq(0)]),
                         ('changed_shared', replacement[replacement.changed & replacement.shared_objects.gt(0)]),
                         ('unchanged', replacement[~replacement.changed])]:
        ix = np.flatnonzero(t.instance_id.isin(select.instance_id))
        for arm in ('external', 'same_scene_replacement'):
            pred = predictions[arm][ix]
            replacement_scopes.append(dict(scope=name, arm=arm, n=len(ix), **evaluate(y[ix], pred), net=int((pred == y[ix]).sum()-(bp[ix] == y[ix]).sum())))
    pd.DataFrame(replacement_scopes).to_csv(HERE/'replacement_scope_effects.csv', index=False)
    render_cases(t, edges, p, transforms, chosen, replacement)
    summary = dict(seconds=time.perf_counter()-started, baseline=baseline_stats, targets=len(t), eligible=int(eligible.sum()),
        review_ids=chosen, score_checks=fold_checks, replacement_windows_checked=checked_windows,
        replacement_changed=int(replacement.changed.sum()), replacement_identical_object_sets=int((replacement.changed & replacement.identical_objects).sum()),
        replacement_disjoint=int((replacement.changed & replacement.shared_objects.eq(0)).sum()),
        external_class_net=class_rows[class_rows.arm.eq('external')][['class_name', 'net', 'recall_delta_pp']].to_dict('records'),
        external_acquisitions=dict(positive=int((acq[acq.arm.eq('external')].net > 0).sum()), negative=int((acq[acq.arm.eq('external')].net < 0).sum()), zero=int((acq[acq.arm.eq('external')].net == 0).sum())),
        no_model_fitting=True, source_script_unchanged=True,
        ci_note='2000 paired cluster bootstrap replicates, fixed seed20261022. Pass and MMSI blocks are separate sensitivity analyses, not joint two-way intervals. Developed-on cohort.',
        control_note='Fixed-external-strength and group deletions are cached posthoc diagnostics, not retrained/source-selected policies.')
    dump(HERE/'analysis_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(pd.DataFrame(fixed_metrics).query('scope == "all"').to_string(index=False), flush=True)
    print(pd.DataFrame(interval_rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
