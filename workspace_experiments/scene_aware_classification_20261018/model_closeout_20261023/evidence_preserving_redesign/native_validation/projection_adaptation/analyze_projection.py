"""Post-training paired analysis and native-chip diagnostics; never fits a model.

Run explicitly after all six fits and evaluation exports have completed.
OA/BA deltas are percentage points; NLL/Brier deltas are candidate minus control.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplcache')
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import train_projection as run

tr = run.tr
SEEDS = (17, 29, 43)
CASE_SEED = 17
BOOTSTRAP_SEED = 1024
N_BOOTSTRAP = 2000
CELLS = {
    'old_full': (HERE.parent, 'old_full'),
    'freeze_moment': (HERE.parent, 'moment'),
    'freeze_axis': (HERE.parent, 'axis_moment'),
    'adapt_moment': (HERE, 'moment'),
    'adapt_axis': (HERE, 'axis_moment'),
}
PAIRS = [(a, 'old_full') for a in CELLS if a != 'old_full'] + [
    ('freeze_axis', 'freeze_moment'), ('adapt_axis', 'adapt_moment'),
    ('adapt_moment', 'freeze_moment'), ('adapt_axis', 'freeze_axis'),
]
INTERACTION = {'adapt_axis': 1, 'adapt_moment': -1,
               'freeze_axis': -1, 'freeze_moment': 1}
METRICS = ('OA', 'BA', 'NLL', 'Brier')
DELTAS = ('OA_pp', 'BA_pp', 'NLL_delta', 'Brier_delta')


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False),
                    encoding='utf-8')


def margin(score, y):
    wrong = np.where(np.arange(8)[None] == y[:, None], -np.inf, score)
    return score[np.arange(len(y)), y] - wrong.max(1)


def observations(score, y):
    lp = score.astype(np.float64) - logsumexp(score.astype(np.float64), axis=1, keepdims=True)
    return np.column_stack((score.argmax(1) == y, -lp[np.arange(len(y)), y],
                            ((np.exp(lp) - np.eye(8)[y]) ** 2).sum(1)))


def aggregate(values, y, delta=False):
    scale = 100 if delta else 1
    keys = DELTAS if delta else METRICS
    return dict(zip(keys, (scale * values[:, 0].mean(),
                          scale * np.mean([values[y == c, 0].mean() for c in np.unique(y)]),
                          values[:, 1].mean(), values[:, 2].mean())))


def paired(a, b, y):
    ac, bc = a[:, 0].astype(bool), b[:, 0].astype(bool)
    rescued, harmed = int((ac & ~bc).sum()), int((bc & ~ac).sum())
    return dict(n=len(y), rescued=rescued, harmed=harmed, net=rescued-harmed,
                **aggregate(a-b, y, delta=True))


def seed_summary(frame, group, columns, path):
    frame.groupby(group, dropna=False)[columns].agg(['mean', 'std']).to_csv(path)


def statistics(data, scores):
    f = data.table.iloc[data.te].reset_index(drop=True)
    y = f.class_id.to_numpy(int)
    values = {(s, a): observations(scores[s, a][data.te], y) for s in SEEDS for a in CELLS}
    detail, overall, effects, interaction, groups, group_effects, group_interaction = [], [], [], [], [], [], []
    strata = [('class', 'class_id'), ('acquisition', 'pass_group')]
    for seed in SEEDS:
        for arm in CELLS:
            v = values[seed, arm]
            score = scores[seed, arm][data.te]
            row = f[['obs_id', 'up_index', 'class_id', 'class_name', 'mmsi', 'pass_group', 'port']].copy()
            row['seed'], row['arm'] = seed, arm
            row['prediction'], row['correct'] = score.argmax(1), v[:, 0].astype(bool)
            row['margin'], row['NLL'], row['Brier'] = margin(score, y), v[:, 1], v[:, 2]
            for c in range(8):
                row[f'score_{c}'] = score[:, c]
            detail.append(row)
            overall.append(dict(seed=seed, arm=arm, n=len(y), **aggregate(v, y)))
        for arm, comparator in PAIRS:
            effects.append(dict(seed=seed, arm=arm, comparator=comparator,
                                **paired(values[seed, arm], values[seed, comparator], y)))
        iv = sum(weight * values[seed, arm] for arm, weight in INTERACTION.items())
        interaction.append(dict(seed=seed, n=len(y), **aggregate(iv, y, delta=True)))
        for kind, column in strata:
            for group, ids in f.groupby(column, dropna=False).indices.items():
                label = dict(kind=kind, group=str(group), seed=seed, n=len(ids),
                             class_count=int(np.unique(y[ids]).size),
                             ships=int(f.iloc[ids].mmsi.nunique()),
                             ports='|'.join(sorted(f.iloc[ids].port.astype(str).unique())))
                for arm in CELLS:
                    groups.append(dict(**label, arm=arm, **aggregate(values[seed, arm][ids], y[ids])))
                for arm, comparator in PAIRS:
                    effect = paired(values[seed, arm][ids], values[seed, comparator][ids], y[ids])
                    effect.pop('n')
                    group_effects.append(dict(**label, arm=arm, comparator=comparator, **effect))
                group_interaction.append(dict(**label, **aggregate(iv[ids], y[ids], delta=True)))
    pd.concat(detail, ignore_index=True).to_csv(HERE / 'predictions.csv.gz', index=False, compression='gzip')
    tables = [('evaluation_metrics', overall), ('paired', effects), ('interaction', interaction),
              ('strata_metrics', groups), ('strata_paired', group_effects), ('strata_interaction', group_interaction)]
    frames = {}
    for name, rows in tables:
        frames[name] = pd.DataFrame(rows)
        frames[name].to_csv(HERE / f'{name}.csv', index=False)
    seed_summary(frames['evaluation_metrics'], ['arm'], list(METRICS), HERE / 'summary.csv')
    seed_summary(frames['paired'], ['arm', 'comparator'], ['rescued', 'harmed', 'net', *DELTAS], HERE / 'paired_summary.csv')
    frames['interaction'][list(DELTAS)].agg(['mean', 'std']).to_csv(HERE / 'interaction_summary.csv')
    for name, keys, columns in [
        ('strata_metrics', ['kind', 'group', 'arm'], list(METRICS)),
        ('strata_paired', ['kind', 'group', 'arm', 'comparator'], ['rescued', 'harmed', 'net', *DELTAS]),
        ('strata_interaction', ['kind', 'group'], list(DELTAS)),
    ]:
        seed_summary(frames[name], keys, columns, HERE / f'{name}_summary.csv')

    # Average per-observation effects across seeds BEFORE resampling acquisitions.
    cluster_ids, cluster_names = pd.factorize(f.pass_group, sort=True)
    if (cluster_ids < 0).any():
        raise ValueError('Missing pass_group prevents acquisition-cluster bootstrap.')
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(cluster_names), size=(N_BOOTSTRAP, len(cluster_names)))
    counts = np.zeros((len(cluster_names), 8), dtype=int)
    np.add.at(counts, (cluster_ids, y), 1)
    sampled_counts = counts[draws].sum(1)
    valid_ba = (sampled_counts > 0).all(1)
    contrasts = [(arm, comparator, {arm: 1, comparator: -1}) for arm, comparator in PAIRS]
    contrasts.append(('adapt_axis-adapt_moment', 'freeze_axis-freeze_moment', INTERACTION))
    intervals = []
    for arm, comparator, coefficients in contrasts:
        v = np.mean([sum(w * values[s, a] for a, w in coefficients.items()) for s in SEEDS], axis=0)
        totals = np.zeros((len(cluster_names), 8, 3))
        np.add.at(totals, (cluster_ids, y), v)
        sample = totals[draws].sum(1)
        n = sampled_counts.sum(1)
        distributions = (100 * sample[:, :, 0].sum(1)/n,
                         100 * (sample[valid_ba, :, 0]/sampled_counts[valid_ba]).mean(1),
                         sample[:, :, 1].sum(1)/n, sample[:, :, 2].sum(1)/n)
        point = aggregate(v, y, delta=True)
        for metric, distribution in zip(DELTAS, distributions):
            lo, hi = np.quantile(distribution, [.025, .975]) if len(distribution) else (None, None)
            intervals.append(dict(arm=arm, comparator=comparator, metric=metric,
                                  estimate=point[metric], low=lo, high=hi,
                                  excludes_zero=bool(lo > 0 or hi < 0) if lo is not None else None,
                                  valid_draws=len(distribution), requested_draws=N_BOOTSTRAP,
                                  bootstrap_seed=BOOTSTRAP_SEED, clusters=len(cluster_names)))
    pd.DataFrame(intervals).to_csv(HERE / 'uncertainty.csv', index=False)
    return dict(queries=len(y), ships=int(f.mmsi.nunique()), acquisitions=len(cluster_names),
                class_counts=np.bincount(y, minlength=8), valid_BA_draws=int(valid_ba.sum()))


def select_cases(data, scores):
    f, ids = data.table, data.te
    y = f.class_id.to_numpy(int)
    chosen, selection, used = [], [], set()
    for arm, comparator, statuses in [
        ('adapt_axis', 'adapt_moment', ('rescue', 'harm', 'CC', 'WW')),
        ('adapt_moment', 'freeze_moment', ('rescue', 'harm')),
    ]:
        a = scores[CASE_SEED, arm][ids].argmax(1) == y[ids]
        b = scores[CASE_SEED, comparator][ids].argmax(1) == y[ids]
        masks = {'rescue': a & ~b, 'harm': ~a & b, 'CC': a & b, 'WW': ~a & ~b}
        for status in statuses:
            pool = sorted(ids[masks[status]].tolist(), key=lambda j: (int(f.iloc[j].obs_id), j))
            eligible = [j for j in pool if int(f.iloc[j].obs_id) not in used]
            rank = (len(eligible)-1)//2 if status in ('CC', 'WW') else 0
            record = dict(seed=CASE_SEED, arm=arm, comparator=comparator, case=status,
                          population_n=len(pool), eligible_n=len(eligible),
                          rule='obs_id lower median' if status in ('CC', 'WW') else 'obs_id first',
                          excluded_obs_ids=sorted(used), status='selected' if eligible else 'absent_or_already_selected')
            if eligible:
                j = eligible[rank]
                record.update(index=int(j), obs_id=int(f.iloc[j].obs_id), rank_zero_based=rank)
                chosen.append(record.copy())
                used.add(record['obs_id'])
            selection.append(record)
    return chosen, selection


def describe_score(score, true, rival):
    return dict(score=score, prediction=int(score.argmax()), correct=bool(score.argmax() == true),
                margin=float(margin(score[None], np.array([true]))[0]),
                fixed_rival_margin=float(score[true]-score[rival]))


@torch.inference_mode()
def source_mean_score(model, data, j, baseline, polarization):
    # Do not inject average evidence into an entirely absent polarization.
    if not data.flags[j, polarization].any():
        return tr.predict(model, data, np.array([j]), baseline)[0]
    results = []
    base = torch.tensor(baseline[[j]], device='cuda')
    columns = list(range(polarization*27, (polarization+1)*27))
    columns += list(range(54+polarization*27, 54+(polarization+1)*27))
    for rotation in range(4):
        b = data.batch(np.array([j]), rotation)
        raw = model.raw(model.describe(b), base, b['flags'])
        z = (raw-model.raw_mu)/model.raw_scale
        z[:, :, columns] = 0
        delta = model.head(z).squeeze(-1)
        delta = delta-delta.mean(1, keepdim=True)
        # Exactly the forward() all-invalid gate; availability and score context stay intact.
        delta = delta*(b['flags'].sum((1, 2)) > 0)[:, None]
        results.append((base+delta)[0].cpu().numpy())
    return np.mean(results, axis=0)


def diagnose_case(case, data, scores, reader, rival):
    j, true = case['index'], int(data.table.iloc[case['index']].class_id)
    seed = CASE_SEED
    baseline = data.baselines[seed]
    np.testing.assert_allclose(baseline[j], scores[seed, 'old_full'][j], atol=3e-5, rtol=0)
    diagnostics = {}
    for cell, (root, arm) in CELLS.items():
        if cell == 'old_full':
            continue
        checkpoint = root / 'models' / f'{arm}_{seed}.pt'
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = (run.Model if cell.startswith('adapt') else tr.Model)(arm, reader).cuda()
        model.load_state_dict(ck['state'], strict=True)
        model.eval().requires_grad_(False)
        expected = scores[seed, cell][j]
        replay = tr.predict(model, data, np.array([j]), baseline)[0]
        np.testing.assert_allclose(replay, expected, atol=3e-5, rtol=0)
        variants = {}
        if cell.startswith('adapt'):
            saved = {name: getattr(model, name).detach().clone() for name in ('W', 'bias')}
            with torch.no_grad():
                for name in saved:
                    getattr(model, name).copy_(torch.as_tensor(reader[name], device='cuda', dtype=getattr(model, name).dtype))
            variants['projection_source_reader'] = tr.predict(model, data, np.array([j]), baseline)[0]
            with torch.no_grad():
                for name, value in saved.items():
                    getattr(model, name).copy_(value)
        if model.conv is not None:
            saved_output = model.conv.output.detach().clone()
            with torch.no_grad():
                model.conv.output.zero_()
            variants['axis_conv_zero'] = tr.predict(model, data, np.array([j]), baseline)[0]
            with torch.no_grad():
                model.conv.output.copy_(saved_output)
        for p, pol in enumerate(('VV', 'VH')):
            variants[f'{pol}_source_mean'] = source_mean_score(model, data, j, baseline, p)
        measurements = {}
        for name, score in variants.items():
            applicable = bool(data.flags[j, ('VV', 'VH').index(name[:2])].any()) if name in ('VV_source_mean', 'VH_source_mean') else True
            measurements[name] = dict(**describe_score(score, true, rival),
                applicable=applicable,
                full_minus_intervention_fixed_margin=float((replay[true]-replay[rival])-(score[true]-score[rival])),
                score_max_change=float(np.abs(replay-score).max()),
                interpretation='Conditional score sensitivity with selected head/scalers fixed; not physical causation.')
        diagnostics[cell] = dict(checkpoint=str(checkpoint), selected_step=int(ck['receipt']['selected_step']),
            replay_max_error=float(np.abs(replay-expected).max()), replay=describe_score(replay, true, rival),
            projection_drift={name: float(np.linalg.norm(getattr(model, name).detach().cpu().numpy()-reader[name]))
                              for name in ('W', 'bias')}, interventions=measurements)
        del model
    return diagnostics


def render_case(case, true, rival, data, scores, geo, up, snapshots, valid, refs, outdir):
    j, i = case['index'], int(data.table.iloc[case['index']].up_index)
    np.testing.assert_equal(int(up.iloc[i].obs_id), case['obs_id'])
    views = [(i, 'QUERY', true)]
    for cls, label in ((true, 'TRUE'), (rival, 'RIVAL')):
        for slot in range(2):
            ri = int(refs[i, cls, slot])
            if 0 <= ri < len(up) and int(up.iloc[ri].class_id) != cls:
                raise ValueError(f'Reference class mismatch: query {i}, class {cls}, ref {ri}')
            views.append((ri, f'{label} REF{slot+1}', cls))
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    provenance = []
    for p, pol in enumerate(('VV', 'VH')):
        for col, (ri, label, cls) in enumerate(views):
            ax = axes[p, col]
            entry = dict(up_index=ri, polarization=pol, label=label, expected_class=cls,
                         refs_source=str(tr.LOCAL / 'scores_layer1.npz'))
            reason = None
            if not 0 <= ri < len(up):
                reason = 'missing_reference_index'
                title = f'{pol} {label}\nMISSING REFERENCE'
            else:
                row = up.iloc[ri]
                obs, path = int(row.obs_id), str(row['image_'+pol.lower()])
                entry.update(obs_id=obs, class_id=int(row.class_id), path=path,
                             role=str(row.get('role', '')), valid_polarization=bool(valid[ri, p]))
                title = f'{pol} {label} {obs}\n{row.class_name}'
                if not valid[ri, p]:
                    reason = 'missing_polarization'
                elif obs not in snapshots or pol.lower() not in snapshots[obs]['images']:
                    reason = 'missing_view_snapshot'
                elif not Path(path).is_file():
                    reason = 'missing_original_chip'
                else:
                    pix, footprint, g, placement = geo.native(up, snapshots, ri, p)
                    ax.imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                    for region, color in zip(geo.soft_regions(g)*footprint, ('#20a780', '#d77329', '#3287d0')):
                        if region.min() < .5 < region.max():
                            ax.contour(region, levels=[.5], colors=[color], linewidths=.55)
                    entry.update(placement=placement, geometry=g, native_canvas=list(pix.shape),
                                 display='grayscale 0..255; native160 crop/pad; no image resize')
            entry['status'] = reason or 'rendered'
            if reason:
                ax.text(.5, .5, reason.replace('_', '\n'), ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(title, fontsize=8)
            ax.set(xticks=[], yticks=[])
            provenance.append(entry)
    before = scores[CASE_SEED, case['comparator']][j]
    after = scores[CASE_SEED, case['arm']][j]
    fig.suptitle(f"{case['arm']} vs {case['comparator']} | {case['case']} | seed17 | obs {case['obs_id']}\n"
                 f"true={true}, fixed rival={rival}; prediction {before.argmax()} -> {after.argmax()}; "
                 f"fixed margin {before[true]-before[rival]:+.5f} -> {after[true]-after[rival]:+.5f}", fontsize=11)
    fig.text(.02, .02, 'Native160 crop/pad, no resize. Green body / orange ends / blue sides. Old M3 and its actual references UNCHANGED.', fontsize=9)
    fig.subplots_adjust(top=.84, bottom=.09, wspace=.12, hspace=.18)
    path = outdir / f"{case['arm']}_vs_{case['comparator']}_{case['case']}_{case['obs_id']}_17.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path), provenance


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # A completed export is required; never analyze partially written new evaluation.
    if not (HERE / 'complete.json').is_file():
        raise RuntimeError('Wait for train_projection.py to finish all fits and evaluation exports.')
    data = tr.Data()
    scores = {}
    for seed in SEEDS:
        for cell, (root, arm) in CELLS.items():
            with np.load(root / 'results' / f'{arm}_{seed}.npz') as z:
                np.testing.assert_array_equal(z['up_index'], data.up)
                score = z['score'].copy()
            if score.shape != (data.n, 8) or not np.isfinite(score).all():
                raise ValueError(f'Invalid score shape/values: {cell}, seed {seed}')
            scores[seed, cell] = score
    receipt = statistics(data, scores)
    chosen, selection = select_cases(data, scores)
    outdir = HERE / 'analysis'
    outdir.mkdir(exist_ok=True)
    write_json(outdir / 'case_selection.json', selection)
    geo = tr.load('projection_analysis_geo', tr.BASE.parent / 'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    snapshot_path = geo.OLD / 'local_condition_module/view_snapshot.json'
    snapshots = {int(s['obs_id']): s for s in json.loads(snapshot_path.read_text(encoding='utf-8'))}
    up = pd.read_csv(tr.UP / 'table.csv')
    with np.load(tr.UP / 'summaries.npz') as z:
        valid = z['valid'].copy()
    with np.load(tr.LOCAL / 'scores_layer1.npz') as z:
        refs = z['refs'].copy()
    with np.load(HERE.parent / 'reader.npz') as z:
        reader = dict(z)
    records = []
    for case in chosen:
        j = case['index']
        row = data.table.iloc[j]
        true = int(row.class_id)
        before = scores[CASE_SEED, case['comparator']][j]
        after = scores[CASE_SEED, case['arm']][j]
        # Use the wrong side of a flip; CC/WW use comparator's strongest wrong class.
        anchor = after if case['case'] == 'harm' else before
        rival = int(np.where(np.arange(8) == true, -np.inf, anchor).argmax())
        diagnostics = diagnose_case(case, data, scores, reader, rival)
        path, provenance = render_case(case, true, rival, data, scores, geo, up, snapshots, valid, refs, outdir)
        records.append(dict(**case, true_class=true, rival=rival, mmsi=str(row.mmsi),
            pass_group=str(row.pass_group), port=str(row.port), up_index=int(row.up_index),
            before=describe_score(before, true, rival), after=describe_score(after, true, rival),
            all_scores={a: describe_score(scores[CASE_SEED, a][j], true, rival) for a in CELLS},
            query_region_flags=data.flags[j], diagnostics=diagnostics, figure=path,
            provenance=provenance, snapshot_source=str(snapshot_path), M3='unchanged',
            visual_review='Panel generated for human review; no automatic physical interpretation.'))
        write_json(outdir / 'cases.json', records)
        print('CASE', case['arm'], case['case'], case['obs_id'], flush=True)
    write_json(outdir / 'cases.json', records)
    receipt.update(seeds=SEEDS, cases=len(records), case_seed=CASE_SEED,
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=N_BOOTSTRAP,
        bootstrap='Acquisition clusters, paired per-observation effects averaged across seeds first; BA only full-eight-class draws. Fixed trained seeds, not training-population uncertainty.',
        interaction='(adapt_axis-adapt_moment)-(freeze_axis-freeze_moment); differences of trained-system increments, not isolated convolution causation.',
        metric_units='OA/BA in [0,1]; OA_pp/BA_pp in percentage points; NLL/Brier deltas candidate minus comparator (negative is better). Stratum BA averages present classes; bootstrap BA requires all eight.',
        seed_std='Sample standard deviation (ddof=1) across seeds 17/29/43; no seed ensemble.',
        source_mean='Zero normalized raw evidence columns for one available polarization (raw and class-centered); keep availability, old scores, head, scalers and all-invalid gate. Entirely absent polarization is unchanged and marked not applicable.',
        projection_reset='Restore only W/bias from original source reader; retain selected head, normalizers, tau, convolution and old full baseline.',
        limitations='Fixed-head interventions may create representation/head mismatch; model-level sensitivity only. Selected cases do not estimate prevalence. Missing chips/references are recorded without substitutes.',
        source_reader=str(HERE.parent / 'reader.npz'), M3='unchanged')
    write_json(HERE / 'analysis_receipt.json', receipt)
    print('Analysis complete:', len(records), 'panels', flush=True)


if __name__ == '__main__':
    main()
