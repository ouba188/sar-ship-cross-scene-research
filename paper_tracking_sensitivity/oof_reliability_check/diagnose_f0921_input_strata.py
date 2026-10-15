"""Matched, descriptive input-stratum diagnosis; no feature reads or fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OLD = ROOT / 'f0921_transfer_20261007'
NEW = ROOT / 'f0921_detector_recrop_20261007'
OUT = ROOT / 'information_loss_diagnosis_20261008/input_strata'
ARMS = {'original_base': 'original_base', 'base64': 'base_visual64',
        'full5': 'visual_handoff5'}
K = 8


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_inputs():
    target = pd.read_csv(NEW / 'target_manifest.csv', dtype={'mmsi': str})
    crop = pd.read_csv(NEW / 'crop_metadata.csv')
    if target.row.duplicated().any() or crop.row.duplicated().any():
        raise ValueError('Original row keys must be unique')
    if set(target.row) != set(crop.row):
        raise ValueError('Crop and manifest original row sets differ')
    frame = target.merge(crop, on='row', validate='one_to_one', suffixes=('', '_crop'))
    for field in ('scene', 'row_in_shp'):
        if not frame[field].equals(frame[field + '_crop']):
            raise ValueError('Original-row join disagrees on ' + field)
    keys = ('sample_ids', 'support_rows', 'eval_masks', 'caps', 'seeds',
            'arm_names', 'predictions', 'selected')
    with np.load(OLD / 'policies.npz') as archive:
        old = {key: archive[key] for key in keys}
    with np.load(NEW / 'policies.npz') as archive:
        new = {key: archive[key] for key in keys}
    for key in keys[:6]:
        np.testing.assert_array_equal(old[key], new[key], err_msg=key)
    if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(old['sample_ids']):
        raise ValueError('Policy and manifest sample identities differ')
    frame = frame.set_index('sample_id').loc[old['sample_ids']].reset_index()
    previous = pd.read_csv(OLD / 'target_manifest.csv', usecols=['sample_id', 'row', 'class_id'])
    previous = previous.set_index('sample_id').loc[frame.sample_id]
    for field in ('row', 'class_id'):
        np.testing.assert_array_equal(previous[field], frame[field], err_msg=field)
    labels = frame.class_id.to_numpy(dtype=int)
    np.testing.assert_array_equal(np.unique(labels), np.arange(K))
    for draw, support in enumerate(old['support_rows']):
        expected = np.ones(len(frame), dtype=bool)
        expected[support] = False
        np.testing.assert_array_equal(old['eval_masks'][draw], expected)
        if len(np.unique(support)) != 64:
            raise ValueError('Expected 64 distinct support positions')
    return frame, old, new


def describe(frame):
    frame['input_change'] = np.where(frame.status.eq('detector_dynamic'), 'changed', 'fallback')
    changed = frame.input_change.eq('changed')
    conf = pd.to_numeric(frame.det_conf, errors='coerce')
    frame['det_conf_bin'] = np.select(
        [conf.isna(), conf.lt(.25), conf.lt(.5), conf.lt(.75)],
        ['missing', '<0.25', '[0.25,0.5)', '[0.5,0.75)'], default='[0.75,+inf)')
    # Fallback side=224 is a retained old window, not a dynamic crop request.
    side = pd.to_numeric(frame.requested_side, errors='coerce')
    frame['requested_side_bin'] = np.select(
        [~changed, side.eq(64), side.between(72, 128), side.gt(128)],
        ['fallback_old_input', '64', '72-128', '>128'], default='other_or_missing')
    frame['change_identity'] = frame.input_change + '|' + frame.identity_join
    frame['change_conf'] = frame.input_change + '|' + frame.det_conf_bin
    frame['change_class'] = frame.input_change + '|class_' + frame.class_id.astype(str)
    frame['mmsi_valid'] = frame.mmsi.notna() & ~frame.mmsi.isin(['', '0', 'nan'])
    valid = frame.loc[frame.mmsi_valid]
    vessels = valid.groupby('mmsi').agg(rows=('row', 'size'), scenes=('scene', 'nunique'),
        label_count=('class_id', 'nunique'),
        class_ids=('class_id', lambda s: '|'.join(map(str, sorted(s.unique())))))
    vessels.to_csv(OUT / 'mmsi_label_summary.csv')
    scene_vessels = valid.groupby(['scene', 'mmsi']).class_id.nunique()
    conflicting = vessels.index[vessels.label_count.gt(1)]
    frame['mmsi_multiple_labels'] = frame.mmsi.isin(conflicting)
    flags = frame[['row', 'sample_id', 'scene', 'mmsi', 'class_id', 'status',
                   'len_m', 'det_len_px', 'requested_side']].copy()
    for field in ('len_m', 'det_len_px'):
        values = pd.to_numeric(flags[field], errors='coerce')
        flags[field + '_missing_nonfinite'] = ~np.isfinite(values)
        flags[field + '_nonpositive'] = np.isfinite(values) & values.le(0)
    flags['det_longside_exceeds_requested_px'] = changed & frame.det_len_px.gt(side)
    flags['mmsi_multiple_labels'] = frame.mmsi_multiple_labels
    flags.to_csv(OUT / 'input_flags.csv', index=False)
    counts = dict(rows=len(frame), scenes=int(frame.scene.nunique()),
        distinct_nonmissing_mmsi=int(len(vessels)), rows_missing_mmsi=int((~frame.mmsi_valid).sum()),
        scene_mmsi_pairs=int(len(scene_vessels)),
        mmsi_with_multiple_class_ids=int(len(conflicting)),
        rows_on_multiple_label_mmsi=int(frame.mmsi_multiple_labels.sum()),
        scene_mmsi_pairs_with_multiple_class_ids=int(scene_vessels.gt(1).sum()),
        changed_rows=int(changed.sum()), fallback_rows=int((~changed).sum()),
        flags={c: int(flags[c].sum()) for c in flags if c.endswith(('_nonfinite', '_nonpositive', '_px'))
               and c != 'det_len_px'},
        interpretation='MMSI counts describe recorded identities, not verified physical vessels. '
        'Multiple labels per MMSI are metadata inconsistencies, not measured label error rates. '
        'len_m is privileged AIS metadata, not a visual measurement or feature in this diagnosis. '
        'len_m and det_len_px are not physically compared without validated pixel-to-meter geometry.')
    return counts


def metadata_summary(frame, dimension):
    records = []
    for value, group in frame.groupby(dimension, sort=True):
        record = dict(dimension=dimension, stratum=str(value), rows=len(group),
            scenes=int(group.scene.nunique()),
            distinct_mmsi=int(group.loc[group.mmsi_valid, 'mmsi'].nunique()),
            scene_mmsi_pairs=int(len(group.loc[group.mmsi_valid, ['scene', 'mmsi']].drop_duplicates())),
            changed_rows=int(group.input_change.eq('changed').sum()),
            multiple_label_mmsi_rows=int(group.mmsi_multiple_labels.sum()))
        for c in range(K):
            record[f'class_{c}_rows'] = int(group.class_id.eq(c).sum())
        for col in ('len_m', 'det_len_px', 'det_conf', 'requested_side',
                    'vv_zero_fraction', 'vh_zero_fraction'):
            x = pd.to_numeric(group[col], errors='coerce')
            x = x[np.isfinite(x)]
            record[col + '_n_finite'] = len(x)
            for q, suffix in ((.1, 'p10'), (.5, 'p50'), (.9, 'p90')):
                record[col + '_' + suffix] = float(x.quantile(q)) if len(x) else np.nan
        records.append(record)
    return records


def stratified_metrics(frame, old, new):
    frame['overall'] = 'all'
    dimensions = ['overall', 'input_change', 'status', 'identity_join', 'det_conf_bin',
                  'requested_side_bin', 'class_id', 'scene', 'change_identity',
                  'change_conf', 'change_class']
    y = frame.class_id.to_numpy(dtype=int)
    records, descriptions, supports = [], [], []
    for draw, seed in enumerate(old['seeds']):
        support = old['support_rows'][draw]
        query = old['eval_masks'][draw]
        support_ids = set(frame.iloc[support].loc[lambda x: x.mmsi_valid, 'mmsi'])
        supports.append(dict(seed=int(seed), support_rows=len(support), query_rows=int(query.sum()),
            changed_support_rows=int(frame.iloc[support].input_change.eq('changed').sum()),
            fallback_support_rows=int(frame.iloc[support].input_change.eq('fallback').sum()),
            support_mmsi=len(support_ids),
            query_rows_sharing_support_mmsi=int((query & frame.mmsi.isin(support_ids)).sum()),
            support_class_counts=np.bincount(y[support], minlength=K).tolist()))
    for dimension in dimensions:
        codes, names = pd.factorize(frame[dimension].astype(str), sort=True)
        size = len(names)
        descriptions.extend(metadata_summary(frame, dimension))
        for draw, seed in enumerate(old['seeds']):
            ev = old['eval_masks'][draw]
            total_class = np.bincount(y[ev], minlength=K)
            if (total_class == 0).any():
                raise ValueError('Fixed-eight BA requires all eight query classes')
            cell = codes[ev] * K + y[ev]
            count = np.bincount(cell, minlength=size*K).reshape(size, K)
            n = count.sum(1)
            for arm, policy_arm in ARMS.items():
                ai = list(old['arm_names']).index(policy_arm)
                a = old['predictions'][draw, ai, ev]
                b = new['predictions'][draw, ai, ev]
                ca, cb = a == y[ev], b == y[ev]
                hits_a = np.bincount(cell, weights=ca, minlength=size*K).reshape(size, K)
                hits_b = np.bincount(cell, weights=cb, minlength=size*K).reshape(size, K)
                ra = np.divide(hits_a, count, out=np.full_like(hits_a, np.nan), where=count > 0)
                rb = np.divide(hits_b, count, out=np.full_like(hits_b, np.nan), where=count > 0)
                flips = np.bincount(codes[ev], weights=(a != b), minlength=size)
                gained = np.bincount(codes[ev], weights=(~ca & cb), minlength=size)
                lost = np.bincount(codes[ev], weights=(ca & ~cb), minlength=size)
                selected_a = np.bincount(codes[ev], weights=old['selected'][draw, ev], minlength=size)
                selected_b = np.bincount(codes[ev], weights=new['selected'][draw, ev], minlength=size)
                for g, name in enumerate(names):
                    present = count[g] > 0
                    row = dict(dimension=dimension, stratum=str(name), seed=int(seed), arm=arm,
                        query_rows=int(n[g]), present_classes=int(present.sum()),
                        present_class_ids='|'.join(map(str, np.flatnonzero(present))),
                        old_oa_pct=100*hits_a[g].sum()/n[g] if n[g] else np.nan,
                        new_oa_pct=100*hits_b[g].sum()/n[g] if n[g] else np.nan,
                        old_ba_present_pct=100*ra[g, present].mean() if present.any() else np.nan,
                        new_ba_present_pct=100*rb[g, present].mean() if present.any() else np.nan,
                        prediction_flips=int(flips[g]), wrong_to_right=int(gained[g]),
                        right_to_wrong=int(lost[g]),
                        old_oa_contribution_pp=100*hits_a[g].sum()/ev.sum(),
                        new_oa_contribution_pp=100*hits_b[g].sum()/ev.sum(),
                        old_fixed8_ba_contribution_pp=100*np.sum(hits_a[g]/total_class)/K,
                        new_fixed8_ba_contribution_pp=100*np.sum(hits_b[g]/total_class)/K,
                        old_admissions=int(selected_a[g]) if arm == 'full5' else 0,
                        new_admissions=int(selected_b[g]) if arm == 'full5' else 0)
                    row['delta_oa_pp'] = row['new_oa_pct'] - row['old_oa_pct']
                    row['delta_ba_present_pp'] = row['new_ba_present_pct'] - row['old_ba_present_pct']
                    row['delta_oa_contribution_pp'] = row['new_oa_contribution_pp'] - row['old_oa_contribution_pp']
                    row['delta_fixed8_ba_contribution_pp'] = row['new_fixed8_ba_contribution_pp'] - row['old_fixed8_ba_contribution_pp']
                    for c in range(K):
                        row[f'class_{c}_query_rows'] = int(count[g, c])
                        row[f'class_{c}_old_recall_pct'] = 100*ra[g, c]
                        row[f'class_{c}_new_recall_pct'] = 100*rb[g, c]
                    records.append(row)
        print(f'METRICS {dimension}: {size} strata', flush=True)
    per_draw = pd.DataFrame(records)
    key = ['dimension', 'stratum', 'arm']
    values = [c for c in per_draw.select_dtypes(include='number') if c != 'seed']
    means = per_draw.groupby(key, sort=False)[values].mean().reset_index()
    spans = per_draw.groupby(key, sort=False).agg(draws=('seed', 'size'),
        nonempty_draws=('query_rows', lambda s: int(s.gt(0).sum())),
        delta_oa_min_pp=('delta_oa_pp', 'min'), delta_oa_max_pp=('delta_oa_pp', 'max'),
        delta_ba_present_min_pp=('delta_ba_present_pp', 'min'),
        delta_ba_present_max_pp=('delta_ba_present_pp', 'max')).reset_index()
    means = means.merge(spans, on=key, validate='one_to_one')
    descriptions = pd.DataFrame(descriptions)
    per_draw.to_csv(OUT / 'strata_per_draw.csv', index=False)
    means.to_csv(OUT / 'strata_means.csv', index=False)
    descriptions.to_csv(OUT / 'strata_metadata.csv', index=False)
    means.loc[means.dimension.eq('scene')].merge(descriptions, on=['dimension', 'stratum']).to_csv(
        OUT / 'scene_summary.csv', index=False)
    pd.DataFrame(supports).to_csv(OUT / 'support_summary.csv', index=False)
    # A missed group or a subgroup-specific BA denominator would break additivity.
    contribution_cols = [c for c in values if 'contribution_pp' in c]
    overall = per_draw.loc[per_draw.dimension.eq('overall')].set_index(['seed', 'arm'])
    for dimension in dimensions[1:]:
        summed = per_draw.loc[per_draw.dimension.eq(dimension)].groupby(['seed', 'arm'])[contribution_cols].sum()
        np.testing.assert_allclose(summed, overall.loc[summed.index, contribution_cols], atol=1e-10)
    for folder, prefix in ((OLD, 'old'), (NEW, 'new')):
        reference = json.loads((folder / 'summary.json').read_text())['means']
        for arm, policy_arm in ARMS.items():
            ref = next(r for r in reference if r['arm'] == policy_arm)
            row = means.loc[means.dimension.eq('overall') & means.arm.eq(arm)].iloc[0]
            np.testing.assert_allclose([row[f'{prefix}_oa_pct'], row[f'{prefix}_ba_present_pct']],
                                       [100*ref['acc'], 100*ref['ba']], atol=1e-9)
    return means, supports


def findings(means, counts, supports):
    def pick(arm, dimension='overall', stratum='all'):
        return means.loc[means.arm.eq(arm) & means.dimension.eq(dimension) & means.stratum.eq(stratum)].iloc[0]

    provenance = dict(status='upstream raw-to-8bit recipe unresolved',
        search_scope='Under two minutes; named source crop scripts, source manifest header, and relevant Hermes/scripts names only; no broad filesystem search.',
        references=[
            dict(path='E:/临时会话/build_v5_context_chips.py', lines='265-285',
                 finding='PIL opens source_tiff, crops existing image, saves PNG; no radiometric stretch in crop path.'),
            dict(path='E:/临时会话/build_v5_context_chips_gdal.py', lines='1-7,92-119',
                 finding='Documents georeferenced 8-bit TIFF inputs; bounded reads copied to PNG with source dtype.'),
            dict(path='E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv', lines='1-2',
                 finding='Source manifest references UTM_8bit tiles; naming does not establish upstream scaling.'),
            dict(path=str(ROOT / 'oof_reliability_check/prepare_f0921_recrop.py'), lines='32-46,130-135',
                 finding='Target uses finite per-chip 2/98 percentile scaling, including zeros, then bilinear128.')],
        limit='Other target crop scripts do not prove the source upstream recipe. Center, window size and stretch effects remain coupled.')
    overall = means.loc[means.dimension.eq('overall')]
    decomposition = means.loc[means.dimension.eq('input_change')]
    class_names = pd.read_csv(NEW / 'target_manifest.csv', usecols=['class_id', 'class_name']).drop_duplicates()
    class_names = {str(r.class_id): r.class_name for r in class_names.itertuples()}
    dump(OUT / 'input_diagnosis.json', dict(counts=counts, supports=supports,
        arms=ARMS, class_names=class_names, overall=json.loads(overall.to_json(orient='records')),
        changed_fallback=json.loads(decomposition.to_json(orient='records')),
        provenance=provenance,
        checks=dict(original_row_join=True, policy_sample_alignment=True, same_labels=True,
            same_support_query_masks=True, all_partition_contributions_add_to_overall=True,
            matches_existing_old_new_summary=True),
        metric_definitions=dict(oa='Correct / query rows; percentages.',
            ba_present='Mean recall over classes present in this subgroup; absent recall is null/blank.',
            weighted_oa='100 * subgroup correct / total query rows, per draw.',
            fixed8_ba='100/8 * sum_c(subgroup correct_c / full-query count_c), per draw.',
            aggregation='Equal mean of five matched overlapping support draws. Not independent replication. '
            'Subgroup conditional BA is not additive; only contributions use fixed full-query denominators.',
            mean_recalls='Class recalls average draws in which the class is present; inspect per-draw denominators.'),
        limits=['Descriptive post-hoc strata, not new tuning or blind confirmation.',
            'Unique MMSI in the current DET source establishes join uniqueness, not ground-truth identity or localization.',
            'No label error rate or semantic foreground claim; no visual evidence collected.',
            'Fallback full5 changes combine support adaptation and global handoff ranking; not isolated counterfactuals.',
            'No source or target feature matrix read; no GPU, model fitting, training or modules.']))
    lines = ['# Input-stratum diagnosis, 2026-10-08', '',
        'The matched input correction recovers discriminative signal before target adaptation. '
        'It does not identify a single physical cause or explain all remaining cross-domain errors.', '',
        '| Arm | Old OA | New OA | Delta OA pp | Delta fixed-8 BA pp |',
        '| --- | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        r = pick(arm)
        lines.append(f'| {arm} | {r.old_oa_pct:.6f} | {r.new_oa_pct:.6f} | {r.delta_oa_pp:+.6f} | {r.delta_ba_present_pp:+.6f} |')
    lines += ['', '## Changed inputs versus fallback', '',
        '| Arm / group | Query rows (draw mean) | Group OA delta pp | Group present-class BA delta pp | Overall OA contribution pp | Fixed-8 BA contribution pp |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        for group in ('changed', 'fallback'):
            r = pick(arm, 'input_change', group)
            lines.append(f'| {arm} / {group} | {r.query_rows:.1f} | {r.delta_oa_pp:+.6f} | '
                         f'{r.delta_ba_present_pp:+.6f} | {r.delta_oa_contribution_pp:+.6f} | {r.delta_fixed8_ba_contribution_pp:+.6f} |')
    fallback = pick('original_base', 'input_change', 'fallback')
    lines += ['', f'Source-only fallback prediction flips: {fallback.prediction_flips:.1f} per draw. '
        'The same frozen source head therefore provides a direct input-pipeline contrast on changed rows. '
        'On fallback rows the base64 difference reflects changed support features and their fitted correction; '
        'full5 also includes expert adaptation and competition for the global handoff cap. '
        'These fallback gains/losses are downstream spillover, not newly recovered pixels.', '',
        'Contributions use the same full-query denominators in each draw and sum exactly to overall changes. '
        'Present-class subgroup BA cannot be summed or compared as if class composition were fixed.', '',
        '## Where the gains concentrate', '']
    changed = pick('original_base', 'input_change', 'changed')
    lines += [f'On changed rows, source-only OA rises {changed.delta_oa_pp:.3f} pp, but '
        f'their own eight-class BA changes {changed.delta_ba_present_pp:+.3f} pp '
        f'({changed.old_ba_present_pct:.3f}% to {changed.new_ba_present_pct:.3f}%). '
        'This differs from their positive contribution to full-cohort fixed-8 BA because each class has '
        'a different changed/fallback share. The gain is class-selective, not a uniform recovery. '
        'Class IDs: ' + ', '.join(f'{k}={v}' for k, v in sorted(class_names.items())) + '.', '']
    for dimension in ('status', 'identity_join', 'requested_side_bin', 'det_conf_bin', 'class_id'):
        group = means.loc[means.dimension.eq(dimension) & means.arm.eq('original_base')].sort_values('delta_oa_contribution_pp', ascending=False)
        lines.append(f'- Source-only {dimension}: ' + '; '.join(
            f'{r.stratum}: OA contribution {r.delta_oa_contribution_pp:+.3f} pp, '
            f'fixed-8 BA contribution {r.delta_fixed8_ba_contribution_pp:+.3f} pp'
            for r in group.itertuples()) + '.')
    scene = means.loc[means.dimension.eq('scene') & means.arm.eq('original_base')]
    lines += ['', f'Source-only scene OA change: {int(scene.delta_oa_pp.gt(0).sum())} positive, '
        f'{int(scene.delta_oa_pp.lt(0).sum())} negative, {int(scene.delta_oa_pp.eq(0).sum())} zero. '
        'scene_summary.csv includes scene size, class composition, MMSIs, crop coverage and metadata quantiles; '
        'small or class-sparse scenes are descriptive, not reliable standalone effect estimates.', '',
        '## Identity and input limits', '',
        f'{counts["rows"]:,} rows; {counts["scenes"]:,} scenes; '
        f'{counts["distinct_nonmissing_mmsi"]:,} distinct recorded MMSIs; '
        f'{counts["scene_mmsi_pairs"]:,} scene/MMSI pairs. '
        f'{counts["mmsi_with_multiple_class_ids"]:,} MMSIs have multiple class IDs '
        f'({counts["rows_on_multiple_label_mmsi"]:,} rows); '
        f'{counts["scene_mmsi_pairs_with_multiple_class_ids"]:,} scene/MMSI pairs have multiple labels. '
        'These are metadata consistency counts, not label error rates or verified vessel counts.', '',
        'A unique MMSI match in current DET records is source uniqueness, not ground-truth identity. '
        'MMSI-plus-location is a heuristic. Detection confidence does not establish semantic foreground. '
        'len_m is privileged AIS metadata, not visual evidence and not used as a predictive feature here. '
        'len_m and det_len_px are recorded separately with missing/nonpositive flags; no meter/pixel ratio '
        'or physical disagreement is inferred. Fallback requested_side=224 is excluded from dynamic >128.', '',
        f'Changed support rows by draw: {[s["changed_support_rows"] for s in supports]}. '
        f'Query rows sharing support MMSI: {[s["query_rows_sharing_support_mmsi"] for s in supports]}. '
        'The split is row-level and the five draws overlap.', '',
        '## Radiometric trace and interpretation', '',
        'The named PIL and GDAL V5 crop producers copy existing 8-bit TIFF pixels into PNG crops. '
        'The source manifest names UTM_8bit tiles, but the raw-to-8bit recipe remains unresolved after a bounded search. '
        'The target independently stretches each chip/band at 2/98 percentiles. Changing the window changes '
        'both content and stretch statistics: centering, identity association, spatial scale and radiometry are not isolated. '
        'The source-only changed-row gain locates a demonstrated loss at input construction; remaining loss cannot '
        'be assigned to raw scaling, encoders or heads from these strata alone. Feature/head probes are separate.', '',
        'Checks passed: original-row joins, sample/label alignment, identical supports/query masks, '
        'published metric reproduction, and additive OA/fixed-8 BA contribution reconciliation for every partition. '
        'No training, feature-matrix reads, GPU use, new model modules or visual adjudication.', '',
        'Outputs: input_diagnosis.json; strata_per_draw.csv; strata_means.csv; strata_metadata.csv; '
        'scene_summary.csv; support_summary.csv; mmsi_label_summary.csv; input_flags.csv.', '']
    (OUT / 'INPUT_FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(overall[['arm', 'old_oa_pct', 'new_oa_pct', 'delta_oa_pp', 'delta_ba_present_pp']].to_string(index=False))
    print(decomposition[['arm', 'stratum', 'prediction_flips', 'delta_oa_contribution_pp',
                         'delta_fixed8_ba_contribution_pp']].to_string(index=False))
    print(json.dumps(counts, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, old, new = load_inputs()
    counts = describe(frame)
    means, supports = stratified_metrics(frame, old, new)
    findings(means, counts, supports)


if __name__ == '__main__':
    main()
