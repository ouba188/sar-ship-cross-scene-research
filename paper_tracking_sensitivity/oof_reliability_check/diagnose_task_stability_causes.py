"""Bounded descriptive source/held-port diagnosis; no model fits or selection."""

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'

import argparse
import inspect
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from run_generic_transfer import metrics
from task_stable_sensitivity import pair_geometry


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
BANKS = ROOT / 'protected_drift_activation_20261008/ports'
PARENT = ROOT / 'task_stable_sensitivity_20261009'
OUT = ROOT / 'task_stability_causes_20261009'
MANIFEST = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv')
KEYS = ['port', 'space', 'a', 'b']


def source_diagnostic(z, labels, domains, a, b):
    geo = pair_geometry(z, labels, domains, a, b)
    if geo is None:
        return None
    common = [p for p in np.unique(domains)
              if all(np.count_nonzero((domains == p) & (labels == c)) >= 5
                     for c in (a, b))]
    selected = np.isin(domains, common) & np.isin(labels, [a, b])
    yy, dd = labels[selected], domains[selected]
    score = (z[selected] - geo['midpoint']) @ geo['direction']
    weights = np.empty(len(score))
    cell_variances, cell_counts = [], []
    for port in common:
        for c in (a, b):
            use = (dd == port) & (yy == c)
            weights[use] = 1. / (2 * len(common) * use.sum())
            cell_variances.append(float(np.var(score[use])))
            cell_counts.append(int(use.sum()))
    class_variances = []
    for c in (a, b):
        use = yy == c
        mean = np.average(score[use], weights=weights[use])
        class_variances.append(np.average((score[use] - mean)**2, weights=weights[use]))
    variance = float(np.mean(class_variances))
    within_cell = float(np.mean(cell_variances))
    between_port = geo['port_sd_along_task']**2
    # This detects a mismatch between sample weighting and the reused helper.
    np.testing.assert_allclose(variance, within_cell + between_port, rtol=1e-10, atol=1e-10)
    return geo, dict(
        source_auc=float(roc_auc_score(yy == a, score, sample_weight=weights)),
        source_gap=geo['source_gap'], source_within_class_variance=variance,
        source_within_cell_variance=within_cell,
        source_between_port_variance=between_port,
        source_standardized_gap=geo['source_gap'] / np.sqrt(variance),
        source_common_ports=len(common), source_common_port_names=';'.join(common),
        source_n_a=int((yy == a).sum()), source_n_b=int((yy == b).sum()),
        source_min_cell_n=min(cell_counts),
        source_effective_n_a=float(.25 / np.square(weights[yy == a]).sum()),
        source_effective_n_b=float(.25 / np.square(weights[yy == b]).sum()),
        source_port_sd_over_gap=geo['port_sd_over_gap'])


def coverage(labels, **key):
    counts = np.bincount(labels, minlength=8)
    present = np.flatnonzero(counts)
    return dict(**key, n_rows=len(labels), n_classes=len(present),
                present_classes=';'.join(map(str, present)),
                missing_classes=';'.join(map(str, np.flatnonzero(counts == 0))),
                **{f'class_{c}': int(n) for c, n in enumerate(counts)})


def self_test():
    z, y, domains = [], [], []
    for p, offset in enumerate((-1., 0., 1.)):
        for c, mean in enumerate((5., -5.)):
            z.extend([[mean + offset + v] for v in (-2., -1., 0., 1., 2.)])
            y.extend([c] * 5)
            domains.extend([str(p)] * 5)
    z, y, domains = np.array(z), np.array(y), np.array(domains)
    geo, result = source_diagnostic(z, y, domains, 0, 1)
    np.testing.assert_allclose([result['source_auc'], result['source_gap'],
                               result['source_within_class_variance'],
                               result['source_standardized_gap']],
                              [1., 10., 8./3., 10./np.sqrt(8./3.)])
    repeat = np.r_[np.arange(len(y)), np.tile(np.arange(5), 9)]
    _, repeated = source_diagnostic(z[repeat], y[repeat], domains[repeat], 0, 1)
    for key in ('source_auc', 'source_gap', 'source_within_class_variance'):
        np.testing.assert_allclose(result[key], repeated[key])
    _, extra = source_diagnostic(np.r_[z, [[1000.]] * 5], np.r_[y, [0] * 5],
                                np.r_[domains, ['unpaired'] * 5], 0, 1)
    np.testing.assert_allclose(extra['source_gap'], result['source_gap'])
    assert roc_auc_score(y == 0, (-z @ geo['direction'])) == 0.
    absent = metrics(np.array([0, 0, 1]), np.array([0, 1, 1]), 8)
    assert absent['ba'] == .75 and absent['class_recall'][2:] == [None] * 6
    print('SELF_TEST: separated source, equal-cell weights, common-port exclusion, '
          'reversed held direction, absent-class BA passed', flush=True)


def run_port(folder, frame):
    labels, domains = frame.class_id.to_numpy(), frame.port.to_numpy()
    source = np.load(folder / 'source_rows.npy')
    target = np.load(folder / 'target_rows.npy')
    with np.load(folder / 'policies.npz') as saved:
        supports, masks = saved['support_rows'], saved['eval_mask']
        np.testing.assert_array_equal(source, saved['source_rows'])
        np.testing.assert_array_equal(target, saved['query_rows'])
    assert not (domains[source] == folder.name).any()
    assert (domains[target] == folder.name).all()
    with (folder / 'geometry.pkl').open('rb') as handle:
        g = pickle.load(handle)
    sb = np.load(folder / 'source_bank.npy', mmap_mode='r')
    tb = np.load(folder / 'target_bank.npy', mmap_mode='r')
    r = g['stable_rank']
    assert len(sb) == len(source) and len(tb) == len(target)
    spaces = {
        'stable': (sb[:, 192:192+r] * g['stable_scale'] + g['stable_center'],
                   tb[:, 192:192+r] * g['stable_scale'] + g['stable_center']),
        'ordinary_rank_matched': (sb[:, :r], tb[:, :r])}
    cov = [coverage(labels[source], port=folder.name, scope='source', draw=-1),
           coverage(labels[target], port=folder.name, scope='target_pool', draw=-1)]
    cells = [coverage(labels[source][domains[source] == p],
                      port=folder.name, source_port=p)
             for p in np.unique(domains[source])]
    for draw, mask in enumerate(masks):
        support = supports[draw]
        assert len(support) == len(np.unique(support)) == 64
        assert np.isin(support, target).all()
        np.testing.assert_array_equal(mask, ~np.isin(target, support))
        cov.append(coverage(labels[support], port=folder.name, scope='support', draw=draw))
        query = labels[target[mask]]
        cov.append(coverage(query, port=folder.name, scope='query', draw=draw,
                            support_unseen_query_rows=int((~np.isin(query, labels[support])).sum())))
    source_records, target_records = [], []
    for space, (zs, zt) in spaces.items():
        for a, b in itertools.combinations(range(8), 2):
            diagnostic = source_diagnostic(zs, labels[source], domains[source], a, b)
            if diagnostic is None:
                continue
            geo, record = diagnostic
            key = dict(port=folder.name, space=space, a=a, b=b)
            source_records.append(dict(**key, rank=r, **record))
            for draw, mask in enumerate(masks):
                use = mask & np.isin(labels[target], [a, b])
                yy = labels[target][use]
                na, nb = int((yy == a).sum()), int((yy == b).sum())
                if min(na, nb) < 5:
                    continue
                z = zt[use]
                scores = (z - geo['midpoint']) @ geo['direction']
                gap = float(scores[yy == a].mean() - scores[yy == b].mean())
                variance = float(.5 * (scores[yy == a].var() + scores[yy == b].var()))
                target_records.append(dict(**key, draw=draw, n_a=na, n_b=nb,
                    held_auc_recomputed=float(roc_auc_score(yy == a, scores)),
                    held_signed_gap=gap, held_within_class_variance=variance,
                    held_standardized_gap=gap / np.sqrt(variance),
                    held_gap_ratio_recomputed=gap / record['source_gap'],
                    held_variance_ratio=variance / record['source_within_class_variance'],
                    held_within_cell_variance_ratio=variance / record['source_within_cell_variance'],
                    held_reversed=float(gap < 0)))
    print(f'{folder.name}: source={len(source)}, target={len(target)}, rank={r}, '
          f'pair-draw records={len(target_records)}', flush=True)
    return cov, cells, source_records, target_records


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def summarize_existing():
    matched = pd.read_csv(OUT / 'matched_port_pair_diagnostics.csv')
    comparisons, port_records = [], []
    for space, table in matched.groupby('space'):
        for left, right in [((0, 1), (6, 7)), ((1, 4), (2, 7))]:
            lhs = table[(table.a == left[0]) & (table.b == left[1])].set_index('port')
            rhs = table[(table.a == right[0]) & (table.b == right[1])].set_index('port')
            common = sorted(set(lhs.index) & set(rhs.index))
            comparison = f'{left[0]}-{left[1]}_vs_{right[0]}-{right[1]}'
            record = dict(space=space, comparison=comparison, common_target_ports=len(common),
                          common_port_names=';'.join(common))
            for side, part in [('left', lhs.loc[common]), ('right', rhs.loc[common])]:
                for column in ('source_auc', 'auc', 'held_minus_source_auc_pp',
                               'source_standardized_gap', 'held_standardized_gap',
                               'target_gap_ratio', 'held_variance_ratio'):
                    record[f'{side}_{column}'] = float(part[column].mean())
            record['left_minus_right_source_auc_pp'] = 100 * (record['left_source_auc'] - record['right_source_auc'])
            record['left_minus_right_held_auc_pp'] = 100 * (record['left_auc'] - record['right_auc'])
            comparisons.append(record)
            for port in common:
                port_records.append(dict(space=space, comparison=comparison, port=port,
                    left_source_auc=lhs.loc[port, 'source_auc'], right_source_auc=rhs.loc[port, 'source_auc'],
                    left_held_auc=lhs.loc[port, 'auc'], right_held_auc=rhs.loc[port, 'auc'],
                    left_eligible_draws=int(lhs.loc[port, 'eligible_draws']),
                    right_eligible_draws=int(rhs.loc[port, 'eligible_draws'])))
    pd.DataFrame(comparisons).to_csv(OUT / 'same_target_port_pair_comparisons.csv', index=False)
    pd.DataFrame(port_records).to_csv(OUT / 'same_target_port_pair_details.csv', index=False)
    cov = pd.read_csv(OUT / 'fold_draw_class_coverage.csv', keep_default_na=False)
    records = []
    for port, part in cov.groupby('port'):
        source = part[part.scope == 'source'].iloc[0]
        query = part[part.scope == 'query'].sort_values('draw')
        support = part[part.scope == 'support'].sort_values('draw')
        records.append(dict(port=port, source_n_rows=int(source.n_rows),
            source_present_classes=source.present_classes, source_n_classes=int(source.n_classes),
            support_n_classes_by_draw=';'.join(map(str, support.n_classes)),
            query_n_classes_by_draw=';'.join(map(str, query.n_classes)),
            query_all8_every_draw=bool((query.n_classes == 8).all()),
            query_class_set_same_every_draw=bool(query.present_classes.nunique() == 1)))
    pd.DataFrame(records).to_csv(OUT / 'port_coverage_summary.csv', index=False)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--summarize-existing', action='store_true')
    args = parser.parse_args()
    if args.summarize_existing:
        summarize_existing()
        return
    self_test()
    if args.self_test:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(MANIFEST, usecols=['port', 'class_id', 'class_name'], keep_default_na=False)
    folders = sorted(p for p in BANKS.iterdir() if (p / 'policies.npz').exists())
    all_cov = pd.DataFrame([coverage(part.class_id.to_numpy(), port=port,
                                    evaluated_target=port in {p.name for p in folders})
                            for port, part in frame.groupby('port')])
    all_cov.to_csv(OUT / 'all24_port_class_counts.csv', index=False)
    frame[['class_id', 'class_name']].drop_duplicates().sort_values('class_id').to_csv(
        OUT / 'class_names.csv', index=False)
    protocol = dict(
        manifest=str(MANIFEST), banks=str(BANKS), reused_target_means=str(PARENT / 'target_port_pair_means.csv'),
        new_model_fits=0, new_permutations=0, no_target_selection=True,
        source_status='Fit-in descriptive, NOT independent or nested OOF evaluation',
        direction='Unmodified task_stable_sensitivity.pair_geometry; equal COMMON ports, >=5/class/port, >=3 ports',
        source_auc='Pooled weighted AUC, each class has mass 1/2, each common port equal within class; includes cross-port pairs',
        variance='Population variance (ddof=0) along source direction; equal classes/equal common source ports. Within-class total = mean within-cell variance + port-mean variance',
        standardized_gap='Signed class mean gap / sqrt(mean within-class variances); source gap positive by construction',
        target='Same saved support-excluding five query masks, >=5 per class, no direction reorientation',
        aggregation='Draw mean per eligible target port/pair, equal eligible ports per pair, equal 28 pairs for overall. Source comparisons restricted to same eligible folds',
        boundaries='Overlapping five draws and folds are NOT independent repeats. Source-vs-held differences mix fit optimism, sample composition and transfer; no physical-cause claim',
        ba_definition='mean recall over classes present in query truth only; absent recalls are None, not 0; output space remains eight classes')
    write_json(OUT / 'PROTOCOL.json', protocol)
    (OUT / 'BA_IMPLEMENTATION.txt').write_text(
        f'{inspect.getsourcefile(metrics)}:{inspect.getsourcelines(metrics)[1]}\n\n'
        + inspect.getsource(metrics), encoding='utf-8')
    coverage_records, cells, source_records, target_records = [], [], [], []
    with threadpool_limits(limits=2):
        for folder in folders:
            c, p, s, t = run_port(folder, frame)
            coverage_records.extend(c)
            cells.extend(p)
            source_records.extend(s)
            target_records.extend(t)
    cov, source, target = map(pd.DataFrame, (coverage_records, source_records, target_records))
    cov.to_csv(OUT / 'fold_draw_class_coverage.csv', index=False)
    pd.DataFrame(cells).to_csv(OUT / 'fold_source_port_class_counts.csv', index=False)
    source.to_csv(OUT / 'source_pair_diagnostics.csv', index=False)
    target.to_csv(OUT / 'held_pair_draw_diagnostics.csv', index=False)
    grouped = target.groupby(KEYS)
    held = grouped.mean(numeric_only=True).drop(columns='draw').reset_index()
    held = held.merge(grouped.agg(eligible_draws=('draw', 'size'),
                                 held_min_n_a=('n_a', 'min'), held_min_n_b=('n_b', 'min')).reset_index(), on=KEYS)
    reused = pd.read_csv(PARENT / 'target_port_pair_means.csv')
    merged = reused.merge(held, on=KEYS, how='outer', validate='one_to_one', indicator=True,
                          suffixes=('', '_recomputed'))
    assert (merged['_merge'] == 'both').all()
    merged = merged.drop(columns='_merge')
    errors = {}
    for original, recomputed in [('auc', 'held_auc_recomputed'), ('target_gap_ratio', 'held_gap_ratio_recomputed'),
                                 ('n_a', 'n_a_recomputed'), ('n_b', 'n_b_recomputed')]:
        errors[original] = float(np.max(np.abs(merged[original] - merged[recomputed])))
        np.testing.assert_allclose(merged[original], merged[recomputed], rtol=1e-10, atol=1e-10)
    merged = merged.merge(source, on=KEYS, validate='one_to_one')
    merged['held_minus_source_auc_pp'] = 100 * (merged['auc'] - merged['source_auc'])
    merged['held_minus_source_standardized_gap'] = merged['held_standardized_gap'] - merged['source_standardized_gap']
    merged.to_csv(OUT / 'matched_port_pair_diagnostics.csv', index=False)
    pair = merged.groupby(['space', 'a', 'b']).mean(numeric_only=True).reset_index()
    pair = pair.merge(merged.groupby(['space', 'a', 'b']).agg(
        eligible_target_ports=('port', 'size'), eligible_pair_draws=('eligible_draws', 'sum'),
        minimum_query_class_a=('held_min_n_a', 'min'), minimum_query_class_b=('held_min_n_b', 'min'),
        minimum_common_source_ports=('source_common_ports', 'min')).reset_index(), on=['space', 'a', 'b'])
    pair.to_csv(OUT / 'all28_pair_summary.csv', index=False)
    overall = pair.drop(columns=['a', 'b']).groupby('space').mean(numeric_only=True)
    overall.to_csv(OUT / 'equal_pair_summary.csv')
    counts = dict(
        manifest_rows=len(frame), manifest_ports=len(all_cov),
        manifest_ports_with_all8=int((all_cov.n_classes == 8).sum()),
        manifest_class_count_distribution=all_cov.n_classes.value_counts().sort_index().to_dict(),
        target_ports=len(folders), target_pool_rows=int(cov[cov.scope == 'target_pool'].n_rows.sum()),
        excluded_manifest_ports=all_cov.loc[~all_cov.evaluated_target, 'port'].tolist(),
        source_folds_with_all8=int(((cov.scope == 'source') & (cov.n_classes == 8)).sum()),
        source_fold_class_count_distribution=cov[cov.scope == 'source'].n_classes.value_counts().sort_index().to_dict())
    for scope in ('target_pool', 'support', 'query'):
        part = cov[cov.scope == scope]
        counts[scope + '_class_count_distribution'] = part.n_classes.value_counts().sort_index().to_dict()
        counts[scope + '_all8'] = int((part.n_classes == 8).sum())
    query = cov[cov.scope == 'query']
    counts['query_draws_support_unseen_classes'] = int((query.support_unseen_query_rows > 0).sum())
    counts['query_support_unseen_rows_overlapping_draw_sum'] = int(query.support_unseen_query_rows.sum())
    counts['query_rows_overlapping_draw_sum'] = int(query.n_rows.sum())
    counts['source_pair_records'] = len(source)
    counts['held_pair_draw_records'] = len(target)
    counts['held_port_pair_records'] = len(merged)
    counts['parent_replay_max_abs_errors'] = errors
    counts['new_model_fits'] = 0
    counts['new_permutations'] = 0
    write_json(OUT / 'COMPLETE.json', counts)
    summarize_existing()
    print(json.dumps(counts, indent=2), flush=True)
    cols = ['space', 'a', 'b', 'eligible_target_ports', 'source_auc', 'auc',
            'held_minus_source_auc_pp', 'source_standardized_gap', 'held_standardized_gap',
            'target_gap_ratio', 'held_variance_ratio']
    wanted = {(0, 1), (1, 4), (1, 5), (2, 3), (2, 7), (6, 7)}
    print(pair[[tuple(x) in wanted for x in pair[['a', 'b']].to_numpy()]][cols].to_string(index=False), flush=True)
    print(overall[['source_auc', 'auc', 'held_minus_source_auc_pp', 'source_standardized_gap',
                   'held_standardized_gap', 'held_variance_ratio']].to_string(), flush=True)


if __name__ == '__main__':
    main()
