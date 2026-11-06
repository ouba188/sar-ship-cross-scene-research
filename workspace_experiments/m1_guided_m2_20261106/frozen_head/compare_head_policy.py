"""CPU-only paired analysis. Reads completed runs; never imports the training stack.

Run: python compare_head_policy.py
Outputs: head_policy_*.csv and head_policy_summary.json beside this script.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
PREVIOUS = PARENT.parent / 'm1_coverage_20261101'
MANIFESTS = (PARENT.parent / 'scene_aware_classification_20261018' /
             'model_closeout_20261023/final_oss2_replication_20261029/manifests')
ARMS = ('mean', 'content', 'class')
SEEDS = (17, 29, 43)
SPLITS = ('validation', 'evaluation')
METRICS = ('oa', 'ba', 'nll', 'balanced_nll')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(y, score):
    """Match model_task.metrics: float64 softmax, 1e-20 probability floor."""
    z = np.asarray(score, dtype=np.float64)
    probability = np.exp(z - z.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    loss = -np.log(probability[np.arange(len(y)), y].clip(1e-20))
    correct = score.argmax(1) == y
    labels = np.unique(y)
    return dict(oa=float(correct.mean() * 100),
                ba=float(np.mean([correct[y == c].mean() for c in labels]) * 100),
                nll=float(loss.mean()),
                balanced_nll=float(np.mean([loss[y == c].mean() for c in labels])))


def load_metadata(groups, n):
    path = PREVIOUS / 'manifest.json'
    rows = read(path)
    require(len(rows) == n, 'Previous manifest length differs from cache.')
    old_groups = read(PREVIOUS / 'groups.json')
    require(all(old_groups[k] == v for k, v in groups.items()),
            'Previous manifest groups differ from cached groups.')
    require(len({r['id'] for r in rows}) == n, 'Manifest has repeated observation IDs.')
    # table_index belongs to the upstream cache, not this concatenated manifest.
    for split in ('source_fit', 'source_adapt', 'validation', 'selection', 'evaluation', 'references'):
        source = read(MANIFESTS / (split + '.json'))
        actual = [(rows[i]['id'], rows[i]['label']) for i in groups[split]]
        require(actual == [(r['id'], r['label']) for r in source],
                f'Manifest identity/label order mismatch: {split}')
    # Existing gallery records provide an independent label/identity cross-check.
    for case in read(PARENT / 'cases.json')['cases']:
        row = rows[case['index']]
        require((row['id'], row['label']) == (case['id'], case['label']),
                f'Previous case disagrees with manifest: {case["id"]}.')
    metadata = []
    for i, row in enumerate(rows):
        paths = row.get('source_paths', {})
        metadata.append(dict(row_index=i, table_index=row['indices']['table_index'],
                             id=row['id'], label=row['label'],
                             class_name=row['class_name'], ship=row.get('mmsi'),
                             orbit='_'.join(row['pass_group'].split('_')[:2]),
                             pass_group=row['pass_group'], scene_id=row.get('scene_id'),
                             vv_path=paths.get('vv'), vh_path=paths.get('vh')))
    return pd.DataFrame(metadata), path


def read_score(path, ids):
    with np.load(path, allow_pickle=False) as saved:
        require(np.array_equal(saved['ids'], ids), f'Prediction order mismatch: {path}')
        score = saved['score'].copy()
    require(score.shape == (len(ids), 4) and np.isfinite(score).all(),
            f'Invalid score matrix: {path}')
    return score


def event_counts(frame):
    return dict(events=len(frame), unique_observations=int(frame.id.nunique()),
                unique_ships=int(frame.ship.nunique()),
                unique_orbits=int(frame.orbit.nunique()))


def main():
    # Receipts are the completion boundary; no polling or partial reports.
    missing = [str(base / f'{arm}_s{seed}' / 'receipt.json')
               for base in (PARENT, ROOT) for seed in SEEDS for arm in ARMS
               if not (base / f'{arm}_s{seed}' / 'receipt.json').exists()]
    if missing:
        raise SystemExit('Runs incomplete; no output written. Rerun after completion:\n'
                         + '\n'.join(missing))
    groups = read(PARENT / 'groups.json')
    cache = np.load(PARENT / 'cache/score.npy', mmap_mode='r', allow_pickle=False)
    metadata, manifest = load_metadata(groups, len(cache))
    labels = metadata.label.to_numpy(dtype=np.int64)
    require(np.isin(labels, np.arange(4)).all(), 'Labels outside the four-class task.')
    entries, receipts, metric_rows, checks, clipping = {}, {}, [], [], []
    for split in SPLITS:
        ids = np.array(groups[split])
        entries[('original', split)] = np.array(cache[ids])
        metric_rows.append(dict(policy='original', arm='baseline', seed=None, split=split,
                                mode='on', **metrics(labels[ids], cache[ids])))
    for seed in SEEDS:
        expected_orders = read(PARENT / f'mean_s{seed}/orders.json')
        for arm in ARMS:
            run = f'{arm}_s{seed}'
            for policy, base in (('parent', PARENT), ('frozen', ROOT)):
                dest = base / run
                receipt = read(dest / 'receipt.json')
                receipts[(policy, arm, seed)] = receipt
                require(receipt['arm'] == arm and receipt['seed'] == seed,
                        f'Run identity mismatch: {dest}')
                require(read(dest / 'orders.json') == expected_orders,
                        f'Training orders differ: {dest}')
                if policy == 'frozen':
                    require(receipt.get('head_frozen') is True,
                            f'Missing/false head_frozen: {dest}; rerun after final receipt.')
                    require(receipt.get('head_max_change') == 0,
                            f'Head changed: {dest}')
                    old = receipts[('parent', arm, seed)]
                    for key in ('steps', 'n_train', 'examples', 'upstream_seed',
                                'checkpoint_selection'):
                        require(receipt[key] == old[key], f'{key} differs: {run}')
                    require(old['trainable'] - receipt['trainable'] == 7170 * 4 + 4,
                            f'Trainable parameter difference is not just the head: {run}')
                for gradient in receipt['gradients']:
                    norms = gradient['norms']
                    if policy == 'frozen':
                        require(not any(k.startswith('head.') for k in norms),
                                f'Frozen head has gradients: {run}')
                        require(all(norms.get(f'output.{p}.weight', 0) > 0 for p in (0, 1)),
                                f'Missing readout gradient: {run}')
                    total = float(np.sqrt(sum(v * v for v in norms.values())))
                    readout = float(np.sqrt(sum(v * v for k, v in norms.items()
                                               if not k.startswith('head.'))))
                    clipping.append(dict(policy=policy, arm=arm, seed=seed,
                                         step=gradient['step'], total_norm=total,
                                         readout_norm=readout,
                                         clip_factor=min(1., 5. / (total + 1e-6))))
                for split in SPLITS:
                    ids = np.array(groups[split])
                    modes = ('on', 'off', 'uniform', 'shuffle') if arm == 'class' else ('on', 'off')
                    for mode in modes:
                        name = split + ('' if mode == 'on' else '_' + mode)
                        score = read_score(dest / (name + '.npz'), ids)
                        entries[(policy, arm, seed, split, mode)] = score
                        result = metrics(labels[ids], score)
                        for key in METRICS:
                            require(np.isclose(result[key], receipt['metrics'][name][key],
                                               rtol=0, atol=1e-10),
                                    f'Recomputed {key} differs from receipt: {dest}/{name}')
                        metric_rows.append(dict(policy=policy, arm=arm, seed=seed,
                                                split=split, mode=mode, **result))
                        if policy == 'frozen' and mode == 'off':
                            baseline = entries[('original', split)]
                            error = float(np.abs(score - baseline).max())
                            same = np.array_equal(score.argmax(1), baseline.argmax(1))
                            require(error < 1e-4 and same,
                                    f'Frozen off does not reproduce original cache: {run}/{split}')
                            require(np.isclose(error, receipt['off_errors'][split],
                                               rtol=0, atol=1e-12),
                                    f'Off error disagrees with receipt: {run}/{split}')
                            checks.append(dict(arm=arm, seed=seed, split=split,
                                               head_frozen=True, head_max_change=0,
                                               orders_exact=True, off_max_error=error,
                                               off_predictions_exact=same))

    pairs, events, tradeoffs = [], [], []
    for split in SPLITS:
        ids = np.array(groups[split])
        y, meta = labels[ids], metadata.iloc[ids].reset_index(drop=True)
        baseline = entries[('original', split)]
        for seed in SEEDS:
            def score(policy, arm, mode='on'):
                return entries[(policy, arm, seed, split, mode)]

            comparisons = []
            for arm in ARMS:
                comparisons.extend([
                    (f'frozen_{arm}_vs_parent_{arm}', score('frozen', arm), score('parent', arm)),
                    (f'frozen_{arm}_vs_original', score('frozen', arm), baseline),
                    (f'parent_{arm}_vs_original', score('parent', arm), baseline),
                ])
            for arm in ('mean', 'content'):
                comparisons.append((f'frozen_class_vs_frozen_{arm}',
                                    score('frozen', 'class'), score('frozen', arm)))
            for mode in ('off', 'uniform', 'shuffle'):
                comparisons.append((f'frozen_class_vs_frozen_class_{mode}',
                                    score('frozen', 'class'), score('frozen', 'class', mode)))
            for comparison, candidate, control in comparisons:
                pred, old_pred = candidate.argmax(1), control.argmax(1)
                good, old = pred == y, old_pred == y
                rescue, harm = good & ~old, ~good & old
                a, b = metrics(y, candidate), metrics(y, control)
                pairs.append(dict(comparison=comparison, split=split, seed=seed, n=len(y),
                                  **{'candidate_' + k: v for k, v in a.items()},
                                  **{'control_' + k: v for k, v in b.items()},
                                  **{'delta_' + k: a[k] - b[k] for k in METRICS},
                                  rescues=int(rescue.sum()), harms=int(harm.sum()),
                                  net_correct=int(rescue.sum() - harm.sum()),
                                  prediction_changes=int((pred != old_pred).sum())))
                for outcome, mask in (('rescue', rescue), ('harm', harm)):
                    selected = meta.loc[mask].copy()
                    selected['comparison'], selected['split'] = comparison, split
                    selected['seed'], selected['outcome'] = seed, outcome
                    selected['candidate_prediction'] = pred[mask]
                    selected['control_prediction'] = old_pred[mask]
                    events.append(selected)
            original_good = baseline.argmax(1) == y
            old_good = score('parent', 'class').argmax(1) == y
            new_good = score('frozen', 'class').argmax(1) == y
            tradeoffs.append(dict(
                split=split, seed=seed,
                parent_direct_rescues=int((~original_good & old_good).sum()),
                frozen_direct_rescues=int((~original_good & new_good).sum()),
                retained_rescues=int((~original_good & old_good & new_good).sum()),
                lost_rescues=int((~original_good & old_good & ~new_good).sum()),
                newly_rescued=int((~original_good & ~old_good & new_good).sum()),
                parent_direct_harms=int((original_good & ~old_good).sum()),
                frozen_direct_harms=int((original_good & ~new_good).sum()),
                repaired_parent_harms=int((original_good & ~old_good & new_good).sum()),
                retained_harms=int((original_good & ~old_good & ~new_good).sum()),
                newly_harmed=int((original_good & old_good & ~new_good).sum())))

    pair_frame = pd.DataFrame(pairs)
    event_frame = pd.concat(events, ignore_index=True)
    group_keys = ['split', 'comparison', 'outcome']
    unique = event_frame.groupby(group_keys + ['id'], sort=True).agg(
        row_index=('row_index', 'first'), label=('label', 'first'),
        class_name=('class_name', 'first'), ship=('ship', 'first'), orbit=('orbit', 'first'),
        pass_group=('pass_group', 'first'), scene_id=('scene_id', 'first'),
        vv_path=('vv_path', 'first'), vh_path=('vh_path', 'first'),
        events=('seed', 'size'), seeds=('seed', lambda v: ','.join(map(str, sorted(set(v))))))
    strata, aggregates = [], []
    for (split, comparison), part in pair_frame.groupby(['split', 'comparison']):
        group_events = event_frame[(event_frame.split == split) & (event_frame.comparison == comparison)]
        counts = {outcome: event_counts(group_events[group_events.outcome == outcome])
                  for outcome in ('rescue', 'harm')}
        rescued = set(group_events.loc[group_events.outcome == 'rescue', 'id'])
        harmed = set(group_events.loc[group_events.outcome == 'harm', 'id'])
        aggregates.append(dict(split=split, comparison=comparison, seeds=len(part),
                               means={k: float(part[k].mean()) for k in
                                      ['delta_' + k for k in METRICS] + ['rescues', 'harms']},
                               positive_oa_seeds=int((part.delta_oa > 0).sum()),
                               positive_ba_seeds=int((part.delta_ba > 0).sum()),
                               **counts, observations_rescued_and_harmed=len(rescued & harmed)))
        cohort = metadata.iloc[groups[split]]
        for dimension in ('class_name', 'orbit', 'ship'):
            for value, population in cohort.groupby(dimension, dropna=False):
                selected = group_events[group_events[dimension].eq(value)]
                if pd.isna(value):
                    selected = group_events[group_events[dimension].isna()]
                for outcome in ('rescue', 'harm'):
                    strata.append(dict(split=split, comparison=comparison,
                                       dimension=dimension, value=value, outcome=outcome,
                                       cohort_observations=len(population),
                                       **event_counts(selected[selected.outcome == outcome])))

    clip_frame = pd.DataFrame(clipping)
    paired_clips = clip_frame[clip_frame.policy == 'parent'].merge(
        clip_frame[clip_frame.policy == 'frozen'], on=['arm', 'seed', 'step'],
        suffixes=('_parent', '_frozen'))
    affected = paired_clips[(paired_clips.clip_factor_parent < 1) &
                            (paired_clips.clip_factor_frozen == 1)]
    summary = dict(
        manifest=str(manifest), groups=str(PARENT / 'groups.json'),
        baseline=str(PARENT / 'cache/score.npy'), runs=9,
        units=dict(oa_ba='percent; deltas are percentage points', nll='natural log',
                   event='one observation in one seed', unique_observation='manifest id across seeds',
                   orbit="task.orbit: first two underscore-separated pass_group components",
                   missing_metadata='excluded from unique ship/orbit counts'),
        checks=checks, comparisons=aggregates, class_per_seed_tradeoffs=tradeoffs,
        review=dict(
            confound='Global norm clipping at 5 includes the head in parent runs but excludes it '
                     'when frozen. This changes effective readout update scaling as well as the '
                     'head policy. The comparison estimates the full policy intervention, not '
                     'the isolated effect of preserving a decision boundary.',
            observed_clipping_changes=affected.to_dict(orient='records'),
            evidence_limit='Only steps 1 and 2 have saved gradient norms; later clipping is unknown. '
                           'Head immutability is attested by receipts, not independently inspected weights.',
            scope='Fixed upstream seed17; seeds vary the new readout. Evaluation was historically '
                  'exposed. Uniform/shuffle are same-weight diagnostics, not separately trained controls. '
                  'Original means the existing score.npy cache, not recomputed checkpoint logits. '
                  'No bootstrap, training, gallery, or physical-cause claims.'),
    )
    outputs = {
        'metrics.csv': pd.DataFrame(metric_rows),
        'paired.csv': pair_frame,
        'events.csv': event_frame,
        'unique_events.csv': unique.reset_index(),
        'strata.csv': pd.DataFrame(strata),
        'class_tradeoffs.csv': pd.DataFrame(tradeoffs),
        'clipping.csv': clip_frame,
    }
    # All input/receipt checks finish before creating any analysis outputs.
    for name, frame in outputs.items():
        frame.to_csv(ROOT / ('head_policy_' + name), index=False)
    summary['outputs'] = [str(ROOT / ('head_policy_' + name)) for name in outputs]
    summary_path = ROOT / 'head_policy_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
                            encoding='utf-8')
    print(pair_frame.loc[pair_frame.comparison.isin([
        'frozen_class_vs_parent_class', 'frozen_class_vs_original']),
        ['split', 'seed', 'comparison', 'delta_oa', 'delta_ba', 'delta_nll', 'rescues', 'harms']]
        .to_string(index=False))
    print(f'Wrote {len(outputs)} CSV files and {summary_path}')


if __name__ == '__main__':
    main()
