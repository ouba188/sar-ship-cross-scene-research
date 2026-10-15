"""Descriptive expansion of the saved known-single common-hard intersection.

Stored predictions only, two CPU threads. The default requires the completed
72-task-seed main run and its original 23-port + F explanation. Explicit
--explain-root/--out inputs are labeled validation, never full-run evidence.
"""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from strong_single_multidata_explain import (
    ALIGN, ARMS, ROOT, aligned, archive, load_single, read_json, truth_for, write_json,
)

MAIN = ROOT / 'strong_single_multidata_20261010'
DEFAULT_EXPLAIN = MAIN / 'explain'
DEFAULT_OUT = MAIN / 'commonhard_expanded'
SEEDS = [20261010, 20261011, 20261012]
ADDED = ('sarclip', 'r50_c4_common_readout', 'sum', 'sum_difference')
CONTROL_ROOTS = (ROOT / 'strong_single_sarclip_20261010/targets',
                 ROOT / 'strong_single_polarization_20261010/targets')
OLD_FLAGS = ('all_tested_singles_wrong', 'dual_rescues_commonhard',
             'all_tested_singles_and_dual_wrong', 'dual_wrong_any_single_correct',
             'dual_wrong_all_singles_correct')


def original_set(seeds, r18):
    return [f'{arm}:seed_{seed}' for arm in ARMS for seed in seeds] + [
        'current_full_R50only:r50_late', 'reconstruction:mlp_late'
    ] + (['current_task_R18:single_readout'] if r18 else [])


def flags_from_counts(count, total, dual_correct):
    hard = count == 0
    return dict(zip(OLD_FLAGS, (hard, hard & dual_correct, hard & ~dual_correct,
                                ~dual_correct & (count > 0),
                                ~dual_correct & (count == total))))


def mode_predictions(data, mode):
    modes = list(data['modes'])
    if modes.count(mode) != 1:
        raise ValueError(f'Expected exactly one prediction mode: {mode}')
    expected = (len(data['support']), len(modes), len(data['rows']))
    if data['predictions'].shape != expected:
        raise ValueError(f'Prediction axes do not match draw/mode/row: {mode}')
    return data['predictions'][:, modes.index(mode)]


def load_aligned(path, anchor):
    data = archive(path)
    aligned(anchor, data)
    for key in ('support_labels', 'source_rows'):
        if key in data:
            aligned(anchor, data, (key,))
    return data


def query_indices(frame, anchor, y):
    """Vectorized row/draw lookup also proves exact query coverage, not a join."""
    rows = pd.Index(anchor['rows'])
    if not rows.is_unique or not pd.Index(anchor['ids']).is_unique:
        raise ValueError('Archive query rows and IDs must be unique')
    for col in ('row', 'draw', 'class_id', 'tested_single_entries', 'correct_single_entries'):
        if not pd.api.types.is_integer_dtype(frame[col]):
            raise ValueError(f'Expected integer query column: {col}')
    ix = rows.get_indexer(frame['row'])
    draw = frame['draw'].to_numpy()
    if (ix < 0).any() or (draw < 0).any() or (draw >= len(anchor['support'])).any():
        raise ValueError('Query table contains unknown rows/draws')
    keys = draw * len(rows) + ix
    np.testing.assert_array_equal(np.sort(keys), np.flatnonzero(anchor['eval_masks'].ravel()),
                                  err_msg='Missing, duplicate or non-evaluation query exposure')
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str), anchor['ids'][ix])
    np.testing.assert_array_equal(frame.class_id.to_numpy(), y[ix])
    return draw, ix


def expand_queries(frame, anchor, y, dual, old_counts, old_total, added):
    draw, ix = query_indices(frame, anchor, y)
    dc = dual[draw, ix] == y[ix]
    count = old_counts[draw, ix]
    np.testing.assert_array_equal(frame.tested_single_entries, np.full(len(frame), old_total))
    np.testing.assert_array_equal(frame.correct_single_entries, count)
    old_flags = flags_from_counts(count, old_total, dc)
    for name, values in old_flags.items():
        if frame[name].dtype != np.bool_:
            raise ValueError(f'Expected boolean original flag: {name}')
        np.testing.assert_array_equal(frame[name], values, err_msg=f'Original flag mismatch: {name}')
    if tuple(added) != ADDED:
        raise ValueError('All four named additions must be present in the fixed order')
    correct = np.stack([added[name][draw, ix] == y[ix] for name in ADDED], axis=1)
    n_correct = correct.sum(1)
    any_correct = n_correct > 0
    old_hard = old_flags['all_tested_singles_wrong']
    result = frame.copy()
    flags = {}
    for prefix, values in [('old', old_flags), ('expanded', flags_from_counts(
            count + n_correct, old_total + len(ADDED), dc))]:
        flags.update({f'{prefix}_{key}': value for key, value in values.items()})
    flags.update(old_commonhard=old_hard, expanded_commonhard=old_hard & ~any_correct,
                 dual_correct=dc, any_added_single_correct=any_correct,
                 old_commonhard_resolved_by_any_added=old_hard & any_correct,
                 old_dual_rescue_eliminated_by_any_added=old_hard & dc & any_correct,
                 newly_dual_wrong_single_correct_by_any_added=old_hard & ~dc & any_correct,
                 dual_rescue_vs_added_union=dc & ~any_correct,
                 dual_harm_vs_added_union=~dc & any_correct)
    for j, name in enumerate(ADDED):
        hit = correct[:, j]
        result[f'{name}_prediction'] = added[name][draw, ix]
        flags.update({f'{name}_correct': hit,
                      f'old_commonhard_resolved_by_{name}': old_hard & hit,
                      f'old_dual_rescue_eliminated_by_{name}': old_hard & dc & hit,
                      f'newly_dual_wrong_single_correct_by_{name}': old_hard & ~dc & hit,
                      f'dual_rescue_vs_{name}': dc & ~hit,
                      f'dual_harm_vs_{name}': ~dc & hit,
                      f'dual_both_correct_vs_{name}': dc & hit,
                      f'dual_both_wrong_vs_{name}': ~dc & ~hit})
    result['dual_prediction'] = dual[draw, ix]
    result['added_correct_entries'] = n_correct
    result['expanded_tested_single_entries'] = old_total + len(ADDED)
    result['expanded_correct_single_entries'] = count + n_correct
    result['added_correct_bitmask'] = (correct * (1 << np.arange(len(ADDED)))).sum(1)
    result = pd.concat([result, pd.DataFrame(flags, index=result.index)], axis=1)
    # Each eliminated old failure is either a removed dual rescue or new reverse evidence.
    np.testing.assert_array_equal(old_hard.astype(int),
        flags['expanded_commonhard'].astype(int) + flags['old_commonhard_resolved_by_any_added'])
    np.testing.assert_array_equal(flags['old_commonhard_resolved_by_any_added'].astype(int),
        flags['old_dual_rescue_eliminated_by_any_added'].astype(int)
        + flags['newly_dual_wrong_single_correct_by_any_added'])
    return result, list(flags)


def summarize(frame, flags):
    keys = ['row', 'sample_id', 'class_id']
    groups = frame.groupby(keys, sort=False)
    unique = groups.size().rename('eligible_draw_exposures').to_frame()
    unique = unique.join(groups[flags].any().add_prefix('any_draw_'))
    unique = unique.join(groups[flags].all().add_prefix('all_draw_')).reset_index()
    records = [dict(unit='draw_exposure', n=len(frame), **frame[flags].sum().astype(int).to_dict())]
    for label, prefix in [('unique_query_any_draw', 'any_draw_'), ('unique_query_all_draw', 'all_draw_')]:
        records.append(dict(unit=label, n=len(unique), **{
            name: int(unique[prefix + name].sum()) for name in flags}))
    by_draw = frame.groupby('draw')[flags].sum().astype(int)
    by_draw.insert(0, 'n', frame.groupby('draw').size())
    return unique, pd.DataFrame(records), by_draw.reset_index()


def expand_task(spec, receipt, explain_root, out, control_roots):
    dataset, task = spec['dataset'], spec['task']
    context = dict(dataset=dataset, task=task)
    seeds = receipt['seeds']
    folder = Path(spec['directory'])
    anchor = load_single(folder / f'seed_{seeds[0]}')
    y, _ = truth_for(spec, anchor)
    if (receipt['n_rows'], receipt['draws']) != (len(y), len(anchor['support'])):
        raise ValueError(f'Original receipt dimensions differ: {dataset}/{task}')
    old_counts = np.zeros(anchor['eval_masks'].shape, dtype=np.int16)
    for seed in seeds:
        seed_dir = folder / f'seed_{seed}'
        read_json(seed_dir / 'COMPLETE.json')
        data = anchor if seed == seeds[0] else load_single(seed_dir)
        aligned(anchor, data, ALIGN + ('support_labels', 'source_rows'))
        old_counts += (data['predictions'] == y[None, None, :]).sum(axis=1).astype(np.int16)
    ref = load_aligned(Path(receipt['reference']), anchor)
    dual = mode_predictions(ref, receipt['reference_mode'])
    old_counts += mode_predictions(ref, 'r50_late') == y[None]
    reconstruction = load_aligned(Path(receipt['reconstruction_path']), anchor)
    old_counts += mode_predictions(reconstruction, 'mlp_late') == y[None]
    r18_root = receipt.get('r18_root')
    if r18_root:
        r18 = load_aligned(Path(r18_root) / dataset / task / 'predictions.npz', anchor)
        old_counts += mode_predictions(r18, 'r18_task_trained') == y[None]
    expected_set = original_set(seeds, bool(r18_root))
    if receipt['commonhard_single_set'] != expected_set:
        raise ValueError(f'Unsupported original single set: {dataset}/{task}')
    additions, sources = {}, []
    for root, names in zip(control_roots, (ADDED[:2], ADDED[2:])):
        path = root / dataset / task / 'predictions.npz'
        predicted = read_json(path.parent / 'PREDICTED.json')
        if (predicted['dataset'], predicted['task']) != (dataset, task):
            raise ValueError(f'Control completion receipt mismatch: {path}')
        data = load_aligned(path, anchor)
        aligned(anchor, data, ('support_labels', 'source_rows'))
        for name in names:
            additions[name] = mode_predictions(data, name)
        sources.append(dict(path=str(path), modes=list(names)))
    source = explain_root / 'tasks' / dataset / task / 'commonhard_queries.csv.gz'
    frame = pd.read_csv(source, dtype={'sample_id': str}, keep_default_na=False)
    expanded, flags = expand_queries(frame, anchor, y, dual, old_counts, len(expected_set), additions)
    unique, summary, by_draw = summarize(expanded, flags)
    dst = out / 'tasks' / dataset / task
    dst.mkdir(parents=True, exist_ok=True)
    compression = dict(method='gzip', compresslevel=1)
    expanded.to_csv(dst / 'commonhard_queries.csv.gz', index=False, compression=compression)
    unique.to_csv(dst / 'commonhard_unique_queries.csv.gz', index=False, compression=compression)
    by_draw.assign(**context).to_csv(dst / 'counts_by_draw.csv', index=False)
    patterns = expanded.groupby('added_correct_bitmask').agg(
        draw_exposures=('row', 'size'), old_commonhard=('old_commonhard', 'sum'),
        eliminated_dual_rescues=('old_dual_rescue_eliminated_by_any_added', 'sum'),
        new_reverse_evidence=('newly_dual_wrong_single_correct_by_any_added', 'sum')).reset_index()
    patterns.to_csv(dst / 'added_overlap_patterns.csv', index=False)
    write_json(dst / 'RECEIPT.json', dict(**context, seeds=seeds, original_table=str(source),
        original_single_set=expected_set, expanded_single_set=expected_set + list(ADDED),
        added_prediction_archives=sources, original_r18_included=bool(r18_root),
        draw_exposures=len(expanded), unique_queries=len(unique), alignment_exact=True,
        no_query_exposures_lost=True, old_flags_verified_from_predictions=True))
    print(f'Expanded {dataset}/{task}: {len(expanded)} exposures, {len(unique)} unique queries', flush=True)
    return summary.assign(**context), unique.assign(**context), flags


def inputs(explain_root, full_main):
    if full_main:
        status = read_json(MAIN / 'STATUS.json')
        if (status.get('status') != 'complete' or status.get('completed_task_seeds') != 72
                or status.get('tasks_per_dataset') != {'self24ports': 23, 'f0921': 1}):
            raise ValueError('Default report requires all 72 main task-seeds; queue remains untouched')
    run = read_json(explain_root / 'RUN_RECEIPT.json')
    seeds = run['requested_seeds']
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Original explanation has empty or repeated seeds')
    if full_main and (seeds != SEEDS or run['analyzed_tasks'] != {'self24ports': 23, 'f0921': 1}):
        raise ValueError('Default report requires original explain for 23 ports + F, all three seeds')
    specs = [s for s in run['discovered_tasks'] if s['status'] == 'ready']
    pairs = [(s['dataset'], s['task']) for s in specs]
    counts = {dataset: sum(d == dataset for d, _ in pairs) for dataset, _ in pairs}
    if len(set(pairs)) != len(pairs) or not pairs or counts != run['analyzed_tasks']:
        raise ValueError('Original explanation inventory and analyzed task counts differ')
    table_pairs = {(p.parent.parent.name, p.parent.name)
                   for p in (explain_root / 'tasks').glob('*/*/commonhard_queries.csv.gz')}
    if set(pairs) != table_pairs:
        raise ValueError('Original query tables and ready-task inventory differ; no task may be dropped')
    receipts = []
    for spec in specs:
        dataset, task = spec['dataset'], spec['task']
        receipt = read_json(explain_root / 'tasks' / dataset / task / 'RECEIPT.json')
        if (receipt['dataset'], receipt['task'], receipt['seeds']) != (dataset, task, seeds):
            raise ValueError(f'Original task receipt mismatch: {dataset}/{task}')
        if full_main:
            if not receipt.get('r18_root'):
                raise ValueError(f'Main original common-hard is missing R18: {dataset}/{task}')
            if Path(spec['directory']).resolve() != (MAIN / 'targets' / dataset / task).resolve():
                raise ValueError('Main explanation points outside the main training run')
            for seed in SEEDS:
                folder = Path(spec['directory']) / f'seed_{seed}'
                complete = read_json(folder / 'COMPLETE.json')
                if (complete['dataset'], complete['task'], complete['seed']) != (dataset, task, seed):
                    raise ValueError(f'Main completion receipt mismatch: {folder}')
                if not (folder / 'predictions.npz').is_file():
                    raise FileNotFoundError(folder / 'predictions.npz')
        receipts.append(receipt)
    return run, specs, receipts


def aggregate(task_summaries, unique_tables, flags):
    tasks = pd.concat(task_summaries, ignore_index=True)
    unique = pd.concat(unique_tables, ignore_index=True)
    records = []
    for dataset, table in unique.groupby('dataset', sort=False):
        exposure = tasks[(tasks.dataset == dataset) & (tasks.unit == 'draw_exposure')]
        records.append(dict(dataset=dataset, unit='draw_exposure', n=int(exposure.n.sum()),
                            **exposure[flags].sum().astype(int).to_dict()))
        # Dataset+row+ID is the query identity; tasks/draws are not independent queries.
        groups = table.groupby(['row', 'sample_id', 'class_id'], sort=False)
        for unit, prefix, op in [('unique_query_any_draw', 'any_draw_', 'any'),
                                 ('unique_query_all_draw', 'all_draw_', 'all')]:
            values = groups[[prefix + name for name in flags]].agg(op)
            records.append(dict(dataset=dataset, unit=unit, n=len(values), **{
                name: int(values[prefix + name].sum()) for name in flags}))
    return tasks, pd.DataFrame(records)


def numeric_report(out, scope, counts, seeds, summary, receipts):
    lines = ['# Expanded Known-Single Common-Hard', '', f'Scope: {scope}.',
             f'Analyzed tasks: {counts}; neural seeds in original set: {seeds}.',
             'Original common-hard flags are preserved and verified against stored predictions.',
             'Added controls: sarclip, r50_c4_common_readout, sum, sum_difference.',
             'This explicit expanded set is descriptive, not all possible or all historical singles.',
             'Methods were already selected for inclusion; no selection or tuning is performed here.',
             f'Original R18 included in {sum(bool(r.get("r18_root")) for r in receipts)}/{len(receipts)} tasks.', '',
             '| Dataset | Unit | N | Old hard | Expanded hard | Old dual rescues | Expanded dual rescues | Eliminated rescues (union) | New reverse evidence (union) | Expanded dual wrong, any single correct |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summary.itertuples():
        lines.append(f'| {r.dataset} | {r.unit} | {r.n} | {r.old_commonhard} | {r.expanded_commonhard} | '
                     f'{r.old_dual_rescues_commonhard} | {r.expanded_dual_rescues_commonhard} | '
                     f'{r.old_dual_rescue_eliminated_by_any_added} | '
                     f'{r.newly_dual_wrong_single_correct_by_any_added} | '
                     f'{r.expanded_dual_wrong_any_single_correct} |')
    lines += ['', '## Added-Control Rescue And Harm', '',
              'Rescue = dual correct / added single wrong; harm = dual wrong / added single correct.',
              'Removed rescue = old common-hard AND dual correct AND added single correct.',
              'New reverse evidence = old common-hard AND dual wrong AND added single correct.', '',
              '| Dataset | Unit | Added single | Removed rescues | New reverse evidence | Dual rescue | Dual harm |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in summary.to_dict('records'):
        for name in ADDED:
            lines.append(f'| {r["dataset"]} | {r["unit"]} | {name} | '
                         f'{r["old_dual_rescue_eliminated_by_" + name]} | '
                         f'{r["newly_dual_wrong_single_correct_by_" + name]} | '
                         f'{r["dual_rescue_vs_" + name]} | {r["dual_harm_vs_" + name]} |')
    lines += ['', 'Per-control counts overlap and must not be summed. Union columns count each exposure once.',
              'Bitmask order is sarclip=1, r50_c4_common_readout=2, sum=4, sum_difference=8.',
              'Unique-query any/all flags are computed over eligible draw exposures; queries can have both rescue and harm across draws.',
              'Unique-query event totals do not form a partition and cannot be subtracted to obtain an accuracy gain.',
              'Only exposure-level old-hard = expanded-hard + resolved is a partition; any/all aggregation does not preserve that identity.',
              'The added union is an oracle correctness set, not a deployable classifier. Counts are not task-macro accuracy.',
              'No independence of draws or neural seeds, architecture necessity, or physical specialization is inferred.',
              'No neural execution, GPU, source fitting, network access or tuning. Original results are unchanged.', '']
    (out / 'NUMERIC_FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')


def run_report(explain_root, out, control_roots=CONTROL_ROOTS):
    explain_root, out = Path(explain_root).resolve(), Path(out).resolve()
    full_main = explain_root == DEFAULT_EXPLAIN.resolve()
    if out == explain_root or out in explain_root.parents or explain_root in out.parents:
        raise ValueError('Output must be separate from the original explanation')
    if not full_main and out == DEFAULT_OUT.resolve():
        raise ValueError('Validation inputs require a separate --out, e.g. commonhard_expanded/validation_oss2')
    # All preflight checks run before creating output; a partial queue cannot look complete.
    run, specs, receipts = inputs(explain_root, full_main)
    if (out / 'RUN_RECEIPT.json').exists():
        previous = read_json(out / 'RUN_RECEIPT.json')
        if Path(previous['explain_root']).resolve() != explain_root:
            raise ValueError('Output belongs to another original explanation; choose a separate --out')
    summaries, unique_tables = [], []
    for spec, receipt in zip(specs, receipts):
        summary, unique, flags = expand_task(spec, receipt, explain_root, out, control_roots)
        summaries.append(summary)
        unique_tables.append(unique)
    task_summary, dataset_summary = aggregate(summaries, unique_tables, flags)
    task_summary.to_csv(out / 'task_counts.csv', index=False)
    dataset_summary.to_csv(out / 'dataset_counts.csv', index=False)
    scope = 'full_main_72_task_seeds_23_ports_plus_F' if full_main else 'explicit_input_validation_not_full_main'
    numeric_report(out, scope, run['analyzed_tasks'], run['requested_seeds'], dataset_summary, receipts)
    write_json(out / 'RUN_RECEIPT.json', dict(status='complete_for_stated_scope', scope=scope,
        full_main=full_main, explain_root=str(explain_root), analyzed_tasks=run['analyzed_tasks'],
        original_neural_seeds=run['requested_seeds'], added_controls=list(ADDED),
        original_sets=[dict(dataset=r['dataset'], task=r['task'], singles=r['commonhard_single_set']) for r in receipts],
        device='cpu', cpu_threads=2, neural_execution=False, gpu_used=False, source_fit=False,
        network_access=False, tuning=False, no_query_exposures_lost=True,
        union_counts_overlap_safe=True, unique_identity=['dataset', 'row', 'sample_id', 'class_id'],
        count_units=['draw_exposure', 'unique_query_any_draw', 'unique_query_all_draw']))
    print(f'COMPLETE for stated scope: {scope}; output={out}', flush=True)
    return task_summary, dataset_summary


def self_test():
    from tempfile import TemporaryDirectory
    from strong_single_multidata_explain import common_hard

    # Query A is a removed dual rescue in draw 0 but new reverse evidence in draw 1.
    y = np.array([0, 0, 0, 0, 0])
    rows = np.array([4, 1, 8, 3, 6])
    ids = np.array(['support', '001', 'NA', 'C', 'D'])
    masks = np.array([[False, True, True, True, True]] * 2)
    support = np.array([[0], [0]])
    meta = dict(rows=rows, ids=ids, support=support, eval_masks=masks,
                support_labels=y[support], source_rows=np.array([10, 11]))
    wrong = np.ones((2, 5), dtype=np.int8)
    dual = np.array([[0, 0, 1, 0, 1], [0, 1, 1, 0, 1]])
    added = {name: wrong.copy() for name in ADDED}
    added['sarclip'][0, 1:3] = 0
    added['r50_c4_common_readout'][0, 1] = 0
    added['sum'][1, 1] = 0
    original = common_hard(wrong[None], wrong[None], wrong, wrong, dual, y, masks, rows, ids, r18=wrong)
    original = original.iloc[::-1].reset_index(drop=True)
    result, flags = expand_queries(original, meta, y, dual, np.zeros((2, 5), dtype=int), 5, added)
    unique, summary, _ = summarize(result, flags)
    exposure = summary.iloc[0]
    assert (exposure.n, exposure.old_commonhard, exposure.expanded_commonhard) == (8, 8, 5)
    assert exposure.old_dual_rescue_eliminated_by_any_added == 1
    assert exposure.newly_dual_wrong_single_correct_by_any_added == 2
    assert sum(exposure[f'old_dual_rescue_eliminated_by_{name}'] for name in ADDED) == 2
    assert exposure.dual_harm_vs_sarclip == 1 and exposure.dual_rescue_vs_sarclip == 2
    a = unique[unique.sample_id == '001'].iloc[0]
    assert a.any_draw_old_dual_rescue_eliminated_by_any_added
    assert a.any_draw_newly_dual_wrong_single_correct_by_any_added
    assert not a.any_draw_expanded_commonhard
    assert summary.iloc[1].n == 4
    assert summary.iloc[1].expanded_commonhard == 3
    assert summary.iloc[2].expanded_commonhard == 2

    def rejected(call):
        try:
            call()
        except (AssertionError, ValueError):
            return
        raise AssertionError('Malformed alignment was accepted')

    rejected(lambda: query_indices(original.iloc[:-1], meta, y))
    rejected(lambda: query_indices(pd.concat([original.iloc[:-1], original.iloc[:1]]), meta, y))
    bad = original.copy()
    bad.loc[0, 'sample_id'] = 'wrong-id'
    rejected(lambda: query_indices(bad, meta, y))
    bad = original.copy()
    bad.loc[0, 'row'] = 999
    rejected(lambda: query_indices(bad, meta, y))
    bad = original.copy()
    bad.loc[0, 'all_tested_singles_wrong'] = False
    rejected(lambda: expand_queries(bad, meta, y, dual, np.zeros((2, 5), dtype=int), 5, added))
    for key in ALIGN + ('support_labels', 'source_rows'):
        bad_archive = {k: v.copy() for k, v in meta.items()}
        if key == 'ids':
            bad_archive[key].flat[0] = 'changed'
        elif key == 'eval_masks':
            bad_archive[key].flat[0] = not bad_archive[key].flat[0]
        else:
            bad_archive[key].flat[0] += 1
        rejected(lambda: aligned(meta, bad_archive, ALIGN + ('support_labels', 'source_rows')))
    # Repeated task identity cannot multiply dataset-level unique-query counts.
    _, dataset = aggregate([summary.assign(dataset='oss2', task='a'), summary.assign(dataset='oss2', task='b')],
        [unique.assign(dataset='oss2', task='a'), unique.assign(dataset='oss2', task='b')], flags)
    assert dataset.n.tolist() == [16, 4, 4]

    with TemporaryDirectory(prefix='commonhard_expanded_') as temporary:
        root = Path(temporary)
        folder = root / 'trained'
        seed_dir = folder / 'seed_11'
        seed_dir.mkdir(parents=True)
        np.savez(seed_dir / 'predictions.npz', **meta, arms=ARMS,
                 predictions=np.stack([wrong, wrong], axis=1))
        write_json(seed_dir / 'COMPLETE.json', dict(dataset='oss2', task='oss2', seed=11))
        manifest = root / 'manifest.csv'
        manifest_ids = np.full(9, 'unused', dtype=object)
        manifest_ids[rows] = ids
        pd.DataFrame(dict(sample_id=manifest_ids, class_id=np.zeros(9, dtype=int))).to_csv(manifest, index=False)
        reference, reconstruction = root / 'reference.npz', root / 'reconstruction.npz'
        np.savez(reference, **meta, modes=['dual_refit_late', 'r50_late'], predictions=np.stack([dual, wrong], axis=1))
        np.savez(reconstruction, **meta, modes=['mlp_late'], predictions=wrong[:, None])
        r18_root = root / 'r18'
        r18_dir = r18_root / 'oss2/oss2'
        r18_dir.mkdir(parents=True)
        np.savez(r18_dir / 'predictions.npz', **meta, modes=['r18_task_trained'], predictions=wrong[:, None])
        controls = [root / 'sarclip', root / 'polarization']
        for control, names in zip(controls, (ADDED[:2], ADDED[2:])):
            dst = control / 'oss2/oss2'
            dst.mkdir(parents=True)
            # Reversed mode order catches positional mode assumptions.
            names = names[::-1]
            np.savez(dst / 'predictions.npz', **meta, modes=names,
                     predictions=np.stack([added[name] for name in names], axis=1))
            write_json(dst / 'PREDICTED.json', dict(dataset='oss2', task='oss2'))
        explain_root = root / 'explain'
        dst = explain_root / 'tasks/oss2/oss2'
        dst.mkdir(parents=True)
        original.to_csv(dst / 'commonhard_queries.csv.gz', index=False)
        write_json(dst / 'RECEIPT.json', dict(dataset='oss2', task='oss2', seeds=[11], n_rows=5, draws=2,
            reference=str(reference), reference_mode='dual_refit_late', reconstruction_path=str(reconstruction),
            r18_root=str(r18_root), commonhard_single_set=original_set([11], True)))
        write_json(explain_root / 'RUN_RECEIPT.json', dict(requested_seeds=[11], analyzed_tasks={'oss2': 1},
            discovered_tasks=[dict(dataset='oss2', task='oss2', directory=str(folder), manifest=str(manifest),
                                   id_column='sample_id', status='ready')]))
        source_bytes = (dst / 'commonhard_queries.csv.gz').read_bytes()
        _, saved_summary = run_report(explain_root, root / 'out', controls)
        assert saved_summary.n.tolist() == [8, 4, 4]
        assert not read_json(root / 'out/RUN_RECEIPT.json')['full_main']
        assert (dst / 'commonhard_queries.csv.gz').read_bytes() == source_bytes
        saved = pd.read_csv(root / 'out/tasks/oss2/oss2/commonhard_queries.csv.gz',
                            dtype={'sample_id': str}, keep_default_na=False)
        pd.testing.assert_frame_equal(saved[original.columns], original)
        rejected(lambda: run_report(explain_root, explain_root, controls))
        rejected(lambda: run_report(explain_root, DEFAULT_OUT, controls))
    print('SELF_TEST passed: nested flags, asymmetric rescue/harm, overlapping controls, unique queries, '
          'shuffled rows/modes, missing/duplicate/unknown rows, ID/support/eval alignment, '
          'original flag validation, preserved original CSV, end-to-end report, validation labeling.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--explain-root', type=Path, default=DEFAULT_EXPLAIN)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    run_report(args.explain_root, args.out)


if __name__ == '__main__':
    main()
