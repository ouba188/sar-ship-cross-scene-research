"""Audit completed three-seed training artifacts only; never evaluate or read test results.

Do not run until all three seeds finish training. Checkpoints use CPU weights_only
loading, one seed at a time. Only aggregate_training_integrity.json is written;
sample/vessel/reference IDs and prediction data never appear in that output.
"""
import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path

# On this Windows/Anaconda installation NumPy must initialize OpenMP first.
import numpy as np  # noqa: F401
import torch


SEEDS = (17, 29, 43)
STAGES = ('baseline_stage1', 'baseline_base', 'disentangle_stage1', 'disentangle_base', 'full_h')
GROUPS = {'projections': 'projections.', 'edge': 'edge_score.',
          'prototype': 'prototype_score.', 'attention': 'attention.', 'dustbin': 'dustbin'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_report(path, report):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def check(result, name, passed, **details):
    result['checks'].append(dict(name=name, passed=bool(passed), **details))
    if not passed:
        result['passed'] = False


def load_checkpoint(directory, stage):
    # No unsafe fallback and no model construction/import from the training package.
    return torch.load(directory / (stage + '.pt'), map_location='cpu', weights_only=True)


def same_tensor_dict(left, right):
    keys_same = left.keys() == right.keys()
    changed = 0
    incompatible = 0
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        compatible = isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        compatible = compatible and a.shape == b.shape and a.dtype == b.dtype
        if not compatible:
            incompatible += 1
        elif not torch.equal(a, b):
            changed += 1
    return dict(passed=bool(left) and keys_same and not changed and not incompatible,
                tensor_count=len(left), right_tensor_count=len(right),
                same_keys=keys_same, changed_tensors=changed, incompatible_tensors=incompatible)


def normalization_snapshot(saved):
    return dict(mean=saved['input_config']['mean'], std=saved['input_config']['std'],
                scene_mean=saved['scene_mean'], scene_std=saved['scene_std'])


def same_normalization(left, right):
    return all(torch.equal(torch.as_tensor(left[key]), torch.as_tensor(right[key])) for key in left)


def stage1_history_checks(result, baseline, disentangle, config, train_ids, rows, saved_history):
    check(result, 'stage1_epoch_counts', len(baseline) == len(disentangle) == 20,
          baseline_epochs=len(baseline), disentangle_epochs=len(disentangle))
    check(result, 'stage1_epoch_sequence', [row['epoch'] for row in baseline] == list(range(1, 21))
          and [row['epoch'] for row in disentangle] == list(range(1, 21)))
    supplemental = config['supplemental_batches']
    extras = [sample_id for batch in supplemental for sample_id in batch]
    class_counts = Counter(rows[sample_id]['label'] for sample_id in extras)
    check(result, 'fixed_training_budget', len(train_ids) == 718 and len(extras) == 32
          and config['stage1_epochs'] == 20 and config['base_epochs'] == 10 and config['h_epochs'] == 20,
          natural_samples=len(train_ids), supplemental_exposures=len(extras))
    check(result, 'supplemental_classes_and_repeats', len(set(extras)) == 8
          and set(Counter(extras).values()) == {4} and len(class_counts) == 4
          and set(class_counts.values()) == {8}, per_class_exposures=dict(sorted(class_counts.items())))
    expected_counter = Counter(train_ids) + Counter(extras)
    batches = math.ceil(len(train_ids) / config['batch_size']) + len(supplemental)
    schedule = config['loss']
    epochs = []
    eligible_epochs = []
    for base, dis in zip(baseline, disentangle):
        epoch = dis['epoch']
        same_order = base.get('query_order') == dis.get('query_order')
        exposure_ok = all(Counter(row.get('query_order', [])) == expected_counter
                          and row.get('natural_query_count') == 718
                          and row.get('supplemental_query_count') == 32 for row in (base, dis))
        exact_segments = all(Counter(row.get('query_order', [])[:718]) == Counter(train_ids)
                             and row.get('query_order', [])[718:] == extras for row in (base, dis))
        strong = dis['controlled_samples'] == 0 and dis['controlled_class_counts'] == [0] * 4
        empirical = dis['regularization_samples'] == 32 and dis['regularization_class_counts'] == [8] * 4
        scene_targets = dis['losses']['scene_targets'] * batches
        scene = scene_targets > 0 and dis['losses']['scene'] > 0
        steps = list(range((epoch - 1) * batches, epoch * batches))
        ramps = [min(1., max(0., (step - schedule['warmup_steps'] + 1) / max(schedule['ramp_steps'], 1))) for step in steps]
        hard = [int(step >= schedule['hard_gate_after']) for step in steps]
        schedule_ok = math.isclose(dis['losses']['hard_gate'], sum(hard) / batches, abs_tol=1e-8)
        schedule_ok = schedule_ok and math.isclose(dis['losses']['constraint_ramp'], sum(ramps) / batches, abs_tol=1e-8)
        eligible = bool(hard[-1] and ramps[-1] == 1.)
        if eligible:
            eligible_epochs.append(dis)
        finite = all(math.isfinite(float(value)) for value in dis['losses'].values())
        baseline_ce = all(base['losses'].get(key) == 0 for key in ('scene', 'invariance', 'leakage', 'discriminator'))
        passed = same_order and exposure_ok and exact_segments and strong and empirical and scene and schedule_ok and finite and baseline_ce
        epochs.append(dict(epoch=epoch, passed=passed, identical_query_order=same_order,
                           exposure_counts_valid=exposure_ok and exact_segments,
                           strong_controlled=dis['controlled_samples'], empirical=dis['regularization_samples'],
                           empirical_class_counts=dis['regularization_class_counts'],
                           scene_targets_from_logged_batch_mean=scene_targets,
                           scene_loss=dis['losses']['scene'], hard_route_at_epoch_end=eligible,
                           schedule_matches_log=schedule_ok, finite_losses=finite, baseline_pure_ce=baseline_ce))
    check(result, 'stage1_epoch_exposure_and_supervision', len(epochs) == 20 and all(row['passed'] for row in epochs))
    result['stage1_epochs'] = epochs
    chosen = saved_history[-1]
    eligible_best = max(eligible_epochs, key=lambda row: row['selection_score']) if eligible_epochs else None
    check(result, 'hard_route_checkpoint_selection', eligible_best is not None
          and chosen['epoch'] == eligible_best['epoch']
          and chosen['selection_score'] == eligible_best['selection_score']
          and saved_history == disentangle[:len(saved_history)]
          and all(row['selection_metric'] == 'balanced_accuracy' for row in disentangle),
          selected_epoch=chosen['epoch'], eligible_epoch_count=len(eligible_epochs),
          expected_selected_epoch=eligible_best['epoch'] if eligible_best else None)


def h_parameter_updates(result, before, after, config):
    check(result, 'h_state_keys_unchanged', before.keys() == after.keys())
    expected = dict(projections=True, dustbin=True, edge=config.get('learn_edge_weights', True),
                    prototype=config.get('prototype_weighting', 'floor') != 'uniform', attention=config['alpha'] > 0)
    updates = {}
    recognized = set()
    for group, prefix in GROUPS.items():
        keys = [key for key in before if key == prefix or key.startswith(prefix)]
        recognized.update(keys)
        squared, changed, finite, compatible = 0., 0, True, True
        for key in keys:
            if key not in after or before[key].shape != after[key].shape or before[key].dtype != after[key].dtype:
                compatible = False
                continue
            difference = after[key].double() - before[key].double()
            finite = finite and bool(torch.isfinite(difference).all())
            if finite:
                squared += float(torch.sum(difference * difference))
            changed += int(not torch.equal(before[key], after[key]))
        l2 = math.sqrt(squared) if finite else None
        active = expected[group]
        passed = compatible and finite and ((bool(keys) and changed > 0 and l2 > 0) if active else not keys)
        updates[group] = dict(configured_active=bool(active), tensor_count=len(keys), changed_tensors=changed,
                              change_l2=l2, finite=finite, passed=passed)
    unknown = [key for key in before if not key.startswith('h0.') and key not in recognized]
    check(result, 'h_active_submodules_updated', not unknown and all(row['passed'] for row in updates.values()),
          unclassified_tensor_count=len(unknown))
    result['h_parameter_updates'] = updates
    result['h_parameter_update_scope'] = 'Selected H checkpoint minus saved base initialization; a parameter delta is not proof of a nonzero task gradient or classification benefit.'


def h_training_and_reference_checks(result, saved, history, rows, train_ids):
    references = saved['reference_ids']
    source_set = set(train_ids)
    known = [rows[sample_id] for sample_id in references if sample_id in rows]
    recorded_rows = {row['id']: row for row in saved['manifest_metadata']}
    valid = bool(references) and len(set(references)) == len(references)
    valid = valid and set(references) <= source_set and len(known) == len(references)
    provenance = all(row['id'] in recorded_rows and all(recorded_rows[row['id']].get(key) == row.get(key)
                     for key in ('label', 'vessel_id', 'group_id', 'port', 'development_split')) for row in known)
    vessels = {row['vessel_id'] for row in known}
    check(result, 'reference_bank_source_and_vessel_independence', valid and provenance and len(vessels) == len(references),
          observations=len(references), independent_vessels=len(vessels), all_from_training=valid,
          checkpoint_provenance_matches_manifest=provenance)
    result['reference_distribution'] = dict(
        by_class=dict(sorted(Counter(row['label'] for row in known).items())),
        by_port=dict(sorted(Counter(str(row['port']) for row in known).items())),
        by_class_port=[dict(class_name=label, port=port, observations=count)
                       for (label, port), count in sorted(Counter((row['label'], str(row['port'])) for row in known).items())])
    # Recompute metadata-only fallback eligibility; never decode images or run H.
    fallback = 0
    metadata_supported = 0
    for sample_id in train_ids:
        query = rows[sample_id]
        legal = [ref for ref in known if ref['vessel_id'] != query['vessel_id']
                 and ref['group_id'] != query['group_id'] and ref['id'] != query['id']]
        positive = [ref for ref in legal if ref['label'] == query['label']]
        fallback += int(not any(ref['port'] != query['port'] for ref in positive))
        metadata_supported += int(bool(positive))
    base_epochs = [row for row in history if row['stage'] == 'base']
    match_epochs = [row for row in history if row['stage'] == 'match']
    check(result, 'h_full_history_epoch_counts', len(base_epochs) == 10 and len(match_epochs) == 20
          and [row['epoch'] for row in match_epochs] == list(range(1, 21)),
          base_epochs=len(base_epochs), match_epochs=len(match_epochs))
    batches = math.ceil(len(train_ids) / saved['h_training_config']['batch_size'])
    records = []
    for row in match_epochs:
        raw_total = row['losses']['match_samples'] * batches
        total = round(raw_total)
        reconstructed = math.isclose(raw_total, total, abs_tol=1e-6)
        logged_fallback = row['source_support_fallback_samples']
        finite = all(math.isfinite(float(value)) for value in row['losses'].values())
        passed = reconstructed and 0 < total <= metadata_supported and logged_fallback == fallback and finite
        records.append(dict(epoch=row['epoch'], passed=passed, same_class_supported_samples=total,
                            source_support_fallback_samples=logged_fallback, batch_count=batches,
                            logged_mean_supported_per_batch=row['losses']['match_samples'], finite_losses=finite))
    result['h_epochs'] = records
    result['h_support_count_scope'] = 'Actual same-class support totals reconstructed from logged per-batch mean times batch count; fallback checked from the selected fixed source bank. Metadata eligibility alone does not guarantee usable feature edges.'
    check(result, 'h_actual_support_and_fallback', len(records) == 20 and all(row['passed'] for row in records),
          metadata_eligible_same_class_samples=metadata_supported, expected_fallback_samples=fallback)


def verify_seed(root, seed, rows):
    result = dict(seed=seed, passed=True, checks=[])
    directory = root / 'runs' / f'seed{seed}'
    try:
        base_stage1 = load_checkpoint(directory, 'baseline_stage1')
        baseline_meta = dict(normalization=normalization_snapshot(base_stage1),
                             source_ids=base_stage1['source_ids'], labels=base_stage1['labels'])
        check(result, 'baseline_classification_configuration', base_stage1['training_objective'] == 'classification_baseline'
              and base_stage1['encoder_config']['gate_enabled'] is False)
        del base_stage1
        gc.collect()
        dis = load_checkpoint(directory, 'disentangle_stage1')
        train_ids = dis['source_ids']
        manifest_train = [sample_id for sample_id, row in rows.items() if row.get('development_split') == 'train']
        check(result, 'training_source_matches_manifest', train_ids == manifest_train and len(set(train_ids)) == 718)
        check(result, 'baseline_disentangle_metadata_equal', baseline_meta['source_ids'] == train_ids
              and baseline_meta['labels'] == dis['labels']
              and same_normalization(baseline_meta['normalization'], normalization_snapshot(dis)))
        check(result, 'disentangle_empirical_configuration', dis['training_objective'] == 'disentanglement'
              and dis['encoder_config']['gate_enabled'] is True and dis['leakage_mode'] == 'empirical_regularization'
              and not dis.get('incomplete_supervision', True))
        stage1_history_checks(result, read_json(directory / 'baseline_stage1.history.json'),
                              read_json(directory / 'disentangle_stage1.history.json'),
                              dis['protocol_config'], train_ids, rows, dis['history'])
        base = load_checkpoint(directory, 'disentangle_base')
        equality = same_tensor_dict(dis['encoder'], base['encoder'])
        check(result, 'encoder_stage1_to_base_exact', **equality)
        check(result, 'disentangle_base_metadata_equal', base['source_ids'] == train_ids and base['labels'] == dis['labels']
              and same_normalization(normalization_snapshot(dis), normalization_snapshot(base)))
        del dis, baseline_meta
        gc.collect()
        full = load_checkpoint(directory, 'full_h')
        check(result, 'encoder_base_to_h_exact', **same_tensor_dict(base['encoder'], full['encoder']))
        old_h0 = {key: value for key, value in base['h'].items() if key.startswith('h0.')}
        new_h0 = {key: value for key, value in full['h'].items() if key.startswith('h0.')}
        check(result, 'h0_base_to_h_exact', **same_tensor_dict(old_h0, new_h0))
        check(result, 'base_to_h_metadata_equal', full['source_ids'] == train_ids and full['labels'] == base['labels']
              and same_normalization(normalization_snapshot(base), normalization_snapshot(full)))
        h_parameter_updates(result, base['h'], full['h'], full['h_config'])
        h_training_and_reference_checks(result, full, read_json(directory / 'full_h.history.json'), rows, train_ids)
        check(result, 'checkpoint_stage_tags', base['stage'] == 'base' and full['stage'] == 'h')
        del base, full, old_h0, new_h0
    except Exception as exc:
        # Do not serialize exception text: malformed artifacts may include private IDs.
        check(result, 'artifact_processing', False, error_type=type(exc).__name__,
              explanation='Artifact could not be processed. Inspect local training artifacts; no IDs copied to this report.')
    finally:
        gc.collect()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = root / 'aggregate_training_integrity.json'
    report = dict(status='checking', seeds=[], scope='Training histories and checkpoints only; no test results or images read.',
                  interpretation='Exact frozen tensors, logged supervision and parameter changes establish engineering integrity, not causal identification or performance gains.')
    try:
        state = read_json(root / 'state_local.json')
    except Exception as exc:
        report.update(status='failed', state_read_error_type=type(exc).__name__)
        write_report(output, report)
        parser.exit(1, 'Cannot read training state. Aggregate report preserved.\n')
    missing = [dict(seed=seed, stage=stage) for seed in SEEDS for stage in STAGES
               if state.get('steps', {}).get(f'seed{seed}/{stage}', {}).get('status') != 'succeeded']
    if missing:
        report.update(status='not_ready', unfinished_training=missing)
        write_report(output, report)
        parser.exit(2, 'Training integrity check remains closed: all three seeds must finish all five training stages.\n')
    try:
        payload = read_json(root / 'experiment_manifest_local.json')
        manifest = payload if isinstance(payload, list) else payload['records']
        rows = {row['id']: row for row in manifest}
    except Exception as exc:
        report.update(status='failed', manifest_read_error_type=type(exc).__name__)
        write_report(output, report)
        parser.exit(1, 'Cannot read training manifest. Aggregate report preserved.\n')
    if len(rows) != len(manifest):
        report.update(status='failed', manifest_unique_observations=False)
        write_report(output, report)
        parser.exit(1, 'Training integrity failed: duplicate manifest entries. Aggregate report preserved.\n')
    for seed in SEEDS:
        result = verify_seed(root, seed, rows)
        report['seeds'].append(result)
        write_report(output, report)
        print(f"seed{seed}: {'passed' if result['passed'] else 'FAILED'}", flush=True)
        gc.collect()
    report['status'] = 'passed' if all(row['passed'] for row in report['seeds']) else 'failed'
    write_report(output, report)
    if report['status'] != 'passed':
        parser.exit(1, 'Training integrity failed. Aggregate report preserved; training artifacts were not changed.\n')
    print('Training integrity passed; aggregate_training_integrity.json written.', flush=True)


if __name__ == '__main__':
    main()
