"""Bounded independent review of adapted-base handoff budgets and policies."""

import argparse
import csv
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits

from verify_residual_expert import archive, close, confusion_metrics, read_json, require
from verify_receiver_consistent_handoff import confidence_parts, replay_mask, replay_score


ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT / 'oof_residual_expert_results_20261007_v2'
FROZEN = ROOT / 'oof_receiver_consistent_results_20261007'
GATE = ROOT / 'oof_context_results_20261007'
ADVANTAGE = ROOT / 'oof_action_advantage_results_20261007'
REFIT_PORT = 'Antwerp-Bruges'
ARMS = ('original_base', 'base_bias', 'base_visual', 'bias_handoff',
        'visual_handoff', 'frozen_handoff', 'advantage_reference')
CONTRASTS = (('visual_handoff', 'base_visual'), ('visual_handoff', 'advantage_reference'),
             ('base_visual', 'original_base'), ('base_visual', 'base_bias'),
             ('base_bias', 'original_base'), ('bias_handoff', 'base_bias'),
             ('visual_handoff', 'bias_handoff'), ('visual_handoff', 'frozen_handoff'))
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'admission_fraction', 'original_change_fraction')
COUNTS = ('evaluated_rows', 'admitted_rows', 'handoff_changed_rows', 'changed_from_original_rows', 'cap')


def fit_svd(design, labels, prior):
    counts = np.bincount(labels, minlength=8)
    weights = 64. / (np.count_nonzero(counts) * counts[labels])
    response = -np.ones((64, 8)) - prior
    response[np.arange(64), labels] += 2.
    augmented = np.vstack((np.sqrt(weights[:, None]) * design, np.sqrt(8.) * np.eye(design.shape[1])))
    target = np.vstack((np.sqrt(weights[:, None]) * response, np.zeros((design.shape[1], 8))))
    coefficient, _, rank, _ = lstsq(augmented, target, lapack_driver='gelsd')
    require(rank == design.shape[1], 'penalized augmented design rank')
    return coefficient


def compare_rows(rows, expected, csv_mode=False):
    require(len(rows) == len(expected), 'metric row count')
    seen = set()
    for row in rows:
        key = row['port'], int(row['seed']), row['arm']
        require(key in expected and key not in seen, 'metric row identity: ' + str(key))
        seen.add(key)
        for field in MEASURES + COUNTS:
            close(float(row[field]), expected[key][field], str(key) + ': ' + field)
        if not csv_mode:
            require(len(row['class_recall']) == 8, str(key) + ': class columns')
            for c, value in enumerate(expected[key]['class_recall']):
                if value is None:
                    require(row['class_recall'][c] is None, str(key) + ': absent class')
                else:
                    close(row['class_recall'][c], value, str(key) + ': class recall')


def verify_summary(summary, expected, ports):
    require(summary['n_ports'] == 23 and summary['contexts'] == 115
            and summary['policy_metric_rows'] == 805 and summary['unique_target_labels'] == 64,
            'summary coverage or unique-label budget')
    require(summary['changed_setting'] ==
            'base may adapt globally; 5% bounds expert handoff, NOT changes versus original base',
            'summary must distinguish handoff cap from original prediction changes')
    require(summary['independent_confirmation'] is False and summary['novelty_established'] is False,
            'summary interpretation')
    values = {arm: {field: np.array([[expected[port, seed, arm][field] for seed in range(5)]
                                     for port in ports]) for field in MEASURES} for arm in ARMS}
    require(len(summary['means']) == len(ARMS), 'summary mean count')
    seen = set()
    for row in summary['means']:
        arm = row['arm']
        require(arm in ARMS and arm not in seen, 'summary mean arm')
        seen.add(arm)
        for field, array in values[arm].items():
            close(row[field], array.mean(1).mean(), arm + ': mean ' + field)
    require(len(summary['comparisons']) == len(CONTRASTS), 'summary comparison count')
    sampled = np.random.default_rng(0).integers(0, 23, (10000, 23))
    seen = set()
    for row in summary['comparisons']:
        key = row['candidate'], row['control']
        require(key in CONTRASTS and key not in seen, 'summary contrast identity')
        seen.add(key)
        candidate, control = key
        delta = values[candidate]['ba'] - values[control]['ba']
        by_port = delta.mean(1)
        measures = dict(mean_delta_ba_pp=100 * by_port.mean(),
                        mean_delta_acc_pp=100 * (values[candidate]['acc'] - values[control]['acc']).mean(1).mean(),
                        positive_ports=int((by_port > 1e-12).sum()),
                        negative_ports=int((by_port < -1e-12).sum()),
                        draw_deltas_pp=100 * delta.mean(0),
                        port_bootstrap_95ci_pp=100 * np.percentile(by_port[sampled].mean(1), [2.5, 97.5]))
        for field, value in measures.items():
            close(row[field], value, str(key) + ': ' + field)
        require(set(row['port_deltas_pp']) == set(ports), 'summary contrast ports')
        close([row['port_deltas_pp'][port] for port in ports], 100 * by_port, str(key) + ': port deltas')
    primary = all((values['visual_handoff']['ba'] - values[control]['ba']).mean() > 0
                  for control in ('base_visual', 'advantage_reference'))
    require(summary['passes_development_screen'] == bool(primary), 'two-control primary screen')


def verify(run):
    require((run / 'POLICIES_SAVED.json').is_file() and (run / 'COMPLETE.json').is_file(),
            'wait for policy freeze and complete readout')
    inventory = read_json(run / 'POLICIES_SAVED.json')
    require(inventory == read_json(PARENT / 'POLICIES_SAVED.json'), 'inherited inventory')
    ports = [item['port'] for item in inventory]
    require(len(ports) == len(set(ports)) == 23 and REFIT_PORT in ports, 'fixed port inventory')
    config = read_json(run / 'RUN_CONFIG.json')
    require(tuple(config['arms']) == ARMS and config['target_labels'] == 64 and config['strength'] == 8.
            and config['expert_admission_cap'] == .05 and config['original_prediction_change_cap'] is False,
            'fixed configuration or budget semantics')
    complete = read_json(run / 'COMPLETE.json')
    require(complete['status'] == 'complete' and complete['target_base_fits'] == 230
            and complete['independent_sklearn_coefficient_checks'] == 230 and complete['new_expert_or_backbone_fits'] == 0
            and complete['policy_metric_rows'] == 805, 'producer completion coverage')
    require(0 <= complete['maximum_sklearn_coefficient_error'] < 1e-10, 'producer sklearn comparison failed')
    expected, refits, replayed_scores, replayed_gates = {}, 0, 0, 0
    errors = dict(coefficient=0., score=0., gate=0.)
    over_original_cap = 0
    for port in ports:
        original = archive(ROOT / 'pred_cache' / (port + '.npz'))
        visual = archive(ROOT / 'pred_cache2' / (port + '.npz'))
        parent = archive(PARENT / port / 'policies.npz')
        frozen = archive(FROZEN / port / 'policies.npz')
        advantage = archive(ADVANTAGE / port / 'policies.npz')
        saved = archive(run / port / 'policies.npz')
        source, q = original['spos'], original['qpos']
        n = len(q)
        require(np.array_equal(q, np.unique(q)) and len(source) == len(np.unique(source))
                and not np.isin(source, q).any(), port + ': source/query row boundary')
        require(np.all(original['ports_all'][source] != port) and np.all(original['ports_all'][q] == port),
                port + ': source/query ports')
        require(np.array_equal(saved['source_rows'], source), port + ': saved source rows')
        for data in (saved, parent, frozen, advantage):
            require(np.array_equal(data['query_rows'], q) and np.array_equal(data['seeds'], np.arange(5)),
                    port + ': parent row/seed identity')
        require(tuple(saved['arm_names']) == ARMS, port + ': saved arms')
        for key, shape in (('scores', (5, 2, n, 8)), ('bias_correction', (5, 1, 8)),
                           ('visual_correction', (5, 129, 8)), ('gate_scores', (5, 2, n)),
                           ('selected', (5, 7, n)), ('fallback_predictions', (5, 7, n)),
                           ('predictions', (5, 7, n)), ('expert_predictions', (5, n)),
                           ('original_base', (n,)), ('support_rows', (5, 64)),
                           ('eval_masks', (5, n)), ('caps', (5,))):
            require(saved[key].shape == shape and np.isfinite(saved[key]).all(), port + ': shape/finite ' + key)
        require(saved['selected'].dtype == saved['eval_masks'].dtype == np.bool_, port + ': boolean masks')
        for key in ('predictions', 'fallback_predictions', 'expert_predictions', 'original_base'):
            require(np.issubdtype(saved[key].dtype, np.integer)
                    and np.all((saved[key] >= 0) & (saved[key] < 8)), port + ': class indices')
        raw = visual['Xs']
        require(raw.shape == (38091, 128), port + ': VV-only geometry')
        center, scale = raw[source].mean(0), raw[source].std(0) + 1e-6
        close(saved['source_center'], center, port + ': source-only mean', atol=0, rtol=0)
        close(saved['source_scale'], scale, port + ': source-only scale', atol=0, rtol=0)
        standardized = (raw[q] - center) / scale
        x = standardized.astype(np.float64) / np.sqrt(128.)
        phi = np.column_stack((x, np.ones(n)))
        designs = (phi[:, -1:], phi)
        prior = original['base_logits'][q].astype(np.float64)
        base = prior.argmax(1)
        require(np.array_equal(saved['original_base'], base), port + ': original base')
        with (GATE / port / 'source_model.pkl').open('rb') as handle:
            gate = pickle.load(handle)['prior']
        for field in ('center', 'scale', 'lr'):
            close(frozen['source_' + field], gate[field], port + ': frozen gate ' + field, atol=0, rtol=0)
        si = list(parent['score_names']).index('anchor_visual64')
        fi = list(frozen['expert_names']).index('anchor_visual64')
        ai = list(advantage['arm_names']).index('advantage_visual')
        bi = list(frozen['budgets']).index(.05)
        for seed in range(5):
            context = port + '/' + str(seed)
            support = np.random.default_rng(seed).choice(q, 64, replace=False)
            require(len(np.unique(support)) == 64, context + ': unique support budget')
            ev = ~np.isin(q, support)
            for data in (saved, parent, frozen, advantage):
                require(np.array_equal(data['support_rows'][seed], support)
                        and np.array_equal(data['eval_masks'][seed], ev), context + ': shared support/eval')
            local = np.searchsorted(q, support)
            require(np.array_equal(q[local], support), context + ': support mapping')
            labels = original['y_ids'][support]
            observed = np.bincount(labels, minlength=8) > 0
            cap = int(np.floor(.05 * ev.sum()))
            require(saved['caps'][seed] == frozen['caps'][seed, bi] == advantage['caps'][seed] == cap,
                    context + ': fixed admission cap')
            adapted = []
            for index, (design, name) in enumerate(zip(designs, ('bias_correction', 'visual_correction'))):
                coefficient = saved[name][seed]
                if port == REFIT_PORT:
                    independent = fit_svd(design[local], labels, prior[local])
                    close(coefficient, independent, context + ': augmented-SVD coefficient', atol=1e-10)
                    errors['coefficient'] = max(errors['coefficient'], float(np.max(np.abs(coefficient - independent))))
                    refits += 1
                scores = prior + design @ coefficient
                close(saved['scores'][seed, index], scores, context + ': base score replay', atol=1e-11)
                errors['score'] = max(errors['score'], float(np.max(np.abs(saved['scores'][seed, index] - scores))))
                adapted.append(scores.argmax(1))
                replayed_scores += 1
            receiving_scores = parent['scores'][seed, si]
            expert = receiving_scores.argmax(1)
            require(np.array_equal(saved['expert_predictions'][seed], expert), context + ': unchanged anchored expert')
            fallback = np.stack((base, adapted[0], adapted[1], adapted[0], adapted[1], base, base))
            require(np.array_equal(saved['fallback_predictions'][seed], fallback), context + ': arm fallback semantics')
            masks = np.zeros((7, n), dtype=bool)
            for index in range(2):
                score = replay_score(confidence_parts(saved['scores'][seed, index]), receiving_scores, gate)
                close(saved['gate_scores'][seed, index], score, context + ': adapted-receiver gate', atol=1e-11)
                errors['gate'] = max(errors['gate'], float(np.max(np.abs(saved['gate_scores'][seed, index] - score))))
                masks[index + 3] = replay_mask(q, ev, fallback[index + 3], expert, observed, score, cap)
                replayed_gates += 1
            frozen_score = replay_score(confidence_parts(prior), receiving_scores, gate)
            masks[5] = replay_mask(q, ev, base, expert, observed, frozen_score, cap)
            masks[6] = advantage['selected'][seed, ai]
            require(np.array_equal(masks[5], frozen['selected'][seed, bi, fi]), context + ': frozen control mask')
            require(np.array_equal(saved['selected'][seed], masks), context + ': all seven masks')
            predictions = np.where(masks, expert[None], fallback)
            require(np.array_equal(saved['predictions'][seed], predictions), context + ': all seven predictions')
            require(np.array_equal(predictions[5], frozen['predictions'][seed, bi, fi])
                    and np.array_equal(predictions[6], advantage['predictions'][seed, ai]), context + ': old controls unchanged')
        # No query truth is consumed by geometry, support refits, or policy replay.
        truth = original['y_ids'][q]
        port_expected = {}
        control_metrics = read_json(ADVANTAGE / port / 'metrics.json')
        for seed in range(5):
            ev = saved['eval_masks'][seed]
            for index, arm in enumerate(ARMS):
                pred = saved['predictions'][seed, index]
                fallback = saved['fallback_predictions'][seed, index]
                selected = saved['selected'][seed, index]
                evaluated = int(ev.sum())
                admitted = int(selected.sum())
                handoff = int(np.count_nonzero(pred[ev] != fallback[ev]))
                changed = int(np.count_nonzero(pred[ev] != base[ev]))
                cap = int(saved['caps'][seed])
                require(not selected[~ev].any() and admitted == handoff <= cap, port + ': adapted-fallback admission cap')
                if index < 3:
                    require(admitted == 0, port + ': baseline alone must have no handoffs')
                over_original_cap += int(changed > cap)
                result = confusion_metrics(truth[ev], base[ev], base[ev], pred[ev])
                result.update(evaluated_rows=evaluated, admitted_rows=admitted,
                              handoff_changed_rows=handoff, changed_from_original_rows=changed,
                              cap=cap, admission_fraction=admitted / evaluated,
                              original_change_fraction=changed / evaluated)
                if index >= 5:
                    name = 'frozen_visual' if index == 5 else 'advantage_visual'
                    old = next(row for row in control_metrics if row['seed'] == seed and row['arm'] == name)
                    for field in ('ba', 'acc'):
                        close(result[field], old[field], port + ': inherited control metric ' + field)
                port_expected[port, seed, arm] = result
        compare_rows(read_json(run / port / 'metrics.json'), port_expected)
        expected.update(port_expected)
        print('VERIFIED ' + port + ': shared64, adapted fallbacks, masks, 35 metric rows', flush=True)
    require(refits == 10 and replayed_scores == 230 and replayed_gates == 230 and len(expected) == 805,
            'bounded replay coverage')
    with (run / 'metrics.csv').open(encoding='utf-8', newline='') as handle:
        compare_rows(list(csv.DictReader(handle)), expected, csv_mode=True)
    verify_summary(read_json(run / 'summary.json'), expected, ports)
    return dict(status='passed', ports=23, contexts=115, independent_fit_port=REFIT_PORT,
                independent_augmented_svd_fits=refits, source_geometry_replays=23,
                coefficient_score_replays=replayed_scores, adapted_gate_mask_replays=replayed_gates,
                original_gate_mask_replays=115, archived_advantage_mask_checks=115,
                seven_arm_policy_replays=805, policy_metric_rows=805, max_absolute_errors=errors,
                unique_shared_target_labels_per_context=64, query_labels_used_only_for_readout=True,
                admission_reference='each arm fallback, adapted for bias_handoff and visual_handoff',
                original_prediction_change_cap=False, policy_rows_exceeding_original_5percent_changes=over_original_cap,
                producer_sklearn_coefficient_checks=230,
                producer_maximum_sklearn_coefficient_error=complete['maximum_sklearn_coefficient_error'],
                producer_check_note='The initial producer receipt mislabeled all sklearn checks primal; its terminology is now corrected. sklearn cholesky uses a dual system for 129 features and 64 rows. Bias checks are primal. This verifier uses augmented SVD for both on one fixed port.',
                producer_functions_used=False, additional_experiments=0, verifier_numpy=np.__version__,
                limitations=['Target coefficient independence relies on producer sklearn checks outside the one fixed refit port.',
                             'Archived expert, source gate, advantage policy and upstream encoder/PCA provenance are inherited.',
                             'Artifact replay does not establish policy-freeze timing or independence of exposed development data.',
                             '5% limits expert handoff only, not all changes from the original base or computation.'])


def verify_coverage(run):
    directory = run / 'coverage_control'
    require((directory / 'POLICIES_SAVED.json').is_file() and (directory / 'summary.json').is_file(),
            'coverage policies and readout must be complete')
    inventory = read_json(directory / 'POLICIES_SAVED.json')
    require(inventory == read_json(run / 'POLICIES_SAVED.json'), 'coverage parent inventory')
    ports = [row['port'] for row in inventory]
    require(len(ports) == len(set(ports)) == 23, 'coverage port count')
    config = read_json(directory / 'RUN_CONFIG.json')
    require(config['post_readout_followup'] is True and config['unique_target_labels'] == 64
            and config['expert_admissions'] == 0, 'coverage follow-up configuration')
    summary = read_json(directory / 'summary.json')
    require(summary['status'] == 'complete' and summary['post_readout_followup'] is True
            and summary['n_ports'] == 23 and summary['contexts'] == 115 and summary['fits'] == 0
            and summary['expert_admissions'] == 0 and summary['original_setting_unchanged'] is False
            and summary['independent_confirmation'] is False, 'coverage readout scope')
    expected, comparison_metrics = {}, {}
    comparison_arms = ('original_base', 'base_bias', 'base_visual', 'visual_handoff')
    for port in ports:
        parent = archive(run / port / 'policies.npz')
        original = archive(ROOT / 'pred_cache' / (port + '.npz'))
        saved = archive(directory / (port + '.npz'))
        q = original['qpos']
        require(np.array_equal(saved['query_rows'], q) and np.array_equal(parent['query_rows'], q),
                port + ': coverage query alignment')
        require(saved['predictions'].shape == (5, len(q)) and saved['observed_classes'].shape == (5, 8)
                and saved['support_rows'].shape == (5, 64) and saved['eval_masks'].shape == (5, len(q)),
                port + ': coverage shape')
        require(saved['observed_classes'].dtype == saved['eval_masks'].dtype == np.bool_,
                port + ': coverage mask dtype')
        base_scores = original['base_logits'][q]
        for seed in range(5):
            support = np.random.default_rng(seed).choice(q, 64, replace=False)
            ev = ~np.isin(q, support)
            require(np.array_equal(saved['support_rows'][seed], support)
                    and np.array_equal(parent['support_rows'][seed], support)
                    and np.array_equal(saved['eval_masks'][seed], ev)
                    and np.array_equal(parent['eval_masks'][seed], ev), port + ': shared coverage support/eval')
            classes = np.unique(original['y_ids'][support])
            observed = np.isin(np.arange(8), classes)
            require(np.array_equal(saved['observed_classes'][seed], observed), port + ': observed classes')
            # Selecting only the observed columns is independent of the producer's -inf masking.
            predicted = classes[base_scores[:, classes].argmax(axis=1)]
            require(np.array_equal(saved['predictions'][seed], predicted), port + ': restricted base prediction')
        truth, base = original['y_ids'][q], base_scores.argmax(axis=1)
        for seed in range(5):
            ev = saved['eval_masks'][seed]
            predicted = saved['predictions'][seed]
            metric = confusion_metrics(truth[ev], base[ev], base[ev], predicted[ev])
            metric.update(admission_fraction=0., original_change_fraction=float(np.mean(predicted[ev] != base[ev])))
            expected[port, seed] = metric
            for arm in comparison_arms:
                index = ARMS.index(arm)
                other = parent['predictions'][seed, index]
                comparison_metrics[port, seed, arm] = confusion_metrics(truth[ev], base[ev], base[ev], other[ev])
    rows = read_json(directory / 'metrics.json')
    require(len(rows) == len(expected) == 115, 'coverage metric count')
    seen = set()
    for row in rows:
        key = row['port'], int(row['seed'])
        require(key in expected and key not in seen, 'coverage metric identity')
        seen.add(key)
        for field in MEASURES:
            close(row[field], expected[key][field], str(key) + ': coverage ' + field)
        require(len(row['class_recall']) == 8, 'coverage class recall columns')
        for c, value in enumerate(expected[key]['class_recall']):
            if value is None:
                require(row['class_recall'][c] is None, 'coverage absent class')
            else:
                close(row['class_recall'][c], value, str(key) + ': coverage recall')
    values = {field: np.array([[expected[port, seed][field] for seed in range(5)] for port in ports])
              for field in ('ba', 'acc', 'original_change_fraction')}
    for field, array in values.items():
        close(summary['mean'][field], array.mean(1).mean(), 'coverage mean ' + field)
    require(len(summary['comparisons']) == 4, 'coverage comparisons count')
    samples = np.random.default_rng(0).integers(0, 23, (10000, 23))
    seen = set()
    for row in summary['comparisons']:
        arm = row['candidate']
        require(arm in comparison_arms and arm not in seen and row['control'] == 'observed_base',
                'coverage comparison identity')
        seen.add(arm)
        other = {field: np.array([[comparison_metrics[port, seed, arm][field] for seed in range(5)]
                                  for port in ports]) for field in ('ba', 'acc')}
        delta = other['ba'] - values['ba']
        by_port = delta.mean(1)
        measures = dict(mean_delta_ba_pp=100 * by_port.mean(),
                        mean_delta_acc_pp=100 * (other['acc'] - values['acc']).mean(1).mean(),
                        positive_ports=int((by_port > 1e-12).sum()), negative_ports=int((by_port < -1e-12).sum()),
                        port_bootstrap_95ci_pp=100 * np.percentile(by_port[samples].mean(1), [2.5, 97.5]),
                        draw_deltas_pp=100 * delta.mean(0))
        for field, value in measures.items():
            close(row[field], value, arm + ': coverage comparison ' + field)
    return dict(status='passed', ports=23, contexts=115, independent_control_prediction_replays=115,
                control_metric_rows=115, parent_comparator_metric_replays=460, summary_contrasts=4,
                fits=0, additional_experiments=0, post_readout_followup=True,
                query_labels_used_only_for_readout=True, unique_shared_target_labels=64,
                original_prediction_change_cap=False, expert_admissions=0,
                observed_base_mean_ba_percent=100 * values['ba'].mean(),
                limitation='Support-observed classes do not establish true class absence; freeze timing and upstream provenance are inherited.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', '--out', dest='run', type=Path,
                        default=ROOT / 'oof_base_adapt_results_20261007')
    parser.add_argument('--coverage-only', action='store_true',
                        help='append the no-fit coverage replay to an already passed main receipt')
    args = parser.parse_args()
    require(args.run.is_dir(), 'run directory unavailable')
    start = time.perf_counter()
    receipt = {}
    previous = read_json(args.run / 'VERIFICATION.json') if (args.run / 'VERIFICATION.json').is_file() else None
    if args.coverage_only:
        require(previous is not None and previous['status'] == 'passed', 'coverage-only requires passed main verification')
        coverage = {}
        try:
            with threadpool_limits(limits=1):
                coverage = verify_coverage(args.run)
        except Exception as exc:
            coverage = dict(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            coverage.update(verifier=Path(__file__).name, seconds=time.perf_counter() - start)
            (args.run / 'coverage_control' / 'VERIFICATION.json').write_text(
                json.dumps(coverage, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            previous['coverage_control'] = coverage
            (args.run / 'VERIFICATION.json').write_text(json.dumps(previous, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            print(json.dumps(coverage, indent=2, allow_nan=False), flush=True)
        return
    try:
        with threadpool_limits(limits=1):
            receipt = verify(args.run)
            if (args.run / 'coverage_control').is_dir():
                receipt['coverage_control'] = verify_coverage(args.run)
    except Exception as exc:
        receipt = dict(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if previous is not None and previous.get('status') == 'failed':
            receipt['previous_failed_attempt'] = previous
        receipt.update(verifier=Path(__file__).name, run=str(args.run.resolve()), seconds=time.perf_counter() - start)
        (args.run / 'VERIFICATION.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
