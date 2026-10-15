"""Independent class-mass, weighted-SVD and policy replay for error focus."""

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

from verify_complementary_expert import (
    BASE, ERRORS, EXPERT_MEASURES, GATE, LOO, MEASURES, RESIDUAL, ROOT,
    archive, check_rows, check_summary, close, exact, gate_score, geometry,
    quality, read_json, require, route,
)


FIRST = ROOT / 'oof_complementary_expert_results_20261007'
ARMS = ('base_visual', 'visual_handoff', 'bias_handoff', 'loo_error_handoff', 'insample_error_handoff')
EXPERTS = ('old_expert', 'loo_error', 'insample_error')
CONTRASTS = (('loo_error_handoff', 'visual_handoff'), ('loo_error_handoff', 'insample_error_handoff'),
             ('loo_error_handoff', 'bias_handoff'), ('loo_error_handoff', 'base_visual'),
             ('insample_error_handoff', 'visual_handoff'))
EXPERT_CONTRASTS = (('loo_error', 'old_expert'), ('loo_error', 'insample_error'))


def error_weights(labels, signal):
    wrong = signal.argmax(axis=1) != labels
    classes = np.unique(labels)
    mass = len(labels) / len(classes)
    weights = np.empty(len(labels), dtype=float)
    for c in classes:
        rows = labels == c
        wrong_count = int(wrong[rows].sum())
        unit = mass / (int(rows.sum()) + wrong_count)
        weights[rows & ~wrong] = unit
        weights[rows & wrong] = 2 * unit
    return weights, wrong


def weighted_svd(design, labels, source_expert, weights):
    width = design.shape[1]
    target = -np.ones((len(labels), 8)) - source_expert
    target[np.arange(len(labels)), labels] += 2.
    root = np.sqrt(weights)[:, None]
    matrix = np.vstack((design * root, np.sqrt(8.) * np.eye(width)))
    response = np.vstack((target * root, np.zeros((width, 8))))
    coefficient, _, rank, _ = lstsq(matrix, response, lapack_driver='gelsd')
    require(rank == width, 'weighted augmented-SVD rank')
    return coefficient


def verify(run):
    ERRORS.clear()
    complete, config = read_json(run / 'COMPLETE.json'), read_json(run / 'RUN_CONFIG.json')
    require(complete['status'] == 'complete' and complete['new_expert_fits'] == 230
            and complete['new_base_or_gate_fits'] == 0 and complete['policy_metric_rows'] == 575
            and complete['expert_metric_rows'] == 345, 'completion coverage')
    require(tuple(config['arms']) == ARMS and tuple(config['experts']) == EXPERTS
            and tuple(map(tuple, config['contrasts'])) == CONTRASTS and config['strength'] == 8
            and config['error_emphasis'] == 2 and config['class_mass'] == '64/K_present per class'
            and config['target_labels'] == 64 and config['expert_admission_cap'] == .05
            and config['post_complementary_readout'] is True
            and Path(config['loo_cache']).resolve() == LOO.resolve(), 'fixed recipe')
    first_receipt, loo_receipt = (read_json(root / 'VERIFICATION.json') for root in (FIRST, LOO))
    require(first_receipt['status'] == loo_receipt['status'] == 'passed'
            and first_receipt['independent_base_loo_svd_fits'] == 8
            and Path(first_receipt['inherited_loo_verification']['path']).resolve()
            == (LOO / 'VERIFICATION.json').resolve(), 'inherited LOO audit')
    inventory = read_json(run / 'POLICIES_SAVED.json')
    require(all(inventory == read_json(root / 'POLICIES_SAVED.json') for root in (BASE, LOO, FIRST)),
            'same inherited inventories')
    ports = [item['port'] for item in inventory]
    require(len(ports) == len(set(ports)) == 23, '23 unique ports')
    policy_expected, expert_expected = {}, {}
    fit_count = gate_count = control_count = old_expert_count = class_mass_count = 0
    signal_disagreements = singleton_mass_count = 0
    for port in ports:
        saved, parent, residual, loo, first = (archive(root / port / 'policies.npz')
                                              for root in (run, BASE, RESIDUAL, LOO, FIRST))
        data, visual = (archive(ROOT / cache / (port + '.npz')) for cache in ('pred_cache', 'pred_cache2'))
        q, source = data['qpos'], data['spos']
        for other in (saved, parent, residual, loo, first):
            for field, value in (('query_rows', q), ('source_rows', source), ('seeds', np.arange(5))):
                exact(other[field], value, port + ': ' + field)
        require(tuple(saved['arm_names']) == ARMS and tuple(saved['expert_names']) == EXPERTS,
                port + ': saved axis identities')
        require(saved['scores'].shape == (5, 3, len(q), 8)
                and saved['coefficients'].shape == (5, 2, 193, 8)
                and saved['support_weights'].shape == (5, 2, 64)
                and saved['base_support_signals'].shape == (5, 2, 64, 8)
                and saved['gate_scores'].shape == (5, 3, len(q)), port + ': expert axes')
        phi, center, scale = geometry(np.column_stack((visual['Xs'], visual['Z'])), source, q)
        for other in (saved, first):
            exact(other['source_center'], center, port + ': source center')
            exact(other['source_scale'], scale, port + ': source scale')
        b0, e0 = (data[key][q].astype(float) for key in ('base_logits', 'expert_logits'))
        exact(first['source_contrast'], e0 - b0, port + ': first source priors')
        original = b0.argmax(1)
        exact(saved['original_base'], original, port + ': original base')
        with (GATE / port / 'source_model.pkl').open('rb') as stream:
            gate = pickle.load(stream)['prior']
        ei = list(residual['score_names']).index('anchor_visual64')
        ci = [list(parent['arm_names']).index(arm) for arm in ARMS[:3]]
        port_policy, port_expert = {}, {}
        for seed in range(5):
            context = f'{port}/{seed}'
            support = np.random.default_rng(seed).choice(q, 64, replace=False)
            ev = ~np.isin(q, support)
            for other in (saved, parent, residual, loo, first):
                exact(other['support_rows'][seed], support, context + ': support order')
                exact(other['eval_masks'][seed], ev, context + ': evaluation')
            local = np.searchsorted(q, support)
            exact(q[local], support, context + ': support mapping')
            labels, truth = data['y_ids'][support], data['y_ids'][q]
            cap = int(np.floor(.05 * ev.sum()))
            require(saved['caps'][seed] == parent['caps'][seed] == cap, context + ': cap')
            base_scores, bias_scores = parent['scores'][seed, 1], parent['scores'][seed, 0]
            base = base_scores.argmax(1)
            loo_scores = b0[local] + loo['loo_residuals'][seed, 0]
            exact(first['loo_base_scores'][seed], loo_scores, context + ': audited base LOO cache')
            exact(first['training_priors'][seed, 1], base_scores[local] + (e0 - b0)[local],
                  context + ': first unchanged B64 support scores')
            signals = (loo_scores, base_scores[local])
            scores = [residual['scores'][seed, ei]]
            exact(saved['scores'][seed, 0], scores[0], context + ': uniform E64')
            exact(first['scores'][seed, 0], scores[0], context + ': first uniform E64')
            old_expert_count += 1
            wrong_signals = []
            for index, (expert, signal) in enumerate(zip(EXPERTS[1:], signals)):
                exact(saved['base_support_signals'][seed, index], signal, context + ': signal ' + expert)
                weights, wrong = error_weights(labels, signal)
                wrong_signals.append(wrong)
                stored = saved['support_weights'][seed, index]
                close(stored, weights, context + ': 2-to-1 weights ' + expert, 'weights')
                classes = np.unique(labels)
                for c in classes:
                    rows = labels == c
                    close(stored[rows].sum(), 64 / len(classes), context + f': class {c} mass', 'class_mass')
                    class_mass_count += 1
                    singleton_mass_count += int(rows.sum() == 1)
                close(stored.sum(), 64., context + ': total weight', 'class_mass')
                coefficient = weighted_svd(phi[local], labels, e0[local], weights)
                close(saved['coefficients'][seed, index], coefficient, context + ': E0-anchored coefficient ' + expert,
                      'coefficient', 1e-10)
                score = e0 + phi @ coefficient
                close(saved['scores'][seed, index + 1], score, context + ': E0 deployment ' + expert, 'score', 1e-10)
                scores.append(score)
                fit_count += 1
            signal_disagreements += int(np.sum(wrong_signals[0] != wrong_signals[1]))
            experts = np.asarray(scores).argmax(2)
            exact(saved['expert_predictions'][seed], experts, context + ': expert predictions')
            masks = np.zeros((5, len(q)), dtype=bool)
            fallback = np.stack((base, base, bias_scores.argmax(1), base, base))
            receiving = experts[[0, 0, 0, 1, 2]]
            for index, score in enumerate(scores):
                gs = gate_score(base_scores, score, gate)
                close(saved['gate_scores'][seed, index], gs, context + ': gate ' + EXPERTS[index], 'gate', 1e-11)
                masks[1 if index == 0 else index + 2] = route(gs, base, experts[index], labels, q, ev, cap)
                gate_count += 1
            gs = gate_score(bias_scores, scores[0], gate)
            masks[2] = route(gs, fallback[2], experts[0], labels, q, ev, cap)
            close(parent['gate_scores'][seed, 0], gs, context + ': bias gate', 'gate', 1e-11)
            gate_count += 1
            predictions = np.where(masks, receiving, fallback)
            for field, value in (('selected', masks), ('fallback_predictions', fallback),
                                 ('receiving_predictions', receiving), ('predictions', predictions)):
                exact(saved[field][seed], value, context + ': ' + field)
                exact(saved[field][seed, :3], first[field][seed, :3], context + ': first controls ' + field)
                if field != 'receiving_predictions':
                    exact(saved[field][seed, :3], parent[field][seed, ci], context + ': parent controls ' + field)
            exact(receiving[:3], np.repeat(parent['expert_predictions'][seed][None], 3, axis=0), context + ': parent receivers')
            control_count += 3
            for index, arm in enumerate(ARMS):
                pred, mask = predictions[index], masks[index]
                admitted = int(mask.sum())
                changed = int(np.sum(pred[ev] != fallback[index, ev]))
                require(not mask[~ev].any() and admitted == changed <= cap, context + ': actual admission cap')
                original_changes = int(np.sum(pred[ev] != original[ev]))
                row = quality(truth, pred, original, ev)
                row.update(admitted_rows=admitted, handoff_changed_rows=changed,
                           changed_from_original_rows=original_changes, cap=cap,
                           admission_fraction=admitted / ev.sum(), original_change_fraction=original_changes / ev.sum())
                port_policy[port, seed, arm] = row
            for index, expert in enumerate(EXPERTS):
                port_expert[port, seed, expert] = quality(truth, experts[index], base, ev, experts[0])
        rows = read_json(run / port / 'metrics.json')
        check_rows(rows, port_policy)
        check_rows(read_json(run / port / 'expert_metrics.json'), port_expert)
        parent_metrics = {(r['seed'], r['arm']): r for r in read_json(BASE / port / 'metrics.json')}
        old_metrics = read_json(RESIDUAL / port / 'metrics.json')
        for row in rows:
            if row['arm'] in ARMS[:3]:
                require(row == parent_metrics[row['seed'], row['arm']], port + ': exact parent metric row')
        for seed in range(5):
            old = old_metrics['seeds'][seed]['metrics']['anchor_visual64']
            for field in ('ba', 'acc'):
                close(port_expert[port, seed, 'old_expert'][field], old[field], port + ': old full expert ' + field)
        policy_expected.update(port_policy)
        expert_expected.update(port_expert)
        print(f'VERIFIED {port}: 10 weighted SVD fits, 20 gate/masks, 25 policy and 15 expert metrics', flush=True)
    require((fit_count, gate_count, control_count, old_expert_count) == (230, 460, 345, 115), 'replay coverage')
    for name, expected in (('metrics.csv', policy_expected), ('expert_metrics.csv', expert_expected)):
        with (run / name).open(encoding='utf-8', newline='') as stream:
            check_rows(list(csv.DictReader(stream)), expected, csv_mode=True)
    summary = read_json(run / 'summary.json')
    require(summary['n_ports'] == 23 and summary['contexts'] == 115 and summary['policy_metric_rows'] == 575
            and summary['expert_metric_rows'] == 345 and summary['target_labels'] == 64, 'summary coverage')
    require(summary['independent_confirmation'] is False and summary['novelty_established'] is False
            and summary['post_complementary_readout'] is True, 'summary claim boundaries')
    require(summary['primary'] == 'loo_error_handoff minus visual_handoff', 'primary identity')
    check_summary(summary, policy_expected, ports, ARMS, CONTRASTS, MEASURES)
    check_summary(summary['full_query_experts'], expert_expected, ports, EXPERTS, EXPERT_CONTRASTS, EXPERT_MEASURES)
    primary = np.mean([policy_expected[p, s, 'loo_error_handoff']['ba'] - policy_expected[p, s, 'visual_handoff']['ba']
                       for p in ports for s in range(5)]) > 0
    require(summary['passes_primary_development_screen'] == bool(primary), 'primary decision')
    return dict(status='passed', ports=23, contexts=115, independently_refit_target_corrections=230,
                difficulty_signal_and_weight_replays=230, class_mass_checks=class_mass_count,
                singleton_class_mass_checks=singleton_mass_count, total_weight_checks=230,
                fixed_error_emphasis=2., class_mass='64/K_present', difficulty_disagreement_support_rows=signal_disagreements,
                training_prior='original source E0, +/-1 targets', deployment='E0 + phi192_with_intercept @ C',
                alphas_used=False, expert_score_replays=230, exact_parent_control_policies=345,
                exact_parent_control_metric_rows=345, exact_old_full_expert_scores=115,
                all_policy_mask_prediction_replays=575, gate_mask_replays=460,
                candidate_gate_mask_replays=230, control_gate_mask_replays=230,
                policy_metric_rows=575, expert_metric_rows=345, csv_metric_rows=920, class_recall_rows=920,
                policy_recover_harm_reference='original source base', expert_recover_harm_reference='adapted B64',
                both_wrong_and_old_expert_recover_harm_rows=345, policy_summary_contrasts=5, expert_summary_contrasts=2,
                independent_base_loo_svd_fits=0, exact_first_study_loo_score_contexts=115,
                inherited_loo_verification=dict(path=str(LOO / 'VERIFICATION.json'), status=loo_receipt['status']),
                inherited_first_study_audit=dict(path=str(FIRST / 'VERIFICATION.json'), status=first_receipt['status'],
                    independent_base_loo_svd_fits=first_receipt['independent_base_loo_svd_fits'],
                    singleton_loo_fits=first_receipt['singleton_loo_fits'], folds=first_receipt['loo_folds']),
                max_absolute_errors=ERRORS, absolute_tolerances=dict(weights=1e-12, class_mass=1e-12,
                    coefficient=1e-10, score=1e-10, gate=1e-11, metrics=1e-12, summary=1e-12),
                independent_helpers='verify_complementary_expert.py', producer_methods_imported=False,
                additional_experiments=0, threadpool_limit=1, verifier_numpy=np.__version__, unique_shared_target_labels=64,
                limitations=['No additional base LOO fits were run: exact cache/row matching inherits the first verifier eight-fold audit and earlier shrinkage verification.',
                             'Original B64, uniform E64, source E0, gate and encoder/PCA provenance are inherited; upstream training is not repeated.',
                             'Expert weights intentionally use all64 support labels; expert support predictions are not held-out validation.',
                             'This single fixed follow-up follows a negative development readout. Artifact replay establishes neither freeze timing nor independent confirmation.',
                             'Support/query split is row-only. The 5% cap is expert admission, not global source-base changes or measured compute; changed-score gate calibration remains unproved.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'oof_error_focused_results_20261007')
    args = parser.parse_args()
    require((args.run / 'COMPLETE.json').is_file(), 'wait for COMPLETE.json before running')
    start = time.perf_counter()
    previous = read_json(args.run / 'VERIFICATION.json') if (args.run / 'VERIFICATION.json').exists() else None
    receipt = {}
    try:
        with threadpool_limits(limits=1):
            receipt = verify(args.run)
    except Exception as exc:
        receipt = dict(status='failed', error_type=type(exc).__name__, error=str(exc),
                       traceback=traceback.format_exc(), max_absolute_errors=ERRORS)
        raise
    finally:
        if previous and previous.get('status') == 'failed':
            receipt['previous_failed_attempt'] = previous
        receipt.update(verifier=Path(__file__).name, run=str(args.run.resolve()), seconds=time.perf_counter() - start)
        (args.run / 'VERIFICATION.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({k: v for k, v in receipt.items() if k != 'inherited_first_study_audit'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
