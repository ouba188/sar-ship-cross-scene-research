"""Independent augmented-SVD and artifact replay for complementary experts."""

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


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'oof_base_adapt_results_20261007'
RESIDUAL = ROOT / 'oof_residual_expert_results_20261007_v2'
LOO = ROOT / 'oof_support_shrinkage_results_20261007'
GATE = ROOT / 'oof_context_results_20261007'
ARMS = ('base_visual', 'visual_handoff', 'bias_handoff', 'crossfit_handoff',
        'insample_handoff', 'no_source_handoff')
EXPERTS = ('old_expert', 'crossfit', 'insample', 'no_source')
CONTRASTS = (('crossfit_handoff', 'visual_handoff'), ('crossfit_handoff', 'insample_handoff'),
             ('crossfit_handoff', 'no_source_handoff'), ('crossfit_handoff', 'bias_handoff'),
             ('crossfit_handoff', 'base_visual'), ('insample_handoff', 'visual_handoff'),
             ('no_source_handoff', 'visual_handoff'))
EXPERT_CONTRASTS = (('crossfit', 'old_expert'), ('crossfit', 'insample'), ('crossfit', 'no_source'))
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'admission_fraction', 'original_change_fraction')
EXPERT_MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'rescues_both_wrong_ba',
                   'old_expert_recover_ba', 'old_expert_harm_ba', 'disagreement_fraction')
ERRORS = {}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def exact(actual, expected, label):
    require(np.array_equal(actual, expected), label + ': exact mismatch')


def close(actual, expected, label, category='metrics', tolerance=1e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
            label + ': shape/nonfinite mismatch')
    error = float(np.max(np.abs(a - b))) if a.size else 0.
    ERRORS[category] = max(ERRORS.get(category, 0.), error)
    require(error <= tolerance, f'{label}: error {error:.17g} exceeds {tolerance}')


def geometry(raw, source, query):
    center, scale = raw[source].mean(0), raw[source].std(0) + 1e-6
    # Preserve the cache's float32 standardization before scalar promotion.
    standardized = (raw[query] - center) / scale
    return np.column_stack((standardized / np.sqrt(raw.shape[1]), np.ones(len(query)))), center, scale


def svd_fit(design, labels, prior):
    n, width = design.shape
    weights = np.empty(n)
    classes = np.unique(labels)
    for c in classes:
        rows = labels == c
        weights[rows] = n / (len(classes) * rows.sum())
    response = -np.ones((n, 8)) - prior
    response[np.arange(n), labels] += 2.
    root = np.sqrt(weights)[:, None]
    matrix = np.vstack((root * design, np.sqrt(8.) * np.eye(width)))
    target = np.vstack((root * response, np.zeros((width, 8))))
    coefficient, _, rank, _ = lstsq(matrix, target, lapack_driver='gelsd')
    require(rank == width, 'augmented-SVD rank')
    return coefficient


def gate_score(base, expert, gate):
    parts = []
    for scores in (expert, base):
        top = np.sort(scores, axis=1)[:, -2:]
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        total = exponentials.sum(axis=1)
        entropy = np.log(total) - (exponentials * shifted).sum(axis=1) / total
        parts.append((top[:, 1] - top[:, 0], entropy))
    features = np.column_stack((parts[0][0], parts[1][0], parts[0][1], parts[1][1]))
    return np.column_stack(((features - gate['center']) / gate['scale'], np.ones(len(base)))) @ gate['lr']


def route(scores, base, expert, labels, query, evaluation, cap):
    eligible = np.flatnonzero(evaluation & (base != expert) & np.isin(expert, labels) & (scores > 0))
    ordered = sorted(eligible, key=lambda row: (-float(scores[row]), int(query[row])))
    selected = np.zeros(len(query), dtype=bool)
    selected[ordered[:cap]] = True
    return selected


def quality(truth, prediction, reference, evaluation, old_expert=None):
    y, p, b = truth[evaluation], prediction[evaluation], reference[evaluation]
    recalls, recover, harm = [None] * 8, [], []
    both, old_recover, old_harm = [], [], []
    old = old_expert[evaluation] if old_expert is not None else None
    for c in np.unique(y):
        rows = y == c
        hit, before = p[rows] == c, b[rows] == c
        recalls[c] = float(hit.mean())
        recover.append(float((hit & ~before).mean()))
        harm.append(float((~hit & before).mean()))
        if old is not None:
            old_hit = old[rows] == c
            both.append(float((hit & ~before & ~old_hit).mean()))
            old_recover.append(float((hit & ~old_hit).mean()))
            old_harm.append(float((~hit & old_hit).mean()))
    result = dict(ba=float(np.mean([r for r in recalls if r is not None])), acc=float((p == y).mean()),
                  recover_ba=float(np.mean(recover)), harm_ba=float(np.mean(harm)),
                  class_recall=recalls, evaluated_rows=len(y))
    if old is not None:
        result.update(rescues_both_wrong_ba=float(np.mean(both)), old_expert_recover_ba=float(np.mean(old_recover)),
                      old_expert_harm_ba=float(np.mean(old_harm)), disagreement_fraction=float((p != b).mean()))
    return result


def check_rows(rows, expected, csv_mode=False):
    require(len(rows) == len(expected), 'metric row count')
    seen = set()
    for row in rows:
        key = row['port'], int(row['seed']), row['arm']
        require(key in expected and key not in seen, 'metric identity ' + str(key))
        seen.add(key)
        for name, value in expected[key].items():
            if name == 'class_recall':
                if csv_mode:
                    continue
                require(len(row[name]) == 8, 'class recall width')
                for c, recall in enumerate(value):
                    if recall is None:
                        require(row[name][c] is None, 'absent class recall')
                    else:
                        close(row[name][c], recall, f'{key}: recall {c}')
            else:
                close(float(row[name]), value, f'{key}: {name}')


def check_summary(summary, expected, ports, arms, contrasts, measures):
    cube = np.array([[[[expected[port, seed, arm][field] for field in measures]
                       for arm in arms] for seed in range(5)] for port in ports])
    require([r['arm'] for r in summary['means']] == list(arms), 'summary mean identities')
    for ai, row in enumerate(summary['means']):
        for mi, field in enumerate(measures):
            close(row[field], cube[:, :, ai, mi].mean(1).mean(), row['arm'] + ': ' + field, 'summary')
    require([(r['candidate'], r['control']) for r in summary['comparisons']] == list(contrasts),
            'summary contrast identities')
    sampled = np.random.default_rng(0).integers(0, len(ports), (10000, len(ports)))
    for row, (candidate, control) in zip(summary['comparisons'], contrasts):
        delta = cube[:, :, arms.index(candidate)] - cube[:, :, arms.index(control)]
        by_port = delta[:, :, 0].mean(1)
        fields = dict(mean_delta_ba_pp=100 * by_port.mean(), mean_delta_acc_pp=100 * delta[:, :, 1].mean(),
                      positive_ports=int((by_port > 1e-12).sum()), negative_ports=int((by_port < -1e-12).sum()),
                      draw_deltas_pp=100 * delta[:, :, 0].mean(0),
                      port_bootstrap_95ci_pp=100 * np.percentile(by_port[sampled].mean(1), [2.5, 97.5]))
        for field, value in fields.items():
            close(row[field], value, candidate + ': ' + field, 'summary')
        require(set(row['port_deltas_pp']) == set(ports), 'summary port identities')
        close([row['port_deltas_pp'][p] for p in ports], 100 * by_port, candidate + ': port deltas', 'summary')


def verify(run):
    complete, config = read_json(run / 'COMPLETE.json'), read_json(run / 'RUN_CONFIG.json')
    require(complete['status'] == 'complete' and complete['new_expert_fits'] == 345
            and complete['reused_base_loo_fits'] == 7360 and complete['new_base_or_gate_fits'] == 0
            and complete['policy_metric_rows'] == 690 and complete['expert_metric_rows'] == 460, 'completion coverage')
    require(tuple(config['arms']) == ARMS and tuple(config['experts']) == EXPERTS
            and tuple(map(tuple, config['contrasts'])) == CONTRASTS
            and tuple(map(tuple, config['expert_contrasts'])) == EXPERT_CONTRASTS
            and config['strength'] == 8 and config['target_labels'] == 64
            and config['expert_admission_cap'] == .05 and config['alphas_used'] is False
            and Path(config['loo_cache']).resolve() == LOO.resolve(), 'fixed recipe')
    inherited = read_json(LOO / 'VERIFICATION.json')
    require(inherited['status'] == 'passed', 'LOO verification must have passed')
    inventory = read_json(run / 'POLICIES_SAVED.json')
    require(inventory == read_json(BASE / 'POLICIES_SAVED.json')
            == read_json(LOO / 'POLICIES_SAVED.json'), 'inherited inventories')
    ports = [item['port'] for item in inventory]
    require(len(ports) == len(set(ports)) == 23, '23 unique ports')
    audit_ports = sorted(ports)[:2]
    policy_expected, expert_expected, folds = {}, {}, []
    fit_count = gate_count = control_count = old_expert_count = 0
    for port in ports:
        saved, parent, residual, loo = (archive(root / port / 'policies.npz') for root in (run, BASE, RESIDUAL, LOO))
        data, visual = (archive(ROOT / cache / (port + '.npz')) for cache in ('pred_cache', 'pred_cache2'))
        q, source = data['qpos'], data['spos']
        for other in (saved, parent, residual, loo):
            for field, value in (('query_rows', q), ('source_rows', source), ('seeds', np.arange(5))):
                exact(other[field], value, port + ': ' + field)
        require(tuple(saved['arm_names']) == ARMS and tuple(saved['expert_names']) == EXPERTS, 'saved axis identities')
        require(saved['scores'].shape == (5, 4, len(q), 8)
                and saved['coefficients'].shape == (5, 3, 193, 8)
                and saved['training_priors'].shape == (5, 3, 64, 8)
                and saved['gate_scores'].shape == (5, 4, len(q)), port + ': expert axes')
        phi, center, scale = geometry(np.column_stack((visual['Xs'], visual['Z'])), source, q)
        exact(saved['source_center'], center, port + ': source center')
        exact(saved['source_scale'], scale, port + ': source scale')
        b0, e0 = (data[key][q].astype(float) for key in ('base_logits', 'expert_logits'))
        difference = e0 - b0
        exact(saved['source_contrast'], difference, port + ': E0-B0')
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
            for other in (saved, parent, residual, loo):
                exact(other['support_rows'][seed], support, context + ': support order')
                exact(other['eval_masks'][seed], ev, context + ': evaluation')
            local = np.searchsorted(q, support)
            exact(q[local], support, context + ': support mapping')
            labels, truth = data['y_ids'][support], data['y_ids'][q]
            cap = int(np.floor(.05 * ev.sum()))
            require(saved['caps'][seed] == parent['caps'][seed] == cap, context + ': cap')
            base_scores, bias_scores = parent['scores'][seed, 1], parent['scores'][seed, 0]
            base = base_scores.argmax(1)
            training_base = b0[local] + loo['loo_residuals'][seed, 0]
            exact(saved['loo_base_scores'][seed], training_base, context + ': unshrunk base LOO')
            if port in audit_ports and seed in (0, 1):
                base_phi = geometry(visual['Xs'], source, q)[0]
                for heldout in (0, 63):
                    train = np.delete(np.arange(64), heldout)
                    coefficient = svd_fit(base_phi[local[train]], labels[train], b0[local[train]])
                    close(loo['loo_residuals'][seed, 0, heldout], base_phi[local[heldout]] @ coefficient,
                          context + f': base LOO {heldout}', 'loo', 1e-10)
                    folds.append(dict(port=port, seed=seed, heldout=heldout, train_rows=63,
                                      training_classes=int(len(np.unique(labels[train]))),
                                      singleton=bool(np.sum(labels == labels[heldout]) == 1)))
            scores = [residual['scores'][seed, ei]]
            exact(saved['scores'][seed, 0], scores[0], context + ': old full expert')
            old_expert_count += 1
            for index, expert in enumerate(EXPERTS[1:]):
                contrast = np.zeros_like(difference) if expert == 'no_source' else difference
                train_base = base_scores[local] if expert == 'insample' else training_base
                prior = train_base + contrast[local]
                exact(saved['training_priors'][seed, index], prior, context + ': training prior ' + expert)
                coefficient = svd_fit(phi[local], labels, prior)
                close(saved['coefficients'][seed, index], coefficient, context + ': coefficient ' + expert,
                      'coefficient', 1e-10)
                score = base_scores + contrast + phi @ coefficient
                close(saved['scores'][seed, index + 1], score, context + ': expert score ' + expert, 'score', 1e-10)
                scores.append(score)
                fit_count += 1
            experts = np.asarray(scores).argmax(2)
            exact(saved['expert_predictions'][seed], experts, context + ': expert predictions')
            masks = np.zeros((6, len(q)), dtype=bool)
            fallback = np.stack((base, base, bias_scores.argmax(1), base, base, base))
            receiving = experts[[0, 0, 0, 1, 2, 3]]
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
                if field != 'receiving_predictions':
                    exact(saved[field][seed, :3], parent[field][seed, ci], context + ': preserved ' + field)
            exact(receiving[:3], np.repeat(parent['expert_predictions'][seed][None], 3, axis=0), context + ': old receivers')
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
        policy_rows = read_json(run / port / 'metrics.json')
        full_rows = read_json(run / port / 'expert_metrics.json')
        check_rows(policy_rows, port_policy)
        check_rows(full_rows, port_expert)
        parent_metrics = {(r['seed'], r['arm']): r for r in read_json(BASE / port / 'metrics.json')}
        old_metrics = read_json(RESIDUAL / port / 'metrics.json')
        for row in policy_rows:
            if row['arm'] in ARMS[:3]:
                require(row == parent_metrics[row['seed'], row['arm']], port + ': exact parent metrics')
        for seed in range(5):
            old = old_metrics['seeds'][seed]['metrics']['anchor_visual64']
            for field in ('ba', 'acc'):
                close(port_expert[port, seed, 'old_expert'][field], old[field], port + ': old full expert ' + field)
        policy_expected.update(port_policy)
        expert_expected.update(port_expert)
        print(f'VERIFIED {port}: 15 SVD fits, 25 gate/masks, 30 policy and 20 expert metrics', flush=True)
    require((fit_count, gate_count, control_count, old_expert_count, len(folds)) == (345, 575, 345, 115, 8),
            'required verification coverage')
    for name, expected in (('metrics.csv', policy_expected), ('expert_metrics.csv', expert_expected)):
        with (run / name).open(encoding='utf-8', newline='') as stream:
            check_rows(list(csv.DictReader(stream)), expected, csv_mode=True)
    summary = read_json(run / 'summary.json')
    require(summary['n_ports'] == 23 and summary['contexts'] == 115 and summary['policy_metric_rows'] == 690
            and summary['expert_metric_rows'] == 460 and summary['target_labels'] == 64, 'summary coverage')
    require(summary['independent_confirmation'] is False and summary['novelty_established'] is False, 'claim boundaries')
    require(summary['primary'] == 'crossfit_handoff minus visual_handoff', 'primary identity')
    check_summary(summary, policy_expected, ports, ARMS, CONTRASTS, MEASURES)
    check_summary(summary['full_query_experts'], expert_expected, ports, EXPERTS, EXPERT_CONTRASTS, EXPERT_MEASURES)
    primary = np.mean([policy_expected[p, s, 'crossfit_handoff']['ba'] - policy_expected[p, s, 'visual_handoff']['ba']
                       for p in ports for s in range(5)]) > 0
    require(summary['passes_primary_development_screen'] == bool(primary), 'primary decision')
    return dict(status='passed', ports=23, contexts=115, independently_refit_target_corrections=fit_count,
                training_prior_replays=345, expert_score_replays=345, alphas_used=False,
                no_source_train_and_deploy_contrast_zero=True, exact_parent_control_policies=control_count,
                exact_parent_control_metric_rows=345, exact_old_full_expert_scores=old_expert_count,
                all_policy_mask_prediction_replays=690, gate_mask_replays=gate_count,
                candidate_gate_mask_replays=345, control_gate_mask_replays=230,
                policy_metric_rows=690, expert_metric_rows=460, csv_metric_rows=1150, class_recall_rows=1150,
                policy_recover_harm_reference='original source base', expert_recover_harm_reference='adapted B64',
                both_wrong_and_old_expert_recover_harm_rows=460, policy_summary_contrasts=7, expert_summary_contrasts=3,
                independent_base_loo_svd_fits=len(folds), loo_folds=folds,
                singleton_loo_fits=sum(f['singleton'] for f in folds),
                loo_training_weights='recomputed within each 63-row training fold',
                inherited_loo_verification=dict(path=str(LOO / 'VERIFICATION.json'), status=inherited['status'],
                    independent_loo_svd_fits=inherited.get('independent_loo_svd_fits'),
                    singleton_loo_fits=inherited.get('singleton_loo_fits')),
                max_absolute_errors=ERRORS, absolute_tolerances=dict(coefficient=1e-10, score=1e-10,
                    loo=1e-10, gate=1e-11, metrics=1e-12, summary=1e-12),
                producer_methods_imported=False, additional_experiments=0, threadpool_limit=1,
                verifier_numpy=np.__version__, unique_shared_target_labels=64,
                limitations=['Only the listed eight base LOO folds were independently refit here; remaining LOO values inherit the prior verification, not a new 14720-fit replay.',
                             'Frozen B64, old E64, source priors, gate and encoder/PCA provenance are inherited; this verifier checks reuse, not upstream retraining.',
                             'C uses all64 labels; its support predictions are not held-out expert validation. The 63/64 base deployment mismatch remains.',
                             'Artifact replay does not establish freeze timing or independent confirmation; these targets have development exposure.',
                             'Support/query separation is row-only. The 5% cap limits expert admission, not source-base prediction changes or measured compute.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'oof_complementary_expert_results_20261007')
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
        print(json.dumps({k: v for k, v in receipt.items() if k != 'loo_folds'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
