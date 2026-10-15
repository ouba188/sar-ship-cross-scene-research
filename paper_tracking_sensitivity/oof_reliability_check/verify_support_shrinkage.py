"""Independent artifact replay and bounded augmented-SVD LOO audit."""

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
GATE = ROOT / 'oof_context_results_20261007'
ARMS = ('base_visual', 'visual_handoff', 'bias_handoff', 'shrink_base',
        'shrink_base_handoff', 'shrink_expert_handoff', 'shrink_both_handoff')
CONTRASTS = (('shrink_base_handoff', 'visual_handoff'),
             ('shrink_expert_handoff', 'visual_handoff'),
             ('shrink_both_handoff', 'visual_handoff'),
             ('shrink_base', 'base_visual'),
             ('shrink_base_handoff', 'shrink_base'),
             ('shrink_both_handoff', 'bias_handoff'))
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'admission_fraction', 'original_change_fraction')
COUNTS = ('evaluated_rows', 'admitted_rows', 'handoff_changed_rows', 'changed_from_original_rows', 'cap')
ERRORS = {}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def close(actual, expected, label, tolerance=1e-12, category=None):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, label + ': shape mismatch')
    require(np.isfinite(a).all() and np.isfinite(b).all(), label + ': nonfinite value')
    error = float(np.max(np.abs(a - b))) if a.size else 0.
    if category:
        ERRORS[category] = max(ERRORS.get(category, 0.), error)
    require(error <= tolerance, f'{label}: maximum absolute error {error:.17g} > {tolerance}')


def exact(actual, expected, label):
    require(np.array_equal(actual, expected), label + ': exact mismatch')


def geometry(raw, source, query):
    center = raw[source].mean(axis=0)
    scale = raw[source].std(axis=0) + 1e-6
    # Preserve float32 standardization, then NumPy scalar division promotion.
    standardized = (raw[query] - center) / scale
    return np.column_stack((standardized / np.sqrt(raw.shape[1]), np.ones(len(query))))


def svd_fit(design, labels, prior):
    n, dim = design.shape
    counts = np.bincount(labels, minlength=8)
    weights = n / (np.count_nonzero(counts) * counts[labels])
    target = -np.ones((n, 8)) - prior
    target[np.arange(n), labels] += 2.
    root = np.sqrt(weights)[:, None]
    augmented = np.concatenate((root * design, np.sqrt(8.) * np.eye(dim)), axis=0)
    response = np.concatenate((root * target, np.zeros((dim, 8))), axis=0)
    coefficient, _, rank, _ = lstsq(augmented, response, lapack_driver='gelsd')
    require(rank == dim, 'augmented SVD rank')
    return coefficient


def gate_score(base, expert, gate):
    parts = []
    for scores in (expert, base):
        top = np.partition(scores, -2, axis=1)[:, -2:]
        margin = top.max(axis=1) - top.min(axis=1)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponential = np.exp(shifted)
        total = exponential.sum(axis=1)
        entropy = np.log(total) - (exponential * shifted).sum(axis=1) / total
        parts.append((margin, entropy))
    x = np.column_stack((parts[0][0], parts[1][0], parts[0][1], parts[1][1]))
    return np.column_stack(((x - gate['center']) / gate['scale'], np.ones(len(x)))) @ gate['lr']


def mask_for(score, base, expert, labels, query, evaluation, cap):
    candidates = np.flatnonzero(evaluation & (base != expert)
                               & np.isin(expert, np.unique(labels)) & (score > 0))
    order = sorted(candidates, key=lambda i: (-float(score[i]), int(query[i])))
    mask = np.zeros(len(query), dtype=bool)
    mask[order[:cap]] = True
    return mask


def metrics(truth, original, prediction, fallback, mask, evaluation, cap):
    y, p, b = truth[evaluation], prediction[evaluation], original[evaluation]
    recalls, recoveries, harms = [None] * 8, [], []
    for c in np.unique(y):
        rows = y == c
        correct, old_correct = p[rows] == c, b[rows] == c
        recalls[c] = float(correct.mean())
        recoveries.append(float((correct & ~old_correct).mean()))
        harms.append(float((~correct & old_correct).mean()))
    admitted = int(mask.sum())
    changed = int(np.count_nonzero(prediction[evaluation] != fallback[evaluation]))
    original_changes = int(np.count_nonzero(p != b))
    require(not mask[~evaluation].any() and admitted == changed <= cap, 'actual handoff/support cap')
    n = len(y)
    return dict(ba=float(np.mean([r for r in recalls if r is not None])), acc=float((p == y).mean()),
                recover_ba=float(np.mean(recoveries)), harm_ba=float(np.mean(harms)), class_recall=recalls,
                evaluated_rows=n, admitted_rows=admitted, handoff_changed_rows=changed,
                changed_from_original_rows=original_changes, cap=cap,
                admission_fraction=admitted / n, original_change_fraction=original_changes / n)


def check_rows(rows, expected, csv_mode=False):
    require(len(rows) == len(expected), 'metric row count')
    seen = set()
    for row in rows:
        key = row['port'], int(row['seed']), row['arm']
        require(key in expected and key not in seen, 'metric row identity ' + str(key))
        seen.add(key)
        for name in MEASURES + COUNTS:
            close(float(row[name]), expected[key][name], f'{key}: {name}', category='metrics')
        if not csv_mode:
            require(len(row['class_recall']) == 8, 'class recall width')
            for c, value in enumerate(expected[key]['class_recall']):
                if value is None:
                    require(row['class_recall'][c] is None, 'absent class recall')
                else:
                    close(row['class_recall'][c], value, f'{key}: class {c}', category='metrics')


def check_summary(summary, expected, ports, alphas):
    require(summary['n_ports'] == 23 and summary['contexts'] == 115
            and summary['policy_metric_rows'] == 805 and summary['target_labels'] == 64, 'summary coverage')
    require(summary['independent_confirmation'] is False and summary['novelty_established'] is False,
            'summary claim boundaries')
    cube = np.array([[[[expected[port, seed, arm][m] for m in MEASURES]
                       for arm in ARMS] for seed in range(5)] for port in ports])
    require([row['arm'] for row in summary['means']] == list(ARMS), 'mean arm order')
    for ai, row in enumerate(summary['means']):
        for mi, field in enumerate(MEASURES):
            close(row[field], cube[:, :, ai, mi].mean(axis=1).mean(), 'summary mean ' + field,
                  category='summary')
    draws = np.random.default_rng(0).integers(0, len(ports), (10000, len(ports)))
    require([(r['candidate'], r['control']) for r in summary['comparisons']] == list(CONTRASTS),
            'summary contrast identities')
    for row, (candidate, control) in zip(summary['comparisons'], CONTRASTS):
        delta = cube[:, :, ARMS.index(candidate), :] - cube[:, :, ARMS.index(control), :]
        by_port = delta[:, :, 0].mean(axis=1)
        checks = dict(mean_delta_ba_pp=100 * by_port.mean(),
                      mean_delta_acc_pp=100 * delta[:, :, 1].mean(axis=1).mean(),
                      positive_ports=int((by_port > 1e-12).sum()), negative_ports=int((by_port < -1e-12).sum()),
                      draw_deltas_pp=100 * delta[:, :, 0].mean(axis=0),
                      port_bootstrap_95ci_pp=100 * np.percentile(np.take(by_port, draws).mean(axis=1), [2.5, 97.5]))
        for field, value in checks.items():
            close(row[field], value, f'{candidate}: {field}', category='summary')
        require(set(row['port_deltas_pp']) == set(ports), 'contrast port identities')
        close([row['port_deltas_pp'][p] for p in ports], 100 * by_port, 'port contrasts', category='summary')
    alpha = np.stack([alphas[p] for p in ports])
    for name, value in (('alpha_mean', alpha.mean(axis=(0, 1))), ('alpha_min', alpha.min(axis=(0, 1))),
                        ('alpha_max', alpha.max(axis=(0, 1)))):
        close(summary[name], value, name, category='summary')
    require(set(summary['alpha_by_port']) == set(ports), 'alpha summary port identities')
    for port in ports:
        close(summary['alpha_by_port'][port], alphas[port], port + ': alpha summary', category='summary')
    primary = (cube[:, :, 4, 0] - cube[:, :, 1, 0]).mean() > 0
    require(summary['passes_primary_development_screen'] == bool(primary), 'primary decision')
    require(summary['primary'] == 'shrink_base_handoff minus visual_handoff', 'primary contrast')


def verify(run):
    complete = read_json(run / 'COMPLETE.json')
    require(complete['status'] == 'complete' and complete['loo_fits'] == 14720
            and complete['all64_refits'] == 230 and complete['policy_metric_rows'] == 805, 'complete run required')
    config = read_json(run / 'RUN_CONFIG.json')
    require(tuple(config['arms']) == ARMS and tuple(map(tuple, config['contrasts'])) == CONTRASTS
            and config['strength'] == 8 and config['target_labels'] == 64 and config['loo_train_rows'] == 63
            and config['expert_admission_cap'] == .05
            and config['fold_class_weights'] == 'recomputed from fitting63', 'fixed recipe')
    inventory = read_json(run / 'POLICIES_SAVED.json')
    require(inventory == read_json(BASE / 'POLICIES_SAVED.json'), 'parent inventory')
    ports = [item['port'] for item in inventory]
    require(len(ports) == len(set(ports)) == 23, '23 unique ports')
    audit_ports = sorted(ports)[:2]
    expected, alphas, folds = {}, {}, []
    fit_count = gate_count = control_count = 0
    for port in ports:
        saved = archive(run / port / 'policies.npz')
        parent = archive(BASE / port / 'policies.npz')
        residual = archive(RESIDUAL / port / 'policies.npz')
        data = archive(ROOT / 'pred_cache' / (port + '.npz'))
        visual = archive(ROOT / 'pred_cache2' / (port + '.npz'))
        q, source = data['qpos'], data['spos']
        require(not np.intersect1d(q, source).size, port + ': source/query overlap')
        exact(saved['source_rows'], source, port + ': source rows')
        exact(saved['seeds'], np.arange(5), port + ': seeds')
        for other in (saved, parent, residual):
            exact(other['query_rows'], q, port + ': query rows')
        require(tuple(saved['arm_names']) == ARMS, port + ': arm order')
        for field, shape in (('scores', (5, 2, len(q), 8)), ('loo_residuals', (5, 2, 64, 8)),
                             ('alpha', (5, 2)), ('alpha_numerator', (5, 2)), ('alpha_denominator', (5, 2)),
                             ('base_correction', (5, 129, 8)), ('expert_correction', (5, 193, 8)),
                             ('gate_scores', (5, 3, len(q))), ('selected', (5, 7, len(q)))):
            require(saved[field].shape == shape and np.isfinite(saved[field]).all(), port + ': ' + field)
        require(saved['selected'].dtype == saved['eval_masks'].dtype == np.bool_, 'boolean masks')
        designs = [geometry(raw, source, q) for raw in (visual['Xs'], np.column_stack((visual['Xs'], visual['Z'])))]
        priors = [data[key][q].astype(float) for key in ('base_logits', 'expert_logits')]
        original = priors[0].argmax(axis=1)
        exact(saved['original_base'], original, port + ': original predictions')
        with (GATE / port / 'source_model.pkl').open('rb') as stream:
            gate = pickle.load(stream)['prior']
        expert_index = list(residual['score_names']).index('anchor_visual64')
        control_indices = [list(parent['arm_names']).index(arm) for arm in ARMS[:3]]
        alpha_port = np.empty((5, 2))
        port_expected = {}
        for seed in range(5):
            context = f'{port}/{seed}'
            support = np.random.default_rng(seed).choice(q, 64, replace=False)
            evaluation = ~np.isin(q, support)
            for other in (saved, parent, residual):
                exact(other['support_rows'][seed], support, context + ': support')
                exact(other['eval_masks'][seed], evaluation, context + ': evaluation')
            local = np.searchsorted(q, support)
            exact(q[local], support, context + ': local support mapping')
            labels = data['y_ids'][support]
            cap = int(np.floor(.05 * evaluation.sum()))
            require(saved['caps'][seed] == parent['caps'][seed] == cap, context + ': cap')
            old_scores = (parent['scores'][seed, 1], residual['scores'][seed, expert_index])
            new_scores = []
            for branch, field in enumerate(('base_correction', 'expert_correction')):
                label = f'{context}/branch{branch}'
                x, prior = designs[branch], priors[branch]
                coefficient = svd_fit(x[local], labels, prior[local])
                close(saved[field][seed], coefficient, label + ': all64 coefficient', 1e-10, 'coefficient')
                close(prior + x @ coefficient, old_scores[branch], label + ': all64 parent score', 1e-10, 'score')
                loo = saved['loo_residuals'][seed, branch]
                # Accumulate class means rather than copying the producer's weight array formula.
                numerator = denominator = 0.
                classes = np.unique(labels)
                for c in classes:
                    rows = labels == c
                    desired = -np.ones((rows.sum(), 8)) - prior[local[rows]]
                    desired[:, c] += 2.
                    numerator += float(np.sum(desired * loo[rows], axis=1).mean())
                    denominator += float(np.sum(loo[rows] * loo[rows], axis=1).mean())
                numerator *= 64 / len(classes)
                denominator *= 64 / len(classes)
                alpha = max(0., min(1., numerator / denominator)) if denominator > 0 else 0.
                close(saved['alpha_numerator'][seed, branch], numerator, label + ': numerator', 1e-10, 'alpha_terms')
                close(saved['alpha_denominator'][seed, branch], denominator, label + ': denominator', 1e-10, 'alpha_terms')
                close(saved['alpha'][seed, branch], alpha, label + ': alpha', category='alpha')
                alpha_port[seed, branch] = alpha
                score = prior + alpha * (x @ coefficient)
                close(saved['scores'][seed, branch], score, label + ': shrunk score', 1e-10, 'score')
                new_scores.append(score)
                fit_count += 1
                if port in audit_ports:
                    counts = np.bincount(labels, minlength=8)
                    singletons = np.flatnonzero(counts[labels] == 1)
                    chosen = sorted(set([0, 63] + ([int(singletons[0])] if len(singletons) else [])))
                    for heldout in chosen:
                        train = np.delete(np.arange(64), heldout)
                        require(len(train) == 63 and heldout not in train, 'LOO training exclusion')
                        fold_coef = svd_fit(x[local[train]], labels[train], prior[local[train]])
                        close(loo[heldout], x[local[heldout]] @ fold_coef, label + f': LOO {heldout}',
                              1e-10, 'loo')
                        folds.append(dict(port=port, seed=seed, branch=branch, heldout=heldout,
                                          singleton=bool(counts[labels[heldout]] == 1), train_rows=63,
                                          training_classes=int(len(np.unique(labels[train])))))
            pairs = {1: (old_scores[0], old_scores[1]), 2: (parent['scores'][seed, 0], old_scores[1]),
                     4: (new_scores[0], old_scores[1]), 5: (old_scores[0], new_scores[1]),
                     6: (new_scores[0], new_scores[1])}
            fallback = np.empty((7, len(q)), dtype=int)
            receiving = np.repeat(old_scores[1].argmax(axis=1)[None], 7, axis=0)
            masks = np.zeros((7, len(q)), dtype=bool)
            fallback[0], fallback[3] = old_scores[0].argmax(axis=1), new_scores[0].argmax(axis=1)
            for ai, (base_score, expert_score) in pairs.items():
                fallback[ai], receiving[ai] = base_score.argmax(axis=1), expert_score.argmax(axis=1)
                score = gate_score(base_score, expert_score, gate)
                masks[ai] = mask_for(score, fallback[ai], receiving[ai], labels, q, evaluation, cap)
                reference = saved['gate_scores'][seed, ai - 4] if ai >= 4 else parent['gate_scores'][seed, 1 if ai == 1 else 0]
                close(reference, score, context + ': gate ' + ARMS[ai], 1e-11, 'gate')
                gate_count += 1
            predictions = fallback.copy()
            predictions[masks] = receiving[masks]
            for field, value in (('selected', masks), ('fallback_predictions', fallback),
                                 ('expert_predictions', receiving), ('predictions', predictions)):
                exact(saved[field][seed], value, context + ': ' + field)
                if field == 'expert_predictions':
                    exact(saved[field][seed, :3], np.repeat(parent[field][seed][None], 3, axis=0), context + ': parent receivers')
                else:
                    exact(saved[field][seed, :3], parent[field][seed, control_indices], context + ': parent ' + field)
            control_count += 3
            truth = data['y_ids'][q]
            for ai, arm in enumerate(ARMS):
                port_expected[port, seed, arm] = metrics(truth, original, predictions[ai], fallback[ai],
                                                        masks[ai], evaluation, cap)
        rows = read_json(run / port / 'metrics.json')
        check_rows(rows, port_expected)
        controls = {(r['seed'], r['arm']): r for r in read_json(BASE / port / 'metrics.json')}
        for row in rows:
            if row['arm'] in ARMS[:3]:
                require(row == controls[row['seed'], row['arm']], port + ': exact parent metric row')
        expected.update(port_expected)
        alphas[port] = alpha_port
        print(f'VERIFIED {port}: 10 fits/alphas, 25 gates, 35 policies/metrics', flush=True)
    require(fit_count == 230 and gate_count == 575 and control_count == 345 and len(expected) == 805,
            'required replay coverage')
    with (run / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
        check_rows(list(csv.DictReader(stream)), expected, csv_mode=True)
    check_summary(read_json(run / 'summary.json'), expected, ports, alphas)
    return dict(status='passed', ports=23, contexts=115, policy_metric_rows=805, csv_metric_rows=805,
                class_recall_rows=805, independently_refit_all64_coefficients=fit_count,
                shrink_score_replays=fit_count, alpha_formula_replays=fit_count,
                gate_mask_replays=gate_count, candidate_gate_mask_replays=345, control_gate_mask_replays=230,
                exact_parent_control_policies=control_count, exact_parent_control_metric_rows=control_count,
                all_policy_mask_prediction_replays=805, summary_contrasts=6,
                loo_audit_ports=audit_ports, independent_loo_svd_fits=len(folds), loo_folds=folds,
                singleton_loo_fits=sum(row['singleton'] for row in folds),
                loo_training_weights='recomputed from each 63-row fold; all64 validation weights select alpha',
                max_absolute_errors=ERRORS, absolute_tolerances=dict(coefficient=1e-10, score=1e-10,
                    loo=1e-10, alpha_terms=1e-10, alpha=1e-12, gate=1e-11, metrics=1e-12, summary=1e-12),
                producer_methods_imported=False, additional_experiments=0, verifier_numpy=np.__version__,
                unique_shared_target_labels=64, threadpool_limit=1,
                limitations=['LOO fitting independently audited only for listed folds, not all14720 fits.',
                             'Source priors, frozen gate and upstream encoder/PCA provenance are inherited.',
                             'Artifact replay cannot establish freeze timing or independent confirmation.',
                             'Support/query separation is row-only; 5% limits expert handoffs, not global changes or compute.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', '--out', dest='run', type=Path, default=ROOT / 'oof_support_shrinkage_results_20261007')
    args = parser.parse_args()
    require((args.run / 'COMPLETE.json').is_file(), 'wait for COMPLETE.json before running verifier')
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
        print(json.dumps({key: value for key, value in receipt.items() if key != 'loo_folds'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
