"""Independent, bounded replay of source-metric residual expert artifacts."""

import argparse
import csv
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from verify_residual_expert import archive, close, confusion_metrics, read_json, require
from verify_receiver_consistent_handoff import confidence_parts, replay_mask, replay_score


ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT / 'oof_residual_expert_results_20261007_v2'
FROZEN = ROOT / 'oof_receiver_consistent_results_20261007'
GATE = ROOT / 'oof_context_results_20261007'
ADVANTAGE = ROOT / 'oof_action_advantage_results_20261007'
NEW_ARMS = ('metric_linear', 'isotropic_rbf', 'metric_rbf')
EXPERT_ARMS = NEW_ARMS + ('linear', 'bias')
BUDGET_ARMS = EXPERT_ARMS + ('advantage_reference',)
REFIT_PORTS = ('Antwerp-Bruges', 'Mombasa')
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'both_wrong_rescue_ba', 'linear_harm_ba')
COUNTS = ('evaluated_rows', 'admitted_rows', 'changed_rows', 'cap', 'admission_fraction')


def geometry_replay(raw, source, source_labels, saved, parent, port):
    """Two covariance refits; source-only bandwidth reconstruction for every port."""
    center, scale = parent['source_center'], parent['source_scale']
    close(saved['source_center'], center, port + ': inherited source center', atol=0, rtol=0)
    close(saved['source_scale'], scale, port + ': inherited source scale', atol=0, rtol=0)
    require(raw.shape[1] == 192 and np.all(scale > 0), port + ': raw geometry')
    pairs = np.random.default_rng(1707).choice(
        source, min(1024, 2 * (len(source) // 2)), replace=False).reshape(-1, 2)
    require(np.array_equal(saved['bandwidth_pairs'], pairs), port + ': source pair sampling')
    require(len(np.unique(pairs)) == pairs.size and np.isin(pairs, source).all(),
            port + ': pair source membership/disjointness')
    transform = saved['transform']
    require(transform.shape == (192, 192) and np.isfinite(transform).all(), port + ': transform')
    # Preserve float32 standardization, then explicit float64 scalar promotion.
    # NumPy 1.x and 2.x differ for array / np.float64(sqrt(d)).
    pair_x = ((raw[pairs] - center) / scale).astype(np.float64) / np.sqrt(192.)
    pair_metric = pair_x @ transform
    bandwidths = np.array([np.median(np.einsum('ij,ij->i', a[:, 0] - a[:, 1], a[:, 0] - a[:, 1]))
                           for a in (pair_x, pair_metric)])
    require(np.all(bandwidths > 0), port + ': positive bandwidths')
    close(saved['bandwidths'], bandwidths, port + ': source-only bandwidths', atol=1e-10)
    if port not in REFIT_PORTS:
        return False
    close(center, raw[source].mean(0), port + ': source center refit')
    close(scale, raw[source].std(0) + 1e-6, port + ': source scale refit')
    x = ((raw[source] - center) / scale).astype(np.float64) / np.sqrt(192.)
    means = np.stack([x[source_labels == c].mean(0) for c in range(8)])
    close(saved['source_class_means'], means, port + ': class means')
    within = x - means[source_labels]
    n, d = within.shape
    empirical = within.T @ within / n
    mu = np.trace(empirical) / d
    delta = (np.square(empirical).sum() - d * mu * mu) / d
    row_energy = np.einsum('ij,ij->i', within, within)
    beta = (np.square(row_energy).mean() - np.square(empirical).sum()) / (d * n)
    shrinkage = min(beta, delta) / delta if delta > 0 else 0.
    covariance = (1 - shrinkage) * empirical + shrinkage * mu * np.eye(d)
    close(saved['shrinkage'], shrinkage, port + ': independent Ledoit-Wolf shrinkage')
    close(saved['covariance'], covariance, port + ': independent covariance')
    values, vectors = eigh(covariance)
    require(np.all(values > 0), port + ': positive covariance')
    inverse_root = (vectors / np.sqrt(values)) @ vectors.T
    energy = np.sqrt(np.square(x @ inverse_root).sum() / np.square(x).sum())
    close(saved['energy_scale'], energy, port + ': source energy scale')
    close(transform, inverse_root / energy, port + ': independent transform', atol=1e-9)
    close(np.square(x @ transform).sum(), np.square(x).sum(), port + ': source energy preserved')
    return True


def independent_kernel(x, support_x, bandwidth):
    if bandwidth is None:
        return np.column_stack((x, np.ones(len(x)))) @ np.column_stack((support_x, np.ones(len(support_x)))).T
    return 1 + np.exp(-cdist(x, support_x, metric='sqeuclidean') / bandwidth)


def support_replay(x, support_positions, support_labels, prior, bandwidth):
    counts = np.bincount(support_labels, minlength=8)
    weights = 64. / (np.count_nonzero(counts) * counts[support_labels])
    response = -np.ones((64, 8)) - prior[support_positions]
    response[np.arange(64), support_labels] += 2.
    support_x = x[support_positions]
    gram = independent_kernel(support_x, support_x, bandwidth)
    # This unscaled weighted-dual system differs from sklearn's sqrt-weight solve.
    dual = solve(gram + np.diag(8. / weights), response, assume_a='pos')
    scores = prior + independent_kernel(x, support_x, bandwidth) @ dual
    return scores, dual


def metric_replay(truth, base, linear, prediction):
    baseline = confusion_metrics(truth, base, base, prediction)
    old = confusion_metrics(truth, base, linear, prediction)
    return {**{k: baseline[k] for k in ('ba', 'acc', 'recover_ba', 'harm_ba', 'class_recall')},
            'both_wrong_rescue_ba': old['rescue_both_wrong_ba'], 'linear_harm_ba': old['harm_ba']}


def metric_key(row):
    return row['port'], int(row['seed']), row['mode'], row['arm']


def compare_metric_rows(rows, expected, csv_mode=False):
    require(len(rows) == len(expected), 'metric table length')
    seen = set()
    for row in rows:
        key = metric_key(row)
        require(key in expected and key not in seen, 'unknown or duplicate metric key: ' + str(key))
        seen.add(key)
        for field in MEASURES + COUNTS:
            value = expected[key][field]
            if value is None:
                require(row[field] == ('' if csv_mode else None), str(key) + ': full-query admission is null')
            else:
                close(float(row[field]), value, str(key) + '/' + field)
        if not csv_mode:
            require(len(row['class_recall']) == 8, str(key) + ': recall columns')
            for c, value in enumerate(expected[key]['class_recall']):
                if value is None:
                    require(row['class_recall'][c] is None, str(key) + ': absent class recall')
                else:
                    close(row['class_recall'][c], value, str(key) + ': recall ' + str(c))


def verify(run):
    require((run / 'POLICIES_SAVED.json').is_file() and (run / 'COMPLETE.json').is_file(),
            'wait for all policies and final readout to complete')
    inventory = read_json(run / 'POLICIES_SAVED.json')
    require(inventory == read_json(PARENT / 'POLICIES_SAVED.json'), 'parent inventory changed')
    ports = [item['port'] for item in inventory]
    require(len(ports) == len(set(ports)) == 23 and set(REFIT_PORTS).issubset(ports), 'fixed port inventory')
    config = read_json(run / 'RUN_CONFIG.json')
    require(tuple(config['new_arms']) == NEW_ARMS and tuple(config['expert_arms']) == EXPERT_ARMS
            and tuple(config['budget_arms']) == BUDGET_ARMS, 'arm configuration changed')
    for key, value in dict(labels=64, strength=8., bandwidth_seed=1707, bandwidth_pair_limit=512, budget=.05).items():
        require(config[key] == value, 'fixed configuration changed: ' + key)
    complete = read_json(run / 'COMPLETE.json')
    require(complete['status'] == 'complete' and complete['source_metric_fits'] == 23
            and complete['new_target_fits'] == 345 and complete['new_backbone_pca_head_fits'] == 0
            and complete['policy_metric_rows'] == 1265, 'completion counts differ')
    expected, fits, refits = {}, 0, []
    errors = dict(score=0., dual=0., gate=0.)
    for port in ports:
        original = archive(ROOT / 'pred_cache' / (port + '.npz'))
        visual = archive(ROOT / 'pred_cache2' / (port + '.npz'))
        parent = archive(PARENT / port / 'policies.npz')
        frozen = archive(FROZEN / port / 'policies.npz')
        advantage = archive(ADVANTAGE / port / 'policies.npz')
        saved = archive(run / port / 'policies.npz')
        geometry = archive(run / port / 'geometry.npz')
        source, q = original['spos'], original['qpos']
        n = len(q)
        require(np.array_equal(q, np.unique(q)) and len(source) == len(np.unique(source))
                and not np.isin(source, q).any(), port + ': source/query rows')
        require(np.all(original['ports_all'][source] != port) and np.all(original['ports_all'][q] == port),
                port + ': outer split')
        require(np.array_equal(geometry['source_rows'], source), port + ': geometry source rows')
        for data in (saved, parent, frozen, advantage):
            require(np.array_equal(data['query_rows'], q) and np.array_equal(data['seeds'], np.arange(5)),
                    port + ': query/seeds alignment')
            require(data['support_rows'].shape == (5, 64) and data['eval_masks'].shape == (5, n)
                    and data['eval_masks'].dtype == np.bool_, port + ': support/eval shape')
        require(tuple(saved['expert_names']) == EXPERT_ARMS and tuple(saved['budget_names']) == BUDGET_ARMS,
                port + ': arm axes')
        for key, shape in (('scores', (5, 5, n, 8)), ('dual_coefficients', (5, 3, 64, 8)),
                           ('gate_scores', (5, 5, n)), ('expert_predictions', (5, 5, n)),
                           ('selected', (5, 6, n)), ('predictions', (5, 6, n)), ('caps', (5,))):
            require(saved[key].shape == shape and np.isfinite(saved[key]).all(), port + ': shape/finite ' + key)
        require(saved['selected'].dtype == np.bool_, port + ': mask dtype')
        for key in ('expert_predictions', 'predictions', 'base_prediction'):
            require(np.issubdtype(saved[key].dtype, np.integer)
                    and np.all((saved[key] >= 0) & (saved[key] < 8)), port + ': predictions ' + key)
        require(visual['Xs'].shape[1] == 128 and visual['Z'].shape[1] == 64, port + ': visual dimensions')
        raw = np.column_stack((visual['Xs'], visual['Z']))
        require(np.array_equal(np.unique(original['y_ids'][source]), np.arange(8)), port + ': source classes')
        if geometry_replay(raw, source, original['y_ids'][source], geometry, parent, port):
            refits.append(port)
        x = ((raw[q] - parent['source_center']) / parent['source_scale']).astype(np.float64) / np.sqrt(192.)
        metric_x = x @ geometry['transform']
        feature_blocks = (metric_x, x, metric_x)
        bandwidths = (None, geometry['bandwidths'][0], geometry['bandwidths'][1])
        prior = original['expert_logits'][q].astype(float)
        base_logits = original['base_logits'][q]
        base = base_logits.argmax(1)
        require(np.array_equal(saved['base_prediction'], base), port + ': frozen base')
        with (GATE / port / 'source_model.pkl').open('rb') as handle:
            gate = pickle.load(handle)['prior']
        for field in ('center', 'scale', 'lr'):
            close(frozen['source_' + field], gate[field], port + ': archived gate ' + field, atol=0, rtol=0)
        base_parts = confidence_parts(base_logits)
        bi = list(frozen['budgets']).index(.05)
        control_names = ('anchor_visual64', 'anchor_bias64')
        control_si = [list(parent['score_names']).index(name) for name in control_names]
        control_ei = [list(frozen['expert_names']).index(name) for name in control_names]
        advantage_i = list(advantage['arm_names']).index('advantage_visual')
        for i in range(5):
            context = port + '/' + str(i)
            support = np.random.default_rng(i).choice(q, 64, replace=False)
            ev = ~np.isin(q, support)
            for data in (saved, parent, frozen, advantage):
                require(np.array_equal(data['support_rows'][i], support)
                        and np.array_equal(data['eval_masks'][i], ev), context + ': support/evaluation changed')
            cap = int(np.floor(.05 * ev.sum()))
            require(saved['caps'][i] == frozen['caps'][i, bi] == advantage['caps'][i] == cap,
                    context + ': cap changed')
            local = np.searchsorted(q, support)
            require(np.array_equal(q[local], support), context + ': support position mapping')
            labels = original['y_ids'][support]
            observed = np.bincount(labels, minlength=8) > 0
            for ai, (features, bandwidth) in enumerate(zip(feature_blocks, bandwidths)):
                scores, dual = support_replay(features, local, labels, prior, bandwidth)
                close(saved['dual_coefficients'][i, ai], dual, context + '/' + NEW_ARMS[ai] + ': dual', atol=1e-9)
                close(saved['scores'][i, ai], scores, context + '/' + NEW_ARMS[ai] + ': scores', atol=1e-9)
                require(np.array_equal(saved['expert_predictions'][i, ai], scores.argmax(1)),
                        context + '/' + NEW_ARMS[ai] + ': independent argmax')
                errors['dual'] = max(errors['dual'], float(np.abs(saved['dual_coefficients'][i, ai] - dual).max()))
                errors['score'] = max(errors['score'], float(np.abs(saved['scores'][i, ai] - scores).max()))
                fits += 1
            for offset, (si, ei) in enumerate(zip(control_si, control_ei)):
                ai = 3 + offset
                require(np.array_equal(saved['scores'][i, ai], parent['scores'][i, si]), context + ': unchanged control scores')
                for key in ('selected', 'predictions'):
                    require(np.array_equal(saved[key][i, ai], frozen[key][i, bi, ei]), context + ': unchanged control ' + key)
                require(np.array_equal(saved['expert_predictions'][i, ai], frozen['expert_predictions'][i, ei]),
                        context + ': unchanged control receiver')
                close(saved['gate_scores'][i, ai], frozen['gate_scores'][i, ei], context + ': unchanged control gate', atol=0, rtol=0)
            for ai, arm in enumerate(EXPERT_ARMS):
                scores = saved['scores'][i, ai]
                receiver = scores.argmax(1)
                require(np.array_equal(saved['expert_predictions'][i, ai], receiver), context + ': receiver argmax')
                score = replay_score(base_parts, scores, gate)
                close(saved['gate_scores'][i, ai], score, context + '/' + arm + ': independent gate', atol=1e-11)
                errors['gate'] = max(errors['gate'], float(np.abs(saved['gate_scores'][i, ai] - score).max()))
                mask = replay_mask(q, ev, base, receiver, observed, score, cap)
                require(np.array_equal(saved['selected'][i, ai], mask), context + '/' + arm + ': ranked mask')
                require(np.array_equal(saved['predictions'][i, ai], np.where(mask, receiver, base)),
                        context + '/' + arm + ': handoff')
            for key in ('selected', 'predictions'):
                require(np.array_equal(saved[key][i, 5], advantage[key][i, advantage_i]),
                        context + ': unchanged advantage reference ' + key)
        # Query truth enters only after all five support fits and policies for this port are checked.
        truth = original['y_ids'][q]
        port_expected = {}
        parent_metrics = read_json(PARENT / port / 'metrics.json')
        frozen_metrics = read_json(FROZEN / port / 'metrics.json')
        advantage_metrics = read_json(ADVANTAGE / port / 'metrics.json')
        for i in range(5):
            ev = saved['eval_masks'][i]
            linear = saved['expert_predictions'][i, 3]
            for mode, arms, predictions in (('full', EXPERT_ARMS, saved['expert_predictions'][i]),
                                             ('budget', BUDGET_ARMS, saved['predictions'][i])):
                for ai, arm in enumerate(arms):
                    pred = predictions[ai]
                    result = metric_replay(truth[ev], base[ev], linear[ev], pred[ev])
                    result.update(evaluated_rows=int(ev.sum()), admitted_rows=None, changed_rows=None,
                                  cap=None, admission_fraction=None)
                    if mode == 'budget':
                        mask = saved['selected'][i, ai]
                        count, changed = int(mask.sum()), int(np.count_nonzero(pred[ev] != base[ev]))
                        require(not mask[~ev].any() and changed == count <= saved['caps'][i], port + ': budget accounting')
                        result.update(admitted_rows=count, changed_rows=changed, cap=int(saved['caps'][i]),
                                      admission_fraction=count / int(ev.sum()))
                    if arm in ('linear', 'bias'):
                        name = control_names[('linear', 'bias').index(arm)]
                        control = (parent_metrics['seeds'][i]['metrics'][name] if mode == 'full' else
                                   next(r for r in frozen_metrics if r['seed'] == i and r['budget'] == .05 and r['expert'] == name))
                    elif arm == 'advantage_reference':
                        control = next(r for r in advantage_metrics if r['seed'] == i and r['arm'] == 'advantage_visual')
                    else:
                        control = None
                    if control is not None:
                        for field in ('ba', 'acc'):
                            close(result[field], control[field], port + ': inherited control metric ' + field)
                    port_expected[(port, i, mode, arm)] = result
        compare_metric_rows(read_json(run / port / 'metrics.json'), port_expected)
        expected.update(port_expected)
        print('VERIFIED ' + port + ': source pairs, 15 dual fits, 25 gates, controls, 55 metric rows', flush=True)
    require(fits == 345 and len(refits) == 2 and len(expected) == 1265, 'replay coverage')
    with (run / 'metrics.csv').open(encoding='utf-8', newline='') as handle:
        compare_metric_rows(list(csv.DictReader(handle)), expected, csv_mode=True)
    verify_summary(read_json(run / 'summary.json'), expected, ports)
    return dict(status='passed', ports=23, contexts=115, source_geometry_refit_ports=refits,
                source_bandwidth_pair_replays=23, independent_weighted_dual_fits=fits,
                gate_and_mask_replays=575, unchanged_budget_controls=345,
                full_metric_rows=575, budget_metric_rows=690, policy_metric_rows=len(expected),
                max_absolute_errors=errors, query_labels_used_only_for_readout=True,
                producer_functions_used=False, additional_experiments=0,
                verifier_numpy=np.__version__,
                verifier_precision_note='Initial bandwidth replay differed by 6.7e-8 under NumPy 1.26 scalar promotion; explicit float64 conversion after float32 standardization reproduces archived bandwidths without tolerance changes.',
                limitations=['Source geometry fully refit for two fixed ports; remaining transforms are trusted exports.',
                             'Archived PCA/source heads, frozen gate, and advantage policy are trusted inputs; no refitting.',
                             'Freeze timing and upstream provenance are not independently established by artifact replay.',
                             'Source energy matching does not match effective kernel regularization.'])


def verify_summary(summary, expected, ports):
    require(summary['n_ports'] == 23 and summary['contexts'] == 115 and summary['policy_metric_rows'] == 1265
            and summary['target_label_budget'] == 64, 'summary coverage')
    require(summary['independent_confirmation'] is False and summary['novelty_established'] is False,
            'summary interpretation changed')
    values = {}
    for mode, arms in (('full', EXPERT_ARMS), ('budget', BUDGET_ARMS)):
        for arm in arms:
            fields = MEASURES + (('admission_fraction',) if mode == 'budget' else ())
            values[mode, arm] = {field: np.array([[expected[port, i, mode, arm][field] for i in range(5)]
                                                 for port in ports]) for field in fields}
    require(len(summary['means']) == len(values), 'summary means count')
    seen = set()
    for row in summary['means']:
        key = row['mode'], row['arm']
        require(key in values and key not in seen, 'summary mean identity')
        seen.add(key)
        for field, array in values[key].items():
            close(row[field], array.mean(1).mean(), str(key) + ': mean ' + field)
    contrasts = [(arm, 'linear') for arm in NEW_ARMS] + [('metric_rbf', arm) for arm in ('bias', 'metric_linear', 'isotropic_rbf')]
    keys = {(mode, a, b) for mode in ('full', 'budget') for a, b in contrasts}
    keys.add(('budget', 'metric_rbf', 'advantage_reference'))
    require(len(summary['comparisons']) == len(keys), 'summary contrast count')
    sampled = np.random.default_rng(0).integers(0, 23, (10000, 23))
    seen = set()
    for row in summary['comparisons']:
        key = row['mode'], row['candidate'], row['control']
        require(key in keys and key not in seen, 'summary contrast identity')
        seen.add(key)
        mode, a, b = key
        delta = values[mode, a]['ba'] - values[mode, b]['ba']
        by_port = delta.mean(1)
        measures = dict(mean_delta_ba_pp=100 * by_port.mean(),
                        mean_delta_acc_pp=100 * (values[mode, a]['acc'] - values[mode, b]['acc']).mean(1).mean(),
                        positive_ports=int((by_port > 1e-12).sum()), negative_ports=int((by_port < -1e-12).sum()),
                        draw_deltas_pp=100 * delta.mean(0),
                        port_bootstrap_95ci_pp=100 * np.percentile(by_port[sampled].mean(1), [2.5, 97.5]))
        for field, value in measures.items():
            close(row[field], value, str(key) + ': ' + field)
        require(set(row['port_deltas_pp']) == set(ports), 'contrast port inventory')
        close([row['port_deltas_pp'][port] for port in ports], 100 * by_port, str(key) + ': port deltas')
    primary = (values['budget', 'metric_rbf']['ba'] - values['budget', 'linear']['ba']).mean() > 0
    components = all((values['budget', 'metric_rbf']['ba'] - values['budget', arm]['ba']).mean() > 0
                     for arm in ('metric_linear', 'isotropic_rbf'))
    require(summary['passes_development_screen'] == bool(primary)
            and summary['both_components_positive'] == bool(components), 'development screen computation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', '--out', dest='run', type=Path,
                        default=ROOT / 'oof_local_metric_expert_results_20261007')
    args = parser.parse_args()
    require(args.run.is_dir(), 'run directory is not available yet')
    start = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            receipt = verify(args.run)
    except Exception as exc:
        receipt = dict(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(verifier=Path(__file__).name, run=str(args.run.resolve()), seconds=time.perf_counter() - start)
        (args.run / 'VERIFICATION.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
