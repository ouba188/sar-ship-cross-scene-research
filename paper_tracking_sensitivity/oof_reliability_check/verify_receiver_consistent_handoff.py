"""Read-only independent replay of receiver-consistent frozen-gate handoff."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from verify_budgeted_residual_handoff import (
    BUDGETS, COUNTS, EXPERTS, GATE, MEASURES, PARENT, ROOT,
    archive, check_metrics, close, read_json, replay_metrics, require,
)


STALE = ROOT/'oof_budgeted_residual_results_20261007'
ARMS = EXPERTS + ('stale_visual',)
CONTROLS = ('fusion', 'anchor_bias64', 'fusion_observed64', 'stale_visual')


def confidence_parts(logits, allowed=None):
    x = np.asarray(logits, dtype=np.float64)
    if allowed is not None:
        x = x[:, np.flatnonzero(allowed)]
    require(x.ndim == 2 and x.shape[1] >= 2 and np.isfinite(x).all(),
            'confidence needs at least two finite score columns')
    top = np.partition(x, -2, axis=1)[:, -2:]
    gap = top.max(axis=1)-top.min(axis=1)
    shifted = x-x.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    normalizer = exponentials.sum(axis=1)
    probability = exponentials/normalizer[:, None]
    entropy = np.log(normalizer)-(probability*shifted).sum(axis=1)
    return gap, entropy


def replay_score(base_parts, expert_logits, prior, allowed=None):
    base_gap, base_entropy = base_parts
    expert_gap, expert_entropy = confidence_parts(expert_logits, allowed)
    features = np.column_stack((expert_gap, base_gap, expert_entropy, base_entropy))
    normalized = (features-prior['center'])/prior['scale']
    return np.column_stack((normalized, np.ones(len(features)))) @ prior['lr']


def replay_mask(q, ev, base, receiving, observed, score, cap):
    candidates = np.flatnonzero(ev & (base != receiving) & observed[receiving] & (score > 0))
    ordered = sorted(candidates, key=lambda k: (-float(score[k]), int(q[k])))
    selected = np.zeros(len(q), dtype=bool)
    selected[ordered[:cap]] = True
    return selected


def row_key(row):
    return (row['port'], int(row['seed']), float(row['budget']), row['expert'])


def policy_metrics(truth, base, pred, ev, selected, cap):
    result = replay_metrics(truth[ev], base[ev], pred[ev])
    count, admitted = int(ev.sum()), int(selected.sum())
    changed = int(np.count_nonzero(pred[ev] != base[ev]))
    require(not selected[~ev].any() and changed <= admitted <= cap,
            'support exclusion or admission/change cap differs')
    result.update(evaluated_rows=count, admitted_rows=admitted, changed_rows=changed, cap=cap,
                  admission_fraction=admitted/count, change_fraction=changed/count)
    return result


def verify(run):
    require((run/'POLICIES_SAVED.json').is_file(), 'await POLICIES_SAVED.json before replay')
    require((run/'COMPLETE.json').is_file(), 'await COMPLETE.json before metric replay')
    inventory = read_json(run/'POLICIES_SAVED.json')
    require(inventory == read_json(STALE/'POLICIES_SAVED.json'), 'fixed inventory changed')
    ports = [item['port'] for item in inventory]
    require(len(ports) == len(set(ports)) == 23, 'expected 23 unique ports')
    config = read_json(run/'RUN_CONFIG.json')
    require(tuple(config['experts']) == EXPERTS and tuple(config['budgets']) == BUDGETS
            and config['labels'] == 64, 'fixed configuration changed')
    require(Path(config['stale']).resolve() == STALE.resolve()
            and Path(config['prior']).resolve() == GATE.resolve(), 'configured archives changed')
    complete = read_json(run/'COMPLETE.json')
    require(complete['status'] == 'complete' and complete['new_fits'] == 0
            and complete['contexts'] == 115 and complete['new_policy_metric_rows'] == 1380
            and complete['inherited_policy_metric_rows'] == 345, 'completion receipt changed')
    expected_rows, max_error = {}, 0.
    for item in inventory:
        port = item['port']
        saved = archive(run/port/'policies.npz')
        old = archive(STALE/port/'policies.npz')
        residual = archive(PARENT/port/'policies.npz')
        original = archive(ROOT/'pred_cache'/f'{port}.npz')
        # This locally produced source-model archive is an explicitly trusted input.
        with (GATE/port/'source_model.pkl').open('rb') as handle:
            prior = pickle.load(handle)['prior']
        for saved_key, prior_key, shape in (('source_center', 'center', (4,)),
                                             ('source_scale', 'scale', (4,)),
                                             ('source_lr', 'lr', (5,))):
            require(saved[saved_key].shape == shape and np.isfinite(saved[saved_key]).all()
                    and np.array_equal(saved[saved_key], prior[prior_key]),
                    f'{port}: source parameters changed: {saved_key}')
        require(np.all(saved['source_scale'] > 0), f'{port}: source scales')
        q, labels = original['qpos'], original['y_ids']
        n = len(q)
        require(np.array_equal(q, np.unique(q)) and item['queries'] == n
                and item['evaluation_rows'] == n-64, f'{port}: query inventory')
        for source in (saved, old, residual):
            require(np.array_equal(source['query_rows'], q)
                    and source['support_rows'].shape == (5, 64)
                    and np.array_equal(source['seeds'], np.arange(5)), f'{port}: rows/support/seeds')
        require(tuple(saved['budgets']) == BUDGETS and tuple(saved['expert_names']) == EXPERTS,
                f'{port}: policy axis order')
        require(saved['selected'].shape == (5, 3, 4, n) and saved['selected'].dtype == np.bool_
                and saved['eval_masks'].shape == (5, n) and saved['eval_masks'].dtype == np.bool_
                and saved['caps'].shape == (5, 3), f'{port}: mask/cap shape')
        require(saved['gate_scores'].shape == (5, 4, n) and np.isfinite(saved['gate_scores']).all(),
                f'{port}: gate scores')
        for key, shape in (('expert_predictions', (5, 4, n)), ('base_prediction', (n,)),
                           ('predictions', (5, 3, 4, n))):
            pred = saved[key]
            require(pred.shape == shape and np.issubdtype(pred.dtype, np.integer)
                    and np.all((pred >= 0) & (pred < 8)), f'{port}: invalid {key}')
        base_logits, fusion_logits = original['base_logits'][q], original['expert_logits'][q]
        base, fusion = base_logits.argmax(1), fusion_logits.argmax(1)
        require(np.array_equal(saved['base_prediction'], base)
                and np.array_equal(old['base_prediction'], base)
                and np.array_equal(saved['expert_predictions'], old['expert_predictions']),
                f'{port}: inherited base/expert predictions changed')
        base_parts = confidence_parts(base_logits)
        for i in range(5):
            context = f'{port}/{i}'
            support = np.random.default_rng(i).choice(q, 64, replace=False)
            ev = ~np.isin(q, support)
            for source in (saved, old, residual):
                require(np.array_equal(source['support_rows'][i], support)
                        and np.array_equal(source['eval_masks'][i], ev), f'{context}: support exclusion')
            observed = np.zeros(8, dtype=bool)
            observed[np.unique(labels[support])] = True
            require(observed.sum() >= 3, f'{context}: declared observed-class inventory changed')
            caps = np.array([int(budget*int(ev.sum())) for budget in BUDGETS])
            require(np.array_equal(saved['caps'][i], caps) and np.array_equal(old['caps'][i], caps),
                    f'{context}: floor cap changed')
            blocks = dict(zip(residual['score_names'], residual['scores'][i]))
            blocks.update(fusion=fusion_logits, fusion_observed64=fusion_logits)
            columns = np.flatnonzero(observed)
            base_ba = replay_metrics(labels[q][ev], base[ev], base[ev])['ba']
            for ei, expert in enumerate(EXPERTS):
                logits = blocks[expert]
                allowed = observed if expert == 'fusion_observed64' else None
                receiving = (columns[logits[:, columns].argmax(1)] if allowed is not None else logits.argmax(1))
                require(np.array_equal(receiving, saved['expert_predictions'][i, ei]),
                        f'{context}/{expert}: actual receiver argmax differs')
                score = replay_score(base_parts, logits, prior, allowed)
                close(saved['gate_scores'][i, ei], score, f'{context}/{expert}: confidence and LR replay',
                      atol=1e-11, rtol=1e-11)
                max_error = max(max_error, float(np.max(np.abs(saved['gate_scores'][i, ei]-score))))
                if expert == 'fusion':
                    close(saved['gate_scores'][i, ei], old['old_lr_score'],
                          f'{context}: unchanged fusion gate', atol=1e-12, rtol=1e-12)
                for bi, budget in enumerate(BUDGETS):
                    cap = int(caps[bi])
                    selected = replay_mask(q, ev, base, receiving, observed, score, cap)
                    require(np.array_equal(saved['selected'][i, bi, ei], selected),
                            f'{context}/{budget}/{expert}: eligibility/ranking differs')
                    pred = base.copy()
                    pred[selected] = receiving[selected]
                    require(np.array_equal(saved['predictions'][i, bi, ei], pred),
                            f'{context}/{budget}/{expert}: handoff differs')
                    if expert == 'fusion':
                        require(np.array_equal(selected, old['selected'][i, 0, bi])
                                and np.array_equal(pred, old['predictions'][i, 0, bi, 0]),
                                f'{context}/{budget}: unchanged fusion policy differs')
                    result = policy_metrics(labels[q], base, pred, ev, selected, cap)
                    require(result['changed_rows'] == result['admitted_rows'],
                            f'{context}/{expert}: receiver-disagreement eligibility differs')
                    close(result['ba']-base_ba, result['recover_ba']-result['harm_ba'],
                          f'{context}/{expert}: base-relative recovery/harm identity')
                    expected_rows[(port, i, budget, expert)] = result
            for bi, budget in enumerate(BUDGETS):
                selected = old['selected'][i, 0, bi]
                pred = base.copy()
                pred[selected] = old['expert_predictions'][i, 3, selected]
                require(np.array_equal(pred, old['predictions'][i, 0, bi, 3]),
                        f'{context}/{budget}: stale visual prediction differs')
                expected_rows[(port, i, budget, 'stale_visual')] = policy_metrics(
                    labels[q], base, pred, ev, selected, int(caps[bi]))
        reported = read_json(run/port/'metrics.json')
        require(len(reported) == 75, f'{port}: metric count')
        seen = set()
        for row in reported:
            key = row_key(row)
            require(key[0] == port and key in expected_rows and key not in seen, f'{port}: metric identity')
            seen.add(key)
            check_metrics(row, expected_rows[key], str(key), MEASURES + COUNTS)
        print(f'VERIFIED {port}: confidence, frozen LR, receiver masks, fusion control and metrics', flush=True)

    with (run/'metrics.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == len(expected_rows) == 1725, 'CSV metric count')
    seen = set()
    for row in rows:
        key = row_key(row)
        require(key in expected_rows and key not in seen, 'CSV duplicate/unknown identity')
        seen.add(key)
        for metric in MEASURES + COUNTS:
            close(float(row[metric]), expected_rows[key][metric], f'CSV {key}/{metric}')
    summary = read_json(run/'summary.json')
    require(summary['n_ports'] == 23 and summary['contexts'] == 115
            and summary['policy_metric_rows'] == 1725 and summary['new_policy_metric_rows'] == 1380,
            'summary coverage')
    require(summary['no_fitting'] is True and summary['independent_confirmation'] is False
            and summary['novelty_established'] is False, 'summary development claims changed')
    values = {(budget, arm): {metric: np.array([
        [expected_rows[(port, i, budget, arm)][metric] for i in range(5)] for port in ports])
        for metric in MEASURES} for budget in BUDGETS for arm in ARMS}
    require(len(summary['means']) == 15, 'summary mean count')
    seen = set()
    for row in summary['means']:
        key = (row['budget'], row['expert'])
        require(key in values and key not in seen, 'summary mean identity')
        seen.add(key)
        for metric in MEASURES:
            close(row[metric], values[key][metric].mean(axis=1).mean(), f'mean {key}/{metric}')
    require(len(summary['comparisons']) == 12, 'comparison count')
    sampled = np.random.default_rng(0).integers(0, 23, (10000, 23))
    seen, primary = set(), {}
    for row in summary['comparisons']:
        key = (row['budget'], row['control'])
        require(row['candidate'] == 'anchor_visual64' and key in values
                and row['control'] in CONTROLS and key not in seen, 'comparison identity')
        seen.add(key)
        a, b = values[(key[0], 'anchor_visual64')], values[key]
        delta = a['ba']-b['ba']
        by_port = delta.mean(axis=1)
        expected = dict(mean_delta_ba_pp=100*by_port.mean(),
                        mean_delta_acc_pp=100*(a['acc']-b['acc']).mean(axis=1).mean(),
                        positive_ports=int((by_port > 1e-12).sum()),
                        negative_ports=int((by_port < -1e-12).sum()),
                        port_bootstrap_95ci_pp=100*np.percentile(by_port[sampled].mean(axis=1), [2.5, 97.5]),
                        draw_deltas_pp=100*delta.mean(axis=0))
        for metric, value in expected.items():
            close(row[metric], value, f'comparison {key}/{metric}')
        if key[0] == .05 and key[1] in ('stale_visual', 'anchor_bias64'):
            primary[key[1]] = float(expected['mean_delta_ba_pp'])
    require(set(primary) == {'stale_visual', 'anchor_bias64'}, 'primary contrast coverage')
    verdict = all(value > 0 for value in primary.values())
    require(summary['passes_development_screen'] is verdict, 'fixed primary screen differs')
    return dict(status='verified', ports=23, contexts=115, policy_metric_rows=1725,
                new_policy_metric_rows=1380, inherited_policy_metric_rows=345, contrasts_replayed=12,
                unchanged_fusion_policies=345, maximum_gate_score_replay_error=max_error,
                passes_development_screen=verdict, primary_delta_ba_pp=primary, fits=0, files_written=0,
                limitation='Trusted archived source LR and residual scores; frozen-gate transfer is not newly '
                           'calibrated reliability. Artifact replay does not prove historical freeze timing. '
                           'Target isolation is row-only; no compute savings measured.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'oof_receiver_consistent_results_20261007')
    args = parser.parse_args()
    print(json.dumps(verify(args.run), indent=2))


if __name__ == '__main__':
    main()
