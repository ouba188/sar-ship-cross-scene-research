"""Read-only independent replay of the fixed budgeted residual handoff."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from verify_residual_expert import archive, close, confusion_metrics, read_json, require


ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT/'oof_residual_expert_results_20261007_v2'
GATE = ROOT/'oof_context_results_20261007'
BUDGETS = (.05, .10, .20)
SELECTORS = ('old_lr_cap', 'base_uncertainty')
EXPERTS = ('fusion', 'fusion_observed64', 'anchor_bias64', 'anchor_visual64')
CONTROLS = ('fusion', 'anchor_bias64', 'fusion_observed64')
METRICS = ('ba', 'acc', 'recover_ba', 'harm_ba')
MEASURES = METRICS + ('admission_fraction', 'change_fraction')
COUNTS = ('evaluated_rows', 'admitted_rows', 'changed_rows', 'cap')


def replay_masks(rows, evaluation, base_logits, fusion, gate_score, observed):
    """Use Python tuple sorting independently of the producer's lexsort."""
    base = base_logits.argmax(axis=1)
    top_two = np.partition(base_logits, -2, axis=1)[:, -2:]
    gap = top_two.max(axis=1)-top_two.min(axis=1)
    eligible = evaluation & (base != fusion) & observed[fusion] & (gate_score > 0)
    ranked = (
        sorted(np.flatnonzero(eligible), key=lambda k: (-float(gate_score[k]), int(rows[k]))),
        sorted(np.flatnonzero(evaluation), key=lambda k: (float(gap[k]), int(rows[k]))),
    )
    caps = np.array([int(budget*int(evaluation.sum())) for budget in BUDGETS])
    selected = np.zeros((2, 3, len(rows)), dtype=bool)
    for si, order in enumerate(ranked):
        for bi, cap in enumerate(caps):
            selected[si, bi, order[:cap]] = True
    return selected, caps


def replay_metrics(truth, base, prediction):
    # The existing confusion counter uses its third argument as recovery reference.
    values = confusion_metrics(truth, base, base, prediction)
    return {key: values[key] for key in METRICS + ('class_recall',)}


def check_metrics(claimed, expected, context, fields=METRICS):
    for key in fields:
        close(claimed[key], expected[key], f'{context}/{key}')
    require(len(claimed['class_recall']) == 8, f'{context}: recall columns')
    for c, recall in enumerate(expected['class_recall']):
        if recall is None:
            require(claimed['class_recall'][c] is None, f'{context}: absent-class recall')
        else:
            close(claimed['class_recall'][c], recall, f'{context}/recall{c}')


def policy_key(row):
    return (row['port'], int(row['seed']), row['selector'], float(row['budget']), row['expert'])


def verify(run):
    require((run/'POLICIES_SAVED.json').is_file(), 'await POLICIES_SAVED.json before replay')
    require((run/'COMPLETE.json').is_file(), 'await COMPLETE.json before metric replay')
    inventory = read_json(run/'POLICIES_SAVED.json')
    ports = [item['port'] for item in inventory]
    require(len(ports) == len(set(ports)) == 23, 'expected 23 unique ports')
    require(ports == [item['port'] for item in read_json(PARENT/'POLICIES_SAVED.json')],
            'port inventory differs from frozen parent')
    config = read_json(run/'RUN_CONFIG.json')
    require(tuple(config['selectors']) == SELECTORS and tuple(config['experts']) == EXPERTS
            and tuple(config['budgets']) == BUDGETS and config['labels'] == 64,
            'configured fixed experiment changed')
    require(Path(config['parent']).resolve() == PARENT.resolve()
            and Path(config['gate']).resolve() == GATE.resolve(), 'configured archive inputs changed')
    complete = read_json(run/'COMPLETE.json')
    require(complete['status'] == 'complete' and complete['new_fits'] == 0
            and complete['ports'] == 23 and complete['contexts'] == 115
            and complete['policy_metric_rows'] == 2760, 'completion receipt differs from protocol')
    expected_rows, references = {}, {}
    for item in inventory:
        port = item['port']
        original = archive(ROOT/'pred_cache'/f'{port}.npz')
        parent = archive(PARENT/port/'policies.npz')
        gate = archive(GATE/port/'policies.npz')
        control = archive(PARENT/'coverage_control'/f'{port}.npz')
        saved = archive(run/port/'policies.npz')
        q, labels = original['qpos'], original['y_ids']
        n = len(q)
        require(np.array_equal(q, np.unique(q)), f'{port}: query order/duplicates')
        require(item['queries'] == n and item['evaluation_rows'] == n-64,
                f'{port}: inventory row counts')
        for source in (parent, gate, control, saved):
            require(np.array_equal(source['query_rows'], q), f'{port}: query alignment')
            require(source['support_rows'].shape == (5, 64), f'{port}: support dimensions')
        require(np.array_equal(saved['seeds'], np.arange(5))
                and np.array_equal(parent['seeds'], saved['seeds'])
                and np.array_equal(gate['seeds'], saved['seeds']), f'{port}: seed order')
        require(tuple(saved['selector_names']) == SELECTORS
                and tuple(saved['expert_names']) == EXPERTS
                and tuple(saved['budgets']) == BUDGETS, f'{port}: axis order')
        require(saved['selected'].shape == (5, 2, 3, n)
                and saved['selected'].dtype == np.bool_
                and saved['eval_masks'].shape == (5, n)
                and saved['eval_masks'].dtype == np.bool_
                and saved['caps'].shape == (5, 3), f'{port}: mask/cap dimensions')
        for field, shape in (('predictions', (5, 2, 3, 4, n)),
                             ('expert_predictions', (5, 4, n)), ('base_prediction', (n,))):
            prediction = saved[field]
            require(prediction.shape == shape and np.issubdtype(prediction.dtype, np.integer)
                    and np.all((prediction >= 0) & (prediction < 8)), f'{port}: invalid {field}')
        base_logits, fusion_logits = original['base_logits'][q], original['expert_logits'][q]
        base, fusion = base_logits.argmax(1), fusion_logits.argmax(1)
        score = gate['scores'][list(gate['score_names']).index('oof_lr0')]
        require(np.isfinite(base_logits).all() and np.isfinite(score).all(), f'{port}: selector inputs')
        require(np.array_equal(saved['base_prediction'], base)
                and np.array_equal(saved['old_lr_score'], score), f'{port}: frozen base/gate changed')
        for i in range(5):
            context = f'{port}/{i}'
            support = np.random.default_rng(i).choice(q, 64, replace=False)
            for source in (parent, gate, control, saved):
                require(np.array_equal(source['support_rows'][i], support),
                        f'{context}: support budget/order changed')
            ev = ~np.isin(q, support)
            for source in (parent, gate, saved):
                require(np.array_equal(source['eval_masks'][i], ev), f'{context}: support exclusion')
            observed = np.zeros(8, dtype=bool)
            observed[np.unique(labels[support])] = True
            columns = np.flatnonzero(observed)
            restricted = columns[fusion_logits[:, columns].argmax(1)]
            require(np.array_equal(control['predictions'][i], restricted)
                    and np.array_equal(control['seen'][i], observed), f'{context}: observed-class control')
            prior = dict(zip(parent['arm_names'], parent['predictions'][i]))
            require(np.array_equal(prior['base'], base) and np.array_equal(prior['fusion'], fusion),
                    f'{context}: original heads changed')
            expected_experts = np.stack((fusion, restricted, prior['anchor_bias64'], prior['anchor_visual64']))
            require(np.array_equal(saved['expert_predictions'][i], expected_experts),
                    f'{context}: inherited expert predictions changed')
            uncapped = base.copy()
            eligible = (base != fusion) & observed[fusion] & (score > 0)
            uncapped[eligible] = fusion[eligible]
            require(np.array_equal(uncapped, prior['oof_lr0'])
                    and np.array_equal(uncapped, gate['predictions'][i, list(gate['arm_names']).index('oof_lr0')]),
                    f'{context}: original LR0 eligibility changed')
            selected, caps = replay_masks(q, ev, base_logits, fusion, score, observed)
            require(np.array_equal(saved['caps'][i], caps), f'{context}: floor cap differs')
            require(np.array_equal(saved['selected'][i], selected), f'{context}: admission ranking/mask differs')
            require(not selected[..., ~ev].any() and np.all(selected.sum(-1) <= caps)
                    and np.array_equal(selected[1].sum(-1), caps), f'{context}: support/cap violation')
            truth, base_eval = labels[q][ev], base[ev]
            for name, pred in [('base', base)] + list(zip(EXPERTS, expected_experts)):
                references[(port, i, name)] = replay_metrics(truth, base_eval, pred[ev])
            base_ba = references[(port, i, 'base')]['ba']
            for si, selector in enumerate(SELECTORS):
                for bi, budget in enumerate(BUDGETS):
                    mask = selected[si, bi]
                    admitted, cap, count = int(mask.sum()), int(caps[bi]), int(ev.sum())
                    for ei, expert in enumerate(EXPERTS):
                        pred = base.copy()
                        pred[mask] = expected_experts[ei, mask]
                        require(np.array_equal(saved['predictions'][i, si, bi, ei], pred),
                                f'{context}/{selector}/{budget}/{expert}: shared-mask handoff differs')
                        result = replay_metrics(truth, base_eval, pred[ev])
                        changes = int(np.count_nonzero(pred[ev] != base_eval))
                        require(changes <= admitted, f'{context}: changes exceed charged admissions')
                        close(result['ba']-base_ba, result['recover_ba']-result['harm_ba'],
                              f'{context}: recovery/harm identity relative to base')
                        result.update(evaluated_rows=count, admitted_rows=admitted, changed_rows=changes,
                                      cap=cap, admission_fraction=admitted/count, change_fraction=changes/count)
                        expected_rows[(port, i, selector, budget, expert)] = result
        reported = read_json(run/port/'metrics.json')
        require(len(reported) == 120, f'{port}: metric row count')
        seen = set()
        for row in reported:
            key = policy_key(row)
            require(key[0] == port and key in expected_rows and key not in seen, f'{port}: metric identity')
            seen.add(key)
            check_metrics(row, expected_rows[key], str(key), MEASURES + COUNTS)
        print(f'VERIFIED {port}: support, rankings, shared masks, costs and confusion metrics', flush=True)

    with (run/'metrics.csv').open(encoding='utf-8', newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == len(expected_rows) == 2760, 'CSV row count')
    seen = set()
    for row in csv_rows:
        key = policy_key(row)
        require(key in expected_rows and key not in seen, 'CSV duplicate/unknown identity')
        seen.add(key)
        for metric in MEASURES + COUNTS:
            close(float(row[metric]), expected_rows[key][metric], f'CSV {key}/{metric}')
    reported = read_json(run/'full_query_references.json')
    require(len(reported) == len(references) == 575, 'full-query reference count')
    seen = set()
    for row in reported:
        key = (row['port'], row['seed'], row['arm'])
        require(key in references and key not in seen, 'full-query reference identity')
        seen.add(key)
        check_metrics(row, references[key], f'reference {key}')

    summary = read_json(run/'summary.json')
    require(summary['n_ports'] == 23 and summary['contexts'] == 115
            and summary['policy_metric_rows'] == 2760, 'summary coverage')
    require(summary['new_fits'] == 0 and summary['independent_confirmation'] is False
            and summary['novelty_established'] is False, 'summary development claims changed')
    values = {(selector, budget, expert): {
        metric: np.array([[expected_rows[(port, seed, selector, budget, expert)][metric]
                           for seed in range(5)] for port in ports]) for metric in MEASURES}
        for selector in SELECTORS for budget in BUDGETS for expert in EXPERTS}
    require(len(summary['means']) == 24, 'summary mean count')
    seen = set()
    for row in summary['means']:
        key = (row['selector'], row['budget'], row['expert'])
        require(key in values and key not in seen, 'summary mean identity')
        seen.add(key)
        for metric in MEASURES:
            close(row[metric], values[key][metric].mean(axis=1).mean(), f'mean {key}/{metric}')
    require(set(summary['full_query_reference_means']) == set(('base',) + EXPERTS),
            'full-query reference arms')
    for name in ('base',) + EXPERTS:
        for metric in ('ba', 'acc'):
            expected = np.array([[references[(port, i, name)][metric] for i in range(5)] for port in ports])
            close(summary['full_query_reference_means'][name][metric], expected.mean(axis=1).mean(),
                  f'reference mean {name}/{metric}')
    sampled = np.random.default_rng(0).integers(0, 23, (10000, 23))
    require(len(summary['comparisons']) == 18, 'comparison count')
    seen, primary = set(), {}
    for row in summary['comparisons']:
        key = (row['selector'], row['budget'], row['control'])
        require(row['candidate'] == 'anchor_visual64' and key in values
                and row['control'] in CONTROLS and key not in seen, 'comparison identity')
        seen.add(key)
        a, b = values[(key[0], key[1], 'anchor_visual64')], values[key]
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
        if key[0] == 'old_lr_cap' and key[1] == .05 and key[2] in ('fusion', 'anchor_bias64'):
            primary[key[2]] = float(expected['mean_delta_ba_pp'])
    require(set(primary) == {'fusion', 'anchor_bias64'}, 'primary contrast coverage')
    verdict = all(delta > 0 for delta in primary.values())
    require(summary['passes_development_screen'] is verdict, 'fixed primary screen differs')
    return dict(status='verified', ports=23, contexts=115, policy_metric_rows=2760,
                shared_masks_replayed=690, full_query_reference_rows=575, contrasts_replayed=18,
                passes_development_screen=verdict, primary_delta_ba_pp=primary,
                fits=0, files_written=0,
                limitation='Prior verified experts and source scores are trusted. Replay checks frozen artifacts, '
                           'not historical readout timing. Target isolation is row-only; no compute savings measured.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'oof_budgeted_residual_results_20261007')
    args = parser.parse_args()
    print(json.dumps(verify(args.run), indent=2))


if __name__ == '__main__':
    main()
