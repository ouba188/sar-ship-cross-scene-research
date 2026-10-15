"""Independently replay the fixed action-advantage study and write its receipt."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from verify_budgeted_residual_handoff import (
    COUNTS, PARENT, ROOT, archive, check_metrics, close, read_json, require,
)
from verify_receiver_consistent_handoff import policy_metrics, replay_mask


FROZEN = ROOT / 'oof_receiver_consistent_results_20261007'
ARMS = ('advantage_bias', 'advantage_visual', 'frozen_bias', 'frozen_visual')
EXPERTS = ('anchor_bias64', 'anchor_visual64')
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'admission_fraction')


def equal(actual, expected, context):
    np.testing.assert_array_equal(actual, expected, err_msg=context)


def replay_scores(logits, base):
    shifted = np.asarray(logits, dtype=float) - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    receiver = logits.argmax(axis=1)
    rows = np.arange(len(logits))
    return probabilities[rows, receiver] - probabilities[rows, base], receiver


def verify_policies(port, saved, frozen, residual, query_rows, base, support_labels):
    """Replay policies with support labels only; no query truth enters this function."""
    n = len(query_rows)
    for data in (saved, frozen, residual):
        equal(data['query_rows'], query_rows, port + ': query alignment')
        equal(data['seeds'], np.arange(5), port + ': fixed support seeds')
    equal(saved['arm_names'], ARMS, port + ': arm order')
    equal(saved['base_prediction'], base, port + ': unchanged base')
    equal(frozen['base_prediction'], base, port + ': frozen base')
    for key, shape in (('support_rows', (5, 64)), ('eval_masks', (5, n)), ('caps', (5,)),
                       ('scores', (5, 2, n)), ('selected', (5, 4, n)),
                       ('predictions', (5, 4, n)), ('receiver_predictions', (5, 4, n))):
        require(saved[key].shape == shape, port + ': policy shape ' + key)
    require(saved['selected'].dtype == saved['eval_masks'].dtype == np.bool_, port + ': boolean masks')
    budget_index = list(frozen['budgets']).index(.05)
    score_indices = [list(residual['score_names']).index(name) for name in EXPERTS]
    receiver_indices = [list(frozen['expert_names']).index(name) for name in EXPERTS]
    maximum_error = 0.
    for seed in range(5):
        context = f'{port}/{seed}'
        support = np.random.default_rng(seed).choice(query_rows, 64, replace=False)
        ev = ~np.isin(query_rows, support)
        for data in (saved, frozen, residual):
            equal(data['support_rows'][seed], support, context + ': ordered 64-label support')
            equal(data['eval_masks'][seed], ev, context + ': support exclusion')
        cap = int(np.floor(.05 * int(ev.sum())))
        require(saved['caps'][seed] == frozen['caps'][seed, budget_index] == cap, context + ': fixed 5% floor cap')
        observed = np.isin(np.arange(8), np.unique(support_labels[seed]))
        for j, expert in enumerate(EXPERTS):
            score, receiver = replay_scores(residual['scores'][seed, score_indices[j]], base)
            close(saved['scores'][seed, j], score, context + '/' + expert + ': same-head T1 advantage',
                  atol=1e-14, rtol=1e-12)
            maximum_error = max(maximum_error, float(np.max(np.abs(saved['scores'][seed, j] - score))))
            ei = receiver_indices[j]
            equal(receiver, frozen['expert_predictions'][seed, ei], context + ': unchanged receiver')
            selected = replay_mask(query_rows, ev, base, receiver, observed, score, cap)
            for arm_index, mask in ((j, selected), (j + 2, frozen['selected'][seed, budget_index, ei])):
                equal(saved['receiver_predictions'][seed, arm_index], receiver, context + ': receiver alignment')
                equal(saved['selected'][seed, arm_index], mask, context + '/' + ARMS[arm_index] + ': ranking/mask')
                require(not mask[~ev].any() and int(mask.sum()) <= cap, context + ': exclusion/cap')
                require(np.all((base != receiver)[mask]) and np.all(observed[receiver[mask]]),
                        context + ': disagreement/observed-class eligibility')
                prediction = base.copy()
                prediction[mask] = receiver[mask]
                equal(saved['predictions'][seed, arm_index], prediction, context + ': base fallback')
                if arm_index >= 2:
                    equal(prediction, frozen['predictions'][seed, budget_index, ei], context + ': full control unchanged')
    return maximum_error


def metric_key(row):
    return row['port'], int(row['seed']), float(row['budget']), row['arm']


def verify_metrics(port, saved, truth, expected, run):
    previous = read_json(FROZEN / port / 'metrics.json')
    base = saved['base_prediction']
    for seed in range(5):
        ev = saved['eval_masks'][seed]
        for ai, arm in enumerate(ARMS):
            result = policy_metrics(truth, base, saved['predictions'][seed, ai], ev,
                                    saved['selected'][seed, ai], int(saved['caps'][seed]))
            require(result['admitted_rows'] == result['changed_rows'], port + ': changes differ from admissions')
            if ai >= 2:
                parent = [r for r in previous if r['seed'] == seed and r['budget'] == .05
                          and r['expert'] == EXPERTS[ai - 2]]
                require(len(parent) == 1, port + ': control metric identity')
                check_metrics(parent[0], result, f'{port}/{seed}/{arm}: parent metrics', MEASURES + COUNTS)
            expected[(port, seed, .05, arm)] = result
    reported = read_json(run / port / 'metrics.json')
    require(len(reported) == 20, port + ': metric count')
    seen = set()
    for row in reported:
        key = metric_key(row)
        require(key[0] == port and key in expected and key not in seen, port + ': metric identity')
        seen.add(key)
        check_metrics(row, expected[key], str(key), MEASURES + COUNTS)


def verify_summary(run, ports, expected):
    summary = read_json(run / 'summary.json')
    require(summary['n_ports'] == 23 and summary['contexts'] == 115
            and summary['policy_metric_rows'] == 460 and summary['budget'] == .05, 'summary coverage')
    require(summary['no_fitting'] is True and summary['independent_confirmation'] is False
            and summary['novelty_established'] is False, 'baseline interpretation changed')
    values = {arm: {metric: np.array([[expected[(port, seed, .05, arm)][metric]
                                     for seed in range(5)] for port in ports])
                    for metric in MEASURES} for arm in ARMS}
    require(len(summary['means']) == 4, 'summary mean count')
    seen = set()
    for row in summary['means']:
        arm = row['arm']
        require(arm in ARMS and arm not in seen, 'summary mean identity')
        seen.add(arm)
        for metric in MEASURES:
            close(row[metric], values[arm][metric].mean(axis=1).mean(), f'mean {arm}/{metric}')
    contrasts = {(a, b) for a in ARMS[:2] for b in ARMS[2:]}
    require(len(summary['comparisons']) == 4, 'comparison count')
    samples = np.random.default_rng(0).integers(0, 23, (10000, 23))
    seen, primary = set(), None
    for row in summary['comparisons']:
        key = row['candidate'], row['control']
        require(key in contrasts and key not in seen, 'comparison identity')
        seen.add(key)
        a, b = values[key[0]], values[key[1]]
        delta = a['ba'] - b['ba']
        port_delta = delta.mean(axis=1)
        replay = dict(mean_delta_ba_pp=100 * port_delta.mean(),
                      mean_delta_acc_pp=100 * (a['acc'] - b['acc']).mean(axis=1).mean(),
                      positive_ports=int((port_delta > 1e-12).sum()),
                      negative_ports=int((port_delta < -1e-12).sum()),
                      port_bootstrap_95ci_pp=100 * np.percentile(port_delta[samples].mean(axis=1), [2.5, 97.5]),
                      draw_deltas_pp=100 * delta.mean(axis=0))
        for metric, value in replay.items():
            close(row[metric], value, f'comparison {key}/{metric}')
        require(set(row['port_deltas_pp']) == set(ports), 'comparison port coverage')
        for port, delta_pp in zip(ports, 100 * port_delta):
            close(row['port_deltas_pp'][port], delta_pp, f'comparison {key}/{port}')
        if key == ('advantage_visual', 'frozen_visual'):
            primary = float(replay['mean_delta_ba_pp'])
    require(primary is not None, 'primary comparison missing')
    verdict = primary > 0
    require(summary['passes_development_screen'] is verdict, 'fixed primary screen differs')
    return dict(passes_development_screen=verdict, primary_delta_ba_pp=primary,
                mean_ba={arm: float(values[arm]['ba'].mean(axis=1).mean()) for arm in ARMS})


def verify(run):
    require((run / 'POLICIES_SAVED.json').is_file(), 'await POLICIES_SAVED.json before replay')
    require((run / 'COMPLETE.json').is_file(), 'await COMPLETE.json before replay')
    complete = read_json(run / 'COMPLETE.json')
    require(complete['status'] == 'complete' and complete['new_fits'] == 0
            and complete['policy_metric_rows'] == 460, 'completion receipt')
    config = read_json(run / 'RUN_CONFIG.json')
    require(config['labels'] == 64 and config['budget'] == .05 and config['temperature'] == 1.,
            'fixed support/budget/temperature')
    equal(config['arms'], ARMS, 'configured arms')
    inventory = read_json(run / 'POLICIES_SAVED.json')
    require(inventory == read_json(FROZEN / 'POLICIES_SAVED.json'), 'inherited inventory changed')
    ports = [item['port'] for item in inventory]
    require(len(ports) == len(set(ports)) == 23, 'expected 23 unique ports')
    expected, maximum_error = {}, 0.
    for item in inventory:
        port = item['port']
        saved = archive(run / port / 'policies.npz')
        frozen = archive(FROZEN / port / 'policies.npz')
        residual = archive(PARENT / port / 'policies.npz')
        with np.load(ROOT / 'pred_cache' / (port + '.npz'), allow_pickle=False) as original:
            q = original['qpos']
            equal(q, np.unique(q), port + ': unique ordered query rows')
            require(item['queries'] == len(q) and item['evaluation_rows'] == len(q) - 64,
                    port + ': inventory counts')
            base = original['base_logits'][q].argmax(axis=1)
            supports = np.stack([np.random.default_rng(seed).choice(q, 64, replace=False) for seed in range(5)])
            support_labels = original['y_ids'][supports]
            maximum_error = max(maximum_error,
                verify_policies(port, saved, frozen, residual, q, base, support_labels))
            # Query truth is first used after policy reconstruction has passed.
            verify_metrics(port, saved, original['y_ids'][q], expected, run)
        print(f'VERIFIED {port}: same-head scores, masks, controls and evaluation metrics', flush=True)
    with (run / 'metrics.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == len(expected) == 460, 'CSV metric count')
    seen = set()
    for row in rows:
        key = metric_key(row)
        require(key in expected and key not in seen, 'CSV duplicate or unknown identity')
        seen.add(key)
        for metric in MEASURES + COUNTS:
            close(float(row[metric]), expected[key][metric], f'CSV {key}/{metric}')
    outcome = verify_summary(run, ports, expected)
    return dict(status='verified', verified_at_utc=datetime.now(timezone.utc).isoformat(),
                run=str(run.resolve()), verifier=str(Path(__file__).resolve()),
                ports=23, contexts=115, policy_metric_rows=460, candidate_policies=230,
                unchanged_control_policies=230, contrasts_replayed=4,
                maximum_score_replay_error=maximum_error, budget=.05, temperature=1.,
                independent_fits=0, runner_modified=False, query_labels_used_for_policy=False,
                **outcome,
                limitation='Archived adapted receiver scores are trusted inputs. Replay does not prove '
                           'historical freeze timing. This is a development baseline, not independent '
                           'confirmation or novelty evidence; no compute savings are measured.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'oof_action_advantage_results_20261007')
    args = parser.parse_args()
    result = verify(args.run)
    receipt = args.run / 'VERIFICATION.json'
    receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    print('RECEIPT ' + str(receipt.resolve()))


if __name__ == '__main__':
    main()
