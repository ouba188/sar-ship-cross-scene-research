"""Read-only independent replay, permitted only after the complete policy freeze."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from build_oof_predictions import load_manifest
from verify_budgeted_residual_handoff import (
    BUDGETS, COUNTS, GATE, MEASURES, PARENT, ROOT,
    archive, check_metrics, close, read_json, require,
)
from verify_receiver_consistent_handoff import (
    confidence_parts, policy_metrics, replay_mask,
)


FROZEN = ROOT / 'oof_receiver_consistent_results_20261007'
ARMS = ('matched_binary', 'gain_3way', 'frozen_bias', 'frozen_visual')
CONTRASTS = (('gain_3way', 'frozen_visual'), ('gain_3way', 'matched_binary'),
             ('gain_3way', 'frozen_bias'), ('matched_binary', 'frozen_visual'),
             ('matched_binary', 'frozen_bias'))


def equal(actual, expected, context):
    np.testing.assert_array_equal(actual, expected, err_msg=context)


def features(base, expert):
    bg, bh = confidence_parts(base)
    eg, eh = confidence_parts(expert)
    return np.column_stack((eg, bg, eh, bh))


def bias_from_support(prior, support, labels):
    # Equal class means independently implement the balanced weighted intercept.
    residuals = []
    for c in np.unique(labels):
        desired = np.full(8, -1.)
        desired[c] = 1.
        residuals.append(desired - prior[support[labels == c]].mean(axis=0))
    return np.mean(residuals, axis=0) * (64 / 72)


def retained_rows(candidates, held, manifest):
    excluded = {key: set(manifest[key][held]) - {''} for key in ('mmsi', 'product')}
    return np.asarray([r for r in candidates
                       if all(manifest[key][r] not in excluded[key] for key in excluded)],
                      dtype=np.int64)


def verify_oof(port, original, manifest):
    oof = archive(ROOT / 'oof_predictions_v1' / (port + '.npz'))
    meta = json.loads(str(oof['metadata_json'].item()))
    require(meta['schema'] == 'oof-reliability-v1' and meta['complete'] is True
            and meta['outer_port'] == port, port + ': incomplete or wrong OOF declaration')
    require(meta['encoder_access'] in ('independent', 'transductive-unlabeled'),
            port + ': missing encoder access declaration')
    require(meta['outer_validation']['base_argmax_mismatches'] == 0
            and meta['outer_validation']['expert_argmax_mismatches'] == 0,
            port + ': failed archived outer-head validation')
    equal(original['y_ids'], manifest['labels'], port + ': manifest labels')
    equal(original['ports_all'], manifest['ports'], port + ': manifest ports')
    outer = np.flatnonzero(manifest['ports'] == port)
    source = retained_rows(np.flatnonzero(manifest['ports'] != port), outer, manifest)
    equal(original['qpos'], outer, port + ': outer query split')
    equal(original['spos'], source, port + ': outer identity exclusion')
    equal(oof['source_rows'], source, port + ': OOF source rows')
    equal(oof['source_labels'], manifest['labels'][source], port + ': OOF labels')
    equal(oof['source_ports'], manifest['ports'][source], port + ': OOF ports')
    require(not np.isin(source, outer).any(), port + ': outer rows in OOF')
    for name in ('oof_base_logits', 'oof_expert_logits'):
        require(oof[name].shape == (len(source), 8) and np.isfinite(oof[name]).all(),
                port + ': OOF score shape/values')
    expected_ports = set(oof['source_ports'])
    require(len(meta['folds']) == len(expected_ports)
            and {f['heldout_port'] for f in meta['folds']} == expected_ports,
            port + ': incomplete or duplicate OOF folds')
    for fold in meta['folds']:
        held_port = fold['heldout_port']
        held = source[manifest['ports'][source] == held_port]
        candidates = source[manifest['ports'][source] != held_port]
        all_held = np.flatnonzero(manifest['ports'] == held_port)
        train = retained_rows(candidates, all_held, manifest)
        equal(fold['heldout_rows'], held, port + '/' + held_port + ': held split')
        equal(fold['train_rows'], train, port + '/' + held_port + ': head identity exclusion')
        require(fold['train_n'] == len(train) and fold['heldout_n'] == len(held),
                port + '/' + held_port + ': fold counts')
    return oof


def verify_bank(port, bank, oof, receipt):
    source = oof['source_rows']
    equal(bank['source_rows'], source, port + ': bank source rows')
    ports, counts = np.unique(oof['source_ports'], return_counts=True)
    available, skipped = ports[counts > 64], ports[counts <= 64]
    equal(bank['skipped_ports'], skipped, port + ': size-only skip rule')
    equal(bank['episode_ports'], np.repeat(available, 5), port + ': episode port order')
    equal(bank['episode_seeds'], np.tile(np.arange(5), len(available)), port + ': episode seeds')
    episodes = len(available) * 5
    require(bank['support_rows'].shape == (episodes, 64)
            and bank['biases'].shape == (episodes, 8), port + ': episode support/bias shape')
    offsets = np.concatenate(([0], np.cumsum(np.repeat(counts[counts > 64] - 64, 5))))
    equal(bank['query_offsets'], offsets, port + ': episode query offsets')
    n = int(offsets[-1])
    for name in ('query_rows', 'class_ids', 'utility', 'eligible', 'weights',
                 'base_prediction', 'receiver_prediction'):
        require(bank[name].shape == (n,), port + ': bank column ' + name)
    require(bank['features'].shape == (n, 4) and bank['eligible'].dtype == np.bool_,
            port + ': bank feature/eligibility shape')
    index = 0
    for held_port in available:
        positions = np.flatnonzero(oof['source_ports'] == held_port)
        rows = source[positions]
        base_logits = oof['oof_base_logits'][positions].astype(float)
        prior = oof['oof_expert_logits'][positions].astype(float)
        labels = oof['source_labels'][positions]
        base = base_logits.argmax(axis=1)
        for seed in range(5):
            context = f'{port}/{held_port}/{seed}'
            support = np.random.default_rng(seed).choice(rows, 64, replace=False)
            equal(bank['support_rows'][index], support, context + ': ordered support')
            local = np.searchsorted(rows, support)
            equal(rows[local], support, context + ': local row mapping')
            evaluation = ~np.isin(rows, support)
            section = slice(offsets[index], offsets[index + 1])
            equal(bank['query_rows'][section], rows[evaluation], context + ': support exclusion')
            bias = bias_from_support(prior, local, labels[local])
            close(bank['biases'][index], bias, context + ': support-only bias', atol=1e-12)
            adapted = prior + bias
            receiver = adapted.argmax(axis=1)
            y = labels[evaluation]
            equal(bank['class_ids'][section], y, context + ': query labels')
            equal(bank['base_prediction'][section], base[evaluation], context + ': base prediction')
            equal(bank['receiver_prediction'][section], receiver[evaluation], context + ': receiver prediction')
            close(bank['features'][section], features(base_logits, adapted)[evaluation],
                  context + ': four confidence features', atol=1e-11, rtol=1e-11)
            utility = ((receiver[evaluation] == y).astype(int)
                       - (base[evaluation] == y).astype(int))
            equal(bank['utility'][section], utility, context + ': true correction utility')
            eligible = ((base != receiver) & np.isin(receiver, np.unique(labels[local])))[evaluation]
            equal(bank['eligible'][section], eligible, context + ': support-observed disagreement')
            classes = np.unique(y)
            expected_weight = np.empty(len(y))
            for c in classes:
                members = y == c
                expected_weight[members] = 1 / (len(available) * 5 * len(classes) * members.sum())
            close(bank['weights'][section], expected_weight, context + ': all-query class weights')
            index += 1
    close(bank['weights'].sum(), 1., port + ': pre-filter source weight sum')
    require(receipt['episodes'] == episodes and receipt['episode_query_rows'] == n
            and receipt['source_ports'] == len(available)
            and receipt['skipped_source_ports'] == skipped.tolist(), port + ': source receipt')
    for value in (-1, 0, 1):
        require(receipt['eligible_source_outcome_counts'][str(value)] ==
                int(np.count_nonzero(bank['eligible'] & (bank['utility'] == value))),
                port + ': eligible outcome counts')


def verify_model(port, model, bank, receipt, refit):
    x, w = bank['features'], bank['weights']
    center = np.sum(x * w[:, None], axis=0) / w.sum()
    scale = np.maximum(np.sqrt(np.sum(w[:, None] * (x - center) ** 2, axis=0) / w.sum()), 1e-6)
    close(model['center'], center, port + ': all-query source center', atol=1e-12)
    close(model['scale'], scale, port + ': all-query source scale', atol=1e-12)
    require(model['center'].shape == model['scale'].shape == (4,), port + ': four-dimensional scaling')
    x = (x - center) / scale
    for name, classes in (('matched_binary', [-1, 1]), ('gain_3way', [-1, 0, 1])):
        use = bank['eligible'].copy()
        if name == 'matched_binary':
            use &= bank['utility'] != 0
        equal(model[name + '_classes'], classes, port + '/' + name + ': class order')
        equal(np.unique(bank['utility'][use]), classes, port + '/' + name + ': source outcomes')
        k = 1 if name == 'matched_binary' else 3
        require(model[name + '_coef'].shape == (k, 4)
                and model[name + '_intercept'].shape == (k,)
                and np.isfinite(model[name + '_coef']).all()
                and np.isfinite(model[name + '_intercept']).all(), port + '/' + name + ': parameters')
        require(receipt['fits'][name]['training_rows'] == int(use.sum())
                and all(0 < i < 2000 for i in receipt['fits'][name]['iterations']),
                port + '/' + name + ': fit receipt')
        if refit:
            fit = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, tol=1e-9)
            fit.fit(x[use], bank['utility'][use], sample_weight=len(x) * w[use])
            require(fit.n_iter_.max() < 2000, port + '/' + name + ': independent convergence')
            close(model[name + '_coef'], fit.coef_, port + '/' + name + ': independent coefficients',
                  atol=1e-7, rtol=1e-7)
            close(model[name + '_intercept'], fit.intercept_, port + '/' + name + ': independent intercept',
                  atol=1e-7, rtol=1e-7)


def gate_scores(model, x):
    normalized = (x - model['center']) / model['scale']
    binary = (normalized @ model['matched_binary_coef'].T + model['matched_binary_intercept'])[:, 0]
    logits = normalized @ model['gain_3way_coef'].T + model['gain_3way_intercept']
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability = exponentials / exponentials.sum(axis=1, keepdims=True)
    return np.stack((binary, probability @ np.array([-1., 0., 1.]))), probability


def metric_key(row):
    return row['port'], int(row['seed']), float(row['budget']), row['arm']


def verify_targets(port, original, saved, model, expected_rows):
    frozen = archive(FROZEN / port / 'policies.npz')
    residual = archive(PARENT / port / 'policies.npz')
    controls = read_json(FROZEN / port / 'metrics.json')
    q = original['qpos']
    n = len(q)
    for data in (saved, frozen, residual):
        equal(data['query_rows'], q, port + ': target row order')
        equal(data['seeds'], np.arange(5), port + ': target seeds')
    equal(saved['arm_names'], ARMS, port + ': arm order')
    equal(saved['score_names'], ARMS[:2], port + ': score order')
    equal(saved['budgets'], BUDGETS, port + ': budgets')
    for name, shape in (('scores', (5, 2, n)), ('gain_probabilities', (5, n, 3)),
                        ('selected', (5, 3, 4, n)), ('predictions', (5, 3, 4, n)),
                        ('receiver_predictions', (5, 4, n)), ('base_prediction', (n,)),
                        ('support_rows', (5, 64)), ('eval_masks', (5, n)), ('caps', (5, 3))):
        require(saved[name].shape == shape, port + ': policy dimensions ' + name)
    require(saved['selected'].dtype == saved['eval_masks'].dtype == np.bool_, port + ': boolean masks')
    base_logits = original['base_logits'][q]
    prior = original['expert_logits'][q].astype(float)
    base = base_logits.argmax(axis=1)
    equal(saved['base_prediction'], base, port + ': unchanged base')
    bias_index = list(residual['score_names']).index('anchor_bias64')
    visual_index = list(residual['score_names']).index('anchor_visual64')
    frozen_indices = [list(frozen['expert_names']).index(name)
                      for name in ('anchor_bias64', 'anchor_visual64')]
    max_error = 0.
    for seed in range(5):
        context = f'{port}/{seed}'
        support = np.random.default_rng(seed).choice(q, 64, replace=False)
        ev = ~np.isin(q, support)
        caps = [int(b * int(ev.sum())) for b in BUDGETS]
        for data in (saved, frozen, residual):
            equal(data['support_rows'][seed], support, context + ': target support')
            equal(data['eval_masks'][seed], ev, context + ': target support exclusion')
        equal(saved['caps'][seed], caps, context + ': floor admission cap')
        equal(frozen['caps'][seed], caps, context + ': control cap')
        support_labels = original['y_ids'][support]
        bias = bias_from_support(prior, np.searchsorted(q, support), support_labels)
        adapted = prior + bias
        close(residual['scores'][seed, bias_index], adapted, context + ': unchanged parent bias scores',
              atol=1e-12, rtol=1e-12)
        close(residual['anchor_bias64'][seed, 0], bias, context + ': parent bias coefficient', atol=1e-12)
        receiver = adapted.argmax(axis=1)
        visual = residual['scores'][seed, visual_index].argmax(axis=1)
        equal(saved['receiver_predictions'][seed], np.stack((receiver, receiver, receiver, visual)),
              context + ': frozen receiving predictions')
        equal(frozen['expert_predictions'][seed, frozen_indices[0]], receiver, context + ': bias control receiver')
        equal(frozen['expert_predictions'][seed, frozen_indices[1]], visual, context + ': visual control receiver')
        scores, probability = gate_scores(model, features(base_logits, adapted))
        close(saved['scores'][seed], scores, context + ': gate scores', atol=1e-10, rtol=1e-10)
        close(saved['gain_probabilities'][seed], probability, context + ': three-outcome probabilities',
              atol=1e-11, rtol=1e-11)
        max_error = max(max_error, float(np.max(np.abs(saved['scores'][seed] - scores))))
        observed = np.isin(np.arange(8), np.unique(support_labels))
        for bi, budget in enumerate(BUDGETS):
            for ai, arm in enumerate(ARMS):
                if ai < 2:
                    selected = replay_mask(q, ev, base, receiver, observed, scores[ai], caps[bi])
                else:
                    ei = frozen_indices[ai - 2]
                    selected = frozen['selected'][seed, bi, ei]
                    equal(saved['predictions'][seed, bi, ai], frozen['predictions'][seed, bi, ei],
                          context + '/' + arm + ': unchanged full control policy')
                equal(saved['selected'][seed, bi, ai], selected, context + '/' + arm + ': ranked mask')
                prediction = base.copy()
                prediction[selected] = saved['receiver_predictions'][seed, ai, selected]
                equal(saved['predictions'][seed, bi, ai], prediction, context + '/' + arm + ': fallback')
                result = policy_metrics(original['y_ids'][q], base, prediction, ev, selected, caps[bi])
                require(result['admitted_rows'] == result['changed_rows'], context + ': disagreement admissions')
                if ai >= 2:
                    expert = ('anchor_bias64', 'anchor_visual64')[ai - 2]
                    previous = [r for r in controls if r['seed'] == seed and r['budget'] == budget
                                and r['expert'] == expert]
                    require(len(previous) == 1, context + ': control metric identity')
                    check_metrics(previous[0], result, context + '/' + arm + ': all control metrics', MEASURES + COUNTS)
                expected_rows[(port, seed, budget, arm)] = result
    return max_error


def verify_summary(run, ports, inventory, expected_rows):
    summary = read_json(run / 'summary.json')
    require(summary['n_ports'] == 23 and summary['contexts'] == 115
            and summary['policy_metric_rows'] == 1380 and summary['new_gate_fits'] == 46
            and summary['source_episodes'] == sum(r['episodes'] for r in inventory), 'summary coverage')
    require(summary['independent_confirmation'] is False and summary['novelty_established'] is False,
            'development-only claims changed')
    values = {(budget, arm): {metric: np.array([
        [expected_rows[(port, seed, budget, arm)][metric] for seed in range(5)] for port in ports])
        for metric in MEASURES} for budget in BUDGETS for arm in ARMS}
    require(len(summary['means']) == len(values), 'summary mean count')
    seen = set()
    for row in summary['means']:
        key = row['budget'], row['arm']
        require(key in values and key not in seen, 'summary mean identity')
        seen.add(key)
        for metric in MEASURES:
            close(row[metric], values[key][metric].mean(axis=1).mean(), f'mean {key}/{metric}')
    expected_contrasts = {(budget, a, b) for budget in BUDGETS for a, b in CONTRASTS}
    require(len(summary['comparisons']) == len(expected_contrasts), 'comparison count')
    samples = np.random.default_rng(0).integers(0, 23, (10000, 23))
    seen, primary = set(), {}
    for row in summary['comparisons']:
        key = row['budget'], row['candidate'], row['control']
        require(key in expected_contrasts and key not in seen, 'comparison identity')
        seen.add(key)
        a, b = values[(key[0], key[1])], values[(key[0], key[2])]
        delta = a['ba'] - b['ba']
        by_port = delta.mean(axis=1)
        expected = dict(mean_delta_ba_pp=100 * by_port.mean(),
                        mean_delta_acc_pp=100 * (a['acc'] - b['acc']).mean(axis=1).mean(),
                        positive_ports=int((by_port > 1e-12).sum()),
                        negative_ports=int((by_port < -1e-12).sum()),
                        port_bootstrap_95ci_pp=100 * np.percentile(by_port[samples].mean(axis=1), [2.5, 97.5]),
                        draw_deltas_pp=100 * delta.mean(axis=0))
        for metric, value in expected.items():
            close(row[metric], value, f'comparison {key}/{metric}')
        if key[0] == .05 and key[1] == 'gain_3way' and key[2] in ('frozen_visual', 'matched_binary'):
            primary[key[2]] = float(expected['mean_delta_ba_pp'])
    require(set(primary) == {'frozen_visual', 'matched_binary'}, 'primary contrast coverage')
    verdict = all(delta > 0 for delta in primary.values())
    require(summary['passes_development_screen'] is verdict, 'fixed development screen differs')
    return verdict, primary


def verify(run):
    # Check both markers before opening any candidate artifact or target outcome.
    require((run / 'POLICIES_SAVED.json').is_file(), 'await POLICIES_SAVED.json before replay')
    require((run / 'COMPLETE.json').is_file(), 'await COMPLETE.json before replay')
    complete = read_json(run / 'COMPLETE.json')
    require(complete['status'] == 'complete', 'run is not complete')
    config = read_json(run / 'RUN_CONFIG.json')
    equal(config['arms'], ARMS, 'configured arms')
    equal(config['contrasts'], CONTRASTS, 'configured contrasts')
    equal(config['budgets'], BUDGETS, 'configured budgets')
    equal(config['source_seeds'], np.arange(5), 'configured source seeds')
    require(config['labels'] == 64 and config['adaptation_strength'] == 8.
            and config['logistic_C'] == 1. and Path(config['frozen']).resolve() == FROZEN.resolve(),
            'fixed experiment configuration changed')
    inventory = read_json(run / 'POLICIES_SAVED.json')
    ports = [r['port'] for r in inventory]
    require(len(ports) == len(set(ports)) == 23, 'expected 23 unique ports')
    equal(ports, [r['port'] for r in read_json(FROZEN / 'POLICIES_SAVED.json')], 'frozen port inventory')
    require(complete['source_episodes'] == sum(r['episodes'] for r in inventory)
            and complete['source_gate_fits'] == 46 and complete['new_target_fits'] == 0
            and complete['new_backbone_pca_or_original_head_fits'] == 0
            and complete['policy_metric_rows'] == 1380, 'completion receipt')
    manifest = load_manifest(Path(config['manifest']))
    expected_rows, maximum_error = {}, 0.
    for index, item in enumerate(inventory):
        port = item['port']
        original = archive(ROOT / 'pred_cache' / (port + '.npz'))
        require(item['queries'] == len(original['qpos']), port + ': query inventory')
        receipt = read_json(run / port / 'source_receipt.json')
        require(receipt == item, port + ': frozen receipt changed')
        oof = verify_oof(port, original, manifest)
        bank = archive(run / port / 'source_episodes.npz')
        verify_bank(port, bank, oof, receipt)
        model = archive(run / port / 'source_models.npz')
        verify_model(port, model, bank, receipt, refit=index == 0)
        saved = archive(run / port / 'policies.npz')
        maximum_error = max(maximum_error, verify_targets(port, original, saved, model, expected_rows))
        reported = read_json(run / port / 'metrics.json')
        require(len(reported) == 60, port + ': metric count')
        seen = set()
        for row in reported:
            key = metric_key(row)
            require(key[0] == port and key in expected_rows and key not in seen, port + ': metric identity')
            seen.add(key)
            check_metrics(row, expected_rows[key], str(key), MEASURES + COUNTS)
        print(f'VERIFIED {port}: OOF splits, source episodes, models, target policies and controls', flush=True)
    with (run / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(expected_rows) == 1380, 'CSV coverage')
    seen = set()
    for row in rows:
        key = metric_key(row)
        require(key in expected_rows and key not in seen, 'CSV identity')
        seen.add(key)
        for metric in MEASURES + COUNTS:
            close(float(row[metric]), expected_rows[key][metric], f'CSV {key}/{metric}')
    verdict, primary = verify_summary(run, ports, inventory, expected_rows)
    return dict(status='verified', ports=23, contexts=115, policy_metric_rows=1380,
                source_episodes=sum(r['episodes'] for r in inventory), independent_gate_fits=2,
                contrasts_replayed=15, unchanged_control_policies=690,
                maximum_gate_score_replay_error=maximum_error,
                passes_development_screen=verdict, primary_delta_ba_pp=primary, files_written=0,
                limitation='OOF logits and visual residual scores are trusted archives; head splits are replayed. '
                           'Only the first port has independent coefficient refits. Artifact replay does not '
                           'prove historical freeze timing. Adaptation/query isolation is row-only.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'oof_episodic_gain_results_20261007')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = verify(args.run)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
