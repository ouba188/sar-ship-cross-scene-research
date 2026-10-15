"""Read-only independent replay of the fixed residual expert experiment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent.parent
ARMS = ('base', 'fusion', 'oof_lr0', 'legacy_target64', 'target64',
        'anchor_bias64', 'anchor_visual64')
NEW_ARMS = ARMS[3:]
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'rescue_both_wrong_ba')
CONTRASTS = (('anchor_visual64', 'fusion'), ('anchor_visual64', 'anchor_bias64'),
             ('anchor_visual64', 'oof_lr0'), ('anchor_visual64', 'target64'),
             ('anchor_visual64', 'legacy_target64'), ('anchor_bias64', 'fusion'))


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def close(actual, expected, context, atol=1e-12, rtol=1e-10):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol,
                               err_msg=context)


def confusion_metrics(truth, base, fusion, prediction):
    # Counts retain the full evaluation class denominator for every rescue subset.
    joint = np.zeros((8, 8, 8, 8), dtype=np.int64)
    np.add.at(joint, (truth, base, fusion, prediction), 1)
    matrix = joint.sum(axis=(1, 2))
    counts = matrix.sum(axis=1)
    present = counts > 0
    recall = np.divide(matrix.diagonal(), counts, out=np.zeros(8), where=present)
    recovery, harm, both = np.zeros((3, 8))
    for c in np.flatnonzero(present):
        others = np.flatnonzero(np.arange(8) != c)
        block = joint[c]
        recovery[c] = block[:, others, c].sum() / counts[c]
        harm[c] = (block[:, c, :].sum() - block[:, c, c].sum()) / counts[c]
        both[c] = block[np.ix_(others, others, [c])].sum() / counts[c]
    same_as_fusion = sum(joint[:, :, c, c].sum() for c in range(8))
    return dict(ba=float(recall[present].mean()), acc=float(matrix.trace()/matrix.sum()),
                recover_ba=float(recovery[present].mean()),
                harm_ba=float(harm[present].mean()),
                rescue_both_wrong_ba=float(both[present].mean()),
                changed_from_fusion=int(matrix.sum()-same_as_fusion),
                class_recall=[float(recall[c]) if present[c] else None for c in range(8)])


def verify(run):
    # Never open candidate policies or readouts before the full freeze marker.
    require((run/'POLICIES_SAVED.json').is_file(), 'await POLICIES_SAVED.json before replay')
    require((run/'COMPLETE.json').is_file(), 'await COMPLETE.json before metric replay')
    inventory = read_json(run/'POLICIES_SAVED.json')
    require(len(inventory) == len({r['port'] for r in inventory}) == 23,
            'expected 23 unique ports')
    config = read_json(run/'RUN_CONFIG.json')
    require(config['strength'] == 8 and config['support_labels'] == 64,
            'fixed strength/support changed')
    require(tuple(config['arms']) == ARMS, 'configured arms changed')
    active = []
    for path in sorted((ROOT/'pred_cache').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            if 'skipped' not in data.files and len(data['qpos']) > 64:
                active.append(path.stem)
    require([r['port'] for r in inventory] == active, 'outer-port inventory differs from cache')
    records, csv_expected = [], {}
    sklearn_fits = 0
    max_score_error = 0.
    for port_index, item in enumerate(inventory):
        port = item['port']
        data = archive(ROOT/'pred_cache'/f'{port}.npz')
        feature = archive(ROOT/'pred_cache2'/f'{port}.npz')
        saved = archive(run/port/'policies.npz')
        parent = archive(ROOT/'oof_context_results_20261007'/port/'policies.npz')
        old_metrics = read_json(ROOT/'oof_reliability_results_v1'/f'{port}.json')
        reported = read_json(run/port/'metrics.json')
        q, source, labels = data['qpos'], data['spos'], data['y_ids']
        require(np.array_equal(q, np.unique(q)), f'{port}: query order/duplicates')
        require(len(source) == len(np.unique(source)) and not np.isin(source, q).any(),
                f'{port}: source/query overlap or duplicated source')
        require(np.all(data['ports_all'][q] == port) and
                np.all(data['ports_all'][source] != port), f'{port}: outer split changed')
        require(np.array_equal(np.unique(labels[source]), np.arange(8)),
                f'{port}: source class columns')
        require(np.array_equal(saved['query_rows'], q) and
                np.array_equal(parent['query_rows'], q) and
                np.array_equal(saved['source_rows'], source), f'{port}: row alignment')
        require(tuple(saved['arm_names']) == ARMS and tuple(saved['score_names']) == NEW_ARMS,
                f'{port}: arm order')
        require(np.array_equal(saved['seeds'], np.arange(5)), f'{port}: support seeds')
        require(saved['scores'].shape == (5, 4, len(q), 8) and
                saved['predictions'].shape == (5, 7, len(q)), f'{port}: policy shape')
        require(np.isfinite(saved['scores']).all(), f'{port}: nonfinite scores')
        require(np.issubdtype(saved['predictions'].dtype, np.integer) and
                np.all((saved['predictions'] >= 0) & (saved['predictions'] < 8)),
                f'{port}: invalid predictions')
        require(feature['Xs'].shape == (len(labels), 128) and
                feature['Z'].shape == (len(labels), 64), f'{port}: visual dimensions')
        raw = np.concatenate((feature['Xs'], feature['Z']), axis=1)
        center, scale = raw[source].mean(0), raw[source].std(0)+1e-6
        close(saved['source_center'], center, f'{port}: source-only center')
        close(saved['source_scale'], scale, f'{port}: source-only scale')
        phi = np.column_stack(((raw[q]-center)/scale/np.sqrt(192), np.ones(len(q))))
        prior = data['expert_logits'][q].astype(float)
        base = data['base_logits'][q].argmax(1)
        fusion = prior.argmax(1)
        require(item['queries'] == len(q) and item['source_rows'] == len(source),
                f'{port}: inventory counts')
        require(0 <= item['source_head_max_error'] < 1e-4 and
                0 <= item['source_head_argmax_mismatch'] <= len(labels) and
                item['replay_near_ties_only'] is True and
                item['original_prior_scores_unchanged'] is True,
                f'{port}: producer source replay failed')
        require(reported['port'] == port and len(reported['seeds']) == 5,
                f'{port}: metric record alignment')
        port_values = []
        for i, seed in enumerate(saved['seeds']):
            context = f'{port}/{seed}'
            support = saved['support_rows'][i]
            expected = np.random.default_rng(int(seed)).choice(q, 64, replace=False)
            require(len(support) == len(np.unique(support)) == 64 and
                    np.array_equal(support, expected) and
                    np.array_equal(support, parent['support_rows'][i]) and
                    np.array_equal(support, old_metrics['seeds'][i]['annotation_rows']),
                    f'{context}: support budget/order/parent changed')
            ev = saved['eval_masks'][i]
            require(ev.dtype == np.bool_ and np.array_equal(ev, ~np.isin(q, support)) and
                    np.array_equal(ev, parent['eval_masks'][i]) and ev.sum() == len(q)-64,
                    f'{context}: evaluation must exclude exactly support')
            local = np.searchsorted(q, support)
            target = labels[support]
            counts = np.bincount(target, minlength=8)
            weights = 64/(np.count_nonzero(counts)*counts[target])
            desired = np.full((64, 8), -1.)
            desired[np.arange(64), target] = 1.
            preds = dict(zip(ARMS, saved['predictions'][i]))
            for name in ARMS[:3]:
                parent_idx = list(parent['arm_names']).index(name)
                require(np.array_equal(preds[name], parent['predictions'][i, parent_idx]),
                        f'{context}/{name}: inherited predictions changed')
            require(np.array_equal(preds['base'], base) and np.array_equal(preds['fusion'], fusion),
                    f'{context}: frozen source predictions changed')
            for j, name in enumerate(NEW_ARMS):
                score = saved['scores'][i, j]
                require(np.array_equal(preds[name], score.argmax(1)),
                        f'{context}/{name}: score-to-prediction mismatch')
                if name == 'legacy_target64':
                    reconstruction = raw[q] @ saved['legacy_coef'][i].T + saved['legacy_intercept'][i]
                    close(score, reconstruction, f'{context}/{name}: score replay', atol=2e-5, rtol=2e-5)
                else:
                    x = phi[:, -1:] if name == 'anchor_bias64' else phi
                    anchor = np.zeros_like(prior) if name == 'target64' else prior
                    delta = saved[name][i]
                    require(delta.shape == (x.shape[1], 8), f'{context}/{name}: correction shape')
                    reconstruction = anchor+x@delta
                    close(score, reconstruction, f'{context}/{name}: prior plus correction')
                    xa = x[local]
                    rhs = xa.T @ (weights[:, None]*(desired-anchor[local]))
                    lhs = xa.T @ (weights[:, None]*(xa@delta)) + 8*delta
                    close(lhs, rhs, f'{context}/{name}: support-only ridge equation', atol=1e-9)
                max_score_error = max(max_score_error, float(np.max(np.abs(score-reconstruction))))
                # One small support draw uses sklearn's SVD solver as an independent fit.
                if port_index == 0 and i == 0:
                    if name == 'legacy_target64':
                        model = Ridge(alpha=1., solver='svd').fit(
                            raw[q][local], np.eye(8)[target], sample_weight=weights/64)
                        reference = model.predict(raw[q])
                    else:
                        model = Ridge(alpha=8., fit_intercept=False, solver='svd').fit(
                            xa, desired-anchor[local], sample_weight=weights)
                        reference = anchor+model.predict(x)
                    close(score, reference, f'{context}/{name}: independent sklearn fit',
                          atol=2e-5 if name == 'legacy_target64' else 1e-9,
                          rtol=2e-5 if name == 'legacy_target64' else 1e-9)
                    require(np.array_equal(score.argmax(1), reference.argmax(1)),
                            f'{context}/{name}: independent fit changes argmax')
                    sklearn_fits += 1
            seed_record = reported['seeds'][i]
            require(seed_record['seed'] == int(seed), f'{context}: metric seed')
            metrics = {}
            for name, pred in preds.items():
                result = confusion_metrics(labels[q][ev], base[ev], fusion[ev], pred[ev])
                metrics[name] = result
                claimed = seed_record['metrics'][name]
                for key in MEASURES + ('changed_from_fusion',):
                    close(claimed[key], result[key], f'{context}/{name}/{key}')
                for c, recall in enumerate(result['class_recall']):
                    if recall is None:
                        require(claimed['class_recall'][c] is None, f'{context}/{name}: absent-class recall')
                    else:
                        close(claimed['class_recall'][c], recall, f'{context}/{name}/recall{c}')
                if name in ARMS[:3]:
                    for key in ('ba', 'acc'):
                        close(result[key], old_metrics['seeds'][i]['metrics'][name][key],
                              f'{context}/{name}: parent {key}')
                csv_expected[(port, int(seed), name)] = dict(result, evaluated_rows=int(ev.sum()))
            for name in ARMS:
                close(metrics[name]['ba']-metrics['fusion']['ba'],
                      metrics[name]['recover_ba']-metrics[name]['harm_ba'],
                      f'{context}/{name}: full-query recovery/harm identity')
            port_values.append(metrics)
        records.append(port_values)
        print(f'VERIFIED {port}: five supports, geometry, score replay, confusion counts', flush=True)

    with (run/'metrics.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == len(csv_expected), 'CSV row count')
    seen = set()
    for row in rows:
        identity = (row['port'], int(row['seed']), row['arm'])
        require(identity in csv_expected and identity not in seen, 'CSV duplicate/unknown row')
        seen.add(identity)
        for key in MEASURES + ('changed_from_fusion', 'evaluated_rows'):
            close(float(row[key]), csv_expected[identity][key], f'CSV {identity}/{key}')
    summary = read_json(run/'summary.json')
    require(summary['n_ports'] == 23 and summary['contexts'] == 115, 'summary coverage')
    values = {name: {metric: np.array([[s[name][metric] for s in p] for p in records])
                     for metric in MEASURES} for name in ARMS}
    for name in ARMS:
        for metric in MEASURES:
            close(summary['means'][name][metric], values[name][metric].mean(axis=1).mean(),
                  f'summary equal-port mean {name}/{metric}')
    sampled = np.random.default_rng(0).integers(0, 23, (10000, 23))
    primary_positive = []
    for a, b in CONTRASTS:
        delta = (values[a]['ba']-values[b]['ba']).mean(axis=1)
        comparison = summary['comparisons'][a+' minus '+b]
        expected = dict(mean_delta_ba_pp=100*delta.mean(),
                        mean_delta_acc_pp=100*(values[a]['acc']-values[b]['acc']).mean(axis=1).mean(),
                        positive_ports=int((delta > 1e-12).sum()),
                        negative_ports=int((delta < -1e-12).sum()),
                        port_bootstrap_95ci_pp=100*np.percentile(delta[sampled].mean(axis=1), [2.5, 97.5]),
                        draw_deltas_pp=100*(values[a]['ba']-values[b]['ba']).mean(axis=0))
        for key, value in expected.items():
            close(comparison[key], value, f'comparison {a}/{b}/{key}')
        primary_positive.append(delta.mean() > 0)
    require(summary['passes_development_screen'] == bool(all(primary_positive[:2])),
            'screen verdict differs from fixed two primary contrasts')
    return dict(status='verified', ports=23, contexts=115, policy_metric_rows=len(rows),
                independent_support_fits=sklearn_fits, maximum_score_replay_error=max_score_error,
                source_heads_retrained=False, files_written=0,
                limitation='Inherited PCA and cached source scores are trusted; source geometry is independently replayed. '
                           'Source-head agreement uses the producer receipt. Target split is row-disjoint, not ship-disjoint.')


def verify_coverage(run):
    out = run/'coverage_control'
    require((run/'POLICIES_SAVED.json').is_file() and
            (out/'POLICIES_SAVED.json').is_file(), 'await both policy freeze markers')
    inventory = read_json(run/'POLICIES_SAVED.json')
    require(len(inventory) == len({r['port'] for r in inventory}) == 23,
            'coverage expected 23 unique ports')
    require(read_json(out/'POLICIES_SAVED.json')['ports'] == 23, 'coverage freeze count')
    reported = read_json(out/'metrics.json')
    summary = read_json(out/'summary.json')
    require(summary['post_readout_followup'] is True and summary['no_fitting'] is True,
            'coverage must remain a post-readout, no-fit follow-up')
    require(summary['n_ports'] == 23 and summary['contexts'] == 115, 'coverage summary count')
    require([r['port'] for r in reported] == [r['port'] for r in inventory],
            'coverage metric port alignment')
    arms = ('fusion', 'oof_lr0', 'anchor_bias64', 'anchor_visual64', 'fusion_observed64')
    records = []
    for item, report in zip(inventory, reported):
        port = item['port']
        cache = archive(ROOT/'pred_cache'/f'{port}.npz')
        parent = archive(run/port/'policies.npz')
        control = archive(out/f'{port}.npz')
        q, labels = cache['qpos'], cache['y_ids']
        require(np.array_equal(control['query_rows'], q) and
                np.array_equal(parent['query_rows'], q) and
                np.array_equal(control['support_rows'], parent['support_rows']),
                f'{port}: coverage row alignment')
        require(control['predictions'].shape == (5, len(q)) and control['seen'].shape == (5, 8)
                and control['seen'].dtype == np.bool_ and len(report['seeds']) == 5,
                f'{port}: coverage policy dimensions')
        base = cache['base_logits'][q].argmax(1)
        fusion = cache['expert_logits'][q].argmax(1)
        port_values = []
        for i in range(5):
            support = control['support_rows'][i]
            expected_support = np.random.default_rng(i).choice(q, 64, replace=False)
            require(np.array_equal(support, expected_support), f'{port}/{i}: coverage support changed')
            ev = ~np.isin(q, support)
            require(np.array_equal(ev, parent['eval_masks'][i]) and ev.sum() == len(q)-64,
                    f'{port}/{i}: coverage evaluation exclusion')
            observed = np.unique(labels[support])
            require(np.array_equal(np.flatnonzero(control['seen'][i]), observed),
                    f'{port}/{i}: observed classes must use support labels only')
            # Select within observed columns directly, independently of the -inf implementation.
            expected = observed[cache['expert_logits'][q][:, observed].argmax(axis=1)]
            require(np.array_equal(control['predictions'][i], expected),
                    f'{port}/{i}: restricted fusion argmax differs')
            predictions = dict(zip(parent['arm_names'], parent['predictions'][i]))
            predictions['fusion_observed64'] = control['predictions'][i]
            seed_values = {}
            for arm in arms:
                result = confusion_metrics(labels[q][ev], base[ev], fusion[ev], predictions[arm][ev])
                claimed = report['seeds'][i][arm]
                for metric in MEASURES + ('changed_from_fusion',):
                    close(claimed[metric], result[metric], f'coverage {port}/{i}/{arm}/{metric}')
                for c, recall in enumerate(result['class_recall']):
                    if recall is None:
                        require(claimed['class_recall'][c] is None, 'coverage absent-class recall')
                    else:
                        close(claimed['class_recall'][c], recall, f'coverage {port}/{i}/{arm}/recall{c}')
                seed_values[arm] = result
            port_values.append(seed_values)
        records.append(port_values)
    values = {arm: {metric: np.array([[s[arm][metric] for s in port] for port in records]).mean(axis=1)
                    for metric in ('ba', 'acc')} for arm in arms}
    for arm in arms:
        for metric in ('ba', 'acc'):
            close(summary['means'][arm][metric], values[arm][metric].mean(),
                  f'coverage equal-port mean {arm}/{metric}')
    samples = np.random.default_rng(0).integers(0, 23, (10000, 23))
    for arm in arms[:-1]:
        delta = values[arm]['ba']-values['fusion_observed64']['ba']
        claimed = summary['contrasts'][arm+' minus fusion_observed64']
        expected = dict(mean_delta_ba_pp=100*delta.mean(), positive_ports=int((delta > 1e-12).sum()),
                        negative_ports=int((delta < -1e-12).sum()),
                        port_bootstrap_95ci_pp=100*np.percentile(delta[samples].mean(axis=1), [2.5, 97.5]))
        for key, value in expected.items():
            close(claimed[key], value, f'coverage comparison {arm}/{key}')
    return dict(status='verified', ports=23, contexts=115, policy_metric_rows=575,
                post_readout_followup=True, fits=0, files_written=0,
                coverage_ba=float(values['fusion_observed64']['ba'].mean()),
                visual_minus_coverage=summary['contrasts']['anchor_visual64 minus fusion_observed64'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'oof_residual_expert_results_20261007_v2')
    parser.add_argument('--coverage-only', action='store_true', help='Replay the added no-fit coverage control only')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.coverage_only:
            result = verify_coverage(args.run)
        else:
            result = verify(args.run)
            if (args.run/'coverage_control').is_dir():
                result['coverage_control'] = verify_coverage(args.run)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
