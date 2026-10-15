"""Train adapted-expert reliability on source ports, freeze, then evaluate."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import budgeted_residual_handoff as budgeted
import episodic_gain_gate as method
from episodic_bias_receiver import make_episode, adapt_bias
import run_oof_reliability as old
from run_budgeted_residual_handoff import ROOT, PARENT, archive, require, write_json


FROZEN = ROOT/'oof_receiver_consistent_results_20261007'
MANIFEST = Path('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv')
ARMS = ('matched_binary', 'gain_3way', 'frozen_bias', 'frozen_visual')
CONTRASTS = (('gain_3way', 'frozen_visual'), ('gain_3way', 'matched_binary'),
             ('gain_3way', 'frozen_bias'), ('matched_binary', 'frozen_visual'),
             ('matched_binary', 'frozen_bias'))
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'admission_fraction', 'change_fraction')


def build_bank(oof):
    ports = sorted(np.unique(oof['source_ports']))
    available = [p for p in ports if np.count_nonzero(oof['source_ports'] == p) > 64]
    skipped = [str(p) for p in ports if p not in available]
    columns = {name: [] for name in ('query_rows', 'class_ids', 'features', 'utility', 'eligible',
                                    'weights', 'base_prediction', 'receiver_prediction')}
    episode_ports, episode_seeds, support_rows, biases, offsets = [], [], [], [], [0]
    for port in available:
        positions = np.flatnonzero(oof['source_ports'] == port)
        global_rows = oof['source_rows'][positions]
        base_logits, expert_logits = oof['oof_base_logits'][positions], oof['oof_expert_logits'][positions]
        labels = oof['source_labels'][positions]
        for seed in range(5):
            support = np.random.default_rng(seed).choice(global_rows, 64, replace=False)
            local = np.searchsorted(global_rows, support)
            require(np.array_equal(global_rows[local], support), 'source support mapping mismatch')
            adapted, bias, ev, observed_ids = make_episode(base_logits, expert_logits, labels, local)
            observed = np.zeros(8, dtype=bool); observed[observed_ids] = True
            b, e = base_logits.argmax(1), adapted.argmax(1)
            y = labels[ev]
            _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
            values = dict(query_rows=global_rows[ev], class_ids=y,
                features=old.confidence_features(base_logits, adapted)[ev],
                utility=old.utility(y, b[ev], e[ev]).astype(np.int8),
                eligible=((b != e) & observed[e])[ev],
                weights=1./(len(available)*5*len(counts)*counts[inverse]),
                base_prediction=b[ev], receiver_prediction=e[ev])
            for name in columns:
                columns[name].append(values[name])
            episode_ports.append(port); episode_seeds.append(seed)
            support_rows.append(support); biases.append(bias); offsets.append(offsets[-1]+len(y))
    require(bool(available), 'no source port supports 64-label episodes')
    bank = {name: np.concatenate(parts) for name, parts in columns.items()}
    bank.update(source_rows=oof['source_rows'], episode_ports=np.asarray(episode_ports),
        episode_seeds=np.asarray(episode_seeds), support_rows=np.stack(support_rows),
        biases=np.stack(biases), query_offsets=np.asarray(offsets), skipped_ports=np.asarray(skipped, dtype=str))
    require(np.isclose(bank['weights'].sum(), 1.), 'episode weights not normalized')
    return bank


def freeze(out):
    inventory = json.loads((FROZEN/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    require(len(inventory) == 23, 'expected 23 target ports')
    receipts = []
    for item in inventory:
        port = item['port']
        path = ROOT/'pred_cache'/(port+'.npz')
        d = archive(path)
        oof = old.load_oof(ROOT/'oof_predictions_v1'/path.name, d, path, port, MANIFEST)
        bank = build_bank(oof)
        require(not np.isin(bank['query_rows'], d['qpos']).any()
                and not np.isin(bank['support_rows'], d['qpos']).any(), 'outer target enters source episodes')
        model = method.fit_models(bank['features'], bank['utility'], bank['eligible'], bank['weights'])
        directory = out/port; directory.mkdir()
        np.savez_compressed(directory/'source_episodes.npz', **bank)
        np.savez_compressed(directory/'source_models.npz', center=model['center'], scale=model['scale'],
                            **{name+'_'+key: value for name, fit in model['models'].items()
                               for key, value in fit.items() if key in ('coef', 'intercept', 'classes')})
        frozen = archive(FROZEN/port/'policies.npz')
        residual = archive(PARENT/port/'policies.npz')
        q = d['qpos']; base = d['base_logits'][q].argmax(1)
        require(np.array_equal(q, frozen['query_rows']) and np.array_equal(q, residual['query_rows']), 'query mismatch')
        require(np.array_equal(frozen['support_rows'], residual['support_rows']), 'support mismatch')
        ei_bias = list(frozen['expert_names']).index('anchor_bias64')
        ei_visual = list(frozen['expert_names']).index('anchor_visual64')
        si_bias = list(residual['score_names']).index('anchor_bias64')
        scores, probabilities, masks, predictions, receivers = [], [], [], [], []
        for i in range(5):
            support, ev = frozen['support_rows'][i], frozen['eval_masks'][i]
            local = np.searchsorted(q, support)
            require(np.array_equal(q[local], support), 'target support mapping')
            adapted, _ = adapt_bias(d['expert_logits'][q], local, d['y_ids'][support])
            expected = residual['scores'][i, si_bias]
            require(np.allclose(adapted, expected, atol=1e-12, rtol=1e-12), 'target bias scores changed')
            receiving = expected.argmax(1)
            require(np.array_equal(receiving, frozen['expert_predictions'][i, ei_bias]), 'target expert predictions changed')
            score, probability = method.predict_scores(model, old.confidence_features(d['base_logits'][q], expected))
            observed = np.bincount(d['y_ids'][support], minlength=8) > 0
            eligible = ev & (base != receiving) & observed[receiving]
            selected = np.stack([np.stack([budgeted.ranked_mask(score[name], eligible & (score[name] > 0), q, cap)
                                             for cap in frozen['caps'][i]]) for name in ARMS[:2]], axis=1)
            selected = np.concatenate((selected, np.take(frozen['selected'][i], [ei_bias, ei_visual], axis=1)), axis=1)
            receiver = np.stack((receiving, receiving, receiving, frozen['expert_predictions'][i, ei_visual]))
            pred = np.where(selected, receiver[None], base[None, None])
            require(np.array_equal(pred[:, 2], frozen['predictions'][i, :, ei_bias])
                    and np.array_equal(pred[:, 3], frozen['predictions'][i, :, ei_visual]), 'frozen control changed')
            scores.append(np.stack([score[name] for name in ARMS[:2]]))
            probabilities.append(probability['gain_3way']); masks.append(selected)
            predictions.append(pred); receivers.append(receiver)
        np.savez_compressed(directory/'policies.npz', query_rows=q, support_rows=frozen['support_rows'],
            eval_masks=frozen['eval_masks'], caps=frozen['caps'], seeds=np.arange(5), budgets=np.asarray(budgeted.BUDGETS),
            arm_names=np.asarray(ARMS), score_names=np.asarray(ARMS[:2]), scores=np.stack(scores),
            gain_probabilities=np.stack(probabilities), selected=np.stack(masks), predictions=np.stack(predictions),
            receiver_predictions=np.stack(receivers), base_prediction=base)
        receipt = dict(port=port, queries=len(q), episodes=len(bank['episode_ports']),
            episode_query_rows=len(bank['query_rows']), source_ports=len(np.unique(bank['episode_ports'])),
            skipped_source_ports=bank['skipped_ports'].tolist(),
            eligible_source_outcome_counts={str(k): int(np.count_nonzero(bank['eligible'] & (bank['utility'] == k))) for k in (-1, 0, 1)},
            fits={name: {key: fit[key] for key in ('training_rows', 'iterations')} for name, fit in model['models'].items()})
        write_json(directory/'source_receipt.json', receipt)
        receipts.append(receipt)
        print(f'FROZEN {port}: {receipt["episodes"]} source episodes, {receipt["episode_query_rows"]} rows', flush=True)
    write_json(out/'POLICIES_SAVED.json', receipts)
    return receipts


def evaluate(out, inventory):
    rows = []
    for item in inventory:
        port = item['port']
        d = archive(ROOT/'pred_cache'/(port+'.npz'))
        p = archive(out/port/'policies.npz')
        parent = json.loads((FROZEN/port/'metrics.json').read_text(encoding='utf-8'))
        base, y = p['base_prediction'], d['y_ids'][p['query_rows']]
        port_rows = []
        for i in range(5):
            ev = p['eval_masks'][i]
            for bi, budget in enumerate(budgeted.BUDGETS):
                for ai, arm in enumerate(ARMS):
                    selected, pred = p['selected'][i, bi, ai], p['predictions'][i, bi, ai]
                    admitted = int(selected.sum()); changes = int(np.count_nonzero((pred != base) & ev))
                    require(not selected[~ev].any() and changes == admitted <= p['caps'][i, bi], 'budget/support mismatch')
                    record = dict(port=port, seed=i, budget=budget, arm=arm,
                        **budgeted.score_policy(pred, base, y, ev), evaluated_rows=int(ev.sum()),
                        admitted_rows=admitted, changed_rows=changes, cap=int(p['caps'][i, bi]),
                        admission_fraction=admitted/ev.sum(), change_fraction=changes/ev.sum())
                    if arm.startswith('frozen_'):
                        expert = 'anchor_bias64' if arm == 'frozen_bias' else 'anchor_visual64'
                        previous = next(r for r in parent if r['seed'] == i and r['budget'] == budget and r['expert'] == expert)
                        require(abs(record['ba']-previous['ba']) < 1e-12 and abs(record['acc']-previous['acc']) < 1e-12,
                                'frozen comparator result changed')
                    port_rows.append(record)
        write_json(out/port/'metrics.json', port_rows); rows.extend(port_rows)
    with (out/'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'class_recall'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    ports = [r['port'] for r in inventory]
    values, means = {}, []
    for budget in budgeted.BUDGETS:
        for arm in ARMS:
            block = [r for r in rows if r['budget'] == budget and r['arm'] == arm]
            values[(budget, arm)] = {m: np.array([[r[m] for r in block if r['port'] == port] for port in ports]) for m in MEASURES}
            means.append(dict(budget=budget, arm=arm, **{m: float(v.mean()) for m, v in values[(budget, arm)].items()}))
    sampled = np.random.default_rng(0).integers(0, 23, (10000, 23))
    comparisons = []
    for budget in budgeted.BUDGETS:
        for a, b in CONTRASTS:
            delta = values[(budget, a)]['ba']-values[(budget, b)]['ba']; by_port = delta.mean(1)
            comparisons.append(dict(budget=budget, candidate=a, control=b, mean_delta_ba_pp=float(100*by_port.mean()),
                mean_delta_acc_pp=float(100*(values[(budget, a)]['acc']-values[(budget, b)]['acc']).mean()),
                positive_ports=int((by_port > 1e-12).sum()), negative_ports=int((by_port < -1e-12).sum()),
                port_bootstrap_95ci_pp=(100*np.percentile(by_port[sampled].mean(1), [2.5, 97.5])).tolist(),
                draw_deltas_pp=(100*delta.mean(0)).tolist()))
    primary = [r for r in comparisons if r['budget'] == .05 and r['candidate'] == 'gain_3way'
               and r['control'] in ('frozen_visual', 'matched_binary')]
    summary = dict(n_ports=23, contexts=115, policy_metric_rows=len(rows), means=means, comparisons=comparisons,
        source_episodes=sum(r['episodes'] for r in inventory), new_gate_fits=46,
        passes_development_screen=all(r['mean_delta_ba_pp'] > 0 for r in primary),
        independent_confirmation=False, novelty_established=False,
        primary='5% gain_3way minus frozen_visual AND minus matched_binary',
        cost='admission cap; expert scores accessed before selection, no measured compute savings',
        statistical_unit='port mean over five overlapping support draws')
    write_json(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); require(not args.out.exists(), 'use a new output directory'); args.out.mkdir(parents=True)
    write_json(args.out/'RUN_CONFIG.json', dict(date='2026-10-07', arms=ARMS, contrasts=CONTRASTS,
        labels=64, budgets=budgeted.BUDGETS, adaptation_strength=8., logistic_C=1., source_seeds=list(range(5)),
        frozen=str(FROZEN), manifest=str(MANIFEST), protocol=str(Path(__file__).with_name('EPISODIC_GAIN_PROTOCOL_20261007.md'))))
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        inventory = freeze(args.out)
        summary = evaluate(args.out, inventory)
    write_json(args.out/'COMPLETE.json', dict(status='complete', seconds=time.perf_counter()-start,
        source_episodes=summary['source_episodes'], source_gate_fits=46, new_target_fits=0,
        new_backbone_pca_or_original_head_fits=0, policy_metric_rows=len(inventory)*5*3*4))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
