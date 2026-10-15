"""One fixed, training-free action-pair ranking control on adapted experts."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax

import budgeted_residual_handoff as budgeted
from run_budgeted_residual_handoff import ROOT, PARENT, archive, require, write_json


FROZEN = ROOT/'oof_receiver_consistent_results_20261007'
ARMS = ('advantage_bias', 'advantage_visual', 'frozen_bias', 'frozen_visual')
EXPERTS = ('anchor_bias64', 'anchor_visual64')
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'admission_fraction')


def action_advantage(receiver_scores, base_prediction):
    probability = softmax(receiver_scores, axis=1)
    row = np.arange(len(probability))
    return probability.max(1)-probability[row, base_prediction]


def freeze(out):
    inventory = json.loads((FROZEN/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    for item in inventory:
        port = item['port']
        frozen = archive(FROZEN/port/'policies.npz')
        residual = archive(PARENT/port/'policies.npz')
        original = archive(ROOT/'pred_cache'/(port+'.npz'))
        q, base = frozen['query_rows'], frozen['base_prediction']
        require(np.array_equal(q, residual['query_rows']) and np.array_equal(q, original['qpos']), 'query mismatch')
        require(np.array_equal(frozen['support_rows'], residual['support_rows']), 'support mismatch')
        bi = list(frozen['budgets']).index(.05)
        score_indices = [list(residual['score_names']).index(e) for e in EXPERTS]
        expert_indices = [list(frozen['expert_names']).index(e) for e in EXPERTS]
        masks, scores, predictions, receivers = [], [], [], []
        for i in range(5):
            ev, support = frozen['eval_masks'][i], frozen['support_rows'][i]
            require(np.array_equal(ev, ~np.isin(q, support)), 'support exclusion mismatch')
            cap = int(frozen['caps'][i, bi])
            require(cap == int(np.floor(.05*ev.sum())), 'cap mismatch')
            observed = np.bincount(original['y_ids'][support], minlength=8) > 0
            receiving = np.take(frozen['expert_predictions'][i], expert_indices, axis=0)
            sc, selected = [], []
            for j, si in enumerate(score_indices):
                logits = residual['scores'][i, si]
                require(np.array_equal(logits.argmax(1), receiving[j]), 'receiver mismatch')
                score = action_advantage(logits, base)
                require(np.isfinite(score).all(), 'nonfinite advantage')
                eligible = ev & (base != receiving[j]) & observed[receiving[j]] & (score > 0)
                sc.append(score)
                selected.append(budgeted.ranked_mask(score, eligible, q, cap))
            mask = np.concatenate((np.stack(selected), np.take(frozen['selected'][i, bi], expert_indices, axis=0)))
            receiver = np.concatenate((receiving, receiving))
            pred = np.where(mask, receiver, base[None])
            require(np.array_equal(pred[2:], np.take(frozen['predictions'][i, bi], expert_indices, axis=0)), 'frozen control changed')
            scores.append(np.stack(sc)); masks.append(mask); predictions.append(pred); receivers.append(receiver)
        directory = out/port
        directory.mkdir()
        np.savez_compressed(directory/'policies.npz', query_rows=q, support_rows=frozen['support_rows'],
            eval_masks=frozen['eval_masks'], caps=frozen['caps'][:, bi], seeds=frozen['seeds'],
            arm_names=np.asarray(ARMS), scores=np.stack(scores), selected=np.stack(masks),
            predictions=np.stack(predictions), receiver_predictions=np.stack(receivers), base_prediction=base)
        print('FROZEN '+port, flush=True)
    write_json(out/'POLICIES_SAVED.json', inventory)
    return inventory


def evaluate(out, inventory):
    rows = []
    for item in inventory:
        port = item['port']
        p = archive(out/port/'policies.npz')
        original = archive(ROOT/'pred_cache'/(port+'.npz'))
        previous = json.loads((FROZEN/port/'metrics.json').read_text(encoding='utf-8'))
        base, y = p['base_prediction'], original['y_ids'][p['query_rows']]
        block = []
        for i in range(5):
            ev = p['eval_masks'][i]
            for ai, arm in enumerate(ARMS):
                selected, pred = p['selected'][i, ai], p['predictions'][i, ai]
                admissions = int(selected.sum())
                changes = int(np.count_nonzero((pred != base) & ev))
                require(not selected[~ev].any() and admissions == changes <= p['caps'][i], 'invalid admission')
                record = dict(port=port, seed=i, budget=.05, arm=arm, **budgeted.score_policy(pred, base, y, ev),
                    evaluated_rows=int(ev.sum()), admitted_rows=admissions, changed_rows=changes,
                    cap=int(p['caps'][i]), admission_fraction=float(admissions/ev.sum()))
                if ai >= 2:
                    parent = next(r for r in previous if r['seed'] == i and r['budget'] == .05 and r['expert'] == EXPERTS[ai-2])
                    require(all(abs(record[m]-parent[m]) < 1e-12 for m in MEASURES), 'frozen metric changed')
                block.append(record)
        write_json(out/port/'metrics.json', block); rows.extend(block)
    with (out/'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'class_recall'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    ports = [item['port'] for item in inventory]
    values = {arm: {m: np.array([[next(r[m] for r in rows if r['arm'] == arm and r['port'] == port and r['seed'] == i)
                                  for i in range(5)] for port in ports]) for m in MEASURES} for arm in ARMS}
    means = [dict(arm=arm, **{m: float(v.mean()) for m, v in values[arm].items()}) for arm in ARMS]
    sampled = np.random.default_rng(0).integers(0, len(ports), (10000, len(ports)))
    comparisons = []
    for a in ARMS[:2]:
        for b in ARMS[2:]:
            delta = values[a]['ba']-values[b]['ba']; by_port = delta.mean(1)
            comparisons.append(dict(candidate=a, control=b, mean_delta_ba_pp=float(100*delta.mean()),
                mean_delta_acc_pp=float(100*(values[a]['acc']-values[b]['acc']).mean()),
                positive_ports=int((by_port > 1e-12).sum()), negative_ports=int((by_port < -1e-12).sum()),
                port_bootstrap_95ci_pp=(100*np.percentile(by_port[sampled].mean(1), [2.5, 97.5])).tolist(),
                draw_deltas_pp=(100*delta.mean(0)).tolist(), port_deltas_pp=dict(zip(ports, (100*by_port).tolist()))))
    primary = next(r for r in comparisons if r['candidate'] == 'advantage_visual' and r['control'] == 'frozen_visual')
    summary = dict(n_ports=len(ports), contexts=len(ports)*5, budget=.05, policy_metric_rows=len(rows),
        means=means, comparisons=comparisons, passes_development_screen=primary['mean_delta_ba_pp'] > 0,
        primary='5% advantage_visual minus frozen_visual', independent_confirmation=False, novelty_established=False,
        no_fitting=True, cost='admission cap only; all expert scores accessed before selection')
    write_json(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'use a new output directory')
    args.out.mkdir(parents=True)
    write_json(args.out/'RUN_CONFIG.json', dict(date='2026-10-07', labels=64, budget=.05, temperature=1., arms=ARMS,
        protocol=str(Path(__file__).with_name('ACTION_ADVANTAGE_PROTOCOL_20261007.md'))))
    start = time.perf_counter()
    inventory = freeze(args.out)
    summary = evaluate(args.out, inventory)
    write_json(args.out/'COMPLETE.json', dict(status='complete', seconds=time.perf_counter()-start, new_fits=0,
        policy_metric_rows=summary['policy_metric_rows']))
    print(json.dumps({k: v for k, v in summary.items() if k != 'comparisons'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
