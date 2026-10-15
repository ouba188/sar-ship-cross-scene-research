"""Fixed input-consistency follow-up after the same-mask budget screen."""

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np

import budgeted_residual_handoff as budgeted
import receiver_consistent_handoff as method
from run_budgeted_residual_handoff import ROOT, PARENT, GATE, archive, require, write_json


STALE = ROOT/'oof_budgeted_residual_results_20261007'
CONTROLS = ('fusion', 'anchor_bias64', 'fusion_observed64', 'stale_visual')


def freeze(out):
    inventory = json.loads((STALE/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    for item in inventory:
        port = item['port']
        old = archive(STALE/port/'policies.npz')
        residual = archive(PARENT/port/'policies.npz')
        original = archive(ROOT/'pred_cache'/(port+'.npz'))
        with (GATE/port/'source_model.pkl').open('rb') as f:
            prior = pickle.load(f)['prior']
        q, base = old['query_rows'], old['base_prediction']
        require(np.array_equal(q, original['qpos']) and np.array_equal(q, residual['query_rows']), 'query mismatch')
        require(np.array_equal(old['support_rows'], residual['support_rows']), 'support mismatch')
        scores, masks, predictions = [], [], []
        for i in range(5):
            ev = old['eval_masks'][i]
            observed = np.bincount(original['y_ids'][old['support_rows'][i]], minlength=8) > 0
            score_blocks = {
                'fusion': original['expert_logits'][q],
                'fusion_observed64': original['expert_logits'][q],
                **dict(zip(residual['score_names'], residual['scores'][i]))}
            sc, mask = [], []
            for ei, expert in enumerate(budgeted.EXPERTS):
                receiving = old['expert_predictions'][i, ei]
                score = method.gate_scores(original['base_logits'][q], score_blocks[expert], prior,
                                            observed if expert == 'fusion_observed64' else None)
                if expert == 'fusion':
                    require(np.allclose(score, old['old_lr_score'], atol=1e-12, rtol=1e-12), 'original fusion gate differs')
                eligible = ev & (base != receiving) & observed[receiving] & (score > 0)
                selected = np.stack([budgeted.ranked_mask(score, eligible, q, cap) for cap in old['caps'][i]])
                if expert == 'fusion':
                    require(np.array_equal(selected, old['selected'][i, 0]), 'original fusion masks differ')
                sc.append(score); mask.append(selected)
            mask = np.stack(mask, axis=1)
            masks.append(mask); scores.append(np.stack(sc))
            predictions.append(np.where(mask, old['expert_predictions'][i][None], base[None, None]))
        directory = out/port
        directory.mkdir()
        np.savez_compressed(directory/'policies.npz', query_rows=q, support_rows=old['support_rows'],
            eval_masks=old['eval_masks'], caps=old['caps'], budgets=old['budgets'], seeds=old['seeds'],
            expert_names=old['expert_names'], expert_predictions=old['expert_predictions'],
            base_prediction=base, gate_scores=np.stack(scores), selected=np.stack(masks), predictions=np.stack(predictions),
            source_center=prior['center'], source_scale=prior['scale'], source_lr=prior['lr'])
        print('FROZEN '+port, flush=True)
    write_json(out/'POLICIES_SAVED.json', inventory)
    return inventory


def evaluate(out, inventory):
    rows = []
    for item in inventory:
        port = item['port']
        original = archive(ROOT/'pred_cache'/(port+'.npz'))
        data = archive(out/port/'policies.npz')
        prior_rows = json.loads((STALE/port/'metrics.json').read_text(encoding='utf-8'))
        y, base = original['y_ids'][data['query_rows']], data['base_prediction']
        port_rows = []
        for i in range(5):
            ev = data['eval_masks'][i]
            for bi, budget in enumerate(budgeted.BUDGETS):
                for ei, expert in enumerate(budgeted.EXPERTS):
                    pred = data['predictions'][i, bi, ei]
                    selected = data['selected'][i, bi, ei]
                    admissions, changes = int(selected.sum()), int(np.count_nonzero((pred != base) & ev))
                    require(not selected[~ev].any() and changes == admissions <= data['caps'][i, bi], 'budget/eligibility mismatch')
                    result = dict(port=port, seed=i, budget=budget, expert=expert,
                        **budgeted.score_policy(pred, base, y, ev), evaluated_rows=int(ev.sum()),
                        admitted_rows=admissions, changed_rows=changes, cap=int(data['caps'][i, bi]),
                        admission_fraction=admissions/ev.sum(), change_fraction=changes/ev.sum())
                    if expert == 'fusion':
                        parent = next(r for r in prior_rows if r['seed'] == i and r['selector'] == 'old_lr_cap'
                                      and r['budget'] == budget and r['expert'] == 'fusion')
                        require(abs(result['ba']-parent['ba']) < 1e-12, 'fusion control metric mismatch')
                    port_rows.append(result)
                parent = next(r for r in prior_rows if r['seed'] == i and r['selector'] == 'old_lr_cap'
                              and r['budget'] == budget and r['expert'] == 'anchor_visual64')
                port_rows.append({k: v for k, v in dict(parent, expert='stale_visual').items() if k != 'selector'})
        write_json(out/port/'metrics.json', port_rows)
        rows.extend(port_rows)
    with (out/'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        fields = [k for k in rows[0] if k != 'class_recall']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    ports = [r['port'] for r in inventory]
    values, means = {}, []
    for budget in budgeted.BUDGETS:
        for expert in budgeted.EXPERTS+('stale_visual',):
            block = [r for r in rows if r['budget'] == budget and r['expert'] == expert]
            values[(budget, expert)] = {m: np.array([[r[m] for r in block if r['port'] == port] for port in ports])
                                        for m in ('ba', 'acc', 'recover_ba', 'harm_ba', 'admission_fraction', 'change_fraction')}
            means.append(dict(budget=budget, expert=expert,
                              **{m: float(v.mean()) for m, v in values[(budget, expert)].items()}))
    sampled = np.random.default_rng(0).integers(0, 23, (10000, 23))
    comparisons = []
    for budget in budgeted.BUDGETS:
        a = values[(budget, 'anchor_visual64')]
        for control in CONTROLS:
            b = values[(budget, control)]
            delta = a['ba']-b['ba']; by_port = delta.mean(1)
            comparisons.append(dict(budget=budget, candidate='anchor_visual64', control=control,
                mean_delta_ba_pp=float(100*by_port.mean()), mean_delta_acc_pp=float(100*(a['acc']-b['acc']).mean()),
                positive_ports=int((by_port > 1e-12).sum()), negative_ports=int((by_port < -1e-12).sum()),
                port_bootstrap_95ci_pp=(100*np.percentile(by_port[sampled].mean(1), [2.5, 97.5])).tolist(),
                draw_deltas_pp=(100*delta.mean(0)).tolist()))
    primary = [r for r in comparisons if r['budget'] == .05 and r['control'] in ('stale_visual', 'anchor_bias64')]
    summary = dict(n_ports=23, contexts=115, policy_metric_rows=len(rows), new_policy_metric_rows=23*5*3*4,
        means=means, comparisons=comparisons, passes_development_screen=all(r['mean_delta_ba_pp'] > 0 for r in primary),
        independent_confirmation=False, novelty_established=False, no_fitting=True,
        primary='5% consistent visual minus stale visual AND minus consistent bias',
        caveat='frozen source gate applied to changed expert score distributions, not newly calibrated OOF reliability',
        cost='admission/replacement cap only; all receiver scores accessed before selection')
    write_json(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'use new output directory')
    args.out.mkdir(parents=True)
    write_json(args.out/'RUN_CONFIG.json', dict(date='2026-10-07', experts=budgeted.EXPERTS,
        budgets=budgeted.BUDGETS, stale=str(STALE), prior=str(GATE), labels=64,
        protocol=str(Path(__file__).with_name('RECEIVER_CONSISTENT_PROTOCOL_20261007.md'))))
    start = time.perf_counter()
    inventory = freeze(args.out)
    summary = evaluate(args.out, inventory)
    write_json(args.out/'COMPLETE.json', dict(status='complete', new_fits=0, seconds=time.perf_counter()-start,
        contexts=115, new_policy_metric_rows=1380, inherited_policy_metric_rows=345))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
