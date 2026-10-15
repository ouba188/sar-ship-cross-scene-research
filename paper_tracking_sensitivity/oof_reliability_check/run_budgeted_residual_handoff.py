"""Freeze same-mask budgeted policies, then evaluate all unchanged experts."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import budgeted_residual_handoff as method
from run_enriched_reliability import write_json
from run_residual_expert import ROOT


PARENT = ROOT/'oof_residual_expert_results_20261007_v2'
GATE = ROOT/'oof_context_results_20261007'
CONTRASTS = ('fusion', 'anchor_bias64', 'fusion_observed64')
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'admission_fraction', 'change_fraction')


def archive(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def freeze(out):
    inventory = json.loads((PARENT/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    require(len(inventory) == 23, 'expected 23 active ports')
    saved = []
    for item in inventory:
        port = item['port']
        original = archive(ROOT/'pred_cache'/(port+'.npz'))
        parent = archive(PARENT/port/'policies.npz')
        gate = archive(GATE/port/'policies.npz')
        control = archive(PARENT/'coverage_control'/(port+'.npz'))
        q = original['qpos']
        for data in (parent, gate, control):
            require(np.array_equal(data['query_rows'], q), 'query alignment mismatch')
            require(np.array_equal(data['support_rows'], parent['support_rows']), 'support alignment mismatch')
        score = gate['scores'][list(gate['score_names']).index('oof_lr0')]
        base_logits = original['base_logits'][q]
        base, fusion = base_logits.argmax(1), original['expert_logits'][q].argmax(1)
        masks, caps, experts, predictions = [], [], [], []
        for i in range(5):
            support, ev = parent['support_rows'][i], parent['eval_masks'][i]
            require(np.array_equal(ev, ~np.isin(q, support)), 'evaluation excludes wrong support')
            require(len(np.unique(support)) == len(support) == 64, 'support budget changed')
            pp = dict(zip(parent['arm_names'], parent['predictions'][i]))
            pp['fusion_observed64'] = control['predictions'][i]
            require(np.array_equal(pp['base'], base) and np.array_equal(pp['fusion'], fusion), 'fixed heads changed')
            candidate = np.stack([pp[name] for name in method.EXPERTS])
            selected, cap = method.admission_masks(base_logits, fusion, score,
                                                   original['y_ids'][support], q, ev)
            observed = np.bincount(original['y_ids'][support], minlength=8) > 0
            uncapped = np.where((base != fusion) & observed[fusion] & (score > 0), fusion, base)
            require(np.array_equal(uncapped, pp['oof_lr0']), 'original uncapped gate changed')
            require(not selected[..., ~ev].any() and np.all(selected.sum(-1) <= cap), 'budget/support violation')
            masks.append(selected); caps.append(cap); experts.append(candidate)
            predictions.append(method.handoff(base, candidate, selected))
        directory = out/port
        directory.mkdir()
        np.savez_compressed(directory/'policies.npz', query_rows=q, support_rows=parent['support_rows'],
            eval_masks=parent['eval_masks'], seeds=np.arange(5), budgets=np.asarray(method.BUDGETS),
            selector_names=np.asarray(method.SELECTORS), expert_names=np.asarray(method.EXPERTS),
            selected=np.stack(masks), caps=np.stack(caps), predictions=np.stack(predictions),
            expert_predictions=np.stack(experts), base_prediction=base, old_lr_score=score)
        saved.append(dict(port=port, queries=len(q), evaluation_rows=len(q)-64))
        print('FROZEN '+port, flush=True)
    write_json(out/'POLICIES_SAVED.json', saved)
    return saved


def evaluate(out, inventory):
    rows, references = [], []
    for item in inventory:
        port = item['port']
        original = archive(ROOT/'pred_cache'/(port+'.npz'))
        policy = archive(out/port/'policies.npz')
        y, base = original['y_ids'][policy['query_rows']], policy['base_prediction']
        port_rows = []
        for i, seed in enumerate(policy['seeds']):
            ev = policy['eval_masks'][i]
            for name, pred in [('base', base)] + list(zip(method.EXPERTS, policy['expert_predictions'][i])):
                references.append(dict(port=port, seed=int(seed), arm=name,
                                       **method.score_policy(pred, base, y, ev)))
            for si, selector in enumerate(method.SELECTORS):
                for bi, budget in enumerate(method.BUDGETS):
                    selected = policy['selected'][i, si, bi]
                    n_admitted, cap = int(selected.sum()), int(policy['caps'][i, bi])
                    for ei, expert in enumerate(method.EXPERTS):
                        pred = policy['predictions'][i, si, bi, ei]
                        changes = int(np.count_nonzero((pred != base) & ev))
                        require(changes <= n_admitted <= cap, 'cost/changes exceed cap')
                        result = dict(port=port, seed=int(seed), selector=selector, budget=budget, expert=expert,
                            **method.score_policy(pred, base, y, ev), evaluated_rows=int(ev.sum()),
                            admitted_rows=n_admitted, changed_rows=changes, cap=cap,
                            admission_fraction=n_admitted/ev.sum(), change_fraction=changes/ev.sum())
                        port_rows.append(result)
        write_json(out/port/'metrics.json', port_rows)
        rows.extend(port_rows)
    with (out/'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        fields = [k for k in rows[0] if k != 'class_recall']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    write_json(out/'full_query_references.json', references)
    ports = [item['port'] for item in inventory]
    values, means = {}, []
    for selector in method.SELECTORS:
        for budget in method.BUDGETS:
            for expert in method.EXPERTS:
                key = (selector, budget, expert)
                block = [r for r in rows if (r['selector'], r['budget'], r['expert']) == key]
                values[key] = {m: np.array([[r[m] for r in block if r['port'] == port] for port in ports]) for m in MEASURES}
                means.append(dict(selector=selector, budget=budget, expert=expert,
                                  **{m: float(v.mean()) for m, v in values[key].items()}))
    sampled = np.random.default_rng(0).integers(0, len(ports), (10000, len(ports)))
    comparisons = []
    for selector in method.SELECTORS:
        for budget in method.BUDGETS:
            a = values[(selector, budget, 'anchor_visual64')]
            for control in CONTRASTS:
                b = values[(selector, budget, control)]
                delta = a['ba']-b['ba']
                by_port = delta.mean(1)
                comparisons.append(dict(selector=selector, budget=budget, candidate='anchor_visual64', control=control,
                    mean_delta_ba_pp=float(100*by_port.mean()), mean_delta_acc_pp=float(100*(a['acc']-b['acc']).mean()),
                    positive_ports=int((by_port > 1e-12).sum()), negative_ports=int((by_port < -1e-12).sum()),
                    port_bootstrap_95ci_pp=(100*np.percentile(by_port[sampled].mean(1), [2.5, 97.5])).tolist(),
                    draw_deltas_pp=(100*delta.mean(0)).tolist()))
    primary = [r for r in comparisons if r['selector'] == 'old_lr_cap' and r['budget'] == .05
               and r['control'] in ('fusion', 'anchor_bias64')]
    require(len(primary) == 2, 'primary contrasts missing')
    reference_means = {a: {m: float(np.mean([r[m] for r in references if r['arm'] == a]))
                            for m in ('ba', 'acc')} for a in ('base',)+method.EXPERTS}
    summary = dict(n_ports=23, contexts=115, policy_metric_rows=len(rows), means=means, comparisons=comparisons,
        full_query_reference_means=reference_means,
        passes_development_screen=all(r['mean_delta_ba_pp'] > 0 for r in primary),
        primary='5% old_lr_cap visual minus fusion AND visual minus bias',
        new_fits=0, independent_confirmation=False, novelty_established=False,
        statistical_unit='port mean over five overlapping support draws',
        costs='admitted-row cap; report actual changes separately; no compute-saving measurement')
    write_json(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'use a new output directory')
    args.out.mkdir(parents=True)
    write_json(args.out/'RUN_CONFIG.json', dict(date='2026-10-07', parent=str(PARENT), gate=str(GATE),
        selectors=method.SELECTORS, experts=method.EXPERTS, budgets=method.BUDGETS, labels=64,
        cap='floor(budget*number_of_non_support_query_rows)',
        protocol=str(Path(__file__).with_name('BUDGETED_RESIDUAL_PROTOCOL_20261007.md'))))
    start = time.perf_counter()
    inventory = freeze(args.out)
    summary = evaluate(args.out, inventory)
    write_json(args.out/'COMPLETE.json', dict(status='complete', new_fits=0, seconds=time.perf_counter()-start,
        ports=23, contexts=115, policy_metric_rows=len(inventory)*5*2*3*4))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
