"""Cross-fitted base-error expert with source-contrast and in-sample controls."""

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import budgeted_residual_handoff as budgeted
from complementary_expert import fit_predict_complement
from receiver_consistent_handoff import gate_scores
from residual_expert import source_geometry
from run_budgeted_residual_handoff import ROOT, PARENT, GATE, archive, require, write_json


BASE = ROOT / 'oof_base_adapt_results_20261007'
LOO = ROOT / 'oof_support_shrinkage_results_20261007'
EXPERTS = ('old_expert', 'crossfit', 'insample', 'no_source')
ARMS = ('base_visual', 'visual_handoff', 'bias_handoff', 'crossfit_handoff', 'insample_handoff', 'no_source_handoff')
CONTRASTS = (('crossfit_handoff', 'visual_handoff'), ('crossfit_handoff', 'insample_handoff'),
             ('crossfit_handoff', 'no_source_handoff'), ('crossfit_handoff', 'bias_handoff'),
             ('crossfit_handoff', 'base_visual'), ('insample_handoff', 'visual_handoff'),
             ('no_source_handoff', 'visual_handoff'))
EXPERT_CONTRASTS = (('crossfit', 'old_expert'), ('crossfit', 'insample'), ('crossfit', 'no_source'))
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'admission_fraction', 'original_change_fraction')
EXPERT_MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'rescues_both_wrong_ba',
                   'old_expert_recover_ba', 'old_expert_harm_ba', 'disagreement_fraction')


def freeze(out):
    inventory = json.loads((BASE / 'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    require(json.loads((LOO / 'VERIFICATION.json').read_text(encoding='utf-8'))['status'] == 'passed', 'unverified LOO cache')
    for item in inventory:
        port = item['port']
        d, pc = archive(ROOT / 'pred_cache' / (port + '.npz')), archive(ROOT / 'pred_cache2' / (port + '.npz'))
        parent, residual, loo = (archive(root / port / 'policies.npz') for root in (BASE, PARENT, LOO))
        q, source = d['qpos'], d['spos']
        for other in (parent, residual, loo):
            require(np.array_equal(q, other['query_rows']) and np.array_equal(source, other['source_rows']), 'source/query alignment')
            require(np.array_equal(parent['support_rows'], other['support_rows']), 'shared support alignment')
            require(np.array_equal(parent['eval_masks'], other['eval_masks']), 'evaluation alignment')
        phi, center, scale = source_geometry(np.column_stack((pc['Xs'], pc['Z'])), source)
        phi = phi[q]
        b0, e0 = (d[key][q].astype(float) for key in ('base_logits', 'expert_logits'))
        difference = e0 - b0
        ei = list(residual['score_names']).index('anchor_visual64')
        ci = [list(parent['arm_names']).index(arm) for arm in ARMS[:3]]
        with (GATE / port / 'source_model.pkl').open('rb') as f:
            gate = pickle.load(f)['prior']
        scores_all, coefficients, training_priors, loo_base_all = [], [], [], []
        gates_all, masks_all, predictions_all, fallback_all, receiving_all = [], [], [], [], []
        for seed in range(5):
            support, ev = parent['support_rows'][seed], parent['eval_masks'][seed]
            local = np.searchsorted(q, support)
            require(np.array_equal(q[local], support) and np.array_equal(ev, ~np.isin(q, support)), 'support exclusion')
            labels = d['y_ids'][support]
            base_scores = parent['scores'][seed, 1]
            base = base_scores.argmax(1)
            training_base = b0[local] + loo['loo_residuals'][seed, 0]
            scores = [residual['scores'][seed, ei]]
            fitted, targets = [], []
            recipes = ((training_base, difference), (base_scores[local], difference), (training_base, np.zeros_like(difference)))
            for train_base, contrast in recipes:
                score, coefficient = fit_predict_complement(phi, local, labels, train_base, base_scores, contrast)
                scores.append(score); fitted.append(coefficient); targets.append(train_base + contrast[local])
            scores = np.asarray(scores)
            require(np.isfinite(scores).all(), 'nonfinite complementary scores')
            experts = scores.argmax(2)
            observed = np.bincount(labels, minlength=8) > 0
            cap = int(parent['caps'][seed])
            require(cap == int(np.floor(.05 * ev.sum())), 'cap mismatch')
            gates, masks = [], []
            for expert, score in zip(experts, scores):
                gate_score = gate_scores(base_scores, score, gate)
                eligible = ev & (base != expert) & observed[expert] & (gate_score > 0)
                masks.append(budgeted.ranked_mask(gate_score, eligible, q, cap)); gates.append(gate_score)
            require(np.array_equal(masks[0], parent['selected'][seed, ci[1]]), 'old gate replay changed')
            selected = np.stack([*parent['selected'][seed, ci], *masks[1:]])
            fallback = np.stack([*parent['fallback_predictions'][seed, ci], base, base, base])
            receiving = np.stack([experts[0], experts[0], experts[0], *experts[1:]])
            pred = np.where(selected, receiving, fallback)
            require(np.array_equal(pred[:3], parent['predictions'][seed, ci]), 'parent policies changed')
            scores_all.append(scores); coefficients.append(fitted); training_priors.append(targets)
            loo_base_all.append(training_base); gates_all.append(gates); masks_all.append(selected)
            predictions_all.append(pred); fallback_all.append(fallback); receiving_all.append(receiving)
        directory = out / port
        directory.mkdir()
        np.savez_compressed(directory / 'policies.npz', query_rows=q, source_rows=source,
            support_rows=parent['support_rows'], eval_masks=parent['eval_masks'], seeds=np.arange(5), caps=parent['caps'],
            arm_names=np.asarray(ARMS), expert_names=np.asarray(EXPERTS), scores=np.asarray(scores_all),
            coefficients=np.asarray(coefficients), training_priors=np.asarray(training_priors),
            loo_base_scores=np.asarray(loo_base_all), source_contrast=difference, source_center=center, source_scale=scale,
            gate_scores=np.asarray(gates_all), selected=np.asarray(masks_all), predictions=np.asarray(predictions_all),
            fallback_predictions=np.asarray(fallback_all), receiving_predictions=np.asarray(receiving_all),
            expert_predictions=np.asarray(scores_all).argmax(3), original_base=parent['original_base'])
        print('FROZEN ' + port, flush=True)
    write_json(out / 'POLICIES_SAVED.json', inventory)
    return inventory


def summarize(rows, names, contrasts, ports, measures):
    lookup = {(r['port'], r['seed'], r['arm']): r for r in rows}
    values = {arm: {m: np.array([[lookup[port, seed, arm][m] for seed in range(5)] for port in ports])
                    for m in measures} for arm in names}
    means = [dict(arm=arm, **{m: float(v.mean()) for m, v in values[arm].items()}) for arm in names]
    sampled = np.random.default_rng(0).integers(0, len(ports), (10000, len(ports)))
    comparisons = []
    for a, b in contrasts:
        delta = values[a]['ba'] - values[b]['ba']
        per_port = delta.mean(1)
        comparisons.append(dict(candidate=a, control=b, mean_delta_ba_pp=float(100 * delta.mean()),
            mean_delta_acc_pp=float(100 * (values[a]['acc'] - values[b]['acc']).mean()),
            positive_ports=int((per_port > 1e-12).sum()), negative_ports=int((per_port < -1e-12).sum()),
            port_bootstrap_95ci_pp=(100 * np.percentile(per_port[sampled].mean(1), [2.5, 97.5])).tolist(),
            draw_deltas_pp=(100 * delta.mean(0)).tolist(), port_deltas_pp=dict(zip(ports, (100 * per_port).tolist()))))
    return dict(means=means, comparisons=comparisons)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'class_recall'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def evaluate(out, inventory):
    rows, full_rows = [], []
    for item in inventory:
        port = item['port']
        p, d = archive(out / port / 'policies.npz'), archive(ROOT / 'pred_cache' / (port + '.npz'))
        y = d['y_ids'][p['query_rows']]
        old = json.loads((BASE / port / 'metrics.json').read_text(encoding='utf-8'))
        old_expert = json.loads((PARENT / port / 'metrics.json').read_text(encoding='utf-8'))
        block, full_block = [], []
        for seed in range(5):
            ev = p['eval_masks'][seed]
            for ai, arm in enumerate(ARMS):
                pred, base, selected = p['predictions'][seed, ai], p['fallback_predictions'][seed, ai], p['selected'][seed, ai]
                admitted = int(selected.sum())
                changed = int(np.count_nonzero((pred != base) & ev))
                require(not selected[~ev].any() and admitted == changed <= p['caps'][seed], 'actual handoff cap violation')
                original_changes = int(np.count_nonzero((pred != p['original_base']) & ev))
                row = dict(port=port, seed=seed, arm=arm, **budgeted.score_policy(pred, p['original_base'], y, ev),
                    evaluated_rows=int(ev.sum()), admitted_rows=admitted, handoff_changed_rows=changed,
                    changed_from_original_rows=original_changes, cap=int(p['caps'][seed]),
                    admission_fraction=float(admitted / ev.sum()), original_change_fraction=float(original_changes / ev.sum()))
                if ai < 3:
                    control = next(r for r in old if r['seed'] == seed and r['arm'] == arm)
                    require(all(abs(row[m] - control[m]) < 1e-12 for m in MEASURES), 'parent metrics changed')
                block.append(row)
            adapted_base = p['predictions'][seed, 0]
            expert_old = p['expert_predictions'][seed, 0]
            truth = y[ev]
            _, inverse, counts = np.unique(truth, return_inverse=True, return_counts=True)
            weights = 1. / (len(counts) * counts[inverse])
            old_correct, base_correct = expert_old[ev] == truth, adapted_base[ev] == truth
            for ei, expert in enumerate(EXPERTS):
                pred = p['expert_predictions'][seed, ei]
                correct = pred[ev] == truth
                row = dict(port=port, seed=seed, arm=expert, **budgeted.score_policy(pred, adapted_base, y, ev),
                    evaluated_rows=int(ev.sum()), rescues_both_wrong_ba=float(weights[correct & ~old_correct & ~base_correct].sum()),
                    old_expert_recover_ba=float(weights[correct & ~old_correct].sum()),
                    old_expert_harm_ba=float(weights[~correct & old_correct].sum()),
                    disagreement_fraction=float((pred[ev] != adapted_base[ev]).mean()))
                if ei == 0:
                    control = old_expert['seeds'][seed]['metrics']['anchor_visual64']
                    require(all(abs(row[m] - control[m]) < 1e-12 for m in ('ba', 'acc')), 'old full expert changed')
                full_block.append(row)
        write_json(out / port / 'metrics.json', block); rows.extend(block)
        write_json(out / port / 'expert_metrics.json', full_block); full_rows.extend(full_block)
    write_csv(out / 'metrics.csv', rows)
    write_csv(out / 'expert_metrics.csv', full_rows)
    ports = [item['port'] for item in inventory]
    policy_summary = summarize(rows, ARMS, CONTRASTS, ports, MEASURES)
    expert_summary = summarize(full_rows, EXPERTS, EXPERT_CONTRASTS, ports, EXPERT_MEASURES)
    summary = dict(n_ports=len(ports), contexts=len(ports)*5, policy_metric_rows=len(rows), expert_metric_rows=len(full_rows),
        **policy_summary, full_query_experts=expert_summary,
        passes_primary_development_screen=policy_summary['comparisons'][0]['mean_delta_ba_pp'] > 0,
        primary='crossfit_handoff minus visual_handoff', target_labels=64, independent_confirmation=False, novelty_established=False,
        budget='5% expert handoff relative to adapted base; not total source-base changes or measured compute',
        expert_metrics_scope='full-query diagnostic, NOT a5% policy; recover/harm reference adapted base')
    write_json(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'use a new output directory')
    args.out.mkdir(parents=True)
    write_json(args.out / 'RUN_CONFIG.json', dict(date='2026-10-07', arms=ARMS, experts=EXPERTS, contrasts=CONTRASTS,
        expert_contrasts=EXPERT_CONTRASTS, strength=8., target_labels=64, expert_admission_cap=.05,
        loo_cache=str(LOO), alphas_used=False, protocol=str(Path(__file__).with_name('COMPLEMENTARY_EXPERT_PROTOCOL_20261007.md'))))
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        inventory = freeze(args.out)
        summary = evaluate(args.out, inventory)
    write_json(args.out / 'COMPLETE.json', dict(status='complete', seconds=time.perf_counter()-start,
        new_expert_fits=len(inventory)*5*3, reused_base_loo_fits=len(inventory)*5*64,
        new_base_or_gate_fits=0, policy_metric_rows=summary['policy_metric_rows'], expert_metric_rows=summary['expert_metric_rows']))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('comparisons', 'full_query_experts')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
