"""Bounded support-LOO shrinkage experiment under the adapted-base setting."""

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import budgeted_residual_handoff as budgeted
from receiver_consistent_handoff import gate_scores
from residual_expert import fit_correction, source_geometry
from run_budgeted_residual_handoff import ROOT, PARENT, GATE, archive, require, write_json
from support_shrinkage import loo_residuals, select_strength


BASE = ROOT / 'oof_base_adapt_results_20261007'
ARMS = ('base_visual', 'visual_handoff', 'bias_handoff', 'shrink_base',
        'shrink_base_handoff', 'shrink_expert_handoff', 'shrink_both_handoff')
CONTRASTS = (('shrink_base_handoff', 'visual_handoff'),
             ('shrink_expert_handoff', 'visual_handoff'),
             ('shrink_both_handoff', 'visual_handoff'),
             ('shrink_base', 'base_visual'),
             ('shrink_base_handoff', 'shrink_base'),
             ('shrink_both_handoff', 'bias_handoff'))
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'admission_fraction', 'original_change_fraction')


def freeze(out):
    inventory = json.loads((BASE / 'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    for item in inventory:
        port = item['port']
        d = archive(ROOT / 'pred_cache' / (port + '.npz'))
        pc = archive(ROOT / 'pred_cache2' / (port + '.npz'))
        parent = archive(BASE / port / 'policies.npz')
        residual = archive(PARENT / port / 'policies.npz')
        q, source = d['qpos'], d['spos']
        require(np.array_equal(q, parent['query_rows']) and np.array_equal(q, residual['query_rows']), 'query alignment')
        require(np.array_equal(parent['support_rows'], residual['support_rows']), 'shared support alignment')
        features = [source_geometry(raw, source)[0][q] for raw in (pc['Xs'], np.column_stack((pc['Xs'], pc['Z'])))]
        priors = [d[key][q].astype(float) for key in ('base_logits', 'expert_logits')]
        ei = list(residual['score_names']).index('anchor_visual64')
        ci = [list(parent['arm_names']).index(arm) for arm in ARMS[:3]]
        with (GATE / port / 'source_model.pkl').open('rb') as f:
            gate = pickle.load(f)['prior']
        scores_all, alphas, numerators, denominators, loo_all = [], [], [], [], []
        coefficients = [[], []]
        gates_all, masks_all, fallback_all, receiving_all, predictions = [], [], [], [], []
        for seed in range(5):
            support, ev = parent['support_rows'][seed], parent['eval_masks'][seed]
            local = np.searchsorted(q, support)
            require(np.array_equal(q[local], support) and np.array_equal(ev, ~np.isin(q, support)), 'support exclusion')
            y = d['y_ids'][support]
            old_scores = [parent['scores'][seed, 1], residual['scores'][seed, ei]]
            new_scores, alpha, numerator, denominator, loofits = [], [], [], [], []
            for branch in range(2):
                x, prior = features[branch], priors[branch]
                loo = loo_residuals(x[local], y, prior[local])
                a, num, den = select_strength(y, prior[local], loo)
                coef = fit_correction(x[local], y, prior[local])
                full = prior + x @ coef
                require(np.max(np.abs(full - old_scores[branch])) < 1e-10, 'all64 parent refit mismatch')
                new_scores.append(prior + a * (x @ coef))
                coefficients[branch].append(coef)
                alpha.append(a); numerator.append(num); denominator.append(den); loofits.append(loo)
            require(np.isfinite(new_scores).all(), 'nonfinite shrunk score')
            observed = np.bincount(y, minlength=8) > 0
            cap = int(parent['caps'][seed])
            require(cap == int(np.floor(.05 * ev.sum())), 'cap mismatch')
            branch_pairs = ((new_scores[0], old_scores[1]), (old_scores[0], new_scores[1]), (new_scores[0], new_scores[1]))
            candidate_gates, candidate_masks, candidate_base, candidate_expert = [], [], [], []
            for base_scores, expert_scores in branch_pairs:
                base, expert = base_scores.argmax(1), expert_scores.argmax(1)
                score = gate_scores(base_scores, expert_scores, gate)
                eligible = ev & (base != expert) & observed[expert] & (score > 0)
                candidate_masks.append(budgeted.ranked_mask(score, eligible, q, cap))
                candidate_gates.append(score); candidate_base.append(base); candidate_expert.append(expert)
            shrunk_base = new_scores[0].argmax(1)
            masks = np.stack([*parent['selected'][seed, ci], np.zeros(len(q), dtype=bool), *candidate_masks])
            fallback = np.stack([*parent['fallback_predictions'][seed, ci], shrunk_base, *candidate_base])
            receivers = np.stack([*[parent['expert_predictions'][seed]] * 3, old_scores[1].argmax(1), *candidate_expert])
            pred = np.where(masks, receivers, fallback)
            require(np.array_equal(pred[:3], parent['predictions'][seed, ci]), 'control predictions changed')
            scores_all.append(new_scores); alphas.append(alpha); numerators.append(numerator); denominators.append(denominator)
            loo_all.append(loofits); gates_all.append(candidate_gates); masks_all.append(masks)
            fallback_all.append(fallback); receiving_all.append(receivers); predictions.append(pred)
        directory = out / port
        directory.mkdir()
        np.savez_compressed(directory / 'policies.npz', query_rows=q, source_rows=source,
            support_rows=parent['support_rows'], eval_masks=parent['eval_masks'], seeds=np.arange(5), caps=parent['caps'],
            arm_names=np.asarray(ARMS), scores=np.asarray(scores_all), alpha=np.asarray(alphas),
            alpha_numerator=np.asarray(numerators), alpha_denominator=np.asarray(denominators),
            loo_residuals=np.asarray(loo_all), base_correction=np.asarray(coefficients[0]), expert_correction=np.asarray(coefficients[1]),
            gate_scores=np.asarray(gates_all), selected=np.asarray(masks_all), fallback_predictions=np.asarray(fallback_all),
            expert_predictions=np.asarray(receiving_all), predictions=np.asarray(predictions), original_base=parent['original_base'])
        print('FROZEN ' + port, flush=True)
    write_json(out / 'POLICIES_SAVED.json', inventory)
    return inventory


def evaluate(out, inventory):
    rows, alpha = [], []
    for item in inventory:
        port = item['port']
        p, d = archive(out / port / 'policies.npz'), archive(ROOT / 'pred_cache' / (port + '.npz'))
        y = d['y_ids'][p['query_rows']]
        old = json.loads((BASE / port / 'metrics.json').read_text(encoding='utf-8'))
        block = []
        alpha.append(p['alpha'])
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
                    require(all(abs(row[m] - control[m]) < 1e-12 for m in MEASURES), 'control metrics changed')
                block.append(row)
        write_json(out / port / 'metrics.json', block)
        rows.extend(block)
    with (out / 'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'class_recall'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    ports = [item['port'] for item in inventory]
    lookup = {(r['port'], r['seed'], r['arm']): r for r in rows}
    values = {arm: {m: np.array([[lookup[port, seed, arm][m] for seed in range(5)] for port in ports])
                    for m in MEASURES} for arm in ARMS}
    means = [dict(arm=arm, **{m: float(v.mean()) for m, v in values[arm].items()}) for arm in ARMS]
    sampled = np.random.default_rng(0).integers(0, len(ports), (10000, len(ports)))
    comparisons = []
    for a, b in CONTRASTS:
        delta = values[a]['ba'] - values[b]['ba']
        port_delta = delta.mean(1)
        comparisons.append(dict(candidate=a, control=b, mean_delta_ba_pp=float(100 * delta.mean()),
            mean_delta_acc_pp=float(100 * (values[a]['acc'] - values[b]['acc']).mean()),
            positive_ports=int((port_delta > 1e-12).sum()), negative_ports=int((port_delta < -1e-12).sum()),
            port_bootstrap_95ci_pp=(100 * np.percentile(port_delta[sampled].mean(1), [2.5, 97.5])).tolist(),
            draw_deltas_pp=(100 * delta.mean(0)).tolist(), port_deltas_pp=dict(zip(ports, (100 * port_delta).tolist()))))
    alpha = np.asarray(alpha)
    summary = dict(n_ports=len(ports), contexts=len(ports)*5, policy_metric_rows=len(rows), means=means, comparisons=comparisons,
        alpha_mean=alpha.mean(axis=(0, 1)).tolist(), alpha_min=alpha.min(axis=(0, 1)).tolist(), alpha_max=alpha.max(axis=(0, 1)).tolist(),
        alpha_by_port={port: alpha[i].tolist() for i, port in enumerate(ports)},
        passes_primary_development_screen=comparisons[0]['mean_delta_ba_pp'] > 0,
        primary='shrink_base_handoff minus visual_handoff', independent_confirmation=False, novelty_established=False,
        target_labels=64, budget='expert handoff <=5%; not total prediction changes',
        cost='all expert scores accessed before admission')
    write_json(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'use a new output directory')
    args.out.mkdir(parents=True)
    write_json(args.out / 'RUN_CONFIG.json', dict(date='2026-10-07', arms=ARMS, contrasts=CONTRASTS,
        strength=8., target_labels=64, loo_train_rows=63, fold_class_weights='recomputed from fitting63',
        expert_admission_cap=.05, protocol=str(Path(__file__).with_name('SUPPORT_SHRINKAGE_PROTOCOL_20261007.md'))))
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        inventory = freeze(args.out)
        summary = evaluate(args.out, inventory)
    write_json(args.out / 'COMPLETE.json', dict(status='complete', seconds=time.perf_counter()-start,
        loo_fits=len(inventory)*5*2*64, all64_refits=len(inventory)*5*2, policy_metric_rows=len(inventory)*5*len(ARMS)))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('comparisons', 'alpha_by_port')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
