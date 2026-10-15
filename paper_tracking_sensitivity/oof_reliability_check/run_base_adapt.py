"""Strong adapted-base controls, explicitly distinct from fixed-base budgets."""

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

import budgeted_residual_handoff as budgeted
from residual_expert import source_geometry, fit_correction
from receiver_consistent_handoff import gate_scores
from run_budgeted_residual_handoff import ROOT, PARENT, GATE, archive, require, write_json


FROZEN = ROOT/'oof_receiver_consistent_results_20261007'
ADVANTAGE = ROOT/'oof_action_advantage_results_20261007'
ARMS = ('original_base', 'base_bias', 'base_visual', 'bias_handoff', 'visual_handoff', 'frozen_handoff', 'advantage_reference')
CONTRASTS = (('visual_handoff', 'base_visual'), ('visual_handoff', 'advantage_reference'),
             ('base_visual', 'original_base'), ('base_visual', 'base_bias'), ('base_bias', 'original_base'),
             ('bias_handoff', 'base_bias'), ('visual_handoff', 'bias_handoff'), ('visual_handoff', 'frozen_handoff'))
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'admission_fraction', 'original_change_fraction')


def freeze(out):
    inventory = json.loads((PARENT/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    max_reference_error = 0.
    for item in inventory:
        port = item['port']
        d, pc = archive(ROOT/'pred_cache'/(port+'.npz')), archive(ROOT/'pred_cache2'/(port+'.npz'))
        residual, frozen = archive(PARENT/port/'policies.npz'), archive(FROZEN/port/'policies.npz')
        advantage = archive(ADVANTAGE/port/'policies.npz')
        q, source = d['qpos'], d['spos']
        require(not np.intersect1d(source, q).size, 'source/target overlap')
        for a in (residual, frozen, advantage):
            require(np.array_equal(q, a['query_rows']) and np.array_equal(residual['support_rows'], a['support_rows']), 'parent alignment')
            require(np.array_equal(residual['eval_masks'], a['eval_masks']), 'evaluation alignment')
        phi, center, scale = source_geometry(pc['Xs'], source)
        phi = phi[q]
        prior = d['base_logits'][q].astype(float)
        base = prior.argmax(1)
        ei = list(residual['score_names']).index('anchor_visual64')
        fi = list(frozen['expert_names']).index('anchor_visual64')
        ai = list(advantage['arm_names']).index('advantage_visual')
        bi = list(frozen['budgets']).index(.05)
        with (GATE/port/'source_model.pkl').open('rb') as stream:
            prior_gate = pickle.load(stream)['prior']
        scores_all, coef_bias, coef_visual, gates_all, masks_all, fallbacks_all, predictions, experts = [], [], [], [], [], [], [], []
        for i in range(5):
            support, ev = residual['support_rows'][i], residual['eval_masks'][i]
            local = np.searchsorted(q, support)
            require(np.array_equal(q[local], support) and np.array_equal(ev, ~np.isin(q, support)), 'support mapping')
            y = d['y_ids'][support]
            _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
            weights = 64/(len(counts)*counts[inverse])
            desired = 2*np.eye(8)[y]-1-prior[local]
            scores, coefficients = [], []
            for features in (phi[:, -1:], phi):
                coefficient = fit_correction(features[local], y, prior[local])
                independent = Ridge(alpha=8., fit_intercept=False, solver='cholesky').fit(features[local], desired, sample_weight=weights).coef_.T
                error = float(np.max(np.abs(coefficient-independent)))
                require(error < 1e-10, 'independent sklearn coefficient check differs')
                max_reference_error = max(max_reference_error, error)
                scores.append(prior+features@coefficient); coefficients.append(coefficient)
            scores = np.stack(scores)
            require(np.isfinite(scores).all(), 'nonfinite adapted base')
            base_bias, base_visual = scores.argmax(2)
            receiving_scores = residual['scores'][i, ei]
            expert = receiving_scores.argmax(1)
            observed = np.bincount(y, minlength=8) > 0
            cap = int(frozen['caps'][i, bi])
            require(cap == int(np.floor(.05*ev.sum())), 'cap mismatch')
            gates, new_masks = [], []
            for index, fallback in enumerate((base_bias, base_visual)):
                score = gate_scores(scores[index], receiving_scores, prior_gate)
                eligible = ev & (fallback != expert) & observed[expert] & (score > 0)
                new_masks.append(budgeted.ranked_mask(score, eligible, q, cap)); gates.append(score)
            masks = np.stack([np.zeros(len(q), dtype=bool) for _ in range(3)]+new_masks+
                             [frozen['selected'][i, bi, fi], advantage['selected'][i, ai]])
            fallbacks = np.stack((base, base_bias, base_visual, base_bias, base_visual, base, base))
            pred = np.where(masks, expert[None], fallbacks)
            require(np.array_equal(pred[5], frozen['predictions'][i, bi, fi]) and np.array_equal(pred[6], advantage['predictions'][i, ai]), 'frozen policy changed')
            scores_all.append(scores); coef_bias.append(coefficients[0]); coef_visual.append(coefficients[1])
            gates_all.append(np.stack(gates)); masks_all.append(masks); fallbacks_all.append(fallbacks)
            predictions.append(pred); experts.append(expert)
        directory = out/port
        directory.mkdir()
        np.savez_compressed(directory/'policies.npz', query_rows=q, source_rows=source,
            support_rows=residual['support_rows'], eval_masks=residual['eval_masks'], seeds=np.arange(5),
            caps=frozen['caps'][:, bi], arm_names=np.asarray(ARMS), scores=np.stack(scores_all),
            bias_correction=np.stack(coef_bias), visual_correction=np.stack(coef_visual), source_center=center, source_scale=scale,
            gate_scores=np.stack(gates_all), selected=np.stack(masks_all), fallback_predictions=np.stack(fallbacks_all),
            predictions=np.stack(predictions), expert_predictions=np.stack(experts), original_base=base)
        print('FROZEN '+port, flush=True)
    write_json(out/'POLICIES_SAVED.json', inventory)
    return inventory, max_reference_error


def evaluate(out, inventory):
    rows = []
    for item in inventory:
        port = item['port']
        p, d = archive(out/port/'policies.npz'), archive(ROOT/'pred_cache'/(port+'.npz'))
        y, original = d['y_ids'][p['query_rows']], p['original_base']
        controls = json.loads((ADVANTAGE/port/'metrics.json').read_text(encoding='utf-8'))
        block = []
        for i in range(5):
            ev = p['eval_masks'][i]
            for ai, arm in enumerate(ARMS):
                pred, fallback, selected = p['predictions'][i, ai], p['fallback_predictions'][i, ai], p['selected'][i, ai]
                admitted = int(selected.sum()); handoff_changes = int(np.count_nonzero((pred != fallback) & ev))
                changed_original = int(np.count_nonzero((pred != original) & ev))
                require(not selected[~ev].any() and admitted == handoff_changes <= p['caps'][i], 'handoff cap violation')
                row = dict(port=port, seed=i, arm=arm, **budgeted.score_policy(pred, original, y, ev),
                    evaluated_rows=int(ev.sum()), admitted_rows=admitted, handoff_changed_rows=handoff_changes,
                    changed_from_original_rows=changed_original, cap=int(p['caps'][i]),
                    admission_fraction=float(admitted/ev.sum()), original_change_fraction=float(changed_original/ev.sum()))
                if ai >= 5:
                    name = 'frozen_visual' if ai == 5 else 'advantage_visual'
                    old = next(r for r in controls if r['seed'] == i and r['arm'] == name)
                    require(all(abs(row[m]-old[m]) < 1e-12 for m in ('ba', 'acc')), 'control metrics changed')
                block.append(row)
        write_json(out/port/'metrics.json', block); rows.extend(block)
    with (out/'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'class_recall'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    ports = [item['port'] for item in inventory]
    values = {arm: {m: np.array([[next(r[m] for r in rows if r['port'] == port and r['seed'] == i and r['arm'] == arm)
                                  for i in range(5)] for port in ports]) for m in MEASURES} for arm in ARMS}
    means = [dict(arm=arm, **{m: float(v.mean()) for m, v in values[arm].items()}) for arm in ARMS]
    sampled = np.random.default_rng(0).integers(0, len(ports), (10000, len(ports)))
    comparisons = []
    for a, b in CONTRASTS:
        delta = values[a]['ba']-values[b]['ba']; by_port = delta.mean(1)
        comparisons.append(dict(candidate=a, control=b, mean_delta_ba_pp=float(100*delta.mean()),
            mean_delta_acc_pp=float(100*(values[a]['acc']-values[b]['acc']).mean()),
            positive_ports=int((by_port > 1e-12).sum()), negative_ports=int((by_port < -1e-12).sum()),
            port_bootstrap_95ci_pp=(100*np.percentile(by_port[sampled].mean(1), [2.5, 97.5])).tolist(),
            draw_deltas_pp=(100*delta.mean(0)).tolist(), port_deltas_pp=dict(zip(ports, (100*by_port).tolist()))))
    primary = [r for r in comparisons if r['candidate'] == 'visual_handoff' and r['control'] in ('base_visual', 'advantage_reference')]
    summary = dict(n_ports=len(ports), contexts=len(ports)*5, policy_metric_rows=len(rows), means=means, comparisons=comparisons,
        passes_development_screen=all(r['mean_delta_ba_pp'] > 0 for r in primary),
        primary='visual_handoff beats adapted base alone AND old advantage reference',
        changed_setting='base may adapt globally; 5% bounds expert handoff, NOT changes versus original base',
        independent_confirmation=False, novelty_established=False, unique_target_labels=64,
        cost='expert admission only, all expert scores accessed before selection')
    write_json(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); require(not args.out.exists(), 'use a new output directory')
    args.out.mkdir(parents=True)
    write_json(args.out/'RUN_CONFIG.json', dict(date='2026-10-07', arms=ARMS, target_labels=64, strength=8., expert_admission_cap=.05,
        original_prediction_change_cap=False, protocol=str(Path(__file__).with_name('BASE_ADAPT_PROTOCOL_20261007.md'))))
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        inventory, error = freeze(args.out)
        summary = evaluate(args.out, inventory)
    write_json(args.out/'COMPLETE.json', dict(status='complete', seconds=time.perf_counter()-start, target_base_fits=230,
        independent_sklearn_coefficient_checks=230, maximum_sklearn_coefficient_error=error, new_expert_or_backbone_fits=0,
        policy_metric_rows=summary['policy_metric_rows']))
    print(json.dumps({k: v for k, v in summary.items() if k != 'comparisons'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
