"""Fixed source-metric/locality ablations of the anchored receiving expert."""

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import budgeted_residual_handoff as budgeted
import local_metric_expert as method
from receiver_consistent_handoff import gate_scores
from run_budgeted_residual_handoff import ROOT, PARENT, GATE, archive, require, write_json


FROZEN = ROOT/'oof_receiver_consistent_results_20261007'
ADVANTAGE = ROOT/'oof_action_advantage_results_20261007'
NEW_ARMS = ('metric_linear', 'isotropic_rbf', 'metric_rbf')
EXPERT_ARMS = NEW_ARMS+('linear', 'bias')
BUDGET_ARMS = EXPERT_ARMS+('advantage_reference',)
MEASURES = ('ba', 'acc', 'recover_ba', 'harm_ba', 'both_wrong_rescue_ba', 'linear_harm_ba')


def freeze(out):
    inventory = json.loads((PARENT/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    for item in inventory:
        port = item['port']
        original = archive(ROOT/'pred_cache'/(port+'.npz'))
        residual = archive(PARENT/port/'policies.npz')
        frozen = archive(FROZEN/port/'policies.npz')
        advantage = archive(ADVANTAGE/port/'policies.npz')
        visual = archive(ROOT/'pred_cache2'/(port+'.npz'))
        raw = np.column_stack((visual['Xs'], visual['Z']))
        q, source = original['qpos'], original['spos']
        for parent in (residual, frozen, advantage):
            require(np.array_equal(q, parent['query_rows']), 'query mismatch')
            require(np.array_equal(parent['support_rows'], residual['support_rows']), 'support mismatch')
            require(np.array_equal(parent['eval_masks'], residual['eval_masks']), 'evaluation mismatch')
        require(not np.intersect1d(source, q).size, 'source/target rows overlap')
        x, metric_x, metadata = method.fit_geometry(raw, source, original['y_ids'][source])
        require(np.array_equal(metadata['source_center'], residual['source_center'])
                and np.array_equal(metadata['source_scale'], residual['source_scale']), 'source standardization changed')
        with (GATE/port/'source_model.pkl').open('rb') as f:
            prior_gate = pickle.load(f)['prior']
        bi = list(frozen['budgets']).index(.05)
        reference_indices = [list(frozen['expert_names']).index(e) for e in ('anchor_visual64', 'anchor_bias64')]
        score_indices = [list(residual['score_names']).index(e) for e in ('anchor_visual64', 'anchor_bias64')]
        advantage_index = list(advantage['arm_names']).index('advantage_visual')
        base_logits, source_scores = original['base_logits'][q], original['expert_logits'][q].astype(float)
        base = base_logits.argmax(1)
        all_scores, all_duals, all_gates, all_masks, all_predictions = [], [], [], [], []
        for i in range(5):
            support, ev = residual['support_rows'][i], residual['eval_masks'][i]
            local = np.searchsorted(q, support)
            require(np.array_equal(q[local], support) and np.array_equal(ev, ~np.isin(q, support)), 'support mapping')
            labels = original['y_ids'][support]
            observed = np.bincount(labels, minlength=8) > 0
            score_list, duals = [], []
            for features, bandwidth in ((metric_x[q], None), (x[q], metadata['bandwidths'][0]),
                                         (metric_x[q], metadata['bandwidths'][1])):
                scores, dual = method.fit_predict(features, local, labels, source_scores, bandwidth)
                score_list.append(scores); duals.append(dual)
            score_list.extend(residual['scores'][i, j] for j in score_indices)
            score_block = np.stack(score_list)
            receivers = score_block.argmax(2)
            cap = int(frozen['caps'][i, bi])
            require(cap == int(np.floor(.05*ev.sum())), 'cap mismatch')
            gates, masks = [], []
            for ai in range(len(EXPERT_ARMS)):
                score = gate_scores(base_logits, score_block[ai], prior_gate)
                eligible = ev & (base != receivers[ai]) & observed[receivers[ai]] & (score > 0)
                selected = budgeted.ranked_mask(score, eligible, q, cap)
                if ai >= 3:
                    require(np.array_equal(selected, frozen['selected'][i, bi, reference_indices[ai-3]]), 'frozen mask changed')
                gates.append(score); masks.append(selected)
            prediction = np.where(np.stack(masks), receivers, base[None])
            for ai, ri in zip((3, 4), reference_indices):
                require(np.array_equal(prediction[ai], frozen['predictions'][i, bi, ri]), 'frozen predictions changed')
            masks.append(advantage['selected'][i, advantage_index])
            prediction = np.concatenate((prediction, advantage['predictions'][i, advantage_index][None]))
            all_scores.append(score_block); all_duals.append(np.stack(duals)); all_gates.append(np.stack(gates))
            all_masks.append(np.stack(masks)); all_predictions.append(prediction)
        directory = out/port
        directory.mkdir()
        np.savez_compressed(directory/'geometry.npz', source_rows=source, **metadata)
        np.savez_compressed(directory/'policies.npz', query_rows=q, support_rows=residual['support_rows'],
            eval_masks=residual['eval_masks'], caps=frozen['caps'][:, bi], seeds=np.arange(5),
            expert_names=np.asarray(EXPERT_ARMS), budget_names=np.asarray(BUDGET_ARMS), scores=np.stack(all_scores),
            dual_coefficients=np.stack(all_duals), gate_scores=np.stack(all_gates),
            expert_predictions=np.stack(all_scores).argmax(3), selected=np.stack(all_masks),
            predictions=np.stack(all_predictions), base_prediction=base)
        print(f'FROZEN {port}: shrinkage={float(metadata["shrinkage"]):.5f}, bandwidths={metadata["bandwidths"]}', flush=True)
    write_json(out/'POLICIES_SAVED.json', inventory)
    return inventory


def diagnostics(pred, base, linear, y, ev):
    result = budgeted.score_policy(pred, base, y, ev)
    truth, p, b, old = y[ev], pred[ev], base[ev], linear[ev]
    classes, inverse, counts = np.unique(truth, return_inverse=True, return_counts=True)
    weights = 1./(len(classes)*counts[inverse])
    result['both_wrong_rescue_ba'] = float(weights[(p == truth) & (b != truth) & (old != truth)].sum())
    result['linear_harm_ba'] = float(weights[(p != truth) & (old == truth)].sum())
    return result


def evaluate(out, inventory):
    rows = []
    for item in inventory:
        port = item['port']
        data = archive(out/port/'policies.npz')
        original = archive(ROOT/'pred_cache'/(port+'.npz'))
        y, base = original['y_ids'][data['query_rows']], data['base_prediction']
        frozen_metrics = json.loads((FROZEN/port/'metrics.json').read_text(encoding='utf-8'))
        full_metrics = json.loads((PARENT/port/'metrics.json').read_text(encoding='utf-8'))
        advantage_metrics = json.loads((ADVANTAGE/port/'metrics.json').read_text(encoding='utf-8'))
        block = []
        for i in range(5):
            ev = data['eval_masks'][i]
            linear = data['expert_predictions'][i, 3]
            for mode, arms, predictions in (('full', EXPERT_ARMS, data['expert_predictions'][i]),
                                             ('budget', BUDGET_ARMS, data['predictions'][i])):
                for ai, arm in enumerate(arms):
                    pred = predictions[ai]
                    row = dict(port=port, seed=i, mode=mode, arm=arm, evaluated_rows=int(ev.sum()),
                               **diagnostics(pred, base, linear, y, ev))
                    if mode == 'budget':
                        mask = data['selected'][i, ai]
                        admitted = int(mask.sum()); changes = int(np.count_nonzero((pred != base) & ev))
                        require(not mask[~ev].any() and admitted == changes <= data['caps'][i], 'budget violation')
                        row.update(admitted_rows=admitted, changed_rows=changes, cap=int(data['caps'][i]),
                                   admission_fraction=float(admitted/ev.sum()))
                    else:
                        row.update(admitted_rows=None, changed_rows=None, cap=None, admission_fraction=None)
                    if arm in ('linear', 'bias'):
                        expert = 'anchor_visual64' if arm == 'linear' else 'anchor_bias64'
                        previous = (full_metrics['seeds'][i]['metrics'][expert] if mode == 'full' else
                                    next(r for r in frozen_metrics if r['seed'] == i and r['budget'] == .05 and r['expert'] == expert))
                        require(all(abs(row[m]-previous[m]) < 1e-12 for m in ('ba', 'acc')), 'control metrics changed')
                    if arm == 'advantage_reference':
                        previous = next(r for r in advantage_metrics if r['seed'] == i and r['arm'] == 'advantage_visual')
                        require(all(abs(row[m]-previous[m]) < 1e-12 for m in ('ba', 'acc')), 'advantage metric changed')
                    block.append(row)
        write_json(out/port/'metrics.json', block); rows.extend(block)
    with (out/'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'class_recall'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    ports = [r['port'] for r in inventory]
    values, means = {}, []
    for mode, arms in (('full', EXPERT_ARMS), ('budget', BUDGET_ARMS)):
        for arm in arms:
            measures = MEASURES+(('admission_fraction',) if mode == 'budget' else ())
            block = [r for r in rows if r['mode'] == mode and r['arm'] == arm]
            values[mode, arm] = {m: np.array([[next(r[m] for r in block if r['port'] == port and r['seed'] == i)
                                               for i in range(5)] for port in ports]) for m in measures}
            means.append(dict(mode=mode, arm=arm, **{m: float(v.mean()) for m, v in values[mode, arm].items()}))
    sampled = np.random.default_rng(0).integers(0, len(ports), (10000, len(ports)))
    comparisons = []
    for mode in ('full', 'budget'):
        contrasts = [(arm, 'linear') for arm in NEW_ARMS]+[('metric_rbf', arm) for arm in ('bias', 'metric_linear', 'isotropic_rbf')]
        if mode == 'budget':
            contrasts.append(('metric_rbf', 'advantage_reference'))
        for a, b in contrasts:
            delta = values[mode, a]['ba']-values[mode, b]['ba']; by_port = delta.mean(1)
            comparisons.append(dict(mode=mode, candidate=a, control=b, mean_delta_ba_pp=float(100*delta.mean()),
                mean_delta_acc_pp=float(100*(values[mode, a]['acc']-values[mode, b]['acc']).mean()),
                positive_ports=int((by_port > 1e-12).sum()), negative_ports=int((by_port < -1e-12).sum()),
                port_bootstrap_95ci_pp=(100*np.percentile(by_port[sampled].mean(1), [2.5, 97.5])).tolist(),
                draw_deltas_pp=(100*delta.mean(0)).tolist(), port_deltas_pp=dict(zip(ports, (100*by_port).tolist()))))
    primary = next(r for r in comparisons if r['mode'] == 'budget' and r['candidate'] == 'metric_rbf' and r['control'] == 'linear')
    components = [r for r in comparisons if r['mode'] == 'budget' and r['candidate'] == 'metric_rbf'
                  and r['control'] in ('metric_linear', 'isotropic_rbf')]
    summary = dict(n_ports=len(ports), contexts=len(ports)*5, policy_metric_rows=len(rows), means=means, comparisons=comparisons,
        passes_development_screen=primary['mean_delta_ba_pp'] > 0,
        both_components_positive=all(r['mean_delta_ba_pp'] > 0 for r in components),
        primary='5% metric_rbf minus unchanged linear expert with receiver-consistent gate',
        independent_confirmation=False, novelty_established=False, target_label_budget=64,
        cost='5% admission cap, all query expert scores computed before selection',
        limitation='source norm matching does not match effective regularization between kernels')
    write_json(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); require(not args.out.exists(), 'use a new output directory')
    args.out.mkdir(parents=True)
    write_json(args.out/'RUN_CONFIG.json', dict(date='2026-10-07', new_arms=NEW_ARMS, expert_arms=EXPERT_ARMS,
        budget_arms=BUDGET_ARMS, labels=64, strength=8., bandwidth_seed=1707, bandwidth_pair_limit=512, budget=.05,
        protocol=str(Path(__file__).with_name('LOCAL_METRIC_EXPERT_PROTOCOL_20261007.md'))))
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        inventory = freeze(args.out)
        summary = evaluate(args.out, inventory)
    write_json(args.out/'COMPLETE.json', dict(status='complete', seconds=time.perf_counter()-start,
        source_metric_fits=len(inventory), new_target_fits=len(inventory)*5*3, new_backbone_pca_head_fits=0,
        policy_metric_rows=summary['policy_metric_rows']))
    print(json.dumps({k: v for k, v in summary.items() if k != 'comparisons'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
