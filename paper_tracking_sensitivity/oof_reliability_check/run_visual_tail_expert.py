"""Fixed omitted-feature residual study with a matched-energy control."""

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import budgeted_residual_handoff as budgeted
import visual_tail_expert as method
from build_oof_predictions import file_sha256, load_features
from residual_expert import source_geometry
from receiver_consistent_handoff import gate_scores
from run_budgeted_residual_handoff import ROOT, PARENT, GATE, archive, require, write_json
from run_local_metric_expert import diagnostics, MEASURES


WS = Path('E:/临时会话/visual_reliable_baseline')
MANIFEST = WS/'artifacts/eight_class_adaptive_20260916/manifest.csv'
VV = MANIFEST.with_name('pooled_features.npy')
VH = WS/'b2_ssl_feats_38091.npz'
FROZEN = ROOT/'oof_receiver_consistent_results_20261007'
ADVANTAGE = ROOT/'oof_action_advantage_results_20261007'
NEW_ARMS = ('visual_tail', 'energy_control')
EXPERT_ARMS = NEW_ARMS+('linear', 'bias')
BUDGET_ARMS = EXPERT_ARMS+('advantage_reference',)


def freeze(out):
    inventory = json.loads((PARENT/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    inputs = {name+'_sha256': file_sha256(path) for name, path in
              (('manifest', MANIFEST), ('vv_features', VV), ('vh_features', VH))}
    vv, vh = load_features(VV), load_features(VH, 'F')
    require(vv.shape == (38091, 2048) and vh.shape == (38091, 512), 'raw feature shape changed')
    for item in inventory:
        port = item['port']
        with np.load(ROOT/'oof_predictions_v1'/(port+'.npz'), allow_pickle=False) as a:
            receipt = json.loads(str(a['metadata_json']))
        require(receipt['complete'] and all(receipt[k] == value for k, value in inputs.items()), 'OOF raw input receipt mismatch')
        original = archive(ROOT/'pred_cache'/(port+'.npz'))
        residual, frozen = archive(PARENT/port/'policies.npz'), archive(FROZEN/port/'policies.npz')
        advantage = archive(ADVANTAGE/port/'policies.npz')
        pc = archive(ROOT/'pred_cache2'/(port+'.npz'))
        q, source = original['qpos'], original['spos']
        require(not np.intersect1d(q, source).size, 'source/target overlap')
        for parent in (residual, frozen, advantage):
            require(np.array_equal(q, parent['query_rows']), 'query order mismatch')
            require(np.array_equal(residual['support_rows'], parent['support_rows']), 'support mismatch')
            require(np.array_equal(residual['eval_masks'], parent['eval_masks']), 'evaluation mismatch')
        phi, center, scale = source_geometry(np.column_stack((pc['Xs'], pc['Z'])), source)
        require(np.array_equal(center, residual['source_center']) and np.array_equal(scale, residual['source_scale']), 'retained geometry changed')
        retained = phi[:, :-1]
        vv_tail, vv_meta = method.fit_tail(vv, retained[:, :128], source, q)
        vh_tail, vh_meta = method.fit_tail(vh, retained[:, 128:], source, q)
        augmented = np.column_stack((retained[q], vv_tail, vh_tail))
        control = np.sqrt(2)*retained[q]
        with (GATE/port/'source_model.pkl').open('rb') as f:
            prior_gate = pickle.load(f)['prior']
        bi = list(frozen['budgets']).index(.05)
        ref_indices = [list(frozen['expert_names']).index(e) for e in ('anchor_visual64', 'anchor_bias64')]
        score_indices = [list(residual['score_names']).index(e) for e in ('anchor_visual64', 'anchor_bias64')]
        adv_index = list(advantage['arm_names']).index('advantage_visual')
        base_logits, prior = original['base_logits'][q], original['expert_logits'][q].astype(float)
        base = base_logits.argmax(1)
        scores_all, corrections, gates_all, masks_all, predictions = [], [], [], [], []
        for i in range(5):
            support, ev = residual['support_rows'][i], residual['eval_masks'][i]
            local = np.searchsorted(q, support)
            require(np.array_equal(q[local], support) and np.array_equal(ev, ~np.isin(q, support)), 'support mapping')
            labels = original['y_ids'][support]
            observed = np.bincount(labels, minlength=8) > 0
            scores, coef = [], {}
            for arm, features in zip(NEW_ARMS, (augmented, control)):
                value, coefficient = method.predict(features, prior, local, labels)
                scores.append(value); coef[arm] = coefficient
            scores.extend(residual['scores'][i, j] for j in score_indices)
            scores = np.stack(scores)
            receivers = scores.argmax(2)
            cap = int(frozen['caps'][i, bi])
            require(cap == int(np.floor(.05*ev.sum())), 'cap mismatch')
            gates, masks = [], []
            for ai in range(len(EXPERT_ARMS)):
                score = gate_scores(base_logits, scores[ai], prior_gate)
                eligible = ev & (base != receivers[ai]) & observed[receivers[ai]] & (score > 0)
                mask = budgeted.ranked_mask(score, eligible, q, cap)
                if ai >= 2:
                    require(np.array_equal(mask, frozen['selected'][i, bi, ref_indices[ai-2]]), 'frozen mask changed')
                gates.append(score); masks.append(mask)
            pred = np.where(np.stack(masks), receivers, base[None])
            for ai, ri in zip((2, 3), ref_indices):
                require(np.array_equal(pred[ai], frozen['predictions'][i, bi, ri]), 'frozen control changed')
            masks.append(advantage['selected'][i, adv_index])
            pred = np.concatenate((pred, advantage['predictions'][i, adv_index][None]))
            scores_all.append(scores); corrections.append(coef); gates_all.append(np.stack(gates))
            masks_all.append(np.stack(masks)); predictions.append(pred)
        directory = out/port
        directory.mkdir()
        np.savez_compressed(directory/'geometry.npz', source_rows=source,
            **{'vv_'+k: v for k, v in vv_meta.items()}, **{'vh_'+k: v for k, v in vh_meta.items()})
        np.savez_compressed(directory/'policies.npz', query_rows=q, support_rows=residual['support_rows'],
            eval_masks=residual['eval_masks'], caps=frozen['caps'][:, bi], seeds=np.arange(5),
            expert_names=np.asarray(EXPERT_ARMS), budget_names=np.asarray(BUDGET_ARMS),
            scores=np.stack(scores_all), expert_predictions=np.stack(scores_all).argmax(3),
            gate_scores=np.stack(gates_all), selected=np.stack(masks_all), predictions=np.stack(predictions),
            base_prediction=base, **{arm+'_correction': np.stack([c[arm] for c in corrections]) for arm in NEW_ARMS})
        print(f'FROZEN {port}: source tail energies VV={float(vv_meta["source_tail_energy"]):.4g}, VH={float(vh_meta["source_tail_energy"]):.4g}', flush=True)
    write_json(out/'POLICIES_SAVED.json', inventory)
    return inventory


def evaluate(out, inventory):
    rows = []
    for item in inventory:
        port = item['port']
        p, original = archive(out/port/'policies.npz'), archive(ROOT/'pred_cache'/(port+'.npz'))
        y, base = original['y_ids'][p['query_rows']], p['base_prediction']
        frozen = json.loads((FROZEN/port/'metrics.json').read_text(encoding='utf-8'))
        full = json.loads((PARENT/port/'metrics.json').read_text(encoding='utf-8'))
        advantage = json.loads((ADVANTAGE/port/'metrics.json').read_text(encoding='utf-8'))
        block = []
        for i in range(5):
            ev, linear = p['eval_masks'][i], p['expert_predictions'][i, 2]
            for mode, arms, predictions in (('full', EXPERT_ARMS, p['expert_predictions'][i]), ('budget', BUDGET_ARMS, p['predictions'][i])):
                for ai, arm in enumerate(arms):
                    pred = predictions[ai]
                    row = dict(port=port, seed=i, mode=mode, arm=arm, evaluated_rows=int(ev.sum()),
                               **diagnostics(pred, base, linear, y, ev))
                    if mode == 'budget':
                        mask = p['selected'][i, ai]
                        admitted, changed = int(mask.sum()), int(np.count_nonzero((pred != base) & ev))
                        require(not mask[~ev].any() and admitted == changed <= p['caps'][i], 'admission violation')
                        row.update(admitted_rows=admitted, changed_rows=changed, cap=int(p['caps'][i]), admission_fraction=float(admitted/ev.sum()))
                    else:
                        row.update(admitted_rows=None, changed_rows=None, cap=None, admission_fraction=None)
                    if arm in ('linear', 'bias'):
                        expert = 'anchor_visual64' if arm == 'linear' else 'anchor_bias64'
                        old = (full['seeds'][i]['metrics'][expert] if mode == 'full' else
                               next(r for r in frozen if r['seed'] == i and r['budget'] == .05 and r['expert'] == expert))
                        require(all(abs(row[m]-old[m]) < 1e-12 for m in ('ba', 'acc')), 'parent metrics changed')
                    if arm == 'advantage_reference':
                        old = next(r for r in advantage if r['seed'] == i and r['arm'] == 'advantage_visual')
                        require(all(abs(row[m]-old[m]) < 1e-12 for m in ('ba', 'acc')), 'advantage metrics changed')
                    block.append(row)
        write_json(out/port/'metrics.json', block); rows.extend(block)
    with (out/'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'class_recall'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    ports = [r['port'] for r in inventory]
    values, means = {}, []
    for mode, arms in (('full', EXPERT_ARMS), ('budget', BUDGET_ARMS)):
        for arm in arms:
            block = [r for r in rows if r['mode'] == mode and r['arm'] == arm]
            measures = MEASURES+(('admission_fraction',) if mode == 'budget' else ())
            values[mode, arm] = {m: np.array([[next(r[m] for r in block if r['port'] == port and r['seed'] == i)
                                               for i in range(5)] for port in ports]) for m in measures}
            means.append(dict(mode=mode, arm=arm, **{m: float(v.mean()) for m, v in values[mode, arm].items()}))
    sampled = np.random.default_rng(0).integers(0, len(ports), (10000, len(ports)))
    comparisons = []
    for mode in ('full', 'budget'):
        contrasts = [('visual_tail', b) for b in ('linear', 'energy_control', 'bias')]+[('energy_control', 'linear')]
        if mode == 'budget':
            contrasts.append(('visual_tail', 'advantage_reference'))
        for a, b in contrasts:
            delta = values[mode, a]['ba']-values[mode, b]['ba']; by_port = delta.mean(1)
            comparisons.append(dict(mode=mode, candidate=a, control=b, mean_delta_ba_pp=float(100*delta.mean()),
                mean_delta_acc_pp=float(100*(values[mode, a]['acc']-values[mode, b]['acc']).mean()),
                positive_ports=int((by_port > 1e-12).sum()), negative_ports=int((by_port < -1e-12).sum()),
                port_bootstrap_95ci_pp=(100*np.percentile(by_port[sampled].mean(1), [2.5, 97.5])).tolist(),
                draw_deltas_pp=(100*delta.mean(0)).tolist(), port_deltas_pp=dict(zip(ports, (100*by_port).tolist()))))
    primary = [r for r in comparisons if r['mode'] == 'budget' and r['candidate'] == 'visual_tail'
               and r['control'] in ('linear', 'energy_control', 'advantage_reference')]
    summary = dict(n_ports=len(ports), contexts=len(ports)*5, policy_metric_rows=len(rows), means=means, comparisons=comparisons,
        passes_development_screen=all(r['mean_delta_ba_pp'] > 0 for r in primary),
        primary='5% visual_tail beats linear, energy_control and advantage_reference',
        independent_confirmation=False, novelty_established=False, target_label_budget=64,
        cost='admission cap only; all raw/retained receiver scores accessed before selection',
        limitation='source energy control does not match every kernel eigenvalue or effective capacity')
    write_json(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); require(not args.out.exists(), 'use a new output directory')
    args.out.mkdir(parents=True)
    write_json(args.out/'RUN_CONFIG.json', dict(date='2026-10-07', expert_arms=EXPERT_ARMS, budget_arms=BUDGET_ARMS,
        labels=64, strength=8., budget=.05, vv_features=str(VV), vh_features=str(VH), manifest=str(MANIFEST),
        source_receipts=str(ROOT/'oof_predictions_v1'), raw_receipt_checks='existing fingerprints replace PCA/head refits',
        protocol=str(Path(__file__).with_name('VISUAL_TAIL_PROTOCOL_20261007.md'))))
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        inventory = freeze(args.out)
        summary = evaluate(args.out, inventory)
    write_json(args.out/'COMPLETE.json', dict(status='complete', seconds=time.perf_counter()-start,
        source_decoder_fits=len(inventory)*2, new_target_fits=len(inventory)*5*2, new_backbone_pca_head_fits=0,
        policy_metric_rows=summary['policy_metric_rows']))
    print(json.dumps({k: v for k, v in summary.items() if k != 'comparisons'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
