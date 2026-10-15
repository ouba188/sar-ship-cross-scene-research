"""Fixed same-support experiment: improve the expert, not the gate."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import residual_expert as method
import run_oof_reliability as old
from run_enriched_reliability import write_json


ROOT = Path(__file__).resolve().parent.parent
NEW_ARMS = ('legacy_target64', 'target64', 'anchor_bias64', 'anchor_visual64')
ARMS = ('base', 'fusion', 'oof_lr0') + NEW_ARMS
CONTRASTS = [('anchor_visual64', 'fusion'), ('anchor_visual64', 'anchor_bias64'),
             ('anchor_visual64', 'oof_lr0'), ('anchor_visual64', 'target64'),
             ('anchor_visual64', 'legacy_target64'), ('anchor_bias64', 'fusion')]


def freeze(out):
    inventory = []
    for path in sorted((ROOT/'pred_cache').glob('*.npz')):
        with np.load(path, allow_pickle=False) as a:
            if 'skipped' in a.files:
                continue
            d = {k: a[k] for k in a.files}
        q, source = d['qpos'], d['spos']
        if len(q) <= old.ANNOT:
            continue
        with np.load(ROOT/'pred_cache2'/path.name, allow_pickle=False) as a:
            raw = np.column_stack((a['Xs'], a['Z']))
        old.require(raw.shape == (len(d['y_ids']), 192), 'unexpected visual feature shape')
        phi, center, scale = method.source_geometry(raw, source)
        replay = method.source_replay(raw, d['y_ids'], source, center, scale)
        head_error = float(np.max(np.abs(replay-d['expert_logits'])))
        changed = replay.argmax(1) != d['expert_logits'].argmax(1)
        mismatch = int(changed.sum())
        margin = np.diff(np.sort(d['expert_logits'], axis=1)[:, -2:], axis=1).ravel()
        row_error = np.max(np.abs(replay-d['expert_logits']), axis=1)
        old.require(head_error < 1e-4 and np.all(margin[changed] <= 2*row_error[changed]),
                    f'cached visual/source-head mismatch {path.stem}: {head_error}/{mismatch}')
        with np.load(ROOT/'oof_context_results_20261007'/path.stem/'policies.npz', allow_pickle=False) as a:
            parent = {k: a[k] for k in a.files}
        old.require(np.array_equal(q, parent['query_rows']), 'query order mismatch')
        scores, predictions, corrections = [], [], []
        for i, seed in enumerate(old.SEEDS):
            ann = np.random.default_rng(seed).choice(q, old.ANNOT, replace=False)
            old.require(np.array_equal(ann, parent['support_rows'][i]), 'support changed')
            local = np.searchsorted(q, ann)
            old.require(np.array_equal(q[local], ann), 'support not in query')
            score, correction = method.predict(phi[q], d['expert_logits'][q].astype(float),
                                                local, d['y_ids'][ann], raw[q])
            pred = {name: s.argmax(1) for name, s in score.items()}
            for name in ARMS[:3]:
                pred[name] = parent['predictions'][i, list(parent['arm_names']).index(name)]
            scores.append(np.stack([score[name] for name in NEW_ARMS]))
            predictions.append(np.stack([pred[name] for name in ARMS]))
            corrections.append(correction)
        directory = out/path.stem
        directory.mkdir()
        np.savez_compressed(directory/'policies.npz', query_rows=q, source_rows=source,
            support_rows=parent['support_rows'], eval_masks=parent['eval_masks'], seeds=np.asarray(old.SEEDS),
            arm_names=np.asarray(ARMS), score_names=np.asarray(NEW_ARMS), predictions=np.stack(predictions),
            scores=np.stack(scores), source_center=center, source_scale=scale,
            **{key: np.stack([c[key] for c in corrections]) for key in corrections[0]})
        inventory.append(dict(port=path.stem, queries=len(q), source_rows=len(source),
                              source_head_max_error=head_error, source_head_argmax_mismatch=mismatch,
                              replay_near_ties_only=True, original_prior_scores_unchanged=True))
        print(f'FROZEN {path.stem}, head replay error={head_error:.3g}', flush=True)
    old.require(len(inventory) == 23, 'expected 23 ports')
    write_json(out/'POLICIES_SAVED.json', inventory)
    return inventory


def metrics(predictions, labels, ev):
    truth = labels[ev]
    classes, inverse, counts = np.unique(truth, return_inverse=True, return_counts=True)
    weight = 1./(len(classes)*counts[inverse])
    expert, base = predictions['fusion'][ev], predictions['base'][ev]
    results = {}
    for name, full in predictions.items():
        p = full[ev]
        correct = p == truth
        recall = [float(correct[truth == c].mean()) if c in classes else None for c in range(8)]
        results[name] = dict(ba=float(weight[correct].sum()), acc=float(correct.mean()),
            recover_ba=float(weight[correct & (expert != truth)].sum()),
            harm_ba=float(weight[~correct & (expert == truth)].sum()),
            rescue_both_wrong_ba=float(weight[correct & (expert != truth) & (base != truth)].sum()),
            changed_from_fusion=int((p != expert).sum()), class_recall=recall)
    return results


def evaluate(out, inventory):
    records, flat = [], []
    for item in inventory:
        port = item['port']
        with np.load(ROOT/'pred_cache'/(port+'.npz'), allow_pickle=False) as a:
            labels = a['y_ids']
        with np.load(out/port/'policies.npz', allow_pickle=False) as a:
            policy = {k: a[k] for k in a.files}
        prior = json.loads((ROOT/'oof_reliability_results_v1'/(port+'.json')).read_text(encoding='utf-8'))
        record = dict(port=port, seeds=[])
        for i, seed in enumerate(old.SEEDS):
            pred = dict(zip(policy['arm_names'], policy['predictions'][i]))
            result = metrics(pred, labels[policy['query_rows']], policy['eval_masks'][i])
            for name in ARMS[:3]:
                for metric in ('ba', 'acc'):
                    old.require(abs(result[name][metric]-prior['seeds'][i]['metrics'][name][metric]) < 1e-12,
                                'parent metric mismatch')
            for name, values in result.items():
                flat.append(dict(port=port, seed=seed, arm=name,
                    **{k: v for k, v in values.items() if k != 'class_recall'},
                    evaluated_rows=int(policy['eval_masks'][i].sum())))
            record['seeds'].append(dict(seed=seed, metrics=result))
        write_json(out/port/'metrics.json', record)
        records.append(record)
    dimensions = ('ba', 'acc', 'recover_ba', 'harm_ba', 'rescue_both_wrong_ba')
    values = {a: {m: np.array([np.mean([s['metrics'][a][m] for s in r['seeds']]) for r in records])
                  for m in dimensions} for a in ARMS}
    sampled = np.random.default_rng(0).integers(0, len(records), (10000, len(records)))
    comparisons = {}
    for a, b in CONTRASTS:
        delta = values[a]['ba']-values[b]['ba']
        comparisons[a+' minus '+b] = dict(mean_delta_ba_pp=float(100*delta.mean()),
            mean_delta_acc_pp=float(100*(values[a]['acc']-values[b]['acc']).mean()),
            positive_ports=int((delta > 1e-12).sum()), negative_ports=int((delta < -1e-12).sum()),
            port_bootstrap_95ci_pp=(100*np.percentile(delta[sampled].mean(1), [2.5, 97.5])).tolist(),
            draw_deltas_pp=[float(100*np.mean([r['seeds'][i]['metrics'][a]['ba']-r['seeds'][i]['metrics'][b]['ba']
                                             for r in records])) for i in range(len(old.SEEDS))])
    summary = dict(n_ports=len(records), contexts=len(records)*len(old.SEEDS),
        means={a: {m: float(v.mean()) for m, v in block.items()} for a, block in values.items()},
        comparisons=comparisons,
        passes_development_screen=all(comparisons[a+' minus '+b]['mean_delta_ba_pp'] > 0 for a, b in CONTRASTS[:2]),
        independent_confirmation=False, novelty_established=False,
        protocol='64 labels, fixed strength8, full-query expert quality; not 5% handoff',
        statistical_unit='port mean over five overlapping support draws')
    with (out/'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    write_json(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    old.require(not args.out.exists(), 'use a new output directory')
    args.out.mkdir(parents=True)
    write_json(args.out/'RUN_CONFIG.json', dict(date='2026-10-07', arms=ARMS, contrasts=CONTRASTS,
        strength=method.STRENGTH, support_labels=64,
        protocol=str(Path(__file__).with_name('RESIDUAL_EXPERT_PROTOCOL_20261007.md'))))
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        inventory = freeze(args.out)
        summary = evaluate(args.out, inventory)
    write_json(args.out/'COMPLETE.json', dict(status='complete', seconds=time.perf_counter()-start,
        contexts=len(inventory)*len(old.SEEDS), target_head_fits=len(inventory)*len(old.SEEDS)*4,
        reproduced_source_heads=len(inventory), backbone_fits=0))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
