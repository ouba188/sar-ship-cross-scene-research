"""Freeze the fixed facility pilot, then read its development results."""

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import context_reliability as method
import run_oof_reliability as old
from run_enriched_reliability import write_json


ROOT = Path(__file__).resolve().parent.parent
CONTRASTS = [('covered_values', 'covered_mask'), ('covered_values', 'oof_lr0'),
             ('covered_mask', 'covered_conf'), ('covered_conf', 'oof_lr0'),
             ('covered_values', 'simple')]
REFERENCES = ('base', 'fusion', 'simple', 's2')


def freeze(out, facility, manifest):
    with np.load(facility, allow_pickle=False) as a:
        values, known = a['values'].astype(float), a['known']
        ids, rows = a['instance_id'], a['row_index']
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        manifest_ids = [r['instance_id'] for r in csv.DictReader(f)]
    old.require(np.array_equal(ids, manifest_ids) and np.array_equal(rows, np.arange(len(ids))),
                'facility row alignment mismatch')
    old.require(values.shape == known.shape and values.shape[1] == 60
                and np.isfinite(values).all(), 'invalid facility array')
    inventory = []
    for path in sorted((ROOT / 'pred_cache').glob('*.npz')):
        with np.load(path, allow_pickle=False) as a:
            if 'skipped' in a.files:
                continue
            d = {k: a[k] for k in a.files}
        q = d['qpos']
        if len(q) <= old.ANNOT:
            continue
        oof = old.load_oof(ROOT / 'oof_predictions_v1' / path.name, d, path, path.stem, manifest)
        source = oof['source_rows']
        model = method.fit_source(oof['oof_base_logits'], oof['oof_expert_logits'],
                                  oof['source_labels'], oof['source_ports'], values[source], known[source])
        scores, covered = method.predict_scores(model, d['base_logits'][q], d['expert_logits'][q], values[q], known[q])
        with np.load(ROOT / 'oof_enriched_results_20261007' / path.stem / 'policies.npz', allow_pickle=False) as a:
            parent = {k: a[k] for k in a.files}
        old.require(np.array_equal(q, parent['query_rows']), 'reference query mismatch')
        predictions = []
        names = list(method.ARMS) + list(REFERENCES)
        for i, seed in enumerate(old.SEEDS):
            ann = np.random.default_rng(seed).choice(q, old.ANNOT, replace=False)
            old.require(np.array_equal(ann, parent['support_rows'][i])
                        and np.array_equal(~np.isin(q, ann), parent['eval_masks'][i]), 'support mismatch')
            pred = method.route(scores, d['base_logits'][q].argmax(1), d['expert_logits'][q].argmax(1), d['y_ids'][ann])
            for name in REFERENCES:
                pred[name] = parent['predictions'][i, list(parent['arm_names']).index(name)]
            for name in method.ARMS[1:]:
                old.require(np.array_equal(pred[name][~covered], pred['oof_lr0'][~covered]), 'missing fallback broken')
            predictions.append(np.stack([pred[name] for name in names]))
        directory = out / path.stem
        directory.mkdir()
        with (directory / 'source_model.pkl').open('wb') as f:
            pickle.dump(model, f)
        np.savez_compressed(directory / 'policies.npz', query_rows=q, source_rows=source,
                            support_rows=parent['support_rows'], eval_masks=parent['eval_masks'],
                            seeds=np.asarray(old.SEEDS), arm_names=np.asarray(names),
                            score_names=np.asarray(method.ARMS), scores=np.stack([scores[a] for a in method.ARMS]),
                            predictions=np.stack(predictions), covered=covered)
        inventory.append(dict(port=path.stem, queries=len(q), covered_queries=int(covered.sum()),
                              covered_source=model['supported_source'], informative_source=model['informative_source'],
                              fallback=model['fallback']))
        print('FROZEN ' + path.stem, flush=True)
    old.require(len(inventory) == 23, 'expected 23 active ports')
    write_json(out / 'POLICIES_SAVED.json', inventory)
    return inventory


def evaluate(out, inventory):
    records, flat = [], []
    max_error = 0.
    for item in inventory:
        port = item['port']
        with np.load(ROOT / 'pred_cache' / (port + '.npz'), allow_pickle=False) as a:
            labels = a['y_ids']
        with np.load(out / port / 'policies.npz', allow_pickle=False) as a:
            policy = {k: a[k] for k in a.files}
        prior = json.loads((ROOT / 'oof_reliability_results_v1' / (port + '.json')).read_text(encoding='utf-8'))
        record = dict(port=port, seeds=[])
        for i, seed in enumerate(old.SEEDS):
            pred = dict(zip(policy['arm_names'], policy['predictions'][i]))
            ev, covered = policy['eval_masks'][i], policy['covered']
            y = labels[policy['query_rows']]
            metrics = old.score_predictions(pred, y, ev)
            supported = old.score_predictions(pred, y, ev & covered) if (ev & covered).any() else None
            for name in ('oof_lr0',) + REFERENCES + ('oracle',):
                for key, value in metrics[name].items():
                    error = abs(value-prior['seeds'][i]['metrics'][name][key])
                    max_error = max(max_error, error)
                    old.require(error < 1e-12, f'parent mismatch {port}/{seed}/{name}/{key}')
            for name, metric in metrics.items():
                flat.append(dict(port=port, seed=seed, arm=name, **metric,
                                 evaluated_rows=int(ev.sum()), covered_rows=int((ev & covered).sum()),
                                 covered_ba=supported[name]['ba'] if supported else None))
            record['seeds'].append(dict(seed=seed, metrics=metrics, supported_metrics=supported))
        write_json(out / port / 'metrics.json', record)
        records.append(record)
    arms = list(records[0]['seeds'][0]['metrics'])
    means = {a: np.array([np.mean([s['metrics'][a]['ba'] for s in r['seeds']]) for r in records]) for a in arms}
    sampled = np.random.default_rng(0).integers(0, len(records), (10000, len(records)))
    comparisons = {}
    for a, b in CONTRASTS:
        delta = means[a]-means[b]
        comparisons[a + ' minus ' + b] = dict(mean_delta_ba_pp=float(100*delta.mean()),
            positive_ports=int((delta > 0).sum()), negative_ports=int((delta < 0).sum()),
            port_bootstrap_95ci_pp=(100*np.percentile(delta[sampled].mean(1), [2.5, 97.5])).tolist(),
            draw_deltas_pp=[float(100*np.mean([r['seeds'][i]['metrics'][a]['ba']-r['seeds'][i]['metrics'][b]['ba']
                                              for r in records])) for i in range(len(old.SEEDS))])
    summary = dict(n_ports=len(records), contexts=len(records)*len(old.SEEDS),
        means_ba={a: float(v.mean()) for a, v in means.items()}, comparisons=comparisons,
        passes_development_screen=all(comparisons[a+' minus '+b]['mean_delta_ba_pp'] > 0 for a, b in CONTRASTS[:2]),
        max_parent_control_error=max_error, independent_confirmation=False, novelty_established=False,
        coverage=inventory, statistical_unit='port mean over five overlapping support draws',
        caveats=['uncapped selection', '64 labels for observed-class guard', 'row-only target isolation',
                 'historically exposed development data', 'map/SAR temporal alignment unresolved'])
    with (out / 'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    write_json(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    old.require(not args.out.exists(), 'use a new output directory')
    args.out.mkdir(parents=True)
    facility = ROOT / 'port_context_audit_20261007/facility_features.npz'
    manifest = Path('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv')
    write_json(args.out / 'RUN_CONFIG.json', dict(date='2026-10-07', arms=method.ARMS,
        contrasts=CONTRASTS, facility=str(facility), manifest=str(manifest), C=1., target_update=False,
        protocol=str(Path(__file__).with_name('CONTEXT_PROTOCOL_20261007.md'))))
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        inventory = freeze(args.out, facility, manifest)
        summary = evaluate(args.out, inventory)
    write_json(args.out / 'COMPLETE.json', dict(status='complete', seconds=time.perf_counter()-start,
        new_head_or_backbone_fits=0, covered_gate_fits=3*sum(not r['fallback'] for r in inventory),
        reproduced_prior_fits=len(inventory)))
    print(json.dumps({k: v for k, v in summary.items() if k != 'coverage'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
