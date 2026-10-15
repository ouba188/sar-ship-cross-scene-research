"""Freeze enriched OOF policies, then evaluate the inherited development queries."""

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import sklearn
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits

import enriched_reliability as method
import run_oof_reliability as old


ROOT = Path(__file__).resolve().parent.parent
CONTRASTS = [('rich_ridge64', 'conf_ridge64'), ('rich_hgb64', 'rich_ridge64'),
             ('rich_ridge0', 'conf_ridge0'), ('rich_hgb0', 'rich_ridge0')]
for family in ('conf_ridge', 'rich_ridge', 'rich_hgb'):
    CONTRASTS.append((family + '64', family + '0'))
for arm in method.ARMS[2:]:
    CONTRASTS.extend([(arm, 'simple'), (arm, 's2'), (arm, 'oof_lr0')])


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def freeze(args):
    paths = sorted(args.pred_cache.glob('*.npz'))
    if args.ports:
        paths = [p for p in paths if p.stem in args.ports]
        old.require({p.stem for p in paths} == set(args.ports), 'missing requested port')
    inventory = []
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            if 'skipped' in a.files:
                continue
            d = {k: a[k] for k in a.files}
        q = d['qpos']
        if len(q) <= old.ANNOT:
            continue
        oof = old.load_oof(args.oof_cache / path.name, d, path, path.stem, args.manifest)
        prior = json.loads((args.parent / (path.stem + '.json')).read_text(encoding='utf-8'))
        models = [method.fit_source(oof['oof_base_logits'], oof['oof_expert_logits'],
                                   oof['source_labels'], oof['source_ports'], rich) for rich in (False, True)]
        directory = args.out / path.stem
        directory.mkdir()
        with (directory / 'source_models.pkl').open('wb') as f:
            pickle.dump(models, f)
        preds, scores, residuals, support, evals = [], [], [], [], []
        arm_names = list(method.ARMS) + ['base', 'fusion', 'simple', 's2']
        for i, seed in enumerate(old.SEEDS):
            ann = np.random.default_rng(seed).choice(q, old.ANNOT, replace=False)
            old.require(np.array_equal(ann, prior['seeds'][i]['annotation_rows'])
                        and prior['seeds'][i]['seed'] == seed, 'parent support mismatch')
            ev = ~np.isin(q, ann)
            local = np.flatnonzero(~ev)
            score, pred, residual = {}, {}, {}
            for model in models:
                ss, pp, rr = method.predict(model, d['base_logits'][q], d['expert_logits'][q],
                                             local, d['y_ids'][q[local]])
                score.update(ss); pred.update(pp); residual.update(rr)
            pred.update(old.reference_predictions(d, ann))
            preds.append(np.stack([pred[a] for a in arm_names]))
            scores.append(np.stack([score[a] for a in method.ARMS]))
            residuals.append(residual)
            support.append(ann)
            evals.append(ev)
        np.savez_compressed(directory / 'policies.npz', query_rows=q,
                            support_rows=np.stack(support), eval_masks=np.stack(evals),
                            arm_names=np.asarray(arm_names), score_names=np.asarray(method.ARMS),
                            seeds=np.asarray(old.SEEDS), predictions=np.stack(preds), scores=np.stack(scores),
                            **{a + '_residual': np.stack([r[a] for r in residuals]) for a in residuals[0]})
        inventory.append(dict(port=path.stem, queries=len(q), seeds=len(old.SEEDS),
                              source_rows=len(oof['source_rows']),
                              source_disagreements=models[0]['source_disagreements']))
        print(f'FROZEN {path.stem}: 5 supports, 6 scores + 4 references', flush=True)
    write_json(args.out / 'POLICIES_SAVED.json', inventory)
    return inventory


def summarize(records):
    arms = list(records[0]['seeds'][0]['metrics'])
    values = {a: np.array([np.mean([s['metrics'][a]['ba'] for s in r['seeds']])
                          for r in records]) for a in arms}
    sampled = np.random.default_rng(0).integers(0, len(records), size=(10000, len(records)))
    comparisons = {}
    for a, b in CONTRASTS:
        diff = values[a] - values[b]
        comparisons[a + ' minus ' + b] = dict(
            mean_delta_ba_pp=float(100 * diff.mean()),
            positive_ports=int((diff > 0).sum()), negative_ports=int((diff < 0).sum()),
            port_bootstrap_95ci_pp=(100 * np.percentile(diff[sampled].mean(1), [2.5, 97.5])).tolist(),
            wilcoxon_p=float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.)
    return dict(n_ports=len(records), seeds_per_port=len(old.SEEDS),
                means_ba={a: float(v.mean()) for a, v in values.items()}, comparisons=comparisons,
                statistical_unit='port mean over five overlapping support draws',
                independent_confirmation=False, novelty_established=False)


def evaluate(args, inventory):
    records, flat = [], []
    max_control_error = 0.
    for item in inventory:
        port = item['port']
        with np.load(args.pred_cache / (port + '.npz'), allow_pickle=False) as a:
            labels = a['y_ids']
        with np.load(args.out / port / 'policies.npz', allow_pickle=False) as a:
            policy = {k: a[k] for k in a.files}
        prior = json.loads((args.parent / (port + '.json')).read_text(encoding='utf-8'))
        q = policy['query_rows']
        record = dict(port=port, seeds=[])
        for i, seed in enumerate(old.SEEDS):
            pred = dict(zip(policy['arm_names'], policy['predictions'][i]))
            ev = policy['eval_masks'][i]
            metrics = old.score_predictions(pred, labels[q], ev)
            mapping = dict(conf_ridge0='oof_utility0', conf_ridge64='oof_utility64',
                           base='base', fusion='fusion', simple='simple', s2='s2', oracle='oracle')
            for new, previous in mapping.items():
                for key, value in metrics[new].items():
                    error = abs(value - prior['seeds'][i]['metrics'][previous][key])
                    max_control_error = max(max_control_error, error)
                    old.require(error < 1e-12, f'control readout mismatch {port}/{seed}/{new}/{key}')
            metrics['oof_lr0'] = prior['seeds'][i]['metrics']['oof_lr0']
            metrics['oof_lr64'] = prior['seeds'][i]['metrics']['oof_lr64']
            for arm, m in metrics.items():
                replacement = int(np.count_nonzero((pred[arm] != pred['base']) & ev)) if arm in pred else None
                flat.append(dict(port=port, seed=seed, arm=arm, **m, replacements=replacement,
                                 evaluated_rows=int(ev.sum()),
                                 inherited_metric=arm in ('oof_lr0', 'oof_lr64')))
            record['seeds'].append(dict(seed=seed, metrics=metrics))
        write_json(args.out / port / 'metrics.json', record)
        records.append(record)
    with (args.out / 'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    summary = summarize(records)
    summary['max_parent_control_error'] = max_control_error
    summary['protocol'] = dict(features='4D versus 82D (4+14+64), identical richer input for ridge/HGB',
                               support_labels=64, source_ridge_lambda=old.SOURCE_RIDGE_LAMBDA,
                               target_prior_strength=old.TARGET_PRIOR_STRENGTH,
                               tree_params=method.TREE_PARAMS, source_models_per_port=3,
                               replacement_budget='uncapped, same as parent; not old 5% protocol',
                               target_isolation='row-only; shared MMSI/products possible',
                               sklearn=sklearn.__version__)
    write_json(args.out / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pred-cache', type=Path, default=ROOT / 'pred_cache')
    p.add_argument('--oof-cache', type=Path, default=ROOT / 'oof_predictions_v1')
    p.add_argument('--parent', type=Path, default=ROOT / 'oof_reliability_results_v1')
    p.add_argument('--manifest', type=Path, default=Path('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ports', nargs='+')
    args = p.parse_args()
    old.require(not args.out.exists(), 'use a new output directory; no result overwrite')
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    write_json(args.out / 'RUN_CONFIG.json', dict(date='2026-10-07', paths={k: str(v) for k, v in vars(args).items()},
                arms=method.ARMS, contrasts=CONTRASTS, tree_params=method.TREE_PARAMS,
                source_ridge_lambda=old.SOURCE_RIDGE_LAMBDA, target_prior_strength=old.TARGET_PRIOR_STRENGTH))
    with threadpool_limits(limits=1):
        inventory = freeze(args)
        summary = evaluate(args, inventory)
    write_json(args.out / 'COMPLETE.json', dict(status='complete', ports=len(inventory),
               contexts=len(inventory) * len(old.SEEDS), new_gate_fits=3 * len(inventory),
               new_encoder_or_classification_head_fits=0, seconds=time.perf_counter() - started,
               max_parent_control_error=summary['max_parent_control_error']))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
