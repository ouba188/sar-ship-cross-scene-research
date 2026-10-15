"""One prespecified post-readout alternative explanation, no fitting."""

import argparse
import json
from pathlib import Path

import numpy as np

from run_enriched_reliability import write_json
from run_residual_expert import ROOT, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    out = args.run/'coverage_control'
    if out.exists():
        raise ValueError('control output already exists')
    out.mkdir()
    inventory = json.loads((args.run/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    for item in inventory:
        port = item['port']
        with np.load(ROOT/'pred_cache'/(port+'.npz'), allow_pickle=False) as a:
            scores, labels = a['expert_logits'], a['y_ids']
        with np.load(args.run/port/'policies.npz', allow_pickle=False) as a:
            q, support = a['query_rows'], a['support_rows']
        predictions, masks = [], []
        for rows in support:
            seen = np.bincount(labels[rows], minlength=8) > 0
            predictions.append(np.where(seen[None], scores[q], -np.inf).argmax(1))
            masks.append(seen)
        np.savez_compressed(out/(port+'.npz'), query_rows=q, support_rows=support,
                            predictions=np.stack(predictions), seen=np.stack(masks))
    write_json(out/'POLICIES_SAVED.json', dict(ports=len(inventory), rule='argmax source fusion over support-observed classes'))
    records = []
    for item in inventory:
        port = item['port']
        with np.load(ROOT/'pred_cache'/(port+'.npz'), allow_pickle=False) as a:
            labels = a['y_ids']
        with np.load(args.run/port/'policies.npz', allow_pickle=False) as a:
            policies = {k: a[k] for k in a.files}
        with np.load(out/(port+'.npz'), allow_pickle=False) as a:
            prediction = a['predictions']
        seeds = []
        for i in range(5):
            pred = dict(zip(policies['arm_names'], policies['predictions'][i]))
            pred['fusion_observed64'] = prediction[i]
            seeds.append(metrics(pred, labels[policies['query_rows']], policies['eval_masks'][i]))
        records.append(dict(port=port, seeds=seeds))
    arms = ('fusion', 'oof_lr0', 'anchor_bias64', 'anchor_visual64', 'fusion_observed64')
    values = {arm: {metric: np.array([np.mean([s[arm][metric] for s in r['seeds']]) for r in records])
                    for metric in ('ba', 'acc')} for arm in arms}
    samples = np.random.default_rng(0).integers(0, 23, (10000, 23))
    contrasts = {}
    for arm in arms[:-1]:
        delta = values[arm]['ba']-values['fusion_observed64']['ba']
        contrasts[arm+' minus fusion_observed64'] = dict(mean_delta_ba_pp=float(100*delta.mean()),
            positive_ports=int((delta > 1e-12).sum()), negative_ports=int((delta < -1e-12).sum()),
            port_bootstrap_95ci_pp=(100*np.percentile(delta[samples].mean(1), [2.5, 97.5])).tolist())
    summary = dict(post_readout_followup=True, no_fitting=True, n_ports=23, contexts=115,
        means={a: {m: float(v.mean()) for m, v in block.items()} for a, block in values.items()},
        contrasts=contrasts)
    write_json(out/'metrics.json', records)
    write_json(out/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
