"""One no-fit support-class restriction control, separately declared."""

import argparse
import json
from pathlib import Path

import numpy as np

from run_budgeted_residual_handoff import ROOT, archive, require, write_json
from budgeted_residual_handoff import score_policy


PARENT = ROOT/'oof_base_adapt_results_20261007'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); require(not args.out.exists(), 'use a new output directory')
    args.out.mkdir(parents=True)
    write_json(args.out/'RUN_CONFIG.json', dict(post_readout_followup=True, unique_target_labels=64,
        recipe='argmax unchanged source base logits restricted to support-observed classes', expert_admissions=0,
        protocol=str(Path(__file__).with_name('BASE_COVERAGE_CONTROL_20261007.md'))))
    inventory = json.loads((PARENT/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    for item in inventory:
        port = item['port']
        p, d = archive(PARENT/port/'policies.npz'), archive(ROOT/'pred_cache'/(port+'.npz'))
        q, support = p['query_rows'], p['support_rows']
        require(np.array_equal(q, d['qpos']), 'query mismatch')
        predictions, seen = [], []
        for i in range(5):
            observed = np.bincount(d['y_ids'][support[i]], minlength=8) > 0
            predictions.append(np.where(observed[None], d['base_logits'][q], -np.inf).argmax(1))
            seen.append(observed)
            require(np.array_equal(p['eval_masks'][i], ~np.isin(q, support[i])), 'support exclusion mismatch')
        np.savez_compressed(args.out/(port+'.npz'), query_rows=q, support_rows=support, eval_masks=p['eval_masks'],
            observed_classes=np.stack(seen), predictions=np.stack(predictions))
    write_json(args.out/'POLICIES_SAVED.json', inventory)
    rows, parent_rows = [], []
    for item in inventory:
        port = item['port']
        p, d = archive(args.out/(port+'.npz')), archive(ROOT/'pred_cache'/(port+'.npz'))
        q = p['query_rows']; base = d['base_logits'][q].argmax(1)
        for i in range(5):
            ev, pred = p['eval_masks'][i], p['predictions'][i]
            metric = score_policy(pred, base, d['y_ids'][q], ev)
            truth, guess = d['y_ids'][q][ev], pred[ev]
            independent_ba = np.mean([np.mean(guess[truth == c] == c) for c in np.unique(truth)])
            require(abs(metric['ba']-independent_ba) < 1e-12, 'independent recall mean mismatch')
            rows.append(dict(port=port, seed=i, **metric, admission_fraction=0.,
                original_change_fraction=float(np.mean(pred[ev] != base[ev]))))
        parent_rows.extend(json.loads((PARENT/port/'metrics.json').read_text(encoding='utf-8')))
    write_json(args.out/'metrics.json', rows)
    ports = [r['port'] for r in inventory]
    values = {m: np.array([[next(r[m] for r in rows if r['port'] == port and r['seed'] == i)
                             for i in range(5)] for port in ports]) for m in ('ba', 'acc', 'original_change_fraction')}
    sampled = np.random.default_rng(0).integers(0, len(ports), (10000, len(ports)))
    comparisons = []
    for arm in ('original_base', 'base_bias', 'base_visual', 'visual_handoff'):
        other = {m: np.array([[next(r[m] for r in parent_rows if r['port'] == port and r['seed'] == i and r['arm'] == arm)
                                for i in range(5)] for port in ports]) for m in ('ba', 'acc')}
        delta = other['ba']-values['ba']; by_port = delta.mean(1)
        comparisons.append(dict(candidate=arm, control='observed_base', mean_delta_ba_pp=float(100*delta.mean()),
            mean_delta_acc_pp=float(100*(other['acc']-values['acc']).mean()), positive_ports=int((by_port > 1e-12).sum()),
            negative_ports=int((by_port < -1e-12).sum()), port_bootstrap_95ci_pp=(100*np.percentile(by_port[sampled].mean(1), [2.5, 97.5])).tolist(),
            draw_deltas_pp=(100*delta.mean(0)).tolist()))
    summary = dict(status='complete', post_readout_followup=True, n_ports=23, contexts=115, fits=0,
        mean={m: float(v.mean()) for m, v in values.items()}, comparisons=comparisons,
        original_setting_unchanged=False, expert_admissions=0, independent_confirmation=False)
    write_json(args.out/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
