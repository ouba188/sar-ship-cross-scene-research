"""Post-freeze matched comparison probe; no target-driven model updates."""

import json
import numpy as np
import pandas as pd
import torch

from frequency_joint_run import OUT, EXP, ROOT, compare
from prepare_learned_frequency import SOURCE, DATASETS, load_npz
from frequency_reliability_core import kernels, coefficients


def main():
    if not json.loads((OUT/'STATUS.json').read_text())['complete']:
        raise ValueError('Complete prediction freeze before label-based diagnosis')
    ys = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    records, selections = [], {}
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        folder = marker.parent
        info = json.loads((folder/'PREPARED.json').read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        dst = OUT/'tasks'/info['dataset']/info['task']
        data = load_npz(folder/'target.npz')
        policies = load_npz(OUT/'targets'/info['dataset']/info['task']/'policies.npz')
        modes = list(policies['modes'])
        old = load_npz(folder/'selective_embedding.npz')['embedding']
        embeddings = dict(learned=old, **{arm: load_npz(dst/f'{arm}_embedding.npz')['embedding']
            for arm in ('continued_ce', 'joint_kernel')})
        y = ys[data['rows']] if info['dataset'] == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        for draw, support in enumerate(data['support']):
            sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
            rng = np.random.default_rng(20261009+draw)
            pool = np.flatnonzero(ev)
            uniform = np.sort(rng.choice(pool, min(512, len(pool)), replace=False))
            disagreement = np.flatnonzero(ev & (policies['base_predictions'][draw, modes.index('learned')] !=
                policies['base_predictions'][draw, modes.index('joint_kernel')]))
            hard = np.sort(rng.choice(disagreement, min(512, len(disagreement)), replace=False))
            rows = np.union1d(uniform, hard)
            ix = np.r_[support, rows]
            selected_data = {k: data[k][ix] for k in ('phi', 'prior', 'mean', 'conditions')}
            _, kernel = kernels(selected_data, old[ix].astype(float), np.arange(64), info['eta'])
            score = selected_data['prior']+kernel@coefficients(kernel[:64], sy, selected_data['prior'][:64])
            counts = np.bincount(sy, minlength=8)
            logits = score[64:].copy()
            rr, labels = np.arange(len(rows)), y[rows]
            logits[:, counts < 2] = -np.inf
            logits[rr, labels] = -np.inf
            negative = logits.argmax(1)
            valid = (counts[labels] >= 2) & np.isfinite(logits[rr, negative])
            for arm, embedding in embeddings.items():
                visual = compare(embedding, data['conditions'], support, rows, sy)
                margin = visual[rr, labels]-visual[rr, negative]
                for cohort, members in [('uniform', uniform), ('disagreement', hard)]:
                    mask = np.isin(rows, members) & valid
                    if not mask.any():
                        continue
                    _, inverse, class_count = np.unique(labels[mask], return_inverse=True, return_counts=True)
                    weight = 1./class_count[inverse]
                    records.append(dict(**meta, draw=draw, mode=arm, cohort=cohort, n=int(mask.sum()),
                        pair_ba=100*float(np.average(margin[mask] > 0, weights=weight)),
                        pair_loss=float(np.average(np.logaddexp(0, -margin[mask]/.1), weights=weight)),
                        mean_margin=float(np.average(margin[mask], weights=weight))))
            selections[f'{info["dataset"]}__{info["task"]}__{draw}__uniform'] = uniform
            selections[f'{info["dataset"]}__{info["task"]}__{draw}__disagreement'] = hard
        print(f'JOINT PROBE {info["task"]}', flush=True)
    result = pd.DataFrame(records)
    result.to_csv(OUT/'comparison_probe.csv', index=False)
    np.savez_compressed(OUT/'comparison_probe_rows.npz', **selections)
    summary = result.groupby(['dataset', 'task', 'cohort', 'mode'])[['pair_ba', 'pair_loss', 'mean_margin']].mean().groupby(
        ['dataset', 'cohort', 'mode']).mean()
    summary.to_csv(OUT/'comparison_probe_summary.csv')
    print(summary.to_string())


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
