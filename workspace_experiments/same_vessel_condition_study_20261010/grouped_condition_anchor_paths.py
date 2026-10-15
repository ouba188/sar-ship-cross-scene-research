"""Fixed-checkpoint prior/kernel attribution; never selects a deployment path."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
import torch
from grouped_condition_anchor import BASE, CACHE, OUT, FactorNet, SEEDS, standardize, metrics, save_json
from conditional_evidence_gate import EvidenceGate, SUPPORTS
from target_response_adapter import residual_head


def changed_features(raw, h, split, fixed, arm, seed):
    model = FactorNet()
    model.load_state_dict(torch.load(OUT/f'{arm}_{seed}_factor.pt', map_location='cpu', weights_only=True)['model'])
    model.eval()
    gate = EvidenceGate(fixed['basis'])
    gate.load_state_dict(torch.load(OUT/f'{arm}_{seed}_gate.pt', map_location='cpu', weights_only=True))
    gate.eval()
    norm = (raw-split['center'])/split['scale']
    with torch.no_grad():
        context = torch.cat([model.condition(torch.from_numpy(norm[i:i+256])) for i in range(0, len(norm), 256)]).numpy()
        context = torch.from_numpy(standardize(context, split['fit']).astype(np.float32))
        return gate(h, context)


def main():
    torch.set_num_threads(1)
    output = OUT/'path_diagnosis'
    output.mkdir(exist_ok=False)
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(OUT/'source_split.npz') as z:
        split = {k: z[k] for k in z.files}
    fixed = torch.load(OUT/'fixed_classifier.pt', map_location='cpu', weights_only=True)
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], np.float32)
    h = (torch.from_numpy(raw)-fixed['center'])/fixed['scale']/np.sqrt(512)
    weight, bias = fixed['weight'], fixed['bias']
    rows, replay = [], []
    for seed in SEEDS:
        old = changed_features(raw, h, split, fixed, 'current', seed)
        for arm in ('swap', 'anchor', 'joint'):
            new = changed_features(raw, h, split, fixed, arm, seed)
            for path in sorted(SUPPORTS.glob('*_split.npz')):
                task = path.stem.removesuffix('_split')
                with np.load(path) as z:
                    support, query, labels = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
                ids = np.r_[support, query]
                features = (old[ids], new[ids])
                truth = meta.iloc[query].class_id.to_numpy()
                saved = np.load(OUT/f'{task}.npz')
                with torch.no_grad():
                    for p, r in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        prior = features[p]@weight+bias
                        phi = torch.cat([features[r], torch.ones((len(ids), 1))], 1)
                        beta, _, _ = residual_head(phi[:64], prior[:64], torch.tensor(labels), balanced=True)
                        score = prior[64:]+phi[64:]@beta
                        pred = score.argmax(1).numpy()
                        if p == r:
                            reference = saved[f'{arm if p else "current"}_{seed}']
                            np.testing.assert_allclose(score.numpy(), reference, atol=1e-6)
                            np.testing.assert_array_equal(pred, reference.argmax(1))
                            replay.append(dict(arm=arm, seed=seed, task=task, mode=f'P{p}R{r}',
                                               max_score_difference=float(np.max(np.abs(score.numpy()-reference)))))
                        rows.append(dict(arm=arm, seed=seed, task=task, port=meta.iloc[query].port.iloc[0],
                                         mode=f'P{p}R{r}', **metrics(pred, truth)))
                saved.close()
            print('PATH', arm, seed, flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(output/'results.csv', index=False)
    summary = data.groupby(['arm', 'mode'])[['OA', 'BA']].mean().reset_index()
    summary.to_csv(output/'summary.csv', index=False)
    data.groupby(['arm', 'port', 'mode'])[['OA', 'BA']].mean().reset_index().to_csv(output/'ports.csv', index=False)
    pd.DataFrame(replay).to_csv(output/'replay.csv', index=False)
    save_json(output/'COMPLETE.json', dict(cells=len(rows), diagonal_replays=len(replay),
        interpretation='Current arm versus each new arm. P=source-prior input,R=support relation geometry. All64 residual heads refit. Fixed computational diagnosis, not target-selected deployment.'))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
