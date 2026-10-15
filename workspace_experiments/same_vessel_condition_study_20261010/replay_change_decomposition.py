"""Reconstruct new evidence and final support readout from raw frozen features."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
import torch
from train_explicit_candidate import load_data, save_json, SEEDS
from run_change_decomposition import OUT, PARENT, ARMS
from replay_crossfit_evidence import encode_state, exported_prediction
from change_decomposition import EvidenceDecomposition, relation_delta
from paired_evidence_factors import score_probe, support_operator, adapted_scores


def encode_evidence(raw, seed, arm):
    with np.load(PARENT/f'full_{seed}_normalization.npz') as f:
        x = torch.tensor((raw-f['raw_center'])/f['raw_scale'], dtype=torch.float32)
    state = torch.load(PARENT/f'full_{seed}_prior.pt', map_location='cpu', weights_only=True)['selected']
    with torch.no_grad():
        z, _ = encode_state(x, state)
    model = EvidenceDecomposition()
    model.load_state_dict(torch.load(OUT/f'full_{seed}_{arm}.pt', map_location='cpu', weights_only=True)['selected'])
    with np.load(OUT/f'full_{seed}_{arm}_map.npz') as f:
        inputs = torch.tensor((z.numpy()-f['center'])@f['whitener'], dtype=torch.float32)
        with torch.no_grad():
            content = model(inputs)['content'].numpy()
        features = score_probe(content, f['probe'])@f['contrasts']/f['rms']
    return content, features


def main():
    torch.set_num_threads(1)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    ids = split['test_source']; labels = meta.class_id.to_numpy()
    position = {int(v): j for j, v in enumerate(ids)}
    rows = []
    for seed in SEEDS:
        base = exported_prediction(raw[ids], seed)
        for arm in ARMS:
            content, features = encode_evidence(raw[ids], seed, arm)
            with np.load(OUT/f'full_{seed}_{arm}_decomposition.npz') as f:
                np.testing.assert_allclose(content, f['content'][ids], atol=1e-5, rtol=1e-5)
            for use in ('global', 'relation'):
                with np.load(OUT/f'correction_{seed}_{arm}_{use}.npz') as f:
                    gain = float(f['selected_gain'])
                    delta = score_probe(features, f['coefficient']) if use == 'global' else relation_delta(features, base['prior'], f['coefficient'])
                    prior = base['prior']+gain*delta
                for draw in range(3):
                    with np.load(OUT/f'{seed}_source_{draw}_predictions.npz') as f:
                        s, q = [np.array([position[int(v)] for v in f[key]]) for key in ('support_obs_ids', 'query_obs_ids')]
                        sy = labels[f['support_obs_ids']]
                        op, present = support_operator(base['kernel'][s], base['kernel'][q], sy)
                        scores = adapted_scores(prior[s], prior[q], op, present, sy)
                        reference = f[f'{arm}_{use}']
                        np.testing.assert_allclose(scores, reference, atol=1e-5, rtol=1e-5)
                        np.testing.assert_array_equal(scores.argmax(1), reference.argmax(1))
                        rows.append(dict(seed=seed, arm=arm, use=use, draw=draw, gain=gain,
                            max_score_error=float(np.abs(scores-reference).max()), changed_predictions=0))
    pd.DataFrame(rows).to_csv(OUT/'deployment_replay.csv', index=False)
    save_json(OUT/'DEPLOYMENT_VERIFIED.json', dict(cases=len(rows), raw_input=True,
        predictions_identical=True, max_score_error=max(r['max_score_error'] for r in rows)))
    print(pd.DataFrame(rows).groupby(['arm', 'use']).max_score_error.max().to_string(), flush=True)


if __name__ == '__main__':
    main()
