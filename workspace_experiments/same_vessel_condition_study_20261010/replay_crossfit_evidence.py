"""Read exported full pipelines and verify genuine raw-feature deployment."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from run_crossfit_evidence import OUT, METHODS, MODES, SEEDS
from train_explicit_candidate import load_data, save_json
from explicit_candidate_projection import identity_episode
from paired_evidence_factors import score_probe, support_operator, adapted_scores


def encode_state(x, state):
    q, r = torch.linalg.qr(state['frame'], mode='reduced')
    q *= torch.where(r.diagonal() < 0, -1., 1.)
    scale = ((state['covariance']@q)*q).sum(0).clamp_min(1e-6).sqrt()
    z = ((x-state['center'])@q)/scale/np.sqrt(q.shape[1])
    return z, F.linear(z, state['head.weight'], state['head.bias'])


def exported_prediction(raw, seed, folder=OUT, name='full'):
    with np.load(folder/f'{name}_{seed}_normalization.npz') as f:
        x = torch.tensor((raw-f['raw_center'])/f['raw_scale'], dtype=torch.float32)
    case = torch.load(folder/f'{name}_{seed}_case.pt', map_location='cpu', weights_only=True)['selected']
    prior = torch.load(folder/f'{name}_{seed}_prior.pt', map_location='cpu', weights_only=True)['selected']
    with torch.no_grad():
        kernel, _ = encode_state(x, case)
        z, score = encode_state(x, prior)
    features = {}
    for method in METHODS:
        with np.load(folder/f'{name}_{seed}_{method}_map.npz') as mapping:
            f = ((z.numpy()-mapping['center'])@mapping['whitener']@mapping['q'])/np.sqrt(7)
            features[method] = score_probe(f, mapping['probe'])@mapping['contrasts']/mapping['rms']
    return dict(prior=score.numpy(), kernel=kernel.numpy(), features=features)


def main():
    torch.set_num_threads(1)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    ids = split['test_source']
    mm = meta.iloc[ids].reset_index(drop=True); labels = mm.class_id.to_numpy()
    s, q = identity_episode(mm, np.random.default_rng(91711), query_size=None)
    rows = []
    for seed in SEEDS:
        replay = exported_prediction(raw[ids], seed)
        with np.load(OUT/f'full_{seed}_outputs.npz') as f:
            for key in ('prior', 'kernel'):
                np.testing.assert_allclose(replay[key], f[key][ids], atol=1e-5, rtol=1e-5)
            np.testing.assert_array_equal(replay['prior'].argmax(1), f['prior'][ids].argmax(1))
            for method in METHODS:
                stored = f['feature_'+method][ids]
                np.testing.assert_allclose(replay['features'][method], stored, atol=1e-5, rtol=1e-5)
                for mode in MODES:
                    with np.load(OUT/f'correction_{mode}_{method}_{seed}.npz') as correction:
                        for gain in sorted(set((0., 1., float(correction['selected_gain'])))):
                            p1 = replay['prior']+gain*score_probe(replay['features'][method], correction['coefficient'])
                            p2 = f['prior'][ids]+gain*score_probe(stored, correction['coefficient'])
                            op1, present = support_operator(replay['kernel'][s], replay['kernel'][q], labels[s])
                            op2, present2 = support_operator(f['kernel'][ids][s], f['kernel'][ids][q], labels[s])
                            a = adapted_scores(p1[s], p1[q], op1, present, labels[s])
                            b = adapted_scores(p2[s], p2[q], op2, present2, labels[s])
                            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
                            np.testing.assert_array_equal(a.argmax(1), b.argmax(1))
                            rows.append(dict(seed=seed, mode=mode, method=method, gain=gain,
                                max_score_error=float(np.abs(a-b).max()), query_chips=len(q), changed_predictions=0))
    pd.DataFrame(rows).to_csv(OUT/'deployment_replay.csv', index=False)
    save_json(OUT/'DEPLOYMENT_VERIFIED.json', dict(tested_cases=len(rows), raw_features=True,
        support_query_corrected_together=True, no_target_statistics=True, source_held_only=True))
    print(pd.DataFrame(rows).groupby(['mode', 'method']).max_score_error.max().to_string(), flush=True)


if __name__ == '__main__':
    main()
