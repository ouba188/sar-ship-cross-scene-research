"""Package lossless 7+25 score decompositions and verify saved-model parity."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from run_paired_evidence_factors import OUT, PARENT, ORIGINS, SEEDS
from train_explicit_candidate import load_data, save_json
from paired_evidence_factors import factorized_scores


def main():
    torch.set_num_threads(1)
    folder = OUT/'modules'; folder.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    held = split['test_source']; checks, examples = [], []
    for origin in ORIGINS:
        for seed in SEEDS:
            state = torch.load(PARENT/f'{origin.removeprefix("split_")}_{seed}.pt', map_location='cpu', weights_only=True)['selected']
            with np.load(PARENT/f'{origin}_{seed}_vectors.npz') as archive:
                old_score = F.linear(torch.tensor(archive['z'][held]), state['head.weight'], state['head.bias']).numpy()
            for method in ('paired', 'head'):
                with np.load(OUT/f'{origin}_{method}_{seed}_mapping.npz') as archive:
                    mapping = {k: archive[k] for k in archive.files}
                mapping.update(head_weight=state['head.weight'].numpy(), head_bias=state['head.bias'].numpy())
                np.savez_compressed(folder/f'{origin}_{method}_{seed}.npz', **mapping)
                result = factorized_scores(raw[held], mapping)
                error = float(np.max(np.abs(old_score-result['scores'])))
                np.testing.assert_allclose(result['scores'], old_score, atol=1e-5, rtol=1e-5)
                np.testing.assert_array_equal(result['scores'].argmax(1), old_score.argmax(1))
                checks.append(dict(origin=origin, seed=seed, method=method, held_chips=len(held),
                                   max_score_error=error, changed_predictions=0))
                for j, obs in enumerate(held):
                    y = int(meta.iloc[obs].class_id)
                    rival_scores = old_score[j].copy(); rival_scores[y] = -np.inf
                    rival = int(rival_scores.argmax())
                    parts = {name: float(result[name][j, y]-result[name][j, rival])
                             for name in ('candidate_score', 'remaining_score')}
                    bias = float(result['intercept'][y]-result['intercept'][rival])
                    examples.append(dict(origin=origin, seed=seed, method=method, obs_id=int(obs),
                        mmsi=int(meta.iloc[obs].mmsi), port=meta.iloc[obs].port,
                        true_class=y, strongest_rival=rival,
                        **parts, intercept_margin=bias,
                        total_margin=float(old_score[j, y]-old_score[j, rival])))
    pd.DataFrame(checks).to_csv(folder/'parity_checks.csv', index=False)
    pd.DataFrame(examples).to_csv(folder/'held_margin_decomposition.csv', index=False)
    save_json(folder/'COMPLETE.json', dict(exported_models=len(checks), tested_held_chips=len(held),
        purpose='Lossless additive decision explanation; no improved accuracy claim, no removal of remaining factors.',
        main_model_modified=False))
    print(pd.DataFrame(checks).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
