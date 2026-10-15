"""Source-calibrated scalar softening separates confidence from conditional learning."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
import torch
from decision_condition_consistency import (
    BASE, CACHE, PREVIOUS, OUT, SEEDS, FactorNet, EvidenceGate, SUPPORTS,
    same_ship_pairs, pair_weights, pair_metrics, readout_inputs, metrics, save_json,
)
from target_response_adapter import residual_head


def soften(scores, ratio):
    center = scores.mean(1, keepdim=True)
    return center+ratio*(scores-center)


def fit_ratio(reference, consistent, pairs, weights):
    energies = []
    for values in (reference, consistent):
        centered = values-values.mean(1, keepdims=True)
        energy = (centered**2).mean(1)
        energies.append(float(weights@(.5*(energy[pairs[:, 0]]+energy[pairs[:, 1]]))))
    return float(np.sqrt(energies[1]/energies[0]))


def main():
    torch.set_num_threads(1)
    output = OUT/'softening_diagnosis'
    output.mkdir(exist_ok=True)
    if (output/'COMPLETE.json').exists():
        raise ValueError('Preserve completed diagnosis')
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(OUT/'source_split.npz') as z:
        split = {k: z[k] for k in z.files}
    fit, test = split['fit'], split['test_source']
    tm = meta.iloc[fit].reset_index(drop=True)
    pairs = same_ship_pairs(tm)
    weights_pair = pair_weights(tm, pairs)
    fixed = torch.load(OUT/'fixed_classifier.pt', map_location='cpu', weights_only=True)
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], np.float32)
    norm = (raw-split['center'])/split['scale']
    h = (torch.from_numpy(raw)-fixed['center'])/fixed['scale']/np.sqrt(512)
    weight, bias = fixed['weight'], fixed['bias']
    rows, source_rows, pair_rows, ratios = [], [], [], []
    for seed in SEEDS:
        with np.load(OUT/f'reference_{seed}_fit_scores.npz') as r, np.load(OUT/f'consistent_{seed}_fit_scores.npz') as c:
            np.testing.assert_array_equal(r['obs_ids'], fit)
            np.testing.assert_array_equal(c['obs_ids'], fit)
            ratio = fit_ratio(r['scores'], c['scores'], pairs, weights_pair)
        ratios.append(dict(seed=seed, ratio=ratio, fit_chips=len(fit), paired_ships=int(tm.iloc[pairs[:, 0]].mmsi.nunique())))
        with np.load(OUT/f'reference_{seed}_test_source_scores.npz') as z:
            scores = torch.from_numpy(z['scores'])
        calibrated = soften(scores, ratio)
        np.testing.assert_array_equal(calibrated.argmax(1).numpy(), scores.argmax(1).numpy())
        source_rows.append(dict(seed=seed, **metrics(calibrated.argmax(1).numpy(), meta.iloc[test].class_id.to_numpy()),
                               **pair_metrics(calibrated, meta.iloc[test].reset_index(drop=True))))
        encoder = FactorNet()
        encoder.load_state_dict(torch.load(PREVIOUS/f'anchor_{seed}_factor.pt', map_location='cpu', weights_only=True)['model'])
        encoder.eval().requires_grad_(False)
        gate = EvidenceGate(fixed['basis'])
        gate.load_state_dict(torch.load(OUT/f'reference_{seed}.pt', map_location='cpu', weights_only=True))
        gate.eval().requires_grad_(False)
        with np.load(OUT/f'context_normalization_{seed}.npz') as z:
            center, scale = z['center'], z['scale']
        with torch.no_grad():
            context = torch.cat([encoder.condition(torch.from_numpy(norm[i:i+256])) for i in range(0, len(norm), 256)]).numpy()
            context = torch.from_numpy(((context-center)/scale).astype(np.float32))
            for path in sorted(SUPPORTS.glob('*_split.npz')):
                task = path.stem.removesuffix('_split')
                with np.load(path) as z:
                    support, query, y = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
                ids = np.r_[support, query]
                changed = gate(h[ids], context[ids])
                qm = meta.iloc[query].reset_index(drop=True)
                paired_query = same_ship_pairs(qm)
                archive = dict(support_obs_ids=support, query_obs_ids=query)
                for route in ('both', 'prior'):
                    phi, prior = readout_inputs(h[ids], changed, weight, bias, route)
                    # The unsoftened diagonal must reproduce the actual reference path.
                    beta, _, _ = residual_head(phi[:64], prior[:64], torch.tensor(y), balanced=True)
                    replay = prior[64:]+phi[64:]@beta
                    with np.load(OUT/f'{task}.npz') as z:
                        np.testing.assert_array_equal(replay.numpy(), z[f'reference_{route}_{seed}'])
                    prior = soften(prior, ratio)
                    beta, _, _ = residual_head(phi[:64], prior[:64], torch.tensor(y), balanced=True)
                    score = prior[64:]+phi[64:]@beta
                    pred = score.argmax(1).numpy()
                    row = dict(arm='softened_'+route, seed=seed, task=task, port=qm.port.iloc[0], **metrics(pred, qm.class_id.to_numpy()))
                    rows.append(row)
                    pair_rows.append(dict(arm='softened_'+route, seed=seed, task=task, **pair_metrics(score, qm, paired_query)))
                    archive[route] = score.numpy()
                np.savez_compressed(output/f'{task}_{seed}.npz', **archive)
        print('SOFTEN', ratios[-1], source_rows[-1], flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(output/'target_results.csv', index=False)
    data.groupby('arm')[['OA', 'BA']].mean().reset_index().to_csv(output/'target_summary.csv', index=False)
    pd.DataFrame(ratios).to_csv(output/'ratios.csv', index=False)
    pd.DataFrame(source_rows).to_csv(output/'source_diagnosis.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(output/'target_pair_diagnosis.csv', index=False)
    save_json(output/'COMPLETE.json', dict(target_rows=len(data), diagonal_replays=36,
        definition='Positive scalar matches source-pair-weighted centered score RMS of consistent to reference. Source-only calibration, no new training, no query selection. Preserve row means. Raw-source argmax unchanged;64-support residual head refit on each route.',
        limitation='Matching source magnitude does not identify all mechanisms; use as competing explanation, not a target-selected replacement.'))
    print(data.groupby('arm')[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
