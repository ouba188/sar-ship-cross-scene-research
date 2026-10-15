"""Locate already-trained correction failures; no new fitting or target choices."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
from train_explicit_candidate import load_data, SEEDS, save_json, metrics
from run_change_decomposition import OUT, PARENT, ARMS
from crossfit_evidence import assemble_training_view
from change_decomposition import relation_delta, relation_features
from paired_evidence_factors import score_probe, support_operator, adapted_scores


def main():
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    fitting = split['factor_fit']; labels = meta.class_id.to_numpy()
    with np.load(PARENT/'source_folds.npz') as f:
        assignment = f['assignment'].copy()
    rows, scales = [], []
    for seed in SEEDS:
        bundles = {}
        for name in ('full', 'fold0', 'fold1', 'fold2'):
            with np.load(PARENT/f'{name}_{seed}_outputs.npz') as f:
                b = dict(prior=f['prior'], kernel=f['kernel'], fitting=f['fit_obs_ids'], features={})
            for arm in ARMS:
                with np.load(OUT/f'{name}_{seed}_{arm}_decomposition.npz') as f, np.load(OUT/f'{name}_{seed}_{arm}_map.npz') as m:
                    b['features'][arm] = score_probe(f['content'], m['probe'])@m['contrasts']/m['rms']
            bundles[name] = b
        position = {int(v): j for j, v in enumerate(fitting)}
        with np.load(PARENT/f'correction_episodes_{seed}.npz') as f:
            training = [(np.array([position[int(v)] for v in s]), np.array([position[int(v)] for v in q]))
                        for s, q in zip(f['support_obs_ids'], f['query_obs_ids'])]
        cohorts = {'oof_correction_train': training}
        with np.load(PARENT/'validation_episodes.npz') as f:
            for role in ('val_new_ship', 'val_new_time'):
                cohorts[role] = [(f[f'{role}_{j}_support'], f[f'{role}_{j}_query']) for j in range(3)]
        cohorts['source_held'] = []
        for j in range(3):
            with np.load(OUT/f'{seed}_source_{j}_predictions.npz') as f:
                cohorts['source_held'].append((f['support_obs_ids'], f['query_obs_ids']))
        full = bundles['full']
        for arm in ARMS:
            train_features, train_prior, train_kernel = assemble_training_view(bundles, fitting, assignment, arm, 'crossfit')
            for use in ('global', 'relation'):
                with np.load(OUT/f'correction_{seed}_{arm}_{use}.npz') as f:
                    coefficient = f['coefficient'].copy()
                for cohort, episodes in cohorts.items():
                    is_train = cohort == 'oof_correction_train'
                    features, prior, kernel, y = ((train_features, train_prior, train_kernel, labels[fitting]) if is_train else
                        (full['features'][arm], full['prior'], full['kernel'], labels))
                    delta = score_probe(features, coefficient) if use == 'global' else relation_delta(features, prior, coefficient)
                    eligible = np.unique(np.concatenate([np.r_[s, q] for s, q in episodes]))
                    scales.append(dict(seed=seed, arm=arm, use=use, cohort=cohort,
                        coefficient_norm=float(np.linalg.norm(coefficient)),
                        delta_rms=float(np.sqrt(np.square(delta[eligible]).mean())),
                        delta_max=float(np.abs(delta[eligible]).max())))
                    for draw, (s, q) in enumerate(episodes):
                        op, present = support_operator(kernel[s], kernel[q], y[s])
                        baseline = adapted_scores(prior[s], prior[q], op, present, y[s])
                        basepred = baseline.argmax(1)
                        corrected = adapted_scores(prior[s]+delta[s], prior[q]+delta[q], op, present, y[s])
                        count = np.bincount(y[q], minlength=8)
                        w = len(q)/((count > 0).sum()*count[y[q]])
                        target = 2*np.eye(8)[y[q]]-1
                        for gain, scores in [(0., baseline), (1., corrected)]:
                            pred = scores.argmax(1)
                            rows.append(dict(seed=seed, arm=arm, use=use, cohort=cohort, draw=draw, gain=gain,
                                **metrics(pred, y[q]), MSE=float((w[:, None]*(scores-target)**2).mean()),
                                rescued=int(((basepred != y[q])&(pred == y[q])).sum()),
                                harmed=int(((basepred == y[q])&(pred != y[q])).sum()),
                                score_change_rms=float(np.sqrt(np.square(scores-baseline).mean()))))
    folder = OUT/'usage'; folder.mkdir(exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(folder/'source_usage_metrics.csv', index=False)
    keys = ['arm', 'use', 'cohort', 'gain']
    values = ['OA', 'BA', 'MSE', 'rescued', 'harmed', 'score_change_rms']
    summary = df.groupby(keys)[values].mean().reset_index()
    summary.to_csv(folder/'source_usage_summary.csv', index=False)
    df.groupby(['seed']+keys)[values].mean().reset_index().to_csv(folder/'source_usage_seed_summary.csv', index=False)
    pd.DataFrame(scales).to_csv(folder/'source_usage_scales.csv', index=False)
    save_json(folder/'COMPLETE.json', dict(no_new_training=True, target_metrics_used=False,
        warning='OOF episodes train correction, not heldcorrector evaluation; forcedgain1 isdiagnosticnotdeploymentselection.'))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
