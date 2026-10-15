"""One bounded loss diagnosis: decision CE versus score MSE, all else fixed."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import time
import numpy as np
import pandas as pd
import torch
from train_explicit_candidate import load_data, SEEDS, SUPPORTS, save_json, metrics
from run_change_decomposition import OUT as ROOT, PARENT, ARMS, classification_rows
from run_crossfit_evidence import choose_gain, evaluate_episodes
from crossfit_evidence import assemble_training_view
from paired_evidence_factors import score_probe
from change_decomposition import train_relation_correction, relation_delta
from run_paired_evidence_factors import target_readout
from replay_crossfit_evidence import exported_prediction
from replay_change_decomposition import encode_evidence

OUT = ROOT/'decision_objective'


def main():
    start = time.perf_counter(); torch.set_num_threads(1); OUT.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    labels = meta.class_id.to_numpy(); fitting = split['factor_fit']
    with np.load(PARENT/'source_folds.npz') as f:
        assignment = f['assignment'].copy()
    validation = {}
    with np.load(PARENT/'validation_episodes.npz') as f:
        for role in ('val_new_ship', 'val_new_time'):
            validation[role] = [(f[f'{role}_{j}_support'], f[f'{role}_{j}_query']) for j in range(3)]
    tasks = {}
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        with np.load(path) as f:
            tasks[path.stem.removesuffix('_split')] = tuple(f[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels'))
    save_json(OUT/'PROTOCOL.json', dict(
        reason='Source OOF correctionTRAIN MSE improved butBA fell underbothglobal/relation. Test loss mismatch without changing decomposition/capacity/gate/bound/supports.',
        change='Only replace adaptedclassbalancedMSE by adaptedclassbalancedcrossentropy temperature1. Keep.25correctbaselinecrossingprotection,.001coefL2,Adam.02,400sameepisodes.',
        scope='4relationcorrectorfits2arms2seeds,NOdecomposition/backbonefits. Same source0/.5/1 gainrule; preserve originalnegative. Exploratory source-diagnostic correction,notpreregisteredindependentconfirm. No further sweep.'))
    choices, validation_rows, traces, train_rows, source_rows, replay_rows = [], [], [], [], [], []
    for seed in SEEDS:
        bundles = {}
        for name in ('full', 'fold0', 'fold1', 'fold2'):
            with np.load(PARENT/f'{name}_{seed}_outputs.npz') as f:
                b = dict(prior=f['prior'], kernel=f['kernel'], fitting=f['fit_obs_ids'], features={})
            for arm in ARMS:
                with np.load(ROOT/f'{name}_{seed}_{arm}_decomposition.npz') as f, np.load(ROOT/f'{name}_{seed}_{arm}_map.npz') as m:
                    b['features'][arm] = score_probe(f['content'], m['probe'])@m['contrasts']/m['rms']
            bundles[name] = b
        full = bundles['full']; position = {int(v): j for j, v in enumerate(fitting)}
        with np.load(PARENT/f'correction_episodes_{seed}.npz') as f:
            episodes = [(np.array([position[int(v)] for v in s]), np.array([position[int(v)] for v in q]))
                        for s, q in zip(f['support_obs_ids'], f['query_obs_ids'])]
        held_episodes = []
        for draw in range(3):
            with np.load(ROOT/f'{seed}_source_{draw}_predictions.npz') as f:
                held_episodes.append((f['support_obs_ids'], f['query_obs_ids']))
        deployment = dict(baseline=full['prior'])
        for arm in ARMS:
            features, prior, kernel = assemble_training_view(bundles, fitting, assignment, arm, 'crossfit')
            coefficient, history = train_relation_correction(features, prior, labels[fitting], kernel, episodes, objective='ce')
            traces.extend(dict(seed=seed, arm=arm, **r) for r in history)
            train_delta = relation_delta(features, prior, coefficient)
            for strength in (0., 1.):
                train_rows.extend(dict(seed=seed, arm=arm, gain=strength, **r) for r in
                    evaluate_episodes(prior+strength*train_delta, kernel, labels[fitting], episodes))
            delta = relation_delta(full['features'][arm], full['prior'], coefficient)
            gain, trials = choose_gain(full['prior'], delta, full['kernel'], labels, validation)
            choices.append(dict(seed=seed, arm=arm, gain=gain))
            validation_rows.extend(dict(seed=seed, arm=arm, **r) for r in trials)
            np.savez_compressed(OUT/f'correction_{seed}_{arm}.npz', coefficient=coefficient, selected_gain=gain)
            deployment[f'{arm}_relation_ce'] = full['prior']+gain*delta
            for strength in sorted(set((0., 1., gain))):
                source_rows.extend(dict(seed=seed, arm=arm, gain=strength, selected=strength == gain, **r) for r in
                    evaluate_episodes(full['prior']+strength*delta, full['kernel'], labels, held_episodes))
            # Exported full feature path is reloaded from raw on source-held rows.
            ids = split['test_source']; lookup = {int(v): j for j, v in enumerate(ids)}
            replay_base = exported_prediction(raw[ids], seed)
            _, replay_features = encode_evidence(raw[ids], seed, arm)
            replay_prior = replay_base['prior']+gain*relation_delta(replay_features, replay_base['prior'], coefficient)
            for draw, (s, q) in enumerate(held_episodes):
                a = target_readout(deployment[f'{arm}_relation_ce'], full['kernel'], s, q, labels[s])
                rs, rq = [np.array([lookup[int(v)] for v in t]) for t in (s, q)]
                b = target_readout(replay_prior, replay_base['kernel'], rs, rq, labels[s])
                np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
                np.testing.assert_array_equal(a.argmax(1), b.argmax(1))
                replay_rows.append(dict(seed=seed, arm=arm, draw=draw, max_error=float(np.abs(a-b).max())))
            print('DECISION_CORRECTION', seed, arm, 'gain', gain, flush=True)
        for task, (s, q, sy) in tasks.items():
            archive = {name: target_readout(p, full['kernel'], s, q, sy) for name, p in deployment.items()}
            np.savez_compressed(OUT/f'{seed}_{task}_predictions.npz', support_obs_ids=s, query_obs_ids=q, **archive)
    for name, rows in [('selection', choices), ('validation', validation_rows), ('training', traces),
                       ('training_metrics', train_rows), ('source_results', source_rows), ('deployment_replay', replay_rows)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv', index=False)
    save_json(OUT/'PREDICTIONS_SAVED.json', dict(query_metrics_read=False, source_choices_frozen=True))
    results, classes = [], []
    for seed in SEEDS:
        for task, (s, q, sy) in tasks.items():
            with np.load(OUT/f'{seed}_{task}_predictions.npz') as f:
                archive = {k: f[k] for k in f.files if k not in ('support_obs_ids', 'query_obs_ids')}
                a, b = classification_rows(archive, labels[q], dict(seed=seed, task=task, port=meta.iloc[q].port.iloc[0]))
                results.extend(a); classes.extend(b)
    df = pd.DataFrame(results); df.to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'target_classes.csv', index=False)
    for suffix, keys in [('summary', ['arm']), ('seed_summary', ['arm', 'seed']), ('port_summary', ['arm', 'port'])]:
        df.groupby(keys)[['OA', 'BA']].mean().reset_index().to_csv(OUT/f'target_{suffix}.csv', index=False)
    pd.DataFrame(source_rows).query('selected').groupby('arm')[['OA', 'BA', 'MSE']].mean().reset_index().to_csv(OUT/'source_summary.csv', index=False)
    pd.DataFrame(train_rows).groupby(['arm', 'gain'])[['OA', 'BA', 'MSE']].mean().reset_index().to_csv(OUT/'training_summary.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-start, correction_fits=4, no_other_fits=True,
        raw_deployment_replays=len(replay_rows), target_choice_used=False, original_results_preserved=True))
    print(df.groupby('arm')[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
