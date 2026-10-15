"""Reuse frozen grouped base models; test explicit decomposition and bounded use."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import time
import numpy as np
import pandas as pd
import torch
from train_explicit_candidate import BASE, SEEDS, SUPPORTS, load_data, metrics, save_json
from run_crossfit_evidence import OUT as PARENT, choose_gain, evaluate_episodes
from replay_crossfit_evidence import encode_state
from crossfit_evidence import semantic_features, assemble_training_view
from paired_evidence_factors import row_weights, fit_episode_correction, score_probe, support_operator, adapted_scores
from change_decomposition import train_decomposition, train_relation_correction, relation_delta
from run_paired_evidence_factors import target_readout
from explicit_candidate_projection import identity_episode

OUT = BASE/'change_decomposition_20261011'
ARMS = ('class_only', 'paired')


def load_base(raw, name, seed):
    with np.load(PARENT/f'{name}_{seed}_outputs.npz') as f:
        bundle = {key: f[key].copy() for key in ('prior', 'kernel')}
        bundle['fitting'], bundle['excluded'] = f['fit_obs_ids'].copy(), f['excluded_obs_ids'].copy()
    with np.load(PARENT/f'{name}_{seed}_normalization.npz') as f:
        x = torch.tensor((raw-f['raw_center'])/f['raw_scale'], dtype=torch.float32)
    state = torch.load(PARENT/f'{name}_{seed}_prior.pt', map_location='cpu', weights_only=True)['selected']
    with torch.no_grad():
        z, score = encode_state(x, state)
    np.testing.assert_array_equal(score.numpy().argmax(1), bundle['prior'].argmax(1))
    with np.load(PARENT/f'{name}_{seed}_paired_map.npz') as f:
        center, whitener = f['center'].copy(), f['whitener'].copy()
    bundle['z'] = (z.numpy()-center)@whitener
    bundle['center'], bundle['whitener'], bundle['features'] = center, whitener, {}
    return bundle


def classification_rows(archive, truth, tag):
    rows, classes = [], []
    baseline = archive['baseline'].argmax(1)
    for arm, scores in archive.items():
        if arm.endswith('_raw'):
            continue
        pred = scores.argmax(1)
        rows.append(dict(**tag, arm=arm, **metrics(pred, truth)))
        for c in np.unique(truth):
            mask = truth == c
            classes.append(dict(**tag, arm=arm, class_id=int(c), count=int(mask.sum()),
                recall=100*float((pred[mask] == c).mean()),
                rescued=int(((baseline != truth)&(pred == truth)&mask).sum()),
                harmed=int(((baseline == truth)&(pred != truth)&mask).sum())))
    return rows, classes


def main():
    started = time.perf_counter(); torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    del norm, fixed, h
    labels = meta.class_id.to_numpy()
    fitting, held = split['factor_fit'], split['test_source']
    with np.load(PARENT/'source_folds.npz') as f:
        np.testing.assert_array_equal(fitting, f['fit_obs_ids'])
        assignment = f['assignment'].copy()
    validation = {}
    with np.load(PARENT/'validation_episodes.npz') as f:
        for role in ('val_new_ship', 'val_new_time'):
            validation[role] = [(f[f'{role}_{j}_support'], f[f'{role}_{j}_query']) for j in range(3)]
    val_ids = np.unique(np.concatenate([np.r_[s, q] for s, q in validation['val_new_ship']]))
    hm = meta.iloc[held].reset_index(drop=True)
    held_episodes = []
    for j in range(3):
        s, q = identity_episode(hm, np.random.default_rng(91011+j), query_size=None)
        held_episodes.append((held[s], held[q]))
    tasks = {}
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        with np.load(path) as f:
            tasks[path.stem.removesuffix('_split')] = tuple(f[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels'))
    save_json(OUT/'PROTOCOL.json', dict(
        objective='Test explicit observed-change residual versus class-only same-capacity decomposition; then ordinary versus bounded relation correction. Not pure causal identification.',
        fixed='Reuse crossfit_evidence_20261011 full/fold base checkpoints, teachers, kernels, whitening, folds,400 correction episodes,64supports,2seeds. No backbone/base retraining or mainmodelchange.',
        decomposition='Source-whitened32 -> learnedorthogonal32 frame plus32-32-32 tanh offset; first7content,last25residual. Orthogonal linear outputcomponents; reconstruction approximate aftertraining. Class-only initialization uses sourceclassbetween eigenspace andridgehead. No hardfeaturediscard.',
        losses='Full SOURCEfit class/ship/acquisitionweighted one-versus-rest targetMSE +.25 reconstructionMSE. paired additionally .25 distinctproduct sameMMSI residualdelta error dividedbyfitdeltaenergy. Pairclasses/ships/acquisitionpairs nestedbalanced. All observedchanges,notprovennuisance; no queryidentityconditioninput.',
        selection='400 Adam.001 steps; every20including0 select independent ridgecontentprobe BA onoriginalnewship sourcevalidation; lowerstep ties. Full/folds never fitheldMMSIs. Samevalidation reusedresearch,notindependentconfirmation. Newtimeexcludedfromdecompositionselection.',
        correction='All corrections TRAIN only fold-unseen source observations. Same400originalepisodes. Global same priorclosedformridge8. Relation28 coefficients, fixedpaircontrast tanh evidence times1/(1+abspriorcontrast), perclassdelta<=.5,400Adam.02updates; adaptedbalancedMSE+.25correctbaseline decisioncrossingpenalty+.001coefL2. Bothsupport/query corrections andabsentsupportmask included.',
        gain='Originalsourcevalidation2roles selects0/.5/1 equalroles/draws. No targetselection. 2factorarmsx2correctionsx2seeds; oldOOFpaired/class baselines replayedwithoutrefit.',
        tests='Source test_source same3identitydisjoint64support draws seeds91011-13. Target original3portsx3supportdraws. Source/targets repeatedlyuseddevelopment,notindependentconfirm. Freeze all targetpredictions beforemetrics.',
        no_search='16 small decomposition fits,8correctionfits; no dimension/coefficient sweep. Selectedstep0 retainedandreported. Mechanism diagnostics separate from accuracy.'))
    trace, choices, selections, validation_rows, correction_trace, source_rows, source_classes = [], [], [], [], [], [], []
    for seed in SEEDS:
        bundles = {}
        for name in ('full', 'fold0', 'fold1', 'fold2'):
            b = load_base(raw, name, seed); bundles[name] = b
            if set(meta.iloc[b['excluded']].mmsi)&set(meta.iloc[np.r_[b['fitting'], val_ids]].mmsi):
                raise ValueError('Fold isolation lost')
            fm = meta.iloc[b['fitting']].reset_index(drop=True)
            for arm in ARMS:
                model, state, output, history, choice, coverage = train_decomposition(
                    b['z'], meta, b['fitting'], val_ids, seed, arm)
                torch.save(state, OUT/f'{name}_{seed}_{arm}.pt')
                feature, reading = semantic_features(output['content'], b['fitting'], labels, row_weights(fm))
                b['features'][arm] = feature
                np.savez_compressed(OUT/f'{name}_{seed}_{arm}_decomposition.npz', **output, z=b['z'],
                    fit_obs_ids=b['fitting'], obs_ids=np.arange(len(meta)))
                np.savez_compressed(OUT/f'{name}_{seed}_{arm}_map.npz', center=b['center'], whitener=b['whitener'], **reading)
                coverage.to_csv(OUT/f'{name}_{seed}_{arm}_pair_coverage.csv', index=False)
                trace.extend(dict(seed=seed, bundle=name, arm=arm, **r) for r in history)
                choices.append(dict(seed=seed, bundle=name, arm=arm, **choice))
                print('DECOMPOSITION', seed, name, arm, choice, flush=True)
        position = {int(v): j for j, v in enumerate(fitting)}
        with np.load(PARENT/f'correction_episodes_{seed}.npz') as f:
            episodes = [(np.array([position[int(v)] for v in s]), np.array([position[int(v)] for v in q]))
                        for s, q in zip(f['support_obs_ids'], f['query_obs_ids'])]
        full = bundles['full']
        deployment = dict(baseline=full['prior'])
        for arm in ARMS:
            features, prior, kernel = assemble_training_view(bundles, fitting, assignment, arm, 'crossfit')
            for use in ('global', 'relation'):
                if use == 'global':
                    coefficient = fit_episode_correction(features, prior, labels[fitting], episodes, kernel)
                    delta = score_probe(full['features'][arm], coefficient)
                else:
                    coefficient, history = train_relation_correction(features, prior, labels[fitting], kernel, episodes)
                    correction_trace.extend(dict(seed=seed, arm=arm, **r) for r in history)
                    delta = relation_delta(full['features'][arm], full['prior'], coefficient)
                gain, trials = choose_gain(full['prior'], delta, full['kernel'], labels, validation)
                validation_rows.extend(dict(seed=seed, arm=arm, use=use, **r) for r in trials)
                selections.append(dict(seed=seed, arm=arm, use=use, gain=gain))
                np.savez_compressed(OUT/f'correction_{seed}_{arm}_{use}.npz', coefficient=coefficient, selected_gain=gain)
                deployment[f'{arm}_{use}'] = full['prior']+gain*delta
                print('CORRECTION', seed, arm, use, 'gain', gain, flush=True)
        with np.load(PARENT/f'full_{seed}_outputs.npz') as old:
            for arm in ARMS:
                with np.load(PARENT/f'correction_crossfit_{arm}_{seed}.npz') as corr:
                    deployment[f'old_{arm}'] = old['prior']+float(corr['selected_gain'])*score_probe(old['feature_'+arm], corr['coefficient'])
        for name, prior in deployment.items():
            source_rows.extend(dict(seed=seed, arm=name, **r) for r in evaluate_episodes(prior, full['kernel'], labels, held_episodes))
        for draw, (s, q) in enumerate(held_episodes):
            archive = {name: target_readout(p, full['kernel'], s, q, labels[s]) for name, p in deployment.items()}
            _, cls = classification_rows(archive, labels[q], dict(seed=seed, draw=draw))
            source_classes.extend(cls)
            np.savez_compressed(OUT/f'{seed}_source_{draw}_predictions.npz', support_obs_ids=s, query_obs_ids=q, **archive)
        for task, (s, q, sy) in tasks.items():
            archive = {}
            for name, prior in deployment.items():
                archive[name] = target_readout(prior, full['kernel'], s, q, sy)
                archive[name+'_raw'] = np.asarray(prior[q], np.float32)
            with np.load(PARENT/f'{seed}_{task}_predictions.npz') as old:
                np.testing.assert_array_equal(archive['baseline'], old['baseline'])
                for arm in ARMS:
                    np.testing.assert_array_equal(archive[f'old_{arm}'], old[f'crossfit_{arm}'])
            np.savez_compressed(OUT/f'{seed}_{task}_predictions.npz', support_obs_ids=s, query_obs_ids=q, **archive)
    for name, rows in [('decomposition_training', trace), ('decomposition_selection', choices),
        ('correction_selection', selections), ('validation', validation_rows), ('correction_training', correction_trace),
        ('source_results', source_rows), ('source_classes', source_classes)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv', index=False)
    save_json(OUT/'PREDICTIONS_SAVED.json', dict(query_metrics_read=False, source_choices_frozen=True))
    results, classes = [], []
    for seed in SEEDS:
        for task, (s, q, sy) in tasks.items():
            with np.load(OUT/f'{seed}_{task}_predictions.npz') as f:
                archive = {name: f[name] for name in f.files if name not in ('support_obs_ids', 'query_obs_ids')}
                rows, cls = classification_rows(archive, labels[q], dict(seed=seed, task=task, port=meta.iloc[q].port.iloc[0]))
                results.extend(rows); classes.extend(cls)
    df = pd.DataFrame(results); df.to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'target_classes.csv', index=False)
    for scope, records in [('target', df), ('source', pd.DataFrame(source_rows))]:
        for suffix, keys in [('summary', ['arm']), ('seed_summary', ['arm', 'seed'])]:
            numeric = ['OA', 'BA']+(['MSE'] if scope == 'source' else [])
            records.groupby(keys)[numeric].mean().reset_index().to_csv(OUT/f'{scope}_{suffix}.csv', index=False)
    df.groupby(['arm', 'port'])[['OA', 'BA']].mean().reset_index().to_csv(OUT/'target_port_summary.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-started, decomposition_fits=16,
        correction_fits=8, main_model_modified=False, old_target_replays=54, target_choice_used=False))
    print(df.groupby('arm')[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
