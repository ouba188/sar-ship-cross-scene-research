"""Matched in-sample, size-matched seen, and ship-held-out correction training."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import time
import numpy as np
import pandas as pd
import torch
from crossfit_evidence import (
    ship_folds, assert_excluded, correction_episodes, fit_pipeline, attach_factors,
    assemble_training_view,
)
from train_explicit_candidate import BASE, OUT as OLD, SEEDS, SUPPORTS, load_data, metrics, save_json
from paired_evidence_factors import fit_episode_correction, score_probe, support_operator, adapted_scores
from run_paired_evidence_factors import target_readout


OUT = BASE/'crossfit_evidence_20261011'
METHODS = ('paired', 'class_only')
MODES = ('full_in', 'size_matched_in', 'crossfit')
GAINS = (0., .5, 1.)


def evaluate_episodes(prior, kernel, labels, episodes):
    rows = []
    for draw, (s, q) in enumerate(episodes):
        op, present = support_operator(kernel[s], kernel[q], labels[s])
        score = adapted_scores(prior[s], prior[q], op, present, labels[s])
        counts = np.bincount(labels[q], minlength=8)
        w = len(q)/((counts > 0).sum()*counts[labels[q]])
        target = 2*np.eye(8)[labels[q]]-1
        rows.append(dict(draw=draw, **metrics(score.argmax(1), labels[q]),
                         MSE=float((w[:, None]*(score-target)**2).mean())))
    return rows


def choose_gain(prior, delta, kernel, labels, validation):
    rows = []
    for gain in GAINS:
        for role, episodes in validation.items():
            rows.extend(dict(role=role, gain=gain, **r) for r in evaluate_episodes(prior+gain*delta, kernel, labels, episodes))
    table = pd.DataFrame(rows)
    means = table.groupby('gain', sort=True).BA.mean()
    return float(means.index[np.argmax(means)]), rows


def export_pipeline(bundle, name, seed):
    for stage in ('case', 'prior'):
        torch.save(bundle[stage+'_state'], OUT/f'{name}_{seed}_{stage}.pt')
    np.savez_compressed(OUT/f'{name}_{seed}_normalization.npz', raw_center=bundle['raw_center'], raw_scale=bundle['raw_scale'],
        teacher_coef=bundle['teacher_coef'], fit_obs_ids=bundle['fitting'], excluded_obs_ids=bundle['excluded'],
        selection_obs_ids=bundle['validation_ids'])
    np.savez_compressed(OUT/f'{name}_{seed}_outputs.npz', prior=bundle['prior'], kernel=bundle['kernel'],
        feature_paired=bundle['features']['paired'], feature_class_only=bundle['features']['class_only'],
        fit_obs_ids=bundle['fitting'], excluded_obs_ids=bundle['excluded'], obs_ids=np.arange(len(bundle['prior'])))
    for method in METHODS:
        np.savez_compressed(OUT/f'{name}_{seed}_{method}_map.npz', **bundle['factor_maps'][method])
    bundle['pair_coverage'].to_csv(OUT/f'{name}_{seed}_pair_coverage.csv', index=False)


def main():
    torch.set_num_threads(1)
    started = time.perf_counter(); OUT.mkdir(exist_ok=False)
    meta, split, roles, source, raw, old_norm, old_fixed, old_h = load_data()
    del old_norm, old_fixed, old_h
    fitting, held = split['factor_fit'], split['test_source']
    fm = meta.iloc[fitting].reset_index(drop=True)
    labels = meta.class_id.to_numpy()
    assignment = ship_folds(fm)
    validation = {}
    with np.load(OLD/'validation_episodes.npz') as f:
        np.savez_compressed(OUT/'validation_episodes.npz', **{k: f[k] for k in f.files})
        for role in ('val_new_ship', 'val_new_time'):
            validation[role] = [(f[f'{role}_{j}_support'], f[f'{role}_{j}_query']) for j in range(3)]
    selection_ids = np.unique(np.concatenate([np.r_[s, q] for s, q in validation['val_new_ship']]))
    fold_rows = []
    for fold in range(3):
        f, e = fitting[assignment != fold], fitting[assignment == fold]
        assert_excluded(meta, f, selection_ids, e)
        fold_rows.append(dict(fold=fold, fit_chips=len(f), fit_ships=meta.iloc[f].mmsi.nunique(),
                              oof_chips=len(e), oof_ships=meta.iloc[e].mmsi.nunique()))
    pd.DataFrame(fold_rows).to_csv(OUT/'fold_coverage.csv', index=False)
    np.savez_compressed(OUT/'source_folds.npz', fit_obs_ids=fitting, assignment=assignment)
    tasks = {}
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        with np.load(path) as f:
            tasks[path.stem.removesuffix('_split')] = tuple(f[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels'))
    save_json(OUT/'PROTOCOL.json', dict(
        seeds=SEEDS, methods=METHODS, modes=MODES, folds=3, fold_seed=20261011, steps=400,
        hypothesis='Does training corrections on vessel-held-out base errors reduce the previous fit/validation degradation?',
        fixed='Frozen cached R18 input features, original1296sourcefit/525ships, original heldsource and64target supports. No backbone retraining; no claim backbone itself never saw these identities historically.',
        upstream='Eachfull/fold independently fits rawnormalization, PCAinit,512ridge teacher, same-vessel other-product teachers, case32 and prior32. No old fullteacher/case/statistics warmstart. Ordinarytask objective supervised+.5teacher+.5adaptation,400steps.',
        checkpoint='case/prior earliest best on THREE ORIGINAL val_new_ship episodes every20steps including0. Originalval_new_time all149identities belongfactorfit; excluded from ALLbasecheckpointselection includingfull. This differs fromhistoricaltwo-role recipe, newfull baseline isprimary.',
        representation='Foldlocal paired7 andclassonly7; independent sourcefitridge8scoreprobe ->fixedHelmert8x7 classcontrasts /SOURCE-fit weightedRMS. Semanticfixedaxis instead of mixing rotatedlatentaxes. Bothcontrols use identical new64parametercorrection input.',
        correction='TOTAL400episodes perarm, cycle originalsource3heldfolds. Each64support96query episode entirelyonefold, shipdisjoint. Same EXACTsampleIDs acrossallthreearms. Closedform ridge8 correction through nativefixedcasekernel; supportprior subtraction/absentclasses retained.',
        arms='full_in:fullmodelonall episode rows; size_matched_in:fold(k+1)%3 model, trainedonbatchk; crossfit:foldk model, excludedallbatchkships. Samefoldmodels supplybothsize-matchedarms. Each foldbase hasabout350ships vs full525.',
        selection='Correctiongain0/.5/1 selectedusingOLDtwo-role sourcevalidation BA, equalroles/draws, lowergain ties. Onlyfulldeployment predictor selects gain; no fold-specific or targetgainselection.',
        deployment='Full-data newlyrefit case/prior/factors plus one64parameterresidual. Original32Ddecisioninformationretained; nohard7Dreplacement. Fold/full errors canstilldiffer despitealignedaxes; diagnose separately.',
        boundary='OOFquerylabels TRAINthecorrector; OOFerrorsnotindependentcorrectorvalidation. Newtime sharescorrectiontrainMMSIs; newshipusedforbaseandgainselection. Target3portsarepreviouslyreuseddevelopment, NOTindependentconfirmation orcausalidentification.',
        no_search='No extra coefficient/rank/network sweep; firstcomplete12corrections,16basefits. Do notpromote merelybecauseOOFtrainlossfalls. Allpredictionssavedbeforequerymetricread.'))
    traces, choices, selections, validation_rows, objective_rows, source_rows = [], [], [], [], [], []
    for seed in SEEDS:
        bundles = {}
        for name, fold in [('full', None)]+[(f'fold{k}', k) for k in range(3)]:
            f = fitting if fold is None else fitting[assignment != fold]
            excluded = np.array([], dtype=int) if fold is None else fitting[assignment == fold]
            b = fit_pipeline(raw, meta, f, validation['val_new_ship'], excluded, seed)
            attach_factors(b, meta); bundles[name] = b
            export_pipeline(b, name, seed)
            for stage in ('case', 'prior'):
                traces.extend(dict(seed=seed, bundle=name, stage=stage, **row) for row in b[stage+'_trace'])
                choices.append(dict(seed=seed, bundle=name, stage=stage, **b[stage+'_choice']))
            for role, ids in [('fit', f), ('source_held', held), ('oof', excluded)]:
                if len(ids):
                    source_rows.append(dict(seed=seed, bundle=name, role=role, **metrics(b['prior'][ids].argmax(1), labels[ids])))
            print('BASE', seed, name, 'case', b['case_choice'], 'prior', b['prior_choice'], flush=True)
        episodes = correction_episodes(fm, assignment, seed)
        for s, q in episodes:
            assert len(np.unique(assignment[np.r_[s, q]])) == 1
            assert not set(fm.iloc[s].mmsi)&set(fm.iloc[q].mmsi)
        np.savez_compressed(OUT/f'correction_episodes_{seed}.npz',
            support_obs_ids=np.stack([fitting[s] for s, q in episodes]),
            query_obs_ids=np.stack([fitting[q] for s, q in episodes]))
        full = bundles['full']
        deployment = dict(baseline=full['prior'])
        for method in METHODS:
            for mode in MODES:
                features, prior, kernel = assemble_training_view(bundles, fitting, assignment, method, mode)
                coefficient = fit_episode_correction(features, prior, labels[fitting], episodes, kernel)
                delta = score_probe(full['features'][method], coefficient)
                gain, trials = choose_gain(full['prior'], delta, full['kernel'], labels, validation)
                tag = dict(seed=seed, method=method, mode=mode)
                validation_rows.extend(dict(**tag, **row) for row in trials)
                selections.append(dict(**tag, gain=gain))
                np.savez_compressed(OUT/f'correction_{mode}_{method}_{seed}.npz', coefficient=coefficient, selected_gain=gain)
                deployment[mode+'_'+method] = full['prior']+gain*delta
                fit_delta = score_probe(features, coefficient)
                for strength in (0., 1.):
                    values = evaluate_episodes(prior+strength*fit_delta, kernel, labels[fitting], episodes)
                    objective_rows.append(dict(**tag, gain=strength,
                        **pd.DataFrame(values)[['OA', 'BA', 'MSE']].mean().to_dict()))
                print('CORRECTION', seed, method, mode, 'gain', gain, flush=True)
        for task, (s, q, sy) in tasks.items():
            archive = dict(support_obs_ids=s, query_obs_ids=q)
            for name, prior in deployment.items():
                archive[name] = target_readout(prior, full['kernel'], s, q, sy)
                archive[name+'_raw'] = np.asarray(prior[q], np.float32)
                if name != 'baseline':
                    entry = next(row for row in selections if row['seed'] == seed and row['mode']+'_'+row['method'] == name)
                    if entry['gain'] == 0:
                        np.testing.assert_array_equal(archive[name], archive['baseline'])
            np.savez_compressed(OUT/f'{seed}_{task}_predictions.npz', **archive)
    for name, rows in [('base_training', traces), ('base_selection', choices), ('correction_selection', selections),
                       ('validation', validation_rows), ('correction_training', objective_rows), ('source_results', source_rows)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv', index=False)
    save_json(OUT/'PREDICTIONS_SAVED.json', dict(query_metrics_read=False, seeds=list(SEEDS), source_gains_frozen=True))
    results, classes = [], []
    for seed in SEEDS:
        for task, (s, q, sy) in tasks.items():
            with np.load(OUT/f'{seed}_{task}_predictions.npz') as f:
                truth = labels[q]; baseline = f['baseline'].argmax(1)
                for name in [key for key in f.files if not key.endswith('_raw') and key not in ('support_obs_ids', 'query_obs_ids')]:
                    pred = f[name].argmax(1)
                    raw_result = metrics(f[name+'_raw'].argmax(1), truth)
                    tag = dict(seed=seed, task=task, port=meta.iloc[q].port.iloc[0], arm=name)
                    results.append(dict(**tag, **metrics(pred, truth), raw_BA=raw_result['BA'], raw_OA=raw_result['OA']))
                    for c in np.unique(truth):
                        mask = truth == c
                        classes.append(dict(**tag, class_id=int(c), recall=float(100*(pred[mask] == c).mean()),
                            rescued=int(((baseline != truth)&(pred == truth)&mask).sum()),
                            harmed=int(((baseline == truth)&(pred != truth)&mask).sum()), count=int(mask.sum())))
    df = pd.DataFrame(results); df.to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'target_classes.csv', index=False)
    for name, keys in [('summary', ['arm']), ('seed_summary', ['arm', 'seed']), ('port_summary', ['arm', 'port'])]:
        df.groupby(keys)[['OA', 'BA', 'raw_OA', 'raw_BA']].mean().reset_index().to_csv(OUT/f'target_{name}.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-started, new_base_fits=16, new_correction_fits=12,
                                       main_model_modified=False, target_choice_used=False))
    print(df.groupby('arm')[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
