"""Compact condition-prediction pilot, frozen base and original support readout."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import time
import numpy as np
import pandas as pd
import torch
from train_explicit_candidate import BASE, SEEDS, SUPPORTS, load_data, save_json, metrics
from run_crossfit_evidence import OUT as PARENT, choose_gain, evaluate_episodes
from run_change_decomposition import load_base, classification_rows
from run_paired_evidence_factors import target_readout
from explicit_candidate_projection import identity_episode
from incidence_transport import ARMS, known_condition, matched_pairs, fit_response, response_metrics, canonical_delta

OUT = BASE/'incidence_transport_20261011'


def prior_transport(model, context, meta, whitened_to_scores):
    delta, valid = canonical_delta(context, meta, model)
    return delta@whitened_to_scores, delta, valid


def main():
    torch.set_num_threads(1); start = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists() or (OUT/'PROTOCOL.json').exists():
        raise ValueError('Do not overwrite an existing response experiment')
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    labels = meta.class_id.to_numpy()
    fit, held = split['factor_fit'], split['test_source']
    validation = {}
    with np.load(PARENT/'validation_episodes.npz') as f:
        for role in ('val_new_ship', 'val_new_time'):
            validation[role] = [(f[f'{role}_{j}_support'], f[f'{role}_{j}_query']) for j in range(3)]
    hm = meta.iloc[held].reset_index(drop=True)
    held_episodes = []
    for j in range(3):
        s, q = identity_episode(hm, np.random.default_rng(91011+j), query_size=None)
        held_episodes.append((held[s], held[q]))
    tasks = {}
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        with np.load(path) as f:
            tasks[path.stem.removesuffix('_split')] = tuple(f[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels'))
    fit_pairs, held_pairs = matched_pairs(meta, fit), matched_pairs(meta, held)
    np.savez_compressed(OUT/'source_cohorts.npz', fit_obs_ids=fit, held_obs_ids=held, fit_pairs=fit_pairs, held_pairs=held_pairs)
    coverage = []
    angle = meta.incidence_deg.to_numpy(float)
    for role, ids in [('fit', fit), ('source_held', held), ('val_new_ship', roles['val_new_ship']), ('val_new_time', roles['val_new_time'])]:
        pairs = matched_pairs(meta, ids)
        gap = np.abs(angle[pairs[:, 1]]-angle[pairs[:, 0]]) if len(pairs) else np.array([])
        coverage.append(dict(role=role, chips=len(ids), ships=meta.iloc[ids].mmsi.nunique(),
            incidence_known=int(np.isfinite(angle[ids]).sum()), metadata_known=int(known_condition(meta.iloc[ids]).sum()),
            pairs=len(pairs), pair_ships=meta.iloc[pairs[:, 0]].mmsi.nunique(),
            gap2_pairs=int((gap >= 2).sum()), median_gap=float(np.median(gap)) if len(gap) else np.nan))
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv', index=False)
    save_json(OUT/'PROTOCOL.json', dict(
        objective='Does metadata-driven, anchor-context-dependent incidence response predict new-vessel changes better than a generic correction or common slope? First test condition learnability, not a new claim of purified content.',
        fixed='Frozen cachedR18, originalfull/3MMSIfold32prior+case pipelines fromcrossfit_evidence_20261011. Source1296/525ships,held823/302ships,2seeds,same64supports. No backbone/projection/classifier refit.',
        context='Existing source-fit class_only7 class-scorecontrast features fromeachbase pipeline. Not groundtruthqueryclass andnotclaimedcausalcontent. Fixedcontext iscalculated ONLY fromAimage for A->B prediction.',
        response='Predict delta_z=(delta_incidence/5)*[1,contextA7]@coef. angle_shared usesdeltaalone; no_condition uses[1,contextA7]withoutdelta. Eachcontextarm256coeffs,shared32. Equalinformationavailable, no_condition isgenericcorrection not physicallyidentifiedcounterfactual.',
        pairs='Knownincidence/orbit/platform; SAME MMSI/port/platform/orbit, DIFFERENTproduct. Bothorientations. Equalclasses,shipswithinclass,pairswithinship; directedduplicates notindependentpairs. Othergeometry/time/sea/state notcontrolled.',
        fit='Fixedridge=.05*trace(weighteddesignGram)/dimension,allcoeffsregularized. TargetBimage appearsONLYasregressionanswer, neverdecoderinput. No fittedvessel-ID or portinput.',
        sham='Onefixedseed20261011 traininganglepermutation WITHINsameMMSI/port/platform/orbit. Preservesstratumanglevalues; breakswhichacquisitionhaswhichangle. Realheldangles forprediction; notglobalpermutationorsignificanceclaim.',
        deployment='Single-image firstorderstep to sourcefitknownmetadata medianangle; deltaangleclippedatsourcefitpairedabsdelta95percentile; unknownmetadata or angleoutsideknownsourceinterval isexactno-op. Canonicalized32vector exportedbutnotpurecausal. Fulloldcasekernelretained; prior_delta=delta_z@inversewhitener@oldheadweight.T.',
        gains='Only0/.5/1 selectedbyOLDnewship+newtime3drawssourceBA; equalroles,lowergainties. NewtimeoverlapsfitMMSIs, otherhistoricalcohortsrepeatedlydeveloped. No targetparameterselection.',
        scope='32smallresponsefits(4armsx4basesx2seeds),noneuralfits. Fold-unseen sourcepairs diagnosepipelinegeneralization; optionalresponse-onlyleaveportout NOTheldportoutbackbone. Threeoldtargetportsx3draws, no otherdatasets.',
        no_search='No extra rank,anglefeature,coefficient,temperature orinteraction sweep. Preserve negative response andrecognition results.'))
    response_rows, choices, validation_rows, source_rows, source_classes, model_rows, movement = [], [], [], [], [], [], []
    for seed in SEEDS:
        deployment = {}
        for name in ('full', 'fold0', 'fold1', 'fold2'):
            b = load_base(raw, name, seed)
            with np.load(PARENT/f'{name}_{seed}_outputs.npz') as f:
                context = f['feature_class_only'].copy()
            state = torch.load(PARENT/f'{name}_{seed}_prior.pt', map_location='cpu', weights_only=True)['selected']
            whitening_to_scores = np.linalg.solve(b['whitener'], state['head.weight'].numpy().T)
            np.savez_compressed(OUT/f'{name}_{seed}_inputs.npz', z=b['z'], context=context, prior=b['prior'], kernel=b['kernel'],
                fit_obs_ids=b['fitting'], excluded_obs_ids=b['excluded'], obs_ids=np.arange(len(meta)),
                whitened_to_scores=whitening_to_scores, center=b['center'], whitener=b['whitener'])
            train_pairs = matched_pairs(meta, b['fitting'])
            evaluations = [('fit', train_pairs), ('source_held', held_pairs)]
            if name != 'full':
                evaluations.append(('oof', matched_pairs(meta, b['excluded'])))
            else:
                deployment['baseline'] = b['prior']
                full_kernel = b['kernel']
            for arm in ARMS:
                model = fit_response(b['z'], context, meta, b['fitting'], arm, train_pairs)
                np.savez_compressed(OUT/f'{name}_{seed}_{arm}.npz', **model)
                model_rows.append(dict(seed=seed, bundle=name, arm=arm, pairs=len(train_pairs),
                    pair_ships=meta.iloc[train_pairs[:, 0]].mmsi.nunique(), coefficients=model['coefficient'].size,
                    coefficient_norm=float(np.linalg.norm(model['coefficient'])), ridge=model['ridge'], gram_rank=model['gram_rank']))
                for role, pairs in evaluations:
                    for subset in ('all', 'gap2'):
                        p = pairs if subset == 'all' else pairs[np.abs(angle[pairs[:, 1]]-angle[pairs[:, 0]]) >= 2]
                        response_rows.append(dict(seed=seed, bundle=name, arm=arm, role=role, subset=subset,
                            **response_metrics(b['z'], context, meta, p, model)))
                if name == 'full':
                    dp, dz, valid = prior_transport(model, context, meta, whitening_to_scores)
                    gain, trials = choose_gain(b['prior'], dp, b['kernel'], labels, validation)
                    choices.append(dict(seed=seed, arm=arm, gain=gain, angle_reference=model['angle_reference'],
                        angle_min=model['angle_min'], angle_max=model['angle_max'], angle_delta_cap=model['angle_delta_cap']))
                    validation_rows.extend(dict(seed=seed, arm=arm, **r) for r in trials)
                    deployment[arm] = b['prior']+gain*dp
                    np.savez_compressed(OUT/f'canonical_{seed}_{arm}.npz', delta_z=dz, valid=valid, gain=gain)
                    for role, ids in [('fit', fit), ('source_held', held)]:
                        movement.append(dict(seed=seed, arm=arm, role=role, valid=int(valid[ids].sum()), count=len(ids),
                            raw_delta_rms=float(np.sqrt(np.square(dp[ids]).mean())), selected_gain=gain))
                    print('FULL_RESPONSE', seed, arm, 'gain', gain, flush=True)
            print('BUNDLE', seed, name, 'complete', flush=True)
        for arm, prior in deployment.items():
            source_rows.extend(dict(seed=seed, arm=arm, **r) for r in evaluate_episodes(prior, full_kernel, labels, held_episodes))
        for draw, (s, q) in enumerate(held_episodes):
            archive = {arm: target_readout(p, full_kernel, s, q, labels[s]) for arm, p in deployment.items()}
            _, classes = classification_rows(archive, labels[q], dict(seed=seed, draw=draw))
            source_classes.extend(classes)
            np.savez_compressed(OUT/f'{seed}_source_{draw}_predictions.npz', support_obs_ids=s, query_obs_ids=q, **archive)
        for task, (s, q, sy) in tasks.items():
            archive = {arm: target_readout(p, full_kernel, s, q, sy) for arm, p in deployment.items()}
            with np.load(PARENT/f'{seed}_{task}_predictions.npz') as old:
                np.testing.assert_array_equal(archive['baseline'], old['baseline'])
            np.savez_compressed(OUT/f'{seed}_{task}_predictions.npz', support_obs_ids=s, query_obs_ids=q, **archive)
    for fname, rows in [('response_results', response_rows), ('models', model_rows), ('selection', choices),
        ('validation', validation_rows), ('source_results', source_rows), ('source_classes', source_classes), ('movement', movement)]:
        pd.DataFrame(rows).to_csv(OUT/f'{fname}.csv', index=False)
    save_json(OUT/'PREDICTIONS_SAVED.json', dict(query_metrics_read=False, source_parameters_frozen=True))
    results, classes = [], []
    for seed in SEEDS:
        for task, (s, q, sy) in tasks.items():
            with np.load(OUT/f'{seed}_{task}_predictions.npz') as f:
                archive = {k: f[k] for k in f.files if k not in ('support_obs_ids', 'query_obs_ids')}
                rows, cls = classification_rows(archive, labels[q], dict(seed=seed, task=task, port=meta.iloc[q].port.iloc[0]))
                results.extend(rows); classes.extend(cls)
    df = pd.DataFrame(results); df.to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'target_classes.csv', index=False)
    for scope, frame in [('target', df), ('source', pd.DataFrame(source_rows))]:
        for suffix, keys in [('summary', ['arm']), ('seed_summary', ['arm', 'seed'])]:
            cols = ['OA', 'BA']+(['MSE'] if scope == 'source' else [])
            frame.groupby(keys)[cols].mean().reset_index().to_csv(OUT/f'{scope}_{suffix}.csv', index=False)
    df.groupby(['arm', 'port'])[['OA', 'BA']].mean().reset_index().to_csv(OUT/'target_port_summary.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-start, response_fits=32, neural_fits=0,
        main_model_modified=False, original_target_diagonals_exact=18, target_choice_used=False))
    print(df.groupby('arm')[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
