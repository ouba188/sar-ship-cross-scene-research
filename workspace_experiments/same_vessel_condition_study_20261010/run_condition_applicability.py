"""One fixed source-only applicability stump; no classification or target ports."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from condition_applicability import (BASE, ARMS, load_features, learn_gate,
    fit_responses, predict_responses, pair_data, predict_gate)
from incidence_transport import matched_pairs

OUT = BASE/'condition_applicability_20261011'


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def rule(model):
    feature = int(model['feature_index'])
    mix = model['leaf_mix']
    if feature == -1:
        return f'root: condition blend={mix[0]:.8f}'
    name = model['feature_names'][feature]
    return (f'{name} <= {model["threshold"]:.8f}: condition blend={mix[0]:.8f}; '
            f'otherwise: condition blend={mix[1]:.8f}')


def add_gate(data, predictions, gate):
    mix = predict_gate(data['inputs'], gate)
    predictions['gated'] = (1-mix)*predictions['anchor']+mix*predictions['condition']
    return mix


def pointwise(meta, split, data, predictions, mix, folds):
    a, b = data['directed'].T
    frame = pd.DataFrame(dict(split=split, a=a, b=b, mmsi=data['ships'],
        class_id=meta.iloc[a].class_id.to_numpy(), port=meta.iloc[a].port.to_numpy(),
        fold=folds, weight=data['weights'], truth=data['truth'], gate_mix=mix))
    for j, name in enumerate(('raw_center_a', 'total_logenergy_a', 'abs_angle_delta', 'mean_angle')):
        frame[name] = data['inputs'][:, j]
    for arm, prediction in predictions.items():
        frame[arm] = prediction
        frame[f'{arm}_squared_error'] = np.square(data['truth']-prediction)
    frame.to_csv(OUT/f'{split}_pointwise.csv', index=False)
    np.savez_compressed(OUT/f'{split}_predictions.npz', directed=data['directed'],
        truth=data['truth'], inputs=data['inputs'], weights=data['weights'],
        gate_mix=mix, fold=folds, **predictions)


def metrics(split, data, predictions):
    errors = {arm: float(data['weights']@np.square(data['truth']-p))
              for arm, p in predictions.items()}
    return [dict(split=split, arm=arm, pairs=len(data['truth'])//2,
        ships=len(np.unique(data['ships'])), mse=value,
        gain_vs_anchor_pct=100*(1-value/max(errors['anchor'], 1e-12)),
        gain_vs_condition_pct=100*(1-value/max(errors['condition'], 1e-12)))
        for arm, value in errors.items()]


def coverage(split, data, gate):
    feature = int(gate['feature_index'])
    leaves = np.zeros(len(data['truth']), dtype=int)
    if feature >= 0:
        leaves = (data['inputs'][:, feature] > gate['threshold']).astype(int)
    rows = []
    for leaf in range(1 if feature < 0 else 2):
        mask = leaves == leaf
        rows.append(dict(split=split, leaf=leaf, directions=int(mask.sum()),
            ships=len(np.unique(data['ships'][mask])), weight_mass=float(data['weights'][mask].sum()),
            condition_mix=float(gate['leaf_mix'][leaf])))
    return rows


def main():
    start = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    if (OUT/'PROTOCOL.json').exists():
        raise ValueError('Preserve previous results; this output already has a protocol')
    meta, fit, held, features = load_features()
    with np.load(BASE/'crossfit_evidence_20261011/source_folds.npz') as z:
        np.testing.assert_array_equal(z['fit_obs_ids'], fit)
        assignment = np.full(len(meta), -1, dtype=int)
        assignment[fit] = z['assignment']
    save_json(OUT/'PROTOCOL.json', dict(
        scope='Source only, fixed center_border_log_ratio absolute-change response. No classification or target-port evaluation.',
        gate_inputs=['raw A center ratio', 'A total log energy', 'absolute incidence gap', 'mean incidence'],
        forbidden_inputs='MMSI, port, true classes; MMSI is used only for splitting/leaf coverage, labels only existing class/ship/pair weights.',
        response='Unchanged condition_evidence_response fit_model/predict_model; fixed ridge and old four arms.',
        gain_target='Squared OOF anchor error minus squared OOF condition error.',
        selection='Root plus one depth1 stump: four features x ordinary directed-training-row quantiles .25/.50/.75. Each leaf >=12 distinct vessels. Select strictly smaller class/ship/pair-weighted gain-target SSE; ties keep first/root. No held choices.',
        blend='clip((leaf weighted anchor MSE - leaf weighted condition MSE)/(their sum +1e-12),0,1). Not a learned optimal convex-blend coefficient.',
        source_assessment='Outer original three vessel folds. For each outer fold, response crossfit within the two remaining folds trains the gate; response models fit all outertrain to evaluate outerval. Pooled metrics recompute original class/ship/pair weights.',
        deployment='Gate from all three source response-OOF folds; full response models from original1296 source rows. Export before reading held outcomes.',
        limitations='Developmental reused held cohort; historically pretrained frozen CNN not refit. Center/border is not true segmentation; magnitude is not recognition risk or causal identification. Both pair directions dependent; leaf ship sets can overlap.',
        neural_fits=0, target_evaluations=0))

    gate, candidates, pairs, data, oof, audit = learn_gate(meta, fit, features, assignment)
    models = fit_responses(meta, fit, features)
    np.savez_compressed(OUT/'gate.npz', **gate)
    for arm, model in models.items():
        np.savez_compressed(OUT/f'{arm}.npz', **model)
    save_json(OUT/'EXPORT_READY.json', dict(rule=rule(gate), held_used=False,
        api='load_features(); gate_inputs(anchor_a,total_logenergy_a,angle_a,angle_b); predict_gate(inputs,model)',
        fit_rows=len(fit), pairs=len(pairs), ships=len(np.unique(data['ships'])),
        response_api='condition_evidence_response.predict_model(anchor_a,total_logenergy_a,angle_a,angle_b,model)'))
    print('EXPORT_READY', rule(gate), flush=True)
    pd.DataFrame(candidates).to_csv(OUT/'full_split_candidates.csv', index=False)
    np.savez_compressed(OUT/'source_gate_training_oof.npz', directed=data['directed'],
        inputs=data['inputs'], truth=data['truth'], weights=data['weights'],
        anchor=oof['anchor'], condition=oof['condition'],
        gain_target=(data['truth']-oof['anchor'])**2-(data['truth']-oof['condition'])**2,
        fold=assignment[data['directed'][:, 0]])
    coverage_rows = coverage('full_gate_training_oof', data, gate)
    audit_rows = [dict(gate_training='full', **row) for row in audit]
    all_rows, rules = [], [dict(split='full', rule=rule(gate))]
    nested = {arm: np.full(len(data['truth']), np.nan) for arm in (*ARMS, 'gated')}
    nested_mix = np.full(len(data['truth']), np.nan)
    for fold in range(3):
        training, validation = fit[assignment[fit] != fold], fit[assignment[fit] == fold]
        local_gate, local_candidates, _, local_data, _, local_audit = learn_gate(
            meta, training, features, assignment)
        local_models = fit_responses(meta, training, features)
        val_pairs = matched_pairs(meta, validation)
        val_data = pair_data(meta, val_pairs, features)
        val_predictions = predict_responses(meta, val_data['directed'], features, local_models)
        mix = add_gate(val_data, val_predictions, local_gate)
        destination = OUT/f'outer{fold}'
        destination.mkdir(exist_ok=True)
        np.savez_compressed(destination/'gate.npz', **local_gate)
        for arm, model in local_models.items():
            np.savez_compressed(destination/f'{arm}.npz', **model)
        pd.DataFrame(local_candidates).to_csv(destination/'split_candidates.csv', index=False)
        # Preserve canonical global pair order; fold-local reverse pairs have their own block.
        index = {tuple(pair): i for i, pair in enumerate(data['directed'])}
        positions = np.asarray([index[tuple(pair)] for pair in val_data['directed']])
        for arm in nested:
            nested[arm][positions] = val_predictions[arm]
        nested_mix[positions] = mix
        all_rows.extend(metrics(f'source_outer{fold}', val_data, val_predictions))
        coverage_rows.extend(coverage(f'outer{fold}_gate_training', local_data, local_gate))
        coverage_rows.extend(coverage(f'outer{fold}_evaluation', val_data, local_gate))
        audit_rows.extend(dict(gate_training=f'outer{fold}', **row) for row in local_audit)
        rules.append(dict(split=f'outer{fold}', rule=rule(local_gate)))
        print('SOURCE_OUTER', fold, rule(local_gate), flush=True)
    assert all(np.isfinite(v).all() for v in nested.values())
    all_rows.extend(metrics('source_nested', data, nested))
    pointwise(meta, 'source_nested', data, nested, nested_mix, assignment[data['directed'][:, 0]])

    held_pairs = matched_pairs(meta, held)
    held_data = pair_data(meta, held_pairs, features)
    held_predictions = predict_responses(meta, held_data['directed'], features, models)
    held_mix = add_gate(held_data, held_predictions, gate)
    all_rows.extend(metrics('held', held_data, held_predictions))
    coverage_rows.extend(coverage('held', held_data, gate))
    pointwise(meta, 'held', held_data, held_predictions, held_mix, np.full(len(held_mix), -1))
    results = pd.DataFrame(all_rows)
    results.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT/'leaf_coverage.csv', index=False)
    pd.DataFrame(audit_rows).to_csv(OUT/'crossfit_audit.csv', index=False)
    pd.DataFrame(rules).to_csv(OUT/'rules.csv', index=False)
    # Replay every saved deployment response and blend, including outer-fold exports.
    max_error, replay_count = 0., 0
    for fold in (-1, 0, 1, 2):
        folder = OUT if fold == -1 else OUT/f'outer{fold}'
        evaluation = held_data if fold == -1 else pair_data(meta, matched_pairs(meta, fit[assignment[fit] == fold]), features)
        loaded = {}
        for arm in ARMS:
            with np.load(folder/f'{arm}.npz') as z:
                loaded[arm] = dict(z)
        replay = predict_responses(meta, evaluation['directed'], features, loaded)
        with np.load(folder/'gate.npz') as z:
            add_gate(evaluation, replay, dict(z))
        positions = np.asarray([{tuple(p): i for i, p in enumerate(data['directed'])}[tuple(p)]
                                for p in evaluation['directed']]) if fold >= 0 else None
        for arm in replay:
            expected = held_predictions[arm] if fold == -1 else nested[arm][positions]
            error = float(np.max(np.abs(replay[arm]-expected)))
            np.testing.assert_array_equal(replay[arm], expected)
            max_error = max(max_error, error)
            replay_count += 1
    receipt = dict(seconds=time.perf_counter()-start, response_fits=34, gate_fits=4,
                   replay_arrays=replay_count, replay_max_error=max_error,
                   neural_fits=0, target_evaluations=0, classification_evaluations=0,
                   full_rule=rule(gate))
    save_json(OUT/'COMPLETE.json', receipt)
    print(results[results.split.isin(['source_nested', 'held'])].to_string(index=False), flush=True)
    print('COMPLETE', receipt, flush=True)


if __name__ == '__main__':
    main()
