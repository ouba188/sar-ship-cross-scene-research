"""Source-only mechanism checks; oracle capacities are NOT DEPLOYABLE results."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from functools import partial
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.optimize import OptimizeWarning

import correct_pair_capacity as capacity
from content_condition_swap_diagnose import reconstruction_diagnostics
from relative_content_condition import load_data, OUT, vessel_targets, old, FactorNet, SEEDS, save_json
from relative_factor_separation import ship_average


ARMS = ('anchor', 'joint', 'self_aux', 'group_aux')
BOUND = .1
MARGIN_TOLERANCE = 1e-7


@torch.no_grad()
def auxiliary_diagnostics(model, state, x, meta):
    """Evaluate existing auxiliary heads on test-only empirical vessel targets."""
    shared, _, valid = vessel_targets(x, meta)
    ships = meta.loc[valid, 'mmsi'].to_numpy()
    stable_outputs, condition_outputs = [], []
    for lo in range(0, len(x), 256):
        batch = torch.from_numpy(x[lo:lo+256])
        stable_outputs.append(F.linear(model.stable(batch), state['shared_head']['weight'],
                                       state['shared_head']['bias']).numpy())
        condition_outputs.append(F.linear(model.condition(batch), state['residual_decoder']['weight'],
                                          state['residual_decoder']['bias']).numpy())
    stable, condition = np.concatenate(stable_outputs), np.concatenate(condition_outputs)
    result = dict(aux_observations=int(valid.sum()), aux_ships=int(pd.Series(ships).nunique()),
                  aux_observation_coverage_fraction=float(valid.mean()))
    for name, prediction, truth in (
        ('shared', stable, shared), ('residual', condition, x-shared),
        ('sum', stable+condition, x),
    ):
        target = truth[valid].astype(np.float64)
        error = ship_average(np.square(prediction[valid].astype(np.float64)-target).mean(1), ships)
        zero = ship_average(np.square(target).mean(1), ships)
        result.update({f'aux_{name}_mse': error, f'aux_{name}_zero_mse': zero,
                       f'aux_{name}_skill': float(100*(1-error/zero)) if zero > 0 else float('nan')})
    return result


def capacity_diagnostics(h, fixed, meta, test, gate_fit):
    """Independent label-informed gains, only for originally wrong source rows."""
    with torch.no_grad():
        basis, weight, bias = [fixed[key] for key in ('basis', 'weight', 'bias')]
        scores = (h[test]@weight+bias).numpy().astype(np.float64)
        coordinates = (h[test]@basis).numpy().astype(np.float64)
        axes = (basis.T@weight).numpy().astype(np.float64)
        amplitude = (h[gate_fit]@basis).square().mean(0).sqrt().numpy().astype(np.float64)
    truth = meta.iloc[test].class_id.to_numpy()
    predictions = scores.argmax(1)
    wrong = np.flatnonzero(predictions != truth)
    if len(test) != 823 or len(wrong) != 464:
        raise ValueError(f'Unexpected source cohort: {len(test)} observations, {len(wrong)} errors; expected 823/464')

    rows, summaries = [], []
    # SciPy forwards the HiGHS-specific option verbatim. Scope the override here
    # to keep the reused oracle single-threaded without editing its source file.
    original_linprog = capacity.linprog
    capacity.linprog = partial(original_linprog, options={'threads': 1})
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected:.*threads.*',
                                    category=OptimizeWarning)
            for operator in ('scale', 'residual'):
                operator_rows = []
                for j, i in enumerate(wrong):
                    label, oid = int(truth[i]), int(test[i])
                    effects = (coordinates[i] if operator == 'scale' else amplitude)[:, None]*axes
                    margin, gain = capacity.oracle_margin(scores[i], effects, label, bound=BOUND)
                    rivals = np.arange(len(scores[i])) != label
                    raw_margin = float(scores[i, label]-scores[i, rivals].max())
                    assert margin >= raw_margin-MARGIN_TOLERANCE, 'Zero gain must remain feasible'
                    assert np.max(np.abs(gain)) <= BOUND+MARGIN_TOLERANCE, 'Oracle gain exceeds fixed bound'
                    row = dict(operator=operator, obs_id=oid, mmsi=meta.iloc[oid].mmsi,
                        product=meta.iloc[oid]['product'], class_id=label,
                        baseline_prediction=int(predictions[i]), raw_correct=0, raw_margin=raw_margin,
                        oracle_margin=margin, oracle_prediction=int((scores[i]+gain@effects).argmax()),
                        correctable=int(margin > MARGIN_TOLERANCE),
                        near_boundary=int(abs(margin) <= MARGIN_TOLERANCE),
                        max_absolute_gain=float(np.max(np.abs(gain))),
                        bound=BOUND, label_informed=True, notdeployable=True)
                    row.update({f'gain_{k}': float(value) for k, value in enumerate(gain)})
                    operator_rows.append(row)
                    if (j+1) % 128 == 0:
                        print('CAPACITY_PROGRESS', operator, j+1, '/', len(wrong), flush=True)
                rows.extend(operator_rows)
                correctable = sum(row['correctable'] for row in operator_rows)
                summary = dict(operator=operator, source_test_observations=len(test),
                    source_test_ships=int(meta.iloc[test].mmsi.nunique()), source_gate_fit_observations=len(gate_fit),
                    baseline_wrong=len(wrong), baseline_error_percent=100*len(wrong)/len(test),
                    correctable_wrong=correctable, correctable_wrong_percent=100*correctable/len(wrong),
                    uncorrectable_wrong=len(wrong)-correctable,
                    uncorrectable_wrong_percent=100*(len(wrong)-correctable)/len(wrong),
                    remaining_error_floor_percent=100*(len(wrong)-correctable)/len(test),
                    near_boundary=sum(row['near_boundary'] for row in operator_rows),
                    bound=BOUND, label_informed=True, notdeployable=True)
                summaries.append(summary)
                print('CAPACITY', summary, flush=True)
    finally:
        capacity.linprog = original_linprog
    return pd.DataFrame(rows), pd.DataFrame(summaries), amplitude


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    meta, split, roles, _, _, norm, fixed, h = load_data()
    fit, test = split['factor_fit'], split['test_source']
    if len(test) != 823:
        raise ValueError(f'Expected the fixed 823-observation source test, got {len(test)}')
    destination = OUT/'mechanism_checks'
    destination.mkdir(exist_ok=False)
    print('SOURCE', dict(test_observations=len(test), test_ships=int(meta.iloc[test].mmsi.nunique()),
                         factor_fit_observations=len(fit), gate_fit_observations=len(roles['fit'])), flush=True)

    rows = []
    test_meta = meta.iloc[test].reset_index(drop=True)
    for arm in ARMS:
        for seed in SEEDS:
            checkpoint = (old.OUT if arm in ('anchor', 'joint') else OUT)/f'{arm}_{seed}_factor.pt'
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model = FactorNet().eval()
            model.load_state_dict(state['model'])
            row = dict(arm=arm, seed=int(seed), checkpoint=str(checkpoint),
                       **reconstruction_diagnostics(model, norm, meta, fit, test))
            if arm in ('self_aux', 'group_aux'):
                row.update(auxiliary_diagnostics(model, state, norm[test], test_meta))
            rows.append(row)
            print('DECODER', arm, seed, {key: row[key] for key in
                  ('self_mse', 'swap_mse', 'condition_delta_skill', 'same_class_other_ship_delta_skill')}, flush=True)
    decoders = pd.DataFrame(rows)
    decoders.to_csv(destination/'decoder.csv', index=False)
    summary = decoders.drop(columns='seed').groupby('arm', sort=False).mean(numeric_only=True)
    summary.insert(0, 'seed_count', decoders.groupby('arm').seed.nunique())
    summary.reset_index().to_csv(destination/'summary.csv', index=False)

    samples, capacities, amplitude = capacity_diagnostics(h, fixed, meta, test, roles['fit'])
    samples.to_csv(destination/'capacity.csv', index=False)
    capacities.to_csv(destination/'capacity_summary.csv', index=False)
    save_json(destination/'COMPLETE.json', dict(
        seconds=time.perf_counter()-start, resources='CPU1; torch/BLAS/HiGHS threads=1',
        source_only=True, training=False, target_metrics_accessed=False, historical_files_modified=False,
        test_observations=len(test), test_ships=int(test_meta.mmsi.nunique()),
        factor_fit_observations=len(fit), source_gate_fit_observations=len(roles['fit']),
        arms=list(ARMS), seeds=[int(seed) for seed in SEEDS], decoder_rows=len(decoders),
        auxiliary_rows=int(decoders.aux_shared_mse.notna().sum()), capacity_rows=len(samples),
        files=['decoder.csv', 'summary.csv', 'capacity.csv', 'capacity_summary.csv'],
        decoder_protocol='Every within-test same-vessel cross-product pair, both directions, equal vessel weights. '
            'Latent replacement means use factor_fit only. Summary is the arithmetic mean over two seeds per arm.',
        donor_protocol='Same-class different-vessel test donors, fixed seed41033, uniform donor vessel then observation. '
            'Products unrestricted; coverage and matched-subset actual-pair references are reported.',
        auxiliary_protocol='Frozen Ds/Dc checkpoints on test-only vessel_targets: equal-product shared mean and x-minus-mean. '
            'Only multi-product vessels, observations averaged within vessel then vessels equally. '
            'Zero-baseline skills in percent; sum reconstructed x is also reported. '
            'self_aux supervises only the sum, so separate components need not be identified. '
            'Shared targets include the focal acquisition and are empirical diagnostic proxies, not causal content.',
        capacity_protocol='Fixed original classifier and basis; original 464 errors only. '
            'scale effects=(h@basis)*basis.T@weight; residual uses source gate-fit RMS(h@basis) instead. '
            'Independent per-image label-informed oracle_margin at fixed closed bounds [-0.1,0.1]. '
            'This is an optimistic outer bound of the tanh gate, ignoring context learnability. '
            'Only margins>1e-7 count as correctable; near-zero margins are recorded separately. '
            'Remaining error floor retains all original correct rows via feasible zero gain; '
            'no capacity LP is run on those rows. Capacity is independent of factor arm and seed. '
            'NOT DEPLOYABLE, not target support-adapted accuracy, no label-based hyperparameter selection.',
        capacity_bound=BOUND, margin_tolerance=MARGIN_TOLERANCE,
        residual_amplitude=amplitude.tolist(), label_informed=True, notdeployable=True,
        capacity_summary=capacities.to_dict('records')))
    display = ['self_mse', 'swap_mse', 'condition_delta_skill', 'same_class_other_ship_delta_skill',
               'mean_condition_output_change_mse', 'mean_stable_output_change_mse',
               'aux_shared_skill', 'aux_residual_skill', 'aux_sum_skill']
    print('DECODER_MEANS\n'+summary[display].to_string(), flush=True)
    print('COMPLETE', destination, 'seconds', round(time.perf_counter()-start, 3), flush=True)


if __name__ == '__main__':
    main()
