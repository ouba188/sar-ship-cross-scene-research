"""Integrate frozen source-selected directional corrections into target policies."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import json
import pickle

import numpy as np
import pandas as pd

from conditional_direction_residual import fit_predict, normalize_direction, rank_conditions
from probe_display_rotation import ROOT, SOURCE, PARENT, REFERENCE, base_phi, replace_r50
from cross_dataset_c4 import (
    OUT as EXTERNAL, DATASETS, base_coordinates, base_phi as external_phi, route,
)

OUT = ROOT/'conditional_direction_residual_20261009'
C4 = ROOT/'c4_base_trial_20261009'
DIAGNOSIS = ROOT/'c4_condition_diagnosis_20261009'
MODES = ('baseline', 'unconditional_selected', 'conditional_selected',
         'unconditional_fixed1', 'conditional_fixed1', 'uniform_condition_fixed1')


def load_npz(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def load_pickle(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def completed(output, etas):
    frozen = output/'FROZEN.json'
    if not frozen.exists():
        return False
    saved = json.loads(frozen.read_text(encoding='utf-8'))
    if saved['modes'] != list(MODES) or saved['selected_etas'] != etas:
        raise ValueError(f'Completed target settings differ: {output}')
    if not (output/'policies.npz').exists():
        raise FileNotFoundError(output/'policies.npz')
    print(f'REUSE TARGET {output}', flush=True)
    return True


def descriptors(dataset, rows, sample_ids):
    frame = pd.read_csv(DIAGNOSIS/'descriptors'/f'{dataset}.csv',
                        usecols=['row_id', 'sample_id', 'elongation_proxy', 'center_excess_proxy'])
    frame = frame.set_index('row_id')
    if not frame.index.is_unique:
        raise ValueError('Descriptor row_id is not unique')
    frame = frame.loc[rows]
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str), sample_ids)
    return rank_conditions(frame)


def integrate(output, phi, prior, delta, conditions, supports, masks,
              support_labels, experts, route_fn, expected, original,
              query_rows, sample_ids, etas, provenance, baseline_scores=None):
    if (output/'FROZEN.json').exists():
        raise ValueError(f'Preserve completed target policies: {output}')
    if supports.shape != (5, 64) or masks.shape != (5, len(prior)):
        raise ValueError('Expected five draws of64 labels and matched masks')
    direction_norm = np.linalg.norm(delta, axis=1)
    direction = normalize_direction(delta)
    configs = [(0., False), (float(etas['unconditional']), False),
               (float(etas['conditional']), True), (1., False), (1., True), (1., 'uniform')]
    predictions = np.empty((5, len(MODES), len(prior)), dtype=np.int8)
    bases = np.empty_like(predictions)
    selected = np.empty(predictions.shape, dtype=bool)
    diagnostics = []
    replay_error = 0.
    for draw, support in enumerate(supports):
        ev, sy = masks[draw], support_labels[draw]
        if len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('Support/query overlap or duplicate support')
        baseline = None
        for arm, (eta, conditional) in enumerate(configs):
            score, parts = fit_predict(phi, prior, direction, conditions, support,
                                      sy, eta, conditional=conditional, strength=8., return_parts=True)
            if not np.isfinite(score).all():
                raise ValueError('Nonfinite target scores')
            pred, sel = route_fn(score, experts[draw], sy, ev)
            bp = score.argmax(1)
            if sel[~ev].any() or sel.sum() > int(np.floor(.05*ev.sum())):
                raise ValueError('Expert handoff exceeded query-only5% cap')
            if arm == 0:
                baseline = score
                for key, actual in [('base_predictions', bp), ('predictions', pred), ('selected', sel)]:
                    np.testing.assert_array_equal(actual, expected[key][draw], err_msg=f'{output} draw{draw} {key}')
                if baseline_scores is not None:
                    error = float(np.max(np.abs(score-baseline_scores[draw])))
                    replay_error = max(replay_error, error)
                    if error > 3e-6:
                        raise ValueError(f'Baseline score replay failed: {error}')
            predictions[draw, arm], bases[draw, arm], selected[draw, arm] = pred, bp, sel
            difference = score[ev]-baseline[ev]
            diagnostics.append(dict(draw=draw, mode=MODES[arm], eta=eta,
                conditional=conditional, query_count=int(ev.sum()),
                score_delta_rms=float(np.sqrt(np.mean(difference**2))),
                base_correction_rms=float(np.sqrt(np.mean(parts['base'][ev]**2))),
                direction_correction_rms=float(np.sqrt(np.mean(parts['direction'][ev]**2))),
                base_argmax_changes=int(np.sum(bp[ev] != bases[draw, 0, ev])),
                full_prediction_changes=int(np.sum(pred[ev] != predictions[draw, 0, ev])),
                selected_count=int(sel.sum())))
        print(f'TARGET {output.name} draw{draw} baseline exact', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'policies.npz', modes=np.asarray(MODES),
        predictions=predictions, base_predictions=bases, selected=selected,
        support_rows=supports, eval_masks=masks, query_rows=query_rows,
        sample_ids=sample_ids, conditions=conditions, direction_norm=direction_norm,
        original_predictions=original['predictions'],
        original_base_predictions=original['base_predictions'])
    pd.DataFrame(diagnostics).to_csv(output/'diagnostics.csv', index=False)
    write_json(output/'FROZEN.json', dict(**provenance, rows=len(prior), draws=5,
        modes=MODES, selected_etas=etas, query_labels_used_in_fit=0,
        labels_per_draw=64, budget=.05, baseline_predictions_and_selection_exact=True,
        baseline_score_replay_max_abs=replay_error if baseline_scores is not None else None))


def run_ports(selection, requested, mean_control=False):
    from direct_incidence_conditioning import condition_features
    from run_conditioned_vessel_drift import SIDECAR
    from receiver_consistent_handoff import confidence_parts
    from evidence_aware_handoff import predict_gate, route_scores

    paths = sorted((DIAGNOSIS/'port_policies').glob('*/policies.npz'))
    if requested:
        available = {path.parent.name for path in paths}
        if set(requested)-available:
            raise ValueError(f'Unknown target ports: {set(requested)-available}')
        paths = [path for path in paths if path.parent.name in requested]
    elif len(paths) != 23:
        raise ValueError('Expected23 evaluable target ports')
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'class_id'])
    side = pd.read_csv(SIDECAR, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    zero = np.load(C4/'pooled_zero.npy', mmap_mode='r')
    average = np.load(C4/'pooled_c4.npy', mmap_mode='r')
    for path in paths:
        port = path.parent.name
        if completed(OUT/'targets/self24ports'/port, selection['ports'][port]):
            continue
        saved = load_npz(path)
        rows, masks = saved['query_rows'], saved['eval_masks']
        supports = pd.Index(rows).get_indexer(saved['support_rows'].ravel()).reshape(5, 64)
        if (supports < 0).any():
            raise ValueError('Support global rows absent from target cohort')
        parent, ref = PARENT/port, REFERENCE/port
        np.testing.assert_array_equal(np.load(parent/'target_rows.npy'), rows)
        g = load_pickle(parent/'geometry.pkl')
        gate = load_pickle(parent/'source_models.pkl')['gates']['joint']
        head = load_pickle(ref/'direct_heads.pkl')['joint_base']
        old = load_npz(ref/'policies.npz')
        np.testing.assert_array_equal(old['query_rows'], rows)
        np.testing.assert_array_equal(old['support_rows'], saved['support_rows'])
        np.testing.assert_array_equal(old['eval_mask'], masks)
        expert = old['expert_scores'][:, list(old['mode_names']).index('direct_condition')]
        bank = np.load(parent/'target_bank.npy', mmap_mode='r')
        bank0 = replace_r50(zero[rows].astype(float), bank, g)
        bank4 = replace_r50(average[rows].astype(float), bank, g)
        u = condition_features(frame.loc[rows, 'incidence_deg'].to_numpy())
        phi = base_phi(bank4, g, u)
        prior = np.c_[bank4, u] @ head['coef']+head['intercept']
        delta = (bank0[:, :128]-bank4[:, :128])/g['base_scale']
        if mean_control:
            delta = (bank4[:, :128]-g['base_center'])/g['base_scale']
        ids = frame.loc[rows, 'instance_id'].to_numpy(str)
        conditions = descriptors('self24ports', rows, ids)
        sy = [frame.loc[s, 'class_id'].to_numpy(int) for s in saved['support_rows']]

        def route_port(score, expert_score, labels, ev):
            em, eh = confidence_parts(expert_score)
            bm, bh = confidence_parts(score)
            result = route_scores(score, expert_score, predict_gate(gate, np.c_[em, bm, eh, bh]),
                                  labels, ev, rows, budget=.05)
            return result['predictions'], result['selected']

        phase = list(saved['phases']).index('support_c4')
        reference = list(saved['phases']).index('reference')
        keys = ('base_predictions', 'predictions', 'selected')
        expected = {key: saved[key][:, phase] for key in keys}
        original = {key: saved[key][:, reference] for key in keys[:2]}
        c4 = load_npz(C4/'ports'/port/'policies.npz')
        np.testing.assert_array_equal(c4['query_rows'], rows)
        np.testing.assert_array_equal(c4['support_rows'], saved['support_rows'])
        np.testing.assert_array_equal(c4['eval_mask'], masks)
        integrate(OUT/'targets/self24ports'/port, phi, prior, delta, conditions,
            supports, masks, sy, expert, route_port, expected, original, rows, ids,
            selection['ports'][port], dict(dataset='self24ports', port=port,
                model_path=str(parent/'geometry.pkl'), baseline_policy_path=str(path),
                direction_scale="g['base_scale']", query_rows_kind='source_manifest_global',
                representation='C4_mean_control' if mean_control else 'direction_residual'),
            baseline_scores=c4['base_scores'][:, list(c4['base_modes']).index('support_c4')])


def run_external(name, selection, mean_control=False):
    if completed(OUT/'targets'/name, selection['external']):
        return
    target_dir, source_dir, joint_dir = DATASETS[name]
    frame = pd.read_csv(ROOT/target_dir/'target_manifest.csv', keep_default_na=False)
    provenance = json.loads((EXTERNAL/name/'FROZEN.json').read_text(encoding='utf-8'))
    model_path = provenance.get('model_path', str(ROOT/source_dir/'source_model.pkl'))
    policy_path = provenance.get('base_policy_path', str(ROOT/target_dir/'policies.npz'))
    with open(model_path, 'rb') as stream:
        model = pickle.load(stream)
    old = load_npz(policy_path)
    saved = load_npz(EXTERNAL/name/'support_refit_diagnostic/policies.npz')
    joint = load_npz(ROOT/joint_dir/'policies.npz')
    ids = frame.sample_id.to_numpy(str)
    np.testing.assert_array_equal(old['sample_ids'], ids)
    np.testing.assert_array_equal(np.load(EXTERNAL/name/'sample_ids.npy'), ids)
    np.testing.assert_array_equal(joint['sample_ids'], ids)
    np.testing.assert_array_equal(saved['support_rows'], old['support_rows'])
    np.testing.assert_array_equal(saved['support_rows'], joint['support_rows'])
    np.testing.assert_array_equal(saved['eval_masks'], old['eval_masks'])
    np.testing.assert_array_equal(saved['eval_masks'], joint['eval_mask'])
    x0 = base_coordinates(np.load(EXTERNAL/name/'pooled_zero.npy', mmap_mode='r'), model)
    x4 = base_coordinates(np.load(EXTERNAL/name/'pooled_c4.npy', mmap_mode='r'), model)
    phi = external_phi(x4, model)
    prior = model['base'].decision_function(x4).astype(np.float32).astype(float)
    delta = (x0-x4)/model['residual_scale']
    if mean_control:
        delta = (x4-model['residual_center'])/model['residual_scale']
    rows = np.arange(len(frame))
    conditions = descriptors(name, rows, ids)
    sy = [frame.loc[s, 'class_id'].to_numpy(int) for s in saved['support_rows']]
    route_ids = frame.target_manifest_row.to_numpy(int)

    def route_external(score, expert, labels, ev):
        return route(score, expert, model['gate'], labels, ev, route_ids)

    reference = load_npz(EXTERNAL/name/'policies.npz')
    np.testing.assert_array_equal(reference['sample_ids'], ids)
    np.testing.assert_array_equal(reference['support_rows'], saved['support_rows'])
    np.testing.assert_array_equal(reference['eval_masks'], saved['eval_masks'])
    slot = list(reference['arms']).index('reference')
    original = {key: reference[key][:, slot] for key in ('predictions', 'base_predictions')}
    integrate(OUT/'targets'/name, phi, prior, delta, conditions,
        saved['support_rows'], saved['eval_masks'], sy, joint['expert_scores'],
        route_external, saved, original, rows, ids, selection['external'],
        dict(dataset=name, model_path=model_path, base_policy_path=policy_path,
             baseline_policy_path=str(EXTERNAL/name/'support_refit_diagnostic/policies.npz'),
             direction_scale="model['residual_scale']", query_rows_kind='target_manifest_local',
             representation='C4_mean_control' if mean_control else 'direction_residual'))


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['self24ports', 'f0921', 'oss2'],
                        default=['self24ports', 'f0921', 'oss2'])
    parser.add_argument('--ports', nargs='+', help='Optional self24ports subset, e.g. Callao')
    parser.add_argument('--mean-control', action='store_true',
                        help='Post-readout representation control; separate outputs, same fixed strengths')
    args = parser.parse_args()
    selection_path = OUT/'source_selection.json'
    if not selection_path.exists():
        raise FileNotFoundError('Complete source selection before target integration')
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if args.mean_control:
        OUT = OUT/'mean_representation_control'
        OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT/'TARGET_PROTOCOL.json', dict(
        scope=['self24ports', 'f0921', 'oss2'], modes=MODES, source_selection=str(selection_path),
        source_selection_snapshot=selection, target_labels_per_draw=64, draws=5,
        support_ridge=8., expert_handoff_fraction=.05, experts_and_gates='Frozen original snapshots',
        baseline='Original source head; C4 support/query features; same64 support refit',
        conditions='Within-target-cohort unlabelled ranks of elongation_proxy and center_excess_proxy',
        direction='Row-L2 normalized C4 mean / residual scale' if args.mean_control else
                  'Row-L2 normalized zero-minus-C4 normalized PCA128 / residual scale',
        post_readout_mechanism_control=args.mean_control,
        strengths_reselected_for_control=False,
        uniform_control='eta1, uniform support-mean offdiagonal condition weight; diagonal unchanged',
        saved_scores=False, diagnostics='Query score-delta RMS, branch RMS, prediction changes; no query labels',
        target_selection=False, independent_confirmation=False, historical_development=True,
        compute='Four R50 passes per sample;5% limits handoff, not feature compute',
        index_convention='support_rows always local; query_rows source-global for24ports, local externally',
        invariance='Display-rotation residual, not identified causal or physical-view invariance'))
    if 'self24ports' in args.datasets:
        run_ports(selection, args.ports, args.mean_control)
    for name in args.datasets:
        if name != 'self24ports':
            run_external(name, selection, args.mean_control)


if __name__ == '__main__':
    main()
