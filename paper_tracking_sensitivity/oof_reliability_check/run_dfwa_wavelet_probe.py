"""Fixed-strength wavelet evidence additions to the frozen C4 classifier."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import json

import numpy as np
import pandas as pd

from conditional_direction_residual import fit_predict, normalize_direction
from dfwa_wavelet_features import OUT, LOW_MASK
from run_conditional_direction_targets import (
    load_npz, load_pickle, descriptors, ROOT, SOURCE, PARENT, REFERENCE, DIAGNOSIS,
    C4, EXTERNAL, DATASETS, base_phi, replace_r50, base_coordinates, external_phi, route)


MODES = ('c4_baseline', 'mean', 'mean_low', 'mean_high', 'mean_all')
ETA = .25
FEATURE_ROOT = OUT/'features'


def wavelet_bank(dataset):
    path = FEATURE_ROOT/dataset
    if not (path/'COMPLETE.json').exists():
        raise ValueError('Complete wavelet extraction before target inference: '+dataset)
    return np.load(path/'features.npy', mmap_mode='r')


def branch_vectors(mean, wavelet, source_wavelet):
    center = source_wavelet.mean(0, dtype=float)
    scale = source_wavelet.std(0, dtype=float)+1e-6
    wave = (wavelet-center)/scale
    mean = normalize_direction(mean)
    vectors = {'c4_baseline': mean, 'mean': mean, 'mean_half': mean}
    for name, mask in [('mean_low', LOW_MASK), ('mean_high', ~LOW_MASK), ('mean_all', slice(None))]:
        vectors[name] = np.c_[mean, normalize_direction(wave[:, mask])]/np.sqrt(2)
    return vectors, center, scale


def infer(folder, phi, prior, mean, wavelet, source_wavelet, conditions,
          support, masks, support_labels, expert, route_fn, expected, rows, ids, source_rows):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'FROZEN.json').exists():
        print('REUSE PROBE '+folder.name, flush=True)
        return
    vectors, center, scale = branch_vectors(mean, wavelet, source_wavelet)
    bases = np.empty((5, len(MODES), len(rows)), dtype=np.int8)
    fulls = np.empty_like(bases)
    selected = np.empty(bases.shape, dtype=bool)
    for draw, s in enumerate(support):
        if len(s) != 64 or masks[draw][s].any():
            raise ValueError('Unmatched support64 or support/query overlap')
        for arm, mode in enumerate(MODES):
            eta = 0 if mode == 'c4_baseline' else ETA/2 if mode == 'mean_half' else ETA
            score = fit_predict(phi, prior, vectors[mode], conditions, s, support_labels[draw], eta, True)
            pred, sel = route_fn(score, expert[draw], support_labels[draw], masks[draw])
            if sel[~masks[draw]].any() or sel.sum() > int(.05*masks[draw].sum()):
                raise ValueError('Handoff budget changed')
            bases[draw, arm], fulls[draw, arm], selected[draw, arm] = score.argmax(1), pred, sel
            if mode == 'c4_baseline':
                for key, actual in [('base_predictions', score.argmax(1)), ('predictions', pred), ('selected', sel)]:
                    np.testing.assert_array_equal(expected[key][draw], actual)
        print(f'PROBE {folder.name} draw{draw} baseline exact', flush=True)
    np.savez_compressed(folder/'policies.npz', modes=np.array(MODES),
        base_predictions=bases, predictions=fulls, selected=selected, query_rows=rows,
        sample_ids=ids, support_rows=support, eval_masks=masks, source_rows=source_rows,
        wavelet_center=center, wavelet_scale=scale)
    (folder/'FROZEN.json').write_text(json.dumps(dict(rows=len(rows), labels_per_draw=64,
        eta=ETA, modes=MODES, exact_baseline=True, source_only_wavelet_normalization=True,
        target_query_labels_used_in_fit=0), indent=2), encoding='utf-8')


def ports():
    from direct_incidence_conditioning import condition_features
    from run_conditioned_vessel_drift import SIDECAR
    from receiver_consistent_handoff import confidence_parts
    from evidence_aware_handoff import predict_gate, route_scores
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'class_id'])
    side = pd.read_csv(SIDECAR, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    wave = wavelet_bank('self24ports')
    np.testing.assert_array_equal(np.load(FEATURE_ROOT/'self24ports/sample_ids.npy'), frame.instance_id)
    average = np.load(C4/'pooled_c4.npy', mmap_mode='r')
    for path in sorted((DIAGNOSIS/'port_policies').glob('*/policies.npz')):
        port = path.parent.name
        saved = load_npz(path)
        rows, masks = saved['query_rows'], saved['eval_masks']
        s = pd.Index(rows).get_indexer(saved['support_rows'].ravel()).reshape(5, 64)
        if (s < 0).any():
            raise ValueError('Missing support row')
        g = load_pickle(PARENT/port/'geometry.pkl')
        gate = load_pickle(PARENT/port/'source_models.pkl')['gates']['joint']
        head = load_pickle(REFERENCE/port/'direct_heads.pkl')['joint_base']
        old = load_npz(REFERENCE/port/'policies.npz')
        expert = old['expert_scores'][:, list(old['mode_names']).index('direct_condition')]
        sr = np.load(PARENT/port/'source_rows.npy')
        if np.intersect1d(sr, rows).size:
            raise ValueError('Target entered source normalization')
        bank = replace_r50(average[rows].astype(float), np.load(PARENT/port/'target_bank.npy'), g)
        u = condition_features(frame.loc[rows, 'incidence_deg'].to_numpy())
        phi = base_phi(bank, g, u)
        prior = np.c_[bank, u]@head['coef']+head['intercept']
        mean = (bank[:, :128]-g['base_center'])/g['base_scale']
        ids = frame.loc[rows, 'instance_id'].to_numpy(str)
        cond = descriptors('self24ports', rows, ids)
        sy = [frame.loc[a, 'class_id'].to_numpy(int) for a in saved['support_rows']]
        def route_port(score, expert_score, labels, ev):
            em, eh = confidence_parts(expert_score)
            bm, bh = confidence_parts(score)
            routed = route_scores(score, expert_score, predict_gate(gate, np.c_[em, bm, eh, bh]),
                                  labels, ev, rows, budget=.05)
            return routed['predictions'], routed['selected']
        slot = list(saved['phases']).index('support_c4')
        expected = {k: saved[k][:, slot] for k in ('base_predictions', 'predictions', 'selected')}
        infer(OUT/'targets/self24ports'/port, phi, prior, mean, wave[rows], wave[sr], cond,
              s, masks, sy, expert, route_port, expected, rows, ids, sr)


def external(name):
    target_dir, source_dir, joint_dir = DATASETS[name]
    frame = pd.read_csv(ROOT/target_dir/'target_manifest.csv', keep_default_na=False)
    provenance = json.loads((EXTERNAL/name/'FROZEN.json').read_text())
    model_path = provenance.get('model_path', str(ROOT/source_dir/'source_model.pkl'))
    from pathlib import Path
    model = load_pickle(Path(model_path))
    saved = load_npz(EXTERNAL/name/'support_refit_diagnostic/policies.npz')
    joint = load_npz(ROOT/joint_dir/'policies.npz')
    sr = load_npz(ROOT/source_dir/'source_split.npz')['source_rows']
    source_wave = wavelet_bank('self24ports')
    wave = wavelet_bank(name)
    ids = frame.sample_id.to_numpy(str)
    np.testing.assert_array_equal(ids, np.load(FEATURE_ROOT/name/'sample_ids.npy'))
    np.testing.assert_array_equal(ids, joint['sample_ids'])
    np.testing.assert_array_equal(saved['support_rows'], joint['support_rows'])
    np.testing.assert_array_equal(saved['eval_masks'], joint['eval_mask'])
    x = base_coordinates(np.load(EXTERNAL/name/'pooled_c4.npy', mmap_mode='r'), model)
    phi = external_phi(x, model)
    prior = model['base'].decision_function(x).astype(np.float32).astype(float)
    mean = (x-model['residual_center'])/model['residual_scale']
    rows = np.arange(len(frame))
    cond = descriptors(name, rows, ids)
    sy = [frame.loc[a, 'class_id'].to_numpy(int) for a in saved['support_rows']]
    def route_external(score, expert_score, labels, ev):
        return route(score, expert_score, model['gate'], labels, ev, frame.target_manifest_row.to_numpy(int))
    infer(OUT/'targets'/name, phi, prior, mean, wave, source_wave[sr], cond,
          saved['support_rows'], saved['eval_masks'], sy, joint['expert_scores'],
          route_external, saved, rows, ids, sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['self24ports', 'oss2', 'f0921'],
                        choices=['self24ports', 'oss2', 'f0921'])
    parser.add_argument('--half-mean-control', action='store_true',
                        help='Isolate reduced mean-kernel strength without wavelet evidence')
    args = parser.parse_args()
    if args.half_mean_control:
        OUT = OUT/'half_mean_control'
        MODES = ('c4_baseline', 'mean_half')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'PROTOCOL.json').write_text(json.dumps(dict(
        modes=MODES, eta=ETA, source='Previously source-selected common eta=.25; no new strength search',
        pilot=True, source_only_wavelet_scaling=True, wavelet_split='LL64 vs axial+diagonal128, two Haar levels',
        half_mean_control=args.half_mean_control,
        controls='All nonzero branches share unit diagonal; mean and wavelet halves weighted equally',
        features='Mean-only vs mean+low vs mean+high vs mean+all; same condition RBF bandwidth .5',
        labels_per_draw=64, draws=5, expert_handoff=.05,
        not_reproduction='No trainable WSA/WRA, no Gabor CNN, no DDF, no detector retraining',
        falsification_scope='Only fixed image-level wavelet statistics probe; failure does not refute learned wavelet features',
        independent_confirmation=False, scope=['self24ports', 'oss2', 'f0921']), indent=2), encoding='utf-8')
    for dataset in args.datasets:
        if dataset == 'self24ports':
            ports()
        else:
            external(dataset)
