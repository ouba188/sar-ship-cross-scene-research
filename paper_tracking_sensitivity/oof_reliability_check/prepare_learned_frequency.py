"""Freeze baseline replay and source episodes before learned frequency fitting."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

from conditional_direction_residual import normalize_direction, rank_conditions
from run_conditional_direction_targets import (
    ROOT, SOURCE, PARENT, REFERENCE, DIAGNOSIS, C4, EXTERNAL, DATASETS,
    load_npz, load_pickle, base_phi, replace_r50, descriptors, base_coordinates, external_phi)

OUT = ROOT/'learned_frequency_residual_20261009'
PREVIOUS = ROOT/'conditional_direction_residual_20261009'
EPISODES = ROOT/'module_completion_20261008/ports_projection'
TRIAL = ROOT/'directional_fusion_trial_20261009'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def save(path, **values):
    # Float64 parent geometry is retained for exact baseline replay.
    np.savez_compressed(path, **values)


def prepare_port(port, selection, frame, averaged, condition_frame, u):
    directory = OUT/'tasks/self24ports'/port
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'PREPARED.json').exists():
        return
    eta = selection['ports'][port]['conditional']
    saved = load_npz(DIAGNOSIS/'port_policies'/port/'policies.npz')
    previous = load_npz(PREVIOUS/'mean_representation_control/targets/self24ports'/port/'policies.npz')
    rows = saved['query_rows']
    sr = np.load(PARENT/port/'source_rows.npy')
    g = load_pickle(PARENT/port/'geometry.pkl')
    head = load_pickle(REFERENCE/port/'direct_heads.pkl')['joint_base']
    bank = replace_r50(np.asarray(averaged[rows], float), np.load(PARENT/port/'target_bank.npy'), g)
    phi = base_phi(bank, g, u[rows])
    prior = np.c_[bank, u[rows]]@head['coef']+head['intercept']
    mean = normalize_direction((bank[:, :128]-g['base_center'])/g['base_scale'])
    conditions = rank_conditions(condition_frame.iloc[rows])
    supports = pd.Index(rows).get_indexer(saved['support_rows'].ravel()).reshape(5, 64)
    slot = list(previous['modes']).index('conditional_selected')
    old = load_npz(REFERENCE/port/'policies.npz')
    save(directory/'target.npz', rows=rows, ids=frame.instance_id.to_numpy(str)[rows], phi=phi,
         prior=prior, mean=mean, conditions=conditions, support=supports,
         support_labels=frame.class_id.to_numpy()[saved['support_rows']], eval_masks=saved['eval_masks'],
         expected_base=previous['base_predictions'][:, slot], expected_full=previous['predictions'][:, slot],
         expected_selected=previous['selected'][:, slot], source_rows=sr,
         expert=old['expert_scores'][:, list(old['mode_names']).index('direct_condition')], route_ids=rows)
    all_ports = sorted(frame.port.unique())
    items = json.loads((EPISODES/port/'EPISODES_COMPLETE.json').read_text())['episodes']
    epdir = directory/'episodes'
    epdir.mkdir(exist_ok=True)
    labels = frame.class_id.to_numpy()
    for i, item in enumerate(items):
        ep = load_npz(Path(item['path']))
        train, er = ep['train_rows'], ep['rows']
        if np.setdiff1d(er, sr).size or np.intersect1d(rows, er).size:
            raise ValueError('Adapter episode includes excluded outer target rows')
        key = '_'.join(f'{j:02d}' for j in sorted([all_ports.index(port), all_ports.index(item['port'])]))
        geo = load_npz(EPISODES/'episode_geometry_cache'/f'{key}.npz')
        pair = load_npz(ROOT/'evidence_aware_handoff_20261008/source_pair_cache'/f'{key}.npz')
        eh = load_npz(TRIAL/'nested_geometry_gpu/pair_fit'/f'{key}.npz')
        for cache in (geo, pair, eh):
            np.testing.assert_array_equal(train, cache['train_rows'])
        raw = np.asarray(averaged[er], float)
        pc = geo['base_components']
        xs = ((raw-geo['base_mean'])@pc.T-geo['base_center'])/geo['base_scale']
        z = (raw-pair['raw_center'])/pair['raw_scale']
        stable = (z@pair['q']-eh['stable_center'])/eh['stable_scale']
        zd = z@eh['qd']
        dxs = (eh['qd'].T*pair['raw_scale'][None, :])@pc.T/geo['base_scale']
        core = xs-.5*zd@dxs
        ep_phi = np.c_[(core-pair['base_residual_center'])/pair['base_residual_scale']/np.sqrt(128),
                       stable/np.sqrt(stable.shape[1]), u[er]/np.sqrt(2), np.ones(len(er))]
        ep_prior = np.c_[xs, stable, zd, u[er]]@eh['original_coef']+eh['original_intercept']
        save(epdir/f'{i:02d}.npz', rows=er, phi=ep_phi, prior=ep_prior,
             mean=normalize_direction((xs-pair['base_residual_center'])/pair['base_residual_scale']),
             conditions=rank_conditions(condition_frame.iloc[er]), support=ep['support'], query=ep['query'],
             labels=labels[er], held_port=item['port'])
    write_json(directory/'PREPARED.json', dict(dataset='self24ports', task=port, eta=eta,
        episodes=len(items), target_query_labels_saved=False, source_geometry='Existing double-port-held geometry',
        source_rows=len(sr), source_training_only=True))
    print('PREPARED '+port, flush=True)


def prepare_external(name, selection, frame, averaged, conditions_source):
    directory = OUT/'tasks'/name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'PREPARED.json').exists():
        return
    target_dir, source_dir, joint_dir = DATASETS[name]
    manifest = pd.read_csv(ROOT/target_dir/'target_manifest.csv', keep_default_na=False)
    provenance = json.loads((EXTERNAL/name/'FROZEN.json').read_text())
    model_path = Path(provenance.get('model_path', str(ROOT/source_dir/'source_model.pkl')))
    model = load_pickle(model_path)
    previous = load_npz(PREVIOUS/'mean_representation_control/targets'/name/'policies.npz')
    saved = load_npz(EXTERNAL/name/'support_refit_diagnostic/policies.npz')
    joint = load_npz(ROOT/joint_dir/'policies.npz')
    sr = load_npz(ROOT/source_dir/'source_split.npz')['source_rows']
    x = base_coordinates(np.load(EXTERNAL/name/'pooled_c4.npy', mmap_mode='r'), model)
    rows = np.arange(len(manifest))
    ids = manifest.sample_id.to_numpy(str)
    np.testing.assert_array_equal(ids, previous['sample_ids'])
    slot = list(previous['modes']).index('conditional_selected')
    save(directory/'target.npz', rows=rows, ids=ids, phi=external_phi(x, model),
         prior=model['base'].decision_function(x).astype(np.float32).astype(float),
         mean=normalize_direction((x-model['residual_center'])/model['residual_scale']),
         conditions=descriptors(name, rows, ids), support=saved['support_rows'],
         support_labels=manifest.class_id.to_numpy()[saved['support_rows']], eval_masks=saved['eval_masks'],
         expected_base=previous['base_predictions'][:, slot], expected_full=previous['predictions'][:, slot],
         expected_selected=previous['selected'][:, slot], source_rows=sr,
         expert=joint['expert_scores'], route_ids=manifest.target_manifest_row.to_numpy(int))
    x4 = base_coordinates(averaged, model)
    x0 = base_coordinates(np.load(SOURCE/'pooled_features.npy', mmap_mode='r'), model)
    phi = external_phi(x4, model)
    mean = normalize_direction((x4-model['residual_center'])/model['residual_scale'])
    epdir = directory/'episodes'
    epdir.mkdir(exist_ok=True)
    labels, domains = frame.class_id.to_numpy(), frame.port.to_numpy()
    rng = np.random.default_rng(20261009)
    count = 0
    for port in sorted(np.unique(domains[sr])):
        er, tr = sr[domains[sr] == port], sr[domains[sr] != port]
        if len(er) <= 64:
            continue
        support = rng.choice(len(er), 64, replace=False)
        query = np.setdiff1d(np.arange(len(er)), support)
        head = RidgeClassifier(alpha=1., class_weight='balanced').fit(x0[tr], labels[tr])
        np.testing.assert_array_equal(head.classes_, np.arange(8))
        save(epdir/f'{count:02d}.npz', rows=er, phi=phi[er], prior=head.decision_function(x4[er]),
             mean=mean[er], conditions=rank_conditions(conditions_source.iloc[er]),
             support=support, query=query, labels=labels[er], held_port=port)
        count += 1
    write_json(directory/'PREPARED.json', dict(dataset=name, task=name,
        eta=selection['external']['conditional'], episodes=count, model_path=str(model_path),
        source_rows=len(sr), target_query_labels_saved=False, source_training_only=True,
        source_geometry='Frozen dataset-purged source geometry; source-episode classifier leaves episode port out',
        limitation='Inner PCA uses all permitted source ports; episodes train adapter, not independent validation'))
    print('PREPARED '+name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['self24ports', 'oss2', 'f0921'])
    parser.add_argument('--ports', nargs='+')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(baseline='C4 mean-representation conditional support correction, original source-selected per-task eta',
        previous=str(PREVIOUS/'mean_representation_control'), modes=['baseline', 'unsplit', 'fixed', 'selective'],
        beta=.25, ridge=8., branch='Frozen R50 layer2 512x16x16; Haar bands; trainable1x1,depthwise3x3,GroupNorm,GELU',
        fusion='Separate unit-L2 embeddings weighted by sqrt(softmax) then block concatenated',
        conditions='Existing unlabelled elongation/center-excess cohort ranks plus learned band summaries',
        source_loss='Balanced query cross-entropy through same64-support joint ridge solve',
        epochs=6, max_query_per_episode=128, learning_rate=.003, seed=20261009,
        precision='bf16 convolution; float32 embedding normalization and source solve; float64 target solve',
        target='Only same64 support labels; adapter frozen; original expert/gate rerouted <=5%',
        baseline_preserved='Mean kernel unchanged; additional beta*frequency kernel; beta0 exact replay',
        readout='All four arms reported, no target-selected checkpoint', independent_confirmation=False,
        scope=['self24ports', 'oss2', 'f0921'], not_dfwa_reproduction=True,
        first_version='No cross-band attention or target-conditioned neural gate fitting; support conditioning via ridge and RBF')
    if not (OUT/'PROTOCOL.json').exists():
        write_json(OUT/'PROTOCOL.json', protocol)
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'port', 'class_id'])
    from run_conditioned_vessel_drift import SIDECAR
    from direct_incidence_conditioning import condition_features
    side = pd.read_csv(SIDECAR, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    u = condition_features(frame.incidence_deg.to_numpy())
    cf = pd.read_csv(DIAGNOSIS/'descriptors/self24ports.csv')
    np.testing.assert_array_equal(cf.instance_id, frame.instance_id)
    averaged = np.load(C4/'pooled_c4.npy', mmap_mode='r')
    selection = json.loads((PREVIOUS/'source_selection.json').read_text())
    if 'self24ports' in args.datasets:
        for port in args.ports or sorted(selection['ports']):
            prepare_port(port, selection, frame, averaged, cf, u)
    for name in args.datasets:
        if name != 'self24ports':
            prepare_external(name, selection, frame, averaged, cf)


if __name__ == '__main__':
    main()
