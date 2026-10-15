"""Prepare matched source-only geometries for the final 2x2 integration pilot."""

import argparse
import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from projection_residual_expert import discarded_coordinates, fit_geometry_statistics
from run_opensarship2_transfer import SOURCE, write_json


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
PREVIOUS = ROOT/'neural_complement_pilot_20261007'
ALIGNED = ROOT/'aligned_residual_pilot_20261008'
PARENT = ROOT/'f0921_transfer_20261007'
TARGET = ROOT/'f0921_detector_recrop_20261007'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise ValueError('use a fresh output directory')
    previous_protocol = json.loads((ALIGNED/'protocol.json').read_text())
    write_json(out/'protocol.json', dict(date='2026-10-08', previous=str(PREVIOUS),
        target=str(TARGET), aligned_parent=str(ALIGNED), source_split=str(PARENT/'source_split.npz'),
        arms=['reference', 'complement_only', 'aligned_only', 'joint'],
        trained_arms=['aligned_only', 'joint'], fixed_training=previous_protocol['fixed_training'],
        source_scope='Existing identity-purged source cohorts; same20source support/query episodes',
        target_scope='CorrectedF0921, historical development data, same5draws,64labels,5%admissions',
        frozen='R50, sourcePCA/statistics/head parameters, base64scores and inherited gate',
        prior_scope='Frozen head parameters, NOT frozen expert score values when R18 features change',
        joint_map='sqrt(.5)*retained192 + sqrt(.5)*standardized_complement2560 + onebias',
        complement_geometry='Pretraining source-only PCA-complement mean/std per fold; never refit after training',
        objective='Class-balanced source queryCE through actual64support ridge8; no extra handoffCE',
        comparison='Matched2x2; same initialization, source episodes, optimizer, sixepochs and finalcheckpoint',
        no_target_checkpoint_selection=True, no_new_target_labels=True, independent_confirmation=False,
        core_original_untouched=True))
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    extra = np.load(PREVIOUS/'frozen_replay/source_extra.npy', mmap_mode='r')
    shutil.copyfile(ALIGNED/'source_model.pkl', out/'source_model.pkl')
    with (out/'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    with np.load(PARENT/'source_split.npz') as saved:
        source_rows = saved['source_rows']
    old_descriptors = json.loads((ALIGNED/'EPISODES_COMPLETE.json').read_text())['episodes']
    directory = out/'episodes'
    directory.mkdir()
    descriptors = []
    with threadpool_limits(limits=4):
        stats = fit_geometry_statistics(base, extra, source_rows, model)
        np.savez_compressed(out/'deployment_complement.npz', source_rows=source_rows, **stats)
        for index, descriptor in enumerate(old_descriptors):
            with np.load(descriptor['path']) as saved:
                arrays = {key: saved[key] for key in saved.files}
            with np.load(PARENT/'source_oof'/f'{descriptor["port"]}.npz') as saved:
                train, held = saved['train_rows'], saved['held_rows']
            np.testing.assert_array_equal(held, arrays['rows'])
            pca = PCA(n_components=128, random_state=0).fit(base[train])
            fitted_x = pca.transform(base[train])
            center, scale = fitted_x.mean(0), fitted_x.std(0)+1e-6
            replay_xs = ((pca.transform(base[held])-center)/scale).astype(np.float32)
            np.testing.assert_allclose(replay_xs, arrays['xs'], rtol=1e-5, atol=2e-5)
            fold_model = dict(pca=pca, pca_ssl=SimpleNamespace(
                mean_=arrays['ssl_mean'], components_=arrays['ssl_components']))
            fold_stats = fit_geometry_statistics(base, extra, train, fold_model)
            mean, std = fold_stats['discarded_center'], fold_stats['discarded_scale']
            bd = (discarded_coordinates(base[held], pca)-mean[:base.shape[1]])/std[:base.shape[1]]
            arrays.update(base_discarded=bd.astype(np.float32),
                discarded_center=mean.astype(np.float32), discarded_scale=std.astype(np.float32))
            path = directory/f'{index:02d}.npz'
            np.savez_compressed(path, **arrays)
            descriptors.append(dict(port=descriptor['port'], path=str(path),
                held=len(held), query=len(arrays['query']), source_fit_rows=len(train)))
            print(f'JOINT GEOMETRY {index+1}/{len(old_descriptors)} {descriptor["port"]}', flush=True)
    np.save(out/'target_feature_rows.npy', np.load(TARGET/'feature_rows.npy'))
    write_json(out/'EPISODES_COMPLETE.json', dict(episodes=descriptors,
        query_rows=sum(d['query'] for d in descriptors), source_only=True,
        frozen_original_extra=str(PREVIOUS/'frozen_replay/source_extra.npy')))
    print('JOINT PREPARATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
