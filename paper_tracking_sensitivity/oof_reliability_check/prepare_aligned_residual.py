"""Freeze source-port episodes and the expert geometry retained at deployment."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from build_oof_predictions import load_manifest
from residual_expert import fit_correction, source_geometry
from run_opensarship2_transfer import SOURCE, write_json

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
PREVIOUS = ROOT/'neural_complement_pilot_20261007'
PARENT = ROOT/'f0921_transfer_20261007'
TARGET = ROOT/'f0921_detector_recrop_20261007'


def geometry_arrays(model):
    return dict(ssl_mean=model['pca_ssl'].mean_, ssl_components=model['pca_ssl'].components_,
        expert_center=model['expert_center'], expert_scale=model['expert_scale'],
        expert_coef=model['expert'].coef_, expert_intercept=model['expert'].intercept_)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'EPISODES_COMPLETE.json').exists():
        raise ValueError('completed source preparation exists')
    write_json(out/'protocol.json', dict(start_date='2026-10-08', target=str(TARGET),
        previous=str(PREVIOUS), source_split=str(PARENT/'source_split.npz'),
        arms=['residual_global', 'residual_handoff'],
        initialization='original e206 layer4, same as frozen replay; no previous trained checkpoint',
        fixed_training=dict(epochs=6, query_batch_size=128, support_size=64,
            optimizer='AdamW', lr=.00001, weight_decay=.0001, gradient_clip=5., seed=20261008,
            batchnorm='running statistics frozen', augmentation='none', checkpoint='last epoch only'),
        global_objective='source-class-balanced CE on actual 64-support adapted ridge expert',
        handoff_objective='same CE plus CE on exact current gate admissions within each query batch',
        selection='same disagreement, support-observed class, positive fixed gate, floor5%, stable row tie',
        backprop='through expert features, support residual labels/prior and differentiable ridge8 solve',
        selection_gradient='discrete selected indices detached; no auxiliary utility/classification head',
        fixed_geometry='PCA, standardization, source ridge and source gate frozen BEFORE encoder training; retained at deployment',
        source_episodes='same previous 64-support source episodes; source port excluded from its prior head fit',
        deployment='retained global frozen_replay source model, updated encoder, existing64 residual and5%handoff; NO post-training refit',
        protocol_change='fixed source expert geometry/gate instead of post-training refit, for head alignment',
        budget_scope='training floor5% per query minibatch, deployment floor5% whole query; not identical population ranking',
        source_training_exposure='source episode labels train encoder; no claim of encoder-level OOF',
        target_scope='historical F0921, inherited unlabeled SSL exposure, row-level query, no new target labels',
        adoption='record positive primary-metric development gains with OA tradeoffs; no all-metric-positive requirement'))
    manifest = load_manifest(SOURCE/'manifest.csv')
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    extra = np.load(PREVIOUS/'frozen_replay/source_extra.npy', mmap_mode='r')
    previous_episodes = json.loads((PREVIOUS/'source_episodes.json').read_text())['episodes']
    episodes = []
    folder = out/'episodes'
    folder.mkdir(exist_ok=True)
    with (PREVIOUS/'frozen_replay/source_model.pkl').open('rb') as stream:
        global_model = pickle.load(stream)
    with (out/'source_model.pkl').open('wb') as stream:
        pickle.dump(global_model, stream)
    np.savez(out/'deployment_geometry.npz', **geometry_arrays(global_model))
    with threadpool_limits(limits=4):
        for index, episode in enumerate(previous_episodes):
            if episode['query'] == 0:
                continue
            port = episode['port']
            with np.load(PARENT/'source_oof'/f'{port}.npz') as saved:
                train, held = saved['train_rows'], saved['held_rows']
                base_prior = saved['base'].astype(float)
            pca = PCA(n_components=128, random_state=0).fit(base[train])
            x = pca.transform(base[train])
            center, scale = x.mean(0), x.std(0)+1e-6
            xs = (x-center)/scale
            held_xs = ((pca.transform(base[held])-center)/scale).astype(np.float32)
            pca_ssl = PCA(n_components=64, random_state=0).fit(extra[train])
            joint = np.column_stack((xs, pca_ssl.transform(extra[train])))
            _, ec, es = source_geometry(joint, np.arange(len(train)))
            expert = RidgeClassifier(alpha=1., class_weight='balanced').fit((joint-ec)/es, manifest['labels'][train])
            arrays = geometry_arrays(dict(pca_ssl=pca_ssl, expert_center=ec, expert_scale=es, expert=expert))
            _, rc, rs = source_geometry(xs, np.arange(len(train)))
            phi = np.column_stack(((held_xs-rc)/rs/np.sqrt(128), np.ones(len(held))))
            support = np.searchsorted(held, np.asarray(episode['support']))
            np.testing.assert_array_equal(held[support], episode['support'])
            support_labels = manifest['labels'][held[support]]
            adapted_base = base_prior+phi@fit_correction(phi[support], support_labels, base_prior[support])
            query = np.flatnonzero(~np.isin(np.arange(len(held)), support))
            path = folder/f'{index:02d}.npz'
            np.savez_compressed(path, rows=held, support=support, query=query, xs=held_xs,
                labels=manifest['labels'][held], adapted_base=adapted_base,
                **{k: v.astype(np.float32) for k, v in arrays.items()})
            episodes.append(dict(port=port, path=str(path), query=len(query), held=len(held)))
            print(f'ALIGNED EPISODE {port} support=64 query={len(query)}', flush=True)
    write_json(out/'EPISODES_COMPLETE.json', dict(episodes=episodes,
        query_rows=sum(e['query'] for e in episodes), source_only=True, retained_deployment_geometry=True))


if __name__ == '__main__':
    main()
