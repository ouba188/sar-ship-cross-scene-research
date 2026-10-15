"""Freeze the pilot and source-only adapted-base episode supervision."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from build_oof_predictions import load_manifest
from residual_expert import source_geometry, fit_correction
from run_opensarship2_transfer import SOURCE, write_json

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
PARENT = ROOT/'f0921_transfer_20261007'
TARGET = ROOT/'f0921_detector_recrop_20261007'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'source_episode_scores.npz').exists():
        raise ValueError('completed source episodes must not be overwritten')
    write_json(out/'protocol.json', dict(start_date='2026-10-07', target=str(TARGET),
        source_split=str(PARENT/'source_split.npz'), arms=['frozen_replay', 'supervised', 'utility_tail'],
        initialization='e206 epoch40 shared; R18 prefix through layer3 frozen; train layer4 and random8class head',
        fixed_training=dict(epochs=6, batch_size=128, optimizer='AdamW', lr=.0001,
            weight_decay=.0001, seed=20261007, augmentation='none; cached deterministic frozen prefix',
            batchnorm='all running statistics frozen in both arms', checkpoint='final fixed epoch; no target selection'),
        control='global source-class-balanced cross entropy',
        candidate='same CE + source recovery/harm/neutral CE + CE on minibatch top5% predicted utility',
        coefficients='all three losses weight1; no search',
        training_budget_proxy='ceil5% of eligible minibatch, at least1; not a deployment-budget guarantee',
        utility_head='512features+8OOFbase+8detached expert logits ->64ReLU->3; auxiliary head discarded at deployment',
        source_episodes='one fixed64label support per sufficiently large source port; other rows supply OOF-base utility targets',
        downstream='same R50 base; refit extra PCA64/expert ridge1 and source head-OOF gate, target residual8 and64labels,5%cap',
        frozen_replay='same recached prefix/tail inference as trained arms to isolate training from cache rounding',
        inner_encoder_oof=False, encoder_scope='supervised layer4 sees source inner-held labels; target identity exclusions retained',
        target_encoder_exposure='inherited e206 F0921 unlabeled pretraining; no target labels train the backbone',
        classification_head='training-only; deployment retains existing PCA+ridge expert architecture',
        scope='minimal representation-learning pilot, not the full proposed attention architecture or proven novel method'))
    manifest = load_manifest(SOURCE/'manifest.csv')
    rows = np.load(PARENT/'source_split.npz')['source_rows']
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    teacher = np.zeros((len(base), 8), dtype=np.float32)
    eligible = np.zeros(len(base), dtype=bool)
    episodes = []
    with threadpool_limits(limits=4):
        for port in np.unique(manifest['ports'][rows]):
            with np.load(PARENT/'source_oof'/f'{port}.npz') as saved:
                train, held, prior = saved['train_rows'], saved['held_rows'], saved['base'].astype(float)
            teacher[held] = prior
            if len(held) <= 64:
                episodes.append(dict(port=port, held=len(held), support=[], query=0, status='too_few_for64plusquery'))
                continue
            pca = PCA(n_components=128, random_state=0).fit(base[train])
            xs = pca.transform(base[train])
            center, scale = xs.mean(0), xs.std(0)+1e-6
            standardized = (xs-center)/scale
            _, rc, rs = source_geometry(standardized, np.arange(len(train)))
            held_x = ((pca.transform(base[held])-center)/scale).astype(np.float32)
            phi = np.column_stack((((held_x-rc)/rs)/np.sqrt(128), np.ones(len(held))))
            support = np.random.default_rng(0).choice(len(held), 64, replace=False)
            correction = fit_correction(phi[support], manifest['labels'][held[support]], prior[support])
            teacher[held] = prior+phi@correction
            query = np.ones(len(held), dtype=bool)
            query[support] = False
            eligible[held[query]] = True
            episodes.append(dict(port=port, held=len(held), support=held[support].tolist(),
                                 query=int(query.sum()), status='source_only64_episode'))
            print(f'SOURCE EPISODE {port} query={query.sum()}', flush=True)
    np.savez_compressed(out/'source_episode_scores.npz', source_rows=rows, base_scores=teacher,
        eligible=eligible, labels=manifest['labels'], ports=manifest['ports'])
    write_json(out/'source_episodes.json', dict(episodes=episodes,
        eligible_query_rows=int(eligible.sum()), source_rows=len(rows),
        no_external_target_labels=True, existing_fold_train_rows_reused=True))


if __name__ == '__main__':
    main()
