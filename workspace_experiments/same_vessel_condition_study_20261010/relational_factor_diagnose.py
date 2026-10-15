"""Frozen diagnostics: compression, prior/kernel paths, and centered relation tests."""

from relational_factors import (
    BASE, OUT, CACHE, OLD, SEED, FactorNet, source_readout, design, encode,
    relation_pools, sample_triplets, angular_gap, residual_head, metrics, save_json,
)
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.decomposition import PCA

DEST = OUT/'diagnosis'


def relation_checks(arm, parts, train_ids, val_ids, obs):
    meta = obs.iloc[val_ids].reset_index(drop=True)
    pairs, pools = relation_pools(meta)
    within = {}
    ports = meta.port.to_numpy()
    for a, (p, n) in pools['scene'].items():
        pp, nn = p[ports[p] == ports[a]], n[ports[n] == ports[a]]
        if len(pp) and len(nn):
            within[a] = (pp, nn)
    scene = sample_triplets(within, np.random.default_rng(SEED+100), 512)
    sem = sample_triplets(pools['semantic'], np.random.default_rng(SEED+100), 512)
    records = []
    for name, values in parts.items():
        mean = values[train_ids].mean(0)
        for center in (False, True):
            v = F.normalize(torch.from_numpy(values[val_ids]-(mean if center else 0)), dim=1)
            distance = (1-(v[pairs[:, 0]]*v[pairs[:, 1]]).sum(1)).numpy()
            same_ship = pd.DataFrame(dict(ship=meta.iloc[pairs[:, 0]].mmsi.to_numpy(), distance=distance)).groupby('ship').distance.mean().mean()
            a, p, _ = sem.T
            same_type = float((1-(v[a]*v[p]).sum(1)).mean())
            gap = angular_gap(v, scene).numpy()
            records.append(dict(arm=arm, factor=name, source_centered=center,
                same_ship_distance=float(same_ship), same_type_other_ship_distance=same_type,
                ship_to_type_distance_ratio=float(same_ship/max(same_type, 1e-8)),
                within_port_scene_gap=float(gap.mean()), within_port_scene_accuracy=float(100*(gap > 0).mean()),
                scene_eligible_anchors=len(within), same_ship_pairs=len(pairs)))
    return records


def run():
    torch.set_num_threads(1)
    DEST.mkdir(exist_ok=True)
    if (DEST/'COMPLETE.json').exists():
        raise ValueError('Preserve completed diagnostics')
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(OUT/'source_split.npz') as z:
        source, train_ids, val_ids = [z[k] for k in ('source_head_obs_ids', 'train_obs_ids', 'diagnostic_obs_ids')]
        center, scale = z['center'], z['scale']
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[obs.manifest_row], dtype=np.float32)
    x = ((raw-center)/scale).astype(np.float32)
    features = {'raw': raw}
    pca = PCA(n_components=128, svd_solver='randomized', random_state=SEED)
    pca.fit(x[train_ids])
    features['pca128'] = pca.transform(x).astype(np.float32)
    np.savez_compressed(DEST/'pca.npz', components=pca.components_, mean=pca.mean_)
    centered = []
    for arm in ('task_only', 'relations'):
        model = FactorNet()
        model.load_state_dict(torch.load(OUT/(arm+'.pt'), map_location='cpu', weights_only=True))
        model.eval()
        parts = encode(model, x)
        features[arm] = parts[0]
        centered.extend(relation_checks(arm, dict(stable=parts[0], condition=parts[1]), train_ids, val_ids, obs))
    pd.DataFrame(centered).to_csv(DEST/'centered_relations.csv', index=False)
    heads = {k: source_readout(v[source], obs.iloc[source].reset_index(drop=True)) for k, v in features.items()}
    all_phi, priors = {}, {}
    with torch.no_grad():
        for k, v in features.items():
            all_phi[k], priors[k] = design(torch.from_numpy(v), heads[k])
    rows, replays = [], []
    for path in sorted(OLD.glob('*_split.npz')):
        tag = path.stem.removesuffix('_split')
        with np.load(path) as z:
            support, query, labels = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
        truth = obs.iloc[query].class_id.to_numpy()
        port, draw = obs.iloc[query].port.iloc[0], int(tag.rsplit('draw', 1)[1])
        for name in ('pca128', 'task_only', 'relations'):
            for pp, kk in (('raw', 'raw'), (name, 'raw'), ('raw', name), (name, name)):
                with torch.no_grad():
                    beta, _, _ = residual_head(all_phi[kk][support], priors[pp][support], torch.from_numpy(labels), balanced=True)
                    score = priors[pp][query]+all_phi[kk][query]@beta
                pred = score.argmax(1).numpy()
                rows.append(dict(port=port, draw=draw, representation=name,
                    prior='raw' if pp == 'raw' else 'new', kernel='raw' if kk == 'raw' else 'new', **metrics(pred, truth)))
                if pp == kk == name and name in ('task_only', 'relations'):
                    with np.load(OUT/f'{name}_{tag}.npz') as z:
                        old = z['stable_scores']
                    errors = int((old.argmax(1) != pred).sum())
                    replays.append(dict(arm=name, task=tag, disagreement=errors, max_score_difference=float(abs(old-score.numpy()).max())))
                    if errors:
                        raise ValueError('Learned stable diagonal failed replay')
                if pp == kk == 'raw' and name == 'pca128':
                    with np.load(BASE/'spectral_residual_20261010'/f'{tag}_models.npz') as z:
                        previous = z['head_only_scores']
                    replays.append(dict(arm='historical_raw_reference', task=tag,
                        disagreement=int((previous.argmax(1) != pred).sum()),
                        max_score_difference=float(abs(previous-score.numpy()).max())))
        print('PATH_DIAGNOSIS', tag, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DEST/'path_results.csv', index=False)
    summary = df.groupby(['representation', 'prior', 'kernel'])[['OA', 'BA']].mean().reset_index()
    summary.to_csv(DEST/'path_summary.csv', index=False)
    save_json(DEST/'COMPLETE.json', dict(new_neural_training=0, tasks=9, learned_diagonal_replays=18,
        replays=replays, pca_fit='Source representation train identities only,128 components, no labels',
        interpretation='Frozen functional path diagnostic; crossed cells are not selected deployed models; centered distances and within-port scenes test simple shortcut explanations, not causal identification'))
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(centered).to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
