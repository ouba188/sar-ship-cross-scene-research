"""One-time migration of frozen M3 assets; never invoked during inference.

legacy_root is model_closeout_20261023, assets_root is the standalone assets
directory. Only 64 strategy_fit source references are exported, no queries.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def scaler(values):
    mu, sd = values.mean(0), values.std(0)
    active = sd > 1e-6
    return dict(mu=mu, scale=np.where(active, sd, 1.), active=active)


def numpy_scaler(value):
    return {k: np.asarray(value[k]) for k in ('mu', 'scale', 'active')}


def export(legacy_root, assets_root):
    root = Path(legacy_root)
    local = root/'local_evidence_transport'
    redesign = root/'evidence_preserving_redesign'
    destination = Path(assets_root)/'m3'
    destination.mkdir(parents=True, exist_ok=True)
    projection = load_npz(local/'projection_layer1.npz')
    ids = projection['bank'].reshape(-1).astype(int)
    if len(ids) != 64 or len(np.unique(ids)) != 64:
        raise ValueError('Expected exactly 64 unique frozen references')
    table = pd.read_csv(local/'paired_scale_readout/table.csv')
    lookup = {int(v): i for i, v in enumerate(table.table_index)}
    indices = np.array([lookup[int(v)] for v in ids])
    selected = table.iloc[indices]
    if not selected.role.eq('strategy_fit').all():
        raise ValueError('Reference bank contains non-source records')
    np.testing.assert_array_equal(selected.class_id.to_numpy(), np.repeat(np.arange(8), 8))
    np.savez_compressed(destination/'projection.npz', mean=projection['mean'],
                        components=projection['components'])
    tok = load_npz(local/'tokens.npz')
    joint = load_npz(local/'spatial_frequency_joint_pooling/features.npz')
    freq = load_npz(local/'frequency_conditioned_matching/features.npz')
    np.testing.assert_array_equal(joint['ids'], table.table_index)
    np.testing.assert_array_equal(freq['ids'], table.table_index)
    projected = (tok['layer1'][ids]-projection['mean'])@projection['components'].T
    x = projected/np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-8)
    mass = tok['mass'][ids]
    x[mass <= 0] = 0
    np.savez_compressed(destination/'bank.npz', x=x.astype(np.float32), mass=mass,
                        mean=joint['mean'][indices], joint=joint['joint'][indices],
                        e_image=freq['e_image'][indices], coverage=freq['coverage'][indices],
                        source_index=ids)
    source_train = table.role.eq('utility_train').to_numpy()
    values = joint['joint'][source_train][joint['mass'][source_train] > 0]
    covariance_scaler = scaler(values)
    np.savez_compressed(destination/'covariance_scaler.npz', **covariance_scaler)
    receipts = []
    for seed in (17, 29, 43):
        def load(path):
            return torch.load(path, map_location='cpu', weights_only=False)
        metric = load(local/'pair_conditioned_joint_metric/models'/f'pair_joint_{seed}.pt')
        vv = load(redesign/'polar_readout_diagnosis_20261025/models'/f'vv_{seed}.pt')
        vh = load(redesign/'null_centered_readout_20261025/models'/f'center_vh_{seed}.pt')
        gate = load(redesign/'anchored_polar_adjustment_20261025/models'/f'frequency_{seed}.pt')
        with np.load(redesign/'polar_readout_diagnosis_20261025/results'/f'inputs_{seed}.npz') as z:
            readout = {k: z[k].copy() for k in ('mu', 'scale', 'active')}
        torch.save(dict(metric_state=metric['state'], metric_scaler=numpy_scaler(metric['scaler']),
                        vv_state=vv['state'], vh_state=vh['state'], readout_scaler=readout,
                        gate_state=gate['state'], gate_scaler=numpy_scaler(gate['scaler']),
                        alpha_anchor=float(gate['receipt']['alpha_anchor'])),
                   destination/f'seed_{seed}.pt')
        receipts.append(dict(seed=seed, alpha_anchor=float(gate['receipt']['alpha_anchor'])))
    manifest = dict(version=1, variant='pair_joint both + old VV + center VH + anchored frequency',
                    bank_size=64, bank_layout='class-major 8 classes x 8 source references',
                    references_per_class=2, projection_dimension=32, token_count=20,
                    gate_dimension=185, seeds=receipts, query_records_exported=0,
                    covariance_scaler_training_rows=int(source_train.sum()),
                    reference_source_indices=ids.tolist())
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('legacy_root')
    parser.add_argument('assets_root')
    args = parser.parse_args()
    print(json.dumps(export(args.legacy_root, args.assets_root), indent=2))
