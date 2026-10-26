"""One-time conversion of existing source caches to a portable training bundle.

This utility reads files, never imports historical experiment code. It does not
export target data or query labels to the inference assets.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def selected(path, table):
    with np.load(path, allow_pickle=False) as z:
        ids = pd.Index(z['obs_id']).get_indexer(table.obs_id)
        if (ids < 0).any():
            raise ValueError(f'Missing observations in {path}')
        return {k: z[k][ids] for k in z.files if len(z[k]) == len(z['obs_id'])}


def export(root, source, output):
    path = root/f'{source}_m123_replication_20261026'
    table = pd.read_csv(path/'stack_table.csv', dtype={'mmsi': str, 'pass_group': str})
    other = 'oss2' if source == 'oss1' else 'oss1'
    protocol = json.loads((root/f'public_m123_cross_dataset_20261026/{source}-to-{other}/protocol.json').read_text(encoding='utf-8'))
    # Resolve within the caller's root, even if the archived JSON has old paths.
    mroot = path/Path(protocol['source_m3']).name
    raw = selected(path/'frontend_cache/frontend_raw.npz', table)
    local = selected(path/'frontend_cache/m3_raw.npz', table)
    chunks = []
    for r in (np.arange(8), np.arange(8, 16), np.arange(16, 20)):
        mass = local['mass'][:, :, r]
        chunks.append((local['layer1'][:, :, r]*mass[..., None]).sum(2)/
                      np.maximum(mass.sum(2)[..., None], 1e-8))
    x = np.c_[np.stack(chunks, 2).reshape(len(table), -1), raw['valid']].astype(np.float32)
    summary = selected(path/'frontend/summaries.npz', table)
    fields = []
    for p in range(2):
        g, q = summary['geometry'][:, p], summary['quality'][:, p]
        fields.extend([np.log1p(g[:, 0]), np.log1p(g[:, 1]), g[:, 2], g[:, 3]])
        for r in range(3):
            fields.extend([np.log1p(q[:, r, 0]), q[:, r, 1], np.log1p(q[:, r, 2]), q[:, r, 3]])
    aux = np.c_[summary['summary1'].reshape(len(table), -1), np.stack(fields, -1)].astype(np.float32)
    with np.load(mroot/'local.npz', allow_pickle=False) as z:
        features = {k: z[k] for k in ('x', 'mean', 'mass', 'joint', 'e_image', 'coverage')}
    with np.load(mroot/'references.npz', allow_pickle=False) as z:
        features.update(refs=z['refs'], bank=z['bank'])
        features.update({'covariance_'+k:z[k] for k in ('mu', 'scale', 'active')})
    if any(len(features[k]) != len(table) for k in ('x', 'mean', 'mass', 'joint', 'refs')):
        raise ValueError('Source feature rows disagree with stack table.')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'features.npz', obs_id=table.obs_id.to_numpy(), raw=x, aux=aux, **features)
    table[['obs_id', 'mmsi', 'pass_group', 'role', 'class_id']].to_csv(output/'table.csv', index=False)
    (output/'provenance.json').write_text(json.dumps(dict(source=source,
        stage='portable frozen frontend features, source only', rows=len(table),
        class_order=['bulk', 'general', 'chemical', 'container']), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--source', choices=('oss1', 'oss2'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.legacy_root, args.source, args.output)
