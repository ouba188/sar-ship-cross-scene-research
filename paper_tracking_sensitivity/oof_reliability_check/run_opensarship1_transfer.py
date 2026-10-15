"""Same64 / 5% current-method transfer to native OSS1 GRD fine-label subset."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from build_oof_predictions import load_manifest
from opensarship1_data import prepare
from run_opensarship2_transfer import SOURCE, WS, write_json, fit_source, freeze, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('source', 'finish'), required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists():
        raise ValueError('completed experiment must not be overwritten')
    with threadpool_limits(limits=4):
        if args.stage == 'source':
            native = prepare(out)
            write_json(out/'RUN_CONFIG.json', dict(dataset='OpenSARShip1 GRDH compatible fine labels',
                global_target_labels=64, support_seeds=list(range(5)), handoff_cap=.05,
                ridge_source=1., ridge_residual=8., base_pca=128, ssl_pca=64,
                encoders='current paired R50 V2 and VH e206 SSL R18; newly extracted',
                historical_development_exposure=True, independent_confirmation=False,
                support_split='row-level, not vessel-disjoint', source_purge='all native GRD+SLC identities',
                no_target_hyperparameter_search=True, old_results_preserved=True))
            manifest = load_manifest(SOURCE/'manifest.csv')
            keep = np.ones(len(manifest['labels']), dtype=bool)
            overlaps = {}
            for key, field in [('mmsi', 'mmsi_or_ship_id'), ('product', 'scene_id')]:
                identities = np.unique(native[field].to_numpy(dtype=str))
                identities = identities[identities != '']
                hit = np.isin(manifest[key], identities)
                keep &= ~hit
                overlaps[key] = dict(source_rows=int(hit.sum()),
                    shared_ids=int(len(np.intersect1d(manifest[key][hit], identities))))
            rows = np.flatnonzero(keep)
            np.savez_compressed(out/'source_split.npz', source_rows=rows,
                                excluded_source_rows=np.flatnonzero(~keep))
            write_json(out/'source_split.json', dict(source_rows=len(rows),
                excluded_source_rows=int((~keep).sum()), overlaps=overlaps,
                policy='exclude all native target MMSI/product identities before source fitting'))
            base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
            with np.load(WS/'b2_ssl_feats_38091.npz') as archive:
                extra = archive['F'].astype(np.float32)
            fit_source(out, manifest, rows, base, extra)
        else:
            with (out/'source_model.pkl').open('rb') as handle:
                model = pickle.load(handle)
            freeze(out, model)
            evaluate(out, target_name='OpenSARShip1 native GRDH fine-label-compatible subset; global64 labels')


if __name__ == '__main__':
    main()
