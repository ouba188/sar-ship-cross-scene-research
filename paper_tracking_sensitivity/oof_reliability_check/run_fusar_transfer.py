"""Frozen source-to-FUSAR transfer using the existing eight-class heads."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from build_oof_predictions import load_manifest
from run_opensarship2_transfer import SOURCE, WS, ARMS, write_json, fit_source, freeze, evaluate

COHORT = WS/'artifacts/bdh_fusar_preparation_20261003/experiment_cohort.csv'
ENCODER_SCOPE = ('Frozen source paired R50 receives the same single FUSAR image in both input '
                 'channels; SSL R18 receives that same image as RGB. No second polarization '
                 'is invented. F0921 SSL pretraining is unchanged.')


def prepare_source(out):
    manifest = load_manifest(SOURCE/'manifest.csv')
    cohort = pd.read_csv(COHORT, dtype={'mmsi': str}, keep_default_na=False)
    eligible = cohort.loc[cohort.eligible]
    target_ids = eligible.mmsi.to_numpy(dtype=str)
    target_ids = np.unique(target_ids[target_ids != ''])
    excluded = np.isin(manifest['mmsi'], target_ids)
    rows = np.flatnonzero(~excluded)
    np.savez_compressed(out/'source_split.npz', source_rows=rows,
                        excluded_source_rows=np.flatnonzero(excluded))
    write_json(out/'source_split.json', dict(source_rows=len(rows), excluded_source_rows=int(excluded.sum()),
        source_ports=len(np.unique(manifest['ports'][rows])), target_identity_scope='ALL eligible five-category rows, before fine-label selection',
        eligible_target_rows=len(eligible), original_target_rows=len(cohort),
        preexisting_target_exclusions=cohort.loc[~cohort.eligible, 'exclusion_reason'].value_counts().to_dict(),
        scene_identity_available=False,
        note='Existing preparation excludes shared-main MMSIs on the target side. Source overlap is still explicitly checked; no invented scene IDs.'))
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    with np.load(WS/'b2_ssl_feats_38091.npz') as archive:
        extra = archive['F'].astype(np.float32)
    return manifest, rows, base, extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('source', 'finish', 'all'), default='all')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'summary.json').exists():
        raise ValueError('completed experiment must not be overwritten')
    write_json(args.out/'RUN_CONFIG.json', dict(date='2026-10-07', dataset='FUSARShip v1.0',
        cohort=str(COHORT), arms=list(ARMS), seeds=list(range(5)), labels_per_draw=64,
        support_sampling='uniform without replacement, no class quotas', base_pca=128, ssl_pca=64,
        source_ridge=1., residual_ridge=8., gate_lr_c=1., budget=.05, gate_threshold=0.,
        target_split='row-level, matching current main experiment', source_label_space=list(range(8)),
        label_rule='eligible rows with explicit compatible fine labels; no coarse-to-fine invention; reject category/subcategory conflicts',
        no_target_tuning=True, input_adapter='same single image copied to both base input channels; same image to SSL',
        encoder_scope=ENCODER_SCOPE, historical_development_dataset=True,
        no_quality_filter_added=True, original_results_untouched=True))
    with threadpool_limits(limits=4):
        if args.stage in ('source', 'all'):
            manifest, rows, base, extra = prepare_source(args.out)
            model = fit_source(args.out, manifest, rows, base, extra)
        else:
            with (args.out/'source_model.pkl').open('rb') as f:
                model = pickle.load(f)
        if args.stage in ('finish', 'all'):
            freeze(args.out, model)
            evaluate(args.out, target_name='FUSARShip fine-label-compatible subset; original eight-output model',
                     encoder_scope=ENCODER_SCOPE)


if __name__ == '__main__':
    main()
