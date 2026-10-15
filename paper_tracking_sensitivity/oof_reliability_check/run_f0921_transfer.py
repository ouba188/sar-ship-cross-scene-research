"""Frozen global-same64 transfer to the existing F0921 clean cohort."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from build_oof_predictions import load_manifest
from f0921_cohort import prepare_target, prepare_source_split
from run_opensarship2_transfer import (SOURCE, WS, ARMS, write_json, fit_source,
                                      target_geometry, freeze_draw, evaluate)

ENCODER_SCOPE = ('Existing frozen paired VV/VH R50 and VH SSL R18. The e206 SSL encoder '
                 'was pretrained on F0921; target-corpus encoder exposure is explicit. '
                 'PCA, source heads and source OOF gate exclude target vessel/scene identities.')


def project_target_batches(model, base, extra, batch_size=2048):
    n = len(base)
    prior_b, prior_e = np.empty((n, 8)), np.empty((n, 8))
    phi_b, phi_e = np.empty((n, 129)), np.empty((n, 193))
    # Batch projection avoids materializing the roughly gigabyte-sized raw matrix.
    for start in range(0, n, batch_size):
        stop = min(start+batch_size, n)
        b, e, pb, pe = target_geometry(model, base[start:stop], extra[start:stop].astype(np.float32))
        prior_b[start:stop], prior_e[start:stop] = b, e
        phi_b[start:stop], phi_e[start:stop] = pb, pe
    return prior_b, prior_e, phi_b, phi_e


def freeze(out, model):
    target = pd.read_csv(out/'target_manifest.csv', keep_default_na=False)
    if not np.array_equal(np.load(out/'feature_rows.npy'), target['row'].to_numpy()):
        raise ValueError('target feature row ordering differs from manifest')
    if not (out/'feature_recipe.json').exists():
        raise ValueError('target feature extraction is incomplete')
    base = np.load(out/'paired_r50.npy', mmap_mode='r')
    extra = np.load(out/'vh_ssl.npy', mmap_mode='r')
    n = len(target)
    prior_b, prior_e, phi_b, phi_e = project_target_batches(model, base, extra)
    del base, extra
    draws = []
    for seed in range(5):
        support = np.random.default_rng(seed).choice(n, 64, replace=False)
        draws.append(freeze_draw(prior_b, prior_e, phi_b, phi_e, model['gate'], support,
            target.iloc[support].class_id.to_numpy(dtype=int), target.target_manifest_row.to_numpy()))
        print(f'FROZEN DRAW {seed}', flush=True)
    del phi_b, phi_e
    np.savez_compressed(out/'policies.npz', **{k: np.stack([d[k] for d in draws]) for k in draws[0]},
        arm_names=np.asarray(ARMS), seeds=np.arange(5), sample_ids=target.sample_id.to_numpy(dtype=str),
        prior_base_scores=prior_b, prior_expert_scores=prior_e)
    write_json(out/'POLICIES_SAVED.json', dict(draws=5, arms=list(ARMS), target_rows=n,
        labels_per_draw=64, label_budget_scope='64 total across this target dataset, not 64 per port',
        split='row-level random support; same-vessel query rows may remain',
        independent_confirmation=False, no_target_query_model_selection=True))
    print('ALL FIVE POLICIES SAVED BEFORE QUERY READOUT', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'source', 'finish', 'all'), default='all')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists():
        raise ValueError('completed experiment must not be overwritten')
    write_json(out/'RUN_CONFIG.json', dict(date='2026-10-07', dataset='F_0921 existing clean compatible cohort',
        arms=list(ARMS), seeds=list(range(5)), labels_per_draw=64, label_budget_scope='global dataset, not per port',
        support_sampling='uniform without replacement, no class quotas', base_pca=128, ssl_pca=64,
        source_ridge=1., residual_ridge=8., gate_lr_c=1., budget=.05, gate_threshold=0.,
        target_split='row-level, matching preceding current-method transfer studies',
        label_rule='explicit fine-label mapping from class_raw, not historical broad class8',
        target_selection='existing paired clean row intersection; no new quality or performance filtering',
        no_target_tuning=True, historical_development_dataset=True, encoder_scope=ENCODER_SCOPE,
        original_results_untouched=True))
    with threadpool_limits(limits=4):
        if args.stage in ('prepare', 'all'):
            prepare_target(out)
        if args.stage == 'prepare':
            return
        if args.stage in ('source', 'all'):
            manifest = load_manifest(SOURCE/'manifest.csv')
            rows = prepare_source_split(out, manifest)
            base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
            with np.load(WS/'b2_ssl_feats_38091.npz') as archive:
                extra = archive['F'].astype(np.float32)
            model = fit_source(out, manifest, rows, base, extra)
            del base, extra
        else:
            with (out/'source_model.pkl').open('rb') as f:
                model = pickle.load(f)
        if args.stage in ('finish', 'all'):
            freeze(out, model)
            evaluate(out, target_name='F_0921 existing clean fine-label-compatible cohort; global64 labels',
                     encoder_scope=ENCODER_SCOPE)


if __name__ == '__main__':
    main()
