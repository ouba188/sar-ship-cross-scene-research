"""Source-label compression of PCA-complement information, fixed64/5% pilot."""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from projection_residual_expert import build_guided_kernel, fit_source_discriminant, freeze_kernel_draw
from run_projection_residual_pilot import DATASETS, ROOT, evaluate_dataset, load_target
from run_opensarship2_transfer import SOURCE, WS, write_json


ARMS = ('pca_reference', 'retained_half_control', 'projection_residual', 'source_discriminant')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out/name).exists() for name in DATASETS):
        raise ValueError('use a new output directory')
    previous = ROOT/'projection_residual_pilot_20261008'
    write_json(args.out/'PROTOCOL.json', dict(date='2026-10-08', datasets=DATASETS, arms=ARMS,
        role='Development follow-up motivated by first pilot, not an independent confirmation',
        source_projection='Balanced ridge1 of8class source residual targets from original source expert, on source-PCA complement',
        projection_labels='Each existing identity-purged source cohort only; no target query labels',
        dimension=8, geometry='Source-only coordinate scales and source-only projected mean/std',
        kernel='.5K_pca + .5K_source_discriminant +1', source_ridge=1., target_ridge=8.,
        labels_per_draw=64, draws=5, budget=.05,
        fixed='Exact cached base64, source prior, gate, support rows, evaluation rows and admission rules',
        controls='Exact policies from preceding pilot, no changes or retraining',
        no_parameter_grid=True, independent_confirmation=False, main_model_unchanged=True))
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    with np.load(WS/'b2_ssl_feats_38091.npz') as saved:
        extra = saved['F'].astype(np.float32)
    source_labels = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy(int)
    with threadpool_limits(limits=4):
        for name in DATASETS:
            started = time.monotonic()
            folder = args.out/name
            folder.mkdir()
            frame, target_base, target_extra, old, model, source_rows, source_folder = load_target(name)
            with np.load(previous/name/'source_geometry.npz') as saved:
                np.testing.assert_array_equal(source_rows, saved['source_rows'])
                stats = {key: saved[key] for key in saved.files if key != 'source_rows'}
            projection = fit_source_discriminant(base, extra, source_rows, source_labels, model, stats)
            np.savez_compressed(folder/'source_projection.npz', **projection, source_rows=source_rows)
            print(f'{name} SOURCE DISCRIMINANT FIT {time.monotonic()-started:.1f}s', flush=True)
            reference = np.unique(old['support_rows'])
            kernel = build_guided_kernel(target_base, target_extra, model, stats, projection, reference)
            draws = []
            for i, support in enumerate(old['support_rows']):
                result = freeze_kernel_draw(old['base_scores'][i], old['prior_expert_scores'],
                    kernel[:, np.searchsorted(reference, support)], support,
                    frame.iloc[support].class_id.to_numpy(int), model['gate'], frame.target_manifest_row.to_numpy())
                np.testing.assert_array_equal(result['eval_mask'], old['eval_masks'][i])
                np.testing.assert_array_equal(result['predictions'] != old['predictions'][i, 1], result['selected'])
                if result['selected'][~result['eval_mask']].any() or result['selected'].sum() > old['caps'][i]:
                    raise ValueError('support exclusion or budget mismatch')
                draws.append(result)
            np.savez_compressed(folder/'source_discriminant_policies.npz',
                **{key: np.stack([d[key] for d in draws]) for key in draws[0]},
                support_rows=old['support_rows'], sample_ids=old['sample_ids'])
            for arm in ARMS[:-1]:
                shutil.copyfile(previous/name/f'{arm}_policies.npz', folder/f'{arm}_policies.npz')
            write_json(folder/'POLICIES_SAVED.json', dict(dataset=name, source_rows=len(source_rows),
                target_rows=len(frame), source_model=str(source_folder/'source_model.pkl'),
                elapsed_seconds=time.monotonic()-started))
            print(f'{name} POLICIES FROZEN {time.monotonic()-started:.1f}s', flush=True)
            del kernel, draws, target_base, target_extra, old
        rows, summaries = [], []
        for name in DATASETS:
            measured, summary = evaluate_dataset(name, args.out, arms=ARMS)
            rows.extend(measured)
            summaries.extend(summary)
        write_json(args.out/'summary.json', summaries)
        pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]).to_csv(
            args.out/'metrics.csv', index=False)
    print('SOURCE DISCRIMINANT PILOT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
