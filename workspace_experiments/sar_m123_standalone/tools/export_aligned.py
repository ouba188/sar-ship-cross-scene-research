"""Export the accepted six configurations, source banks and warm-start assets.

Historical paths are confined to this offline tool. No query cache, raw chip,
or dataset is copied into inference assets. Training outputs can be installed
with --training-root and --source after exporting the frozen source frontend.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

SEEDS = (17, 29, 43)
FEATURES = ('x', 'mean', 'mass', 'joint', 'e_image', 'coverage')
CLASSES = ('bulk', 'general', 'chemical', 'container')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def pack(folder, metric, metric_scaler):
    vv, vh, gate = (load(folder/name) for name in ('vv.pt', 'vh_centered.pt', 'frequency.pt'))
    return dict(metric_state=metric['state'], metric_scaler=metric_scaler,
                vv_state=vv['state'], vh_state=vh['state'],
                readout_scaler=load(folder/'M3_readout_scaler.pt'),
                gate_state=gate['state'], gate_scaler=gate['scaler'], alpha_anchor=float(gate['alpha']))


def install_training(assets, training_root, seeds=SEEDS):
    """Install main runner's seed*/ outputs into an existing source asset tree."""
    assets, training_root = Path(assets), Path(training_root)
    manifest = json.loads((assets/'manifest.json').read_text(encoding='utf-8'))
    manifest['complete'] = False
    write(assets/'manifest.json', manifest)
    for seed in seeds:
        folder = training_root/f'seed{seed}'
        for name in ('raw_scaler.npz', 'summary_scaler.npz'):
            incoming = folder/name if (folder/name).exists() else training_root/name
            if seed != seeds[0]:
                for key, value in npz(incoming).items():
                    np.testing.assert_array_equal(value, npz(assets/'readout'/name)[key])
            copy(incoming, assets/'readout'/name)
        copy(folder/'raw_summary.pt', assets/f'readout/raw_summary_seed{seed}.pt')
        metric = load(folder/'aligned_metric.pt')
        metric_scaler = load(assets/f'warm/seed{seed}/metric_scaler.pt')
        torch.save(pack(folder, metric, metric_scaler), assets/f'm3/seed_{seed}.pt')
        gate = load(folder/'frequency.pt')
        manifest['selections'][str(seed)] = dict(metric_step=metric['selected_step'],
            vv_step=load(folder/'vv.pt')['selected_step'],
            vh_step=load(folder/'vh_centered.pt')['selected_step'],
            frequency_step=gate['selected_step'], alpha=gate['alpha'])
    manifest.update(seeds=list(seeds), complete=True, weights_origin='standalone training output')
    write(assets/'manifest.json', manifest)
    return manifest


def export(legacy_root, assets_root, sources=('oss1', 'oss2')):
    root, output = Path(legacy_root), Path(assets_root)
    output.mkdir(parents=True, exist_ok=True)
    (output/'.gitignore').write_text('*\n!.gitignore\n', encoding='ascii')
    receipts = []
    for source in sources:
        target = 'oss2' if source == 'oss1' else 'oss1'
        direction = f'{source}-to-{target}'
        protocol = json.loads((root/'public_m123_cross_dataset_20261026'/direction/'protocol.json').read_text(encoding='utf-8'))
        source_root = root/f'{source}_m123_replication_20261026'
        mroot = Path(protocol['source_m3'])
        frontend = Path(protocol['source_frontend'])
        table = pd.read_csv(source_root/'stack_table.csv', dtype={'mmsi': str})
        refs = npz(mroot/'references.npz')
        rows = refs['bank'].reshape(-1)
        selected = table.iloc[rows].copy()
        if refs['bank'].shape != (4, 8) or len(np.unique(rows)) != 32:
            raise ValueError('Expected 32 unique class-major source bank rows.')
        if not selected.role.eq('fit_train').all() or selected.mmsi.nunique() != 32:
            raise ValueError('Bank must contain 32 distinct fit_train ships.')
        np.testing.assert_array_equal(selected.class_id, np.repeat(np.arange(4), 8))
        destination = output/source
        destination.mkdir(parents=True, exist_ok=True)
        manifest = dict(version=1, variant='raw_summary + adaptedPairMetric + paired readoutrefit',
                        source=source, endpoint='frequency', class_names=list(CLASSES), seeds=list(SEEDS),
                        bank_size=32, references_per_class=2, raw_dimension=1538, aux_dimension=104,
                        query_records_exported=0, complete=False, selections={})
        write(destination/'manifest.json', manifest)
        copy(Path(protocol['weights']), destination/'backbone/resnet50.pth')
        copy(frontend/'models/reader_layer1.npz', destination/'upstream/models/reader_layer1.npz')
        copy(frontend/'scales.json', destination/'upstream/scales.json')
        (destination/'m3').mkdir(exist_ok=True)
        projection = npz(mroot/'projection.npz')
        np.savez_compressed(destination/'m3/projection.npz', **{k: projection[k] for k in ('mean', 'components')})
        local = npz(mroot/'local.npz')
        np.savez_compressed(destination/'m3/bank.npz', **{k: local[k][rows] for k in FEATURES},
                            source_index=rows, obs_id=selected.obs_id.to_numpy(),
                            mmsi=selected.mmsi.to_numpy(str), pass_group=selected.pass_group.to_numpy(str),
                            class_id=selected.class_id.to_numpy())
        np.savez_compressed(destination/'m3/covariance_scaler.npz', **{k: refs[k] for k in ('mu', 'scale', 'active')})
        selected = selected[['obs_id', 'mmsi', 'pass_group', 'role', 'class_id']]
        selected.insert(0, 'source_index', rows)
        selected.insert(0, 'bank_index', np.arange(32))
        selected.to_csv(destination/'m3/reference_bank.csv', index=False)
        readout = root/'raw_region_bypass_20261026'/source
        for name in ('raw_scaler.npz', 'summary_scaler.npz'):
            copy(readout/name, destination/'readout'/name)
        for seed in SEEDS:
            copy(readout/f'raw_summary_seed{seed}.pt', destination/f'readout/raw_summary_seed{seed}.pt')
            warm = destination/f'warm/seed{seed}'
            warm.mkdir(parents=True, exist_ok=True)
            old = root/'raw_region_bypass_20261026/downstream/raw_summary'/direction/f'seed{seed}'
            for name in ('M3_readout_scaler.pt', 'vv.pt', 'vh_centered.pt', 'frequency.pt', 'selection.json'):
                copy(old/name, warm/name)
            copy(mroot/f'seed{seed}/both_metric.pt', warm/'both_metric.pt')
            metric_scaler = load(mroot/f'seed{seed}/scalers.pt')['metric_scaler']
            torch.save(metric_scaler, warm/'metric_scaler.pt')
            final = root/'m3_aligned_training_20261026'/direction/f'seed{seed}'
            torch.save(pack(final, load(final/'aligned_metric.pt'), metric_scaler), destination/f'm3/seed_{seed}.pt')
            manifest['selections'][str(seed)] = json.loads((final/'selection.json').read_text(encoding='utf-8'))
        manifest['complete'] = True
        write(destination/'manifest.json', manifest)
        receipts.append(dict(source=source, assets=str(destination.resolve()), seeds=list(SEEDS), bank=32))
    write(output/'export.json', receipts)
    return receipts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path)
    parser.add_argument('--assets-root', type=Path, default=Path(__file__).resolve().parents[1]/'assets_aligned')
    parser.add_argument('--source', choices=('oss1', 'oss2'))
    parser.add_argument('--training-root', type=Path)
    args = parser.parse_args()
    if args.training_root:
        if not args.source:
            parser.error('--source is required for installing training output')
        result = install_training(args.assets_root/args.source, args.training_root)
    else:
        if not args.legacy_root:
            parser.error('--legacy-root is required for historical export')
        result = export(args.legacy_root, args.assets_root, (args.source,) if args.source else ('oss1', 'oss2'))
    print(json.dumps(result, indent=2))
