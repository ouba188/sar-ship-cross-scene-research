"""Post-hoc frozen-feature diagnostics, fixed seed 17, train/validation only.

Run AFTER the planned comparison completes. This is not another experimental
arm and never selects checkpoints or tunes probe hyperparameters on validation.
The training discriminator is not called. No test images/predictions are read.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
CODE = Path('E:/临时会话/port_prototype_model_20261121')
BRANCHES = ('baseline_P', 'disentangle_Fc', 'disentangle_Fs')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def scalar_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(count=int(values.size), mean=float(values.mean()),
                minimum=float(values.min()), median=float(np.median(values)),
                maximum=float(values.max()))


def feature_variance(values):
    # Natural, unstandardized GAP features: StandardScaler must not hide collapse.
    variance = values.astype(np.float64).var(axis=0, ddof=0)
    return dict(dimensions=int(len(variance)), population_variance=scalar_summary(variance),
                exactly_zero_variance_dimensions=int(np.count_nonzero(variance == 0)),
                variance_le_1e_minus_12_dimensions=int(np.count_nonzero(variance <= 1e-12)),
                descriptive_near_zero_threshold=1e-12)


@torch.inference_mode()
def extract(model, samples, ordered_ids, rows, device, batch_size, batches, collate):
    n = len(ordered_ids)
    locations = {i: position for position, i in enumerate(ordered_ids)}
    features = {name: np.zeros((n, 1536), np.float32) for name in ('Fc', 'Fs')}
    values = {name: np.zeros((n, 3, 2), np.float64) for name in
              ('gate_nonzero_fraction', 'residual_gate_nonzero_fraction',
               'Fc_energy', 'Fs_energy', 'P_energy', 'Fs_energy_fraction',
               'supported_channel_elements', 'valid_weight_sum')}
    for ids in batches(samples, ordered_ids, batch_size, shuffle=False):
        if any(rows[i]['development_split'] not in ('train', 'val', 'validation') for i in ids):
            raise RuntimeError('Diagnostic extraction is restricted to train and validation')
        batch = collate([samples[i] for i in ids], device)
        output = model(batch['images'], batch['valid'], hard_gate=True)
        positions = [locations[i] for i in ids]
        for level in range(3):
            for pol in range(2):
                fc, fs = output['content'][level][:, pol], output['residual'][level][:, pol]
                gate, weight = output['gates'][level][:, pol], output['valid'][level][:, pol]
                support = (weight > 0).expand_as(gate)
                count = support.sum((1, 2, 3))
                if not bool((count > 0).all()):
                    raise ValueError('This locked dual-polarization diagnostic expects both views available')
                denominator = weight.sum((1, 2, 3))
                energy_denominator = denominator * fc.shape[1]
                start = (level*2+pol)*256
                for name, tensor in (('Fc', fc), ('Fs', fs)):
                    gap = (tensor*weight).sum((2, 3))/denominator[:, None]
                    if gap.shape[1] != 256 or not bool(torch.isfinite(gap).all()):
                        raise ValueError('Invalid diagnostic GAP representation')
                    features[name][positions, start:start+256] = gap.cpu().numpy()
                    values[name+'_energy'][positions, level, pol] = (
                        (tensor.square()*weight).sum((1, 2, 3))/energy_denominator).cpu().numpy()
                p_energy = ((fc+fs).square()*weight).sum((1, 2, 3))/energy_denominator
                fs_energy = (fs.square()*weight).sum((1, 2, 3))/energy_denominator
                values['P_energy'][positions, level, pol] = p_energy.cpu().numpy()
                values['Fs_energy_fraction'][positions, level, pol] = torch.where(
                    p_energy > 0, fs_energy/p_energy.clamp_min(1e-30), torch.zeros_like(p_energy)).cpu().numpy()
                values['gate_nonzero_fraction'][positions, level, pol] = (
                    ((gate != 0) & support).sum((1, 2, 3))/count).cpu().numpy()
                values['residual_gate_nonzero_fraction'][positions, level, pol] = (
                    (((1-gate) != 0) & support).sum((1, 2, 3))/count).cpu().numpy()
                values['supported_channel_elements'][positions, level, pol] = count.cpu().numpy()
                values['valid_weight_sum'][positions, level, pol] = denominator.cpu().numpy()
        del output, batch
    return features, values


def summarize_maps(features, values, indices):
    reports = []
    for level, stride in enumerate((4, 8, 16)):
        for pol, polarization in enumerate(('VV', 'VH')):
            start = (level*2+pol)*256
            reports.append(dict(scale=level+1, stride=stride, polarization=polarization,
                measures={name: scalar_summary(array[indices, level, pol]) for name, array in values.items()},
                Fc_cross_sample_GAP_variance=feature_variance(features['Fc'][indices, start:start+256]),
                Fs_cross_sample_GAP_variance=feature_variance(features['Fs'][indices, start:start+256])))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--code-root', type=Path, default=CODE)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if importlib.util.find_spec('sklearn') is None:
        raise RuntimeError('scikit-learn is not installed; stop and report rather than installing automatically')
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if args.batch_size <= 0 or args.threads <= 0:
        raise ValueError('batch-size and threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    np.random.seed(17)
    sys.path.insert(0, str(args.code_root.resolve()))
    from sar_port.two_stage.cli import (load_records, load_checkpoint, restore_encoder,
        prepare, normalize_samples, batches, classification_metrics, metadata)
    from sar_port.two_stage.data import InputConfig, collate
    from sar_port.two_stage.geometry import GeometryConfig

    root = args.root.resolve()
    os.environ['SAR_PORT_PREPARE_CACHE'] = str(root/'prepared_cache_local')
    # The manifest includes held-out metadata. Filter immediately; only these rows
    # can enter prepare, feature extraction, labels, or any fitted statistic.
    rows = [r for r in load_records(root/'experiment_manifest_local.json')
            if r.get('development_split') in ('train', 'val', 'validation')]
    train_ids = [i for i,r in enumerate(rows) if r['development_split']=='train']
    val_ids = [i for i,r in enumerate(rows) if r['development_split'] in ('val','validation')]
    if (len(train_ids), len(val_ids)) != (718, 227):
        raise ValueError('Expected the locked 718 train / 227 validation observations')
    ordered_ids = train_ids + val_ids
    split_positions = {'train': np.arange(len(train_ids)),
                       'validation': np.arange(len(train_ids),len(ordered_ids))}
    expected_metadata = metadata(rows)
    paths = {name: root/'runs'/'seed17'/f'{name}_stage1.pt' for name in ('baseline','disentangle')}
    checkpoint_digests = {name: digest(path) for name,path in paths.items()}
    private = root/'branch_diagnostics_features_local'
    private.mkdir(exist_ok=True)
    all_features, map_reports, snapshots = {}, {}, {}
    samples, common_input, common_geometry, common_labels = None, None, None, None
    for arm in ('baseline','disentangle'):
        saved = load_checkpoint(paths[arm])
        if saved['stage'] != 'stage1' or saved['protocol_config'].get('seed') != 17:
            raise ValueError('Only fixed seed-17 stage-one checkpoints are supported')
        gated = saved['encoder_config'].get('gate_enabled', True)
        if gated != (arm == 'disentangle'):
            raise ValueError('Baseline/disentangle checkpoint gate settings mismatch')
        kept_metadata = [r for r in saved['manifest_metadata']
                         if r.get('development_split') in ('train','val','validation')]
        if kept_metadata != expected_metadata or saved['source_ids'] != [rows[i]['id'] for i in train_ids]:
            raise ValueError('Checkpoint training/validation identities differ from the locked manifest')
        input_cfg, geometry_cfg, labels = saved['input_config'], saved['geometry_config'], saved['labels']
        if samples is None:
            common_input, common_geometry, common_labels = input_cfg, geometry_cfg, labels
            raw_input = dict(input_cfg, mean=(0.,0.), std=(1.,1.))
            samples = prepare(rows, root, InputConfig(**raw_input), GeometryConfig(**geometry_cfg), ordered_ids)
            # The saved normalization was fitted on training only. Reuse those
            # exact statistics; do not re-fit on validation or on both splits.
            normalize_samples(samples, torch.tensor(input_cfg['mean']), torch.tensor(input_cfg['std']))
        elif (input_cfg != common_input or geometry_cfg != common_geometry or labels != common_labels):
            raise ValueError('The two fixed arms must share normalization, geometry and class ordering')
        model = restore_encoder(saved, args.device).freeze_for_h()
        model.requires_grad_(False)
        features, statistics = extract(model, samples, ordered_ids, rows, args.device,
                                       args.batch_size, batches, collate)
        if arm == 'baseline':
            if np.count_nonzero(features['Fs']):
                raise ValueError('Gate-disabled baseline residual is expected to be identically zero')
            all_features['baseline_P'] = features['Fc']
        else:
            all_features['disentangle_Fc'], all_features['disentangle_Fs'] = features['Fc'], features['Fs']
        map_reports[arm] = {split:summarize_maps(features, statistics, positions)
                            for split,positions in split_positions.items()}
        snapshots[arm] = dict(checkpoint_sha256=checkpoint_digests[arm],
            selected_training_epoch=saved['history'][-1].get('epoch'),
            checkpoint_objective=saved.get('training_objective'),
            normalization=dict(mean=list(input_cfg['mean']),std=list(input_cfg['std'])))
        np.savez_compressed(private/f'{arm}_natural_features_local.npz',
            ids=np.asarray([rows[i]['id'] for i in ordered_ids], dtype=np.str_),
            split=np.asarray(['train']*len(train_ids)+['validation']*len(val_ids), dtype=np.str_),
            label=np.asarray([rows[i]['label'] for i in ordered_ids], dtype=np.str_),
            Fc=features['Fc'], Fs=features['Fs'], **statistics)
        del model, saved, features, statistics
        print(f'Extracted {arm}: 718 train + 227 validation natural views', flush=True)
    for arm,path in paths.items():
        if digest(path) != checkpoint_digests[arm]:
            raise RuntimeError('Checkpoint changed while diagnostics ran; discard these diagnostic features')

    label_map = {str(v):i for i,v in enumerate(common_labels)}
    y = np.asarray([label_map[str(rows[i]['label'])] for i in ordered_ids], dtype=np.int64)
    train, validation = split_positions['train'], split_positions['validation']
    probe_results = {}
    for branch in BRANCHES:
        x = all_features[branch]
        scaler = StandardScaler()
        train_x = scaler.fit_transform(x[train])
        val_x = scaler.transform(x[validation])
        probe = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                                   solver='lbfgs', random_state=17)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            probe.fit(train_x, y[train])
        predictions = {'train':probe.predict(train_x), 'validation':probe.predict(val_x)}
        probe_results[branch] = dict(
            metrics={split:classification_metrics(y[positions],predictions[split],len(common_labels))
                     for split,positions in split_positions.items()},
            convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in caught),
            warning_messages=[str(w.message) for w in caught],
            iterations=probe.n_iter_.tolist(),
            natural_feature_variance={split:feature_variance(x[positions]) for split,positions in split_positions.items()})
        np.savez_compressed(private/f'{branch}_linear_probe_local.npz',
            scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,scaler_variance=scaler.var_,
            coefficients=probe.coef_,intercept=probe.intercept_,classes=probe.classes_,
            train_predictions=predictions['train'],validation_predictions=predictions['validation'])
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Post-hoc diagnostics only; not a new comparison arm or checkpoint selection criterion',
        seed=17, counts=dict(train=718,validation=227,test_images_read=0,test_predictions_read=0),
        class_order=[str(v) for v in common_labels],
        class_counts={split:dict(Counter(map(int,y[positions]))) for split,positions in split_positions.items()},
        checkpoints=snapshots, frozen_encoder=True, hard_gate=True, natural_views_only=True,
        training_discriminator_called=False, representation_dimension=1536,
        feature_order='scale1_VV,scale1_VH,scale2_VV,scale2_VH,scale3_VV,scale3_VH; 256 channels each',
        pooling='Sum(feature * valid_fraction) / sum(valid_fraction); original fused P for gate-disabled baseline',
        probe_spec=dict(StandardScaler='Separate train-only fit per branch; identical recipe',
                        LogisticRegression=dict(C=1,max_iter=1000,class_weight='balanced',solver='lbfgs',random_state=17),
                        hyperparameter_selection=False, validation_used_for_fit=False),
        uniform_guess_reference=dict(expected_accuracy=1/len(common_labels),expected_balanced_accuracy=1/len(common_labels),
                                     interpretation='Descriptive reference, not a significance test or an Fs leakage threshold'),
        probes=probe_results, scale_polarization_diagnostics=map_reports,
        interpretation_notes=[
            'Independent held-out linear probes assess natural-view linearly decodable class information, not success against the jointly trained discriminator.',
            'Fs probe performance above uniform guessing is evidence to investigate residual class information; it alone does not establish an implementation bug or disprove useful regularization.',
            'Low probe accuracy cannot establish causal independence or absence of nonlinear/spatial class information.',
            'Gate occupancy is measured over valid channel-position elements; zero/near-zero energy and unstandardized cross-sample GAP variance help identify collapse.',
            'Cross-sample variance is of valid-weighted GAP channels, not registered map pixels; the 1e-12 count is descriptive, not an automatic scientific verdict.',
            'The original checkpoints were selected by their existing validation protocol, so this validation probe is a development diagnostic, not an untouched external test.',
            'No training discriminator score is used; this diagnostic alone cannot attribute a discrepancy specifically to fooling that discriminator.'
        ],
        provenance=dict(manifest_sha256=digest(root/'experiment_manifest_local.json'),
                        diagnostic_script_sha256=digest(Path(__file__)),
                        code_sha256={name:digest(args.code_root/'sar_port'/'two_stage'/name)
                                     for name in ('stage1.py','data.py','geometry.py','contracts.py','cli.py')},
                        sklearn_version=importlib.metadata.version('scikit-learn'),
                        numpy_version=np.__version__,torch_version=str(torch.__version__),device=args.device))
    encoded = json.dumps(result, ensure_ascii=False)
    if 'pub2_opensarship2_' in encoded:
        raise ValueError('Aggregate diagnostic output must not contain sample IDs')
    write_json(root/'aggregate_branch_diagnostics.json', result)
    print(json.dumps({k:v['metrics']['validation'] for k,v in probe_results.items()},indent=2),flush=True)


if __name__ == '__main__':
    main()
