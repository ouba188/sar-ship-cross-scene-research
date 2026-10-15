"""CPU-only paired diagnostics for completed task-trained R50 prediction caches.

No training, target tuning, pixel extraction or neural forward pass. See
strong_single_multidata_20261010/EXPLAIN_DESIGN_ZH.md for the data contract.
"""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
DEFAULT_OUT = ROOT / 'strong_single_multidata_20261010/explain'
ARMS = ('frozen_single_same_readout', 'task_adapted_single_same_readout')
ALIGN = ('rows', 'ids', 'support', 'eval_masks')
CLASSES = ('bulk_carrier', 'fishing_vessel', 'general_cargo',
           'product_chemical_tanker', 'container_ship', 'crude_oil_tanker',
           'tug_towing', 'offshore_supply')
STRATA = ('gap_lt_005', 'gap_005_020', 'gap_ge_020')
EVENTS = ('rescue', 'harm', 'both_correct', 'both_wrong')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def aligned(a, b, keys=ALIGN):
    for key in keys:
        np.testing.assert_array_equal(a[key], b[key], err_msg=f'Misaligned {key}')


def default_manifest(dataset):
    if dataset == 'self24ports':
        return Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/'
                    'eight_class_adaptive_20260916/manifest.csv'), 'instance_id'
    folder = {'oss2': 'opensarship2_transfer_20261007', 'f0921': 'f0921_transfer_20261007'}[dataset]
    return ROOT / folder / 'target_manifest.csv', 'sample_id'


def discover(roots, specs_path, seeds):
    """Explicit specs cover new layouts; standard layouts use COMPLETE metadata."""
    if specs_path:
        specs = read_json(specs_path)
    else:
        found = {}
        for root in roots:
            if not root.exists():
                raise FileNotFoundError(root)
            for seed_folder in sorted(root.rglob('seed_*')):
                if not seed_folder.is_dir() or not seed_folder.name[5:].isdigit():
                    continue
                folder = seed_folder.parent
                if folder in found:
                    continue
                receipt = seed_folder / 'COMPLETE.json'
                meta = read_json(receipt) if receipt.exists() else {}
                dataset = meta.get('dataset')
                if dataset is None:
                    dataset = next((s for s in ('self24ports', 'f0921', 'oss2') if s in folder.parts), None)
                if dataset is None:
                    raise ValueError(f'Cannot infer dataset for {folder}; use --task-specs')
                task = meta.get('task', folder.name if dataset == 'self24ports' else dataset)
                found[folder] = dict(dataset=dataset, task=task, directory=str(folder))
        specs = list(found.values())
    ready, inventory, seen = [], [], set()
    for spec in specs:
        dataset, task = spec['dataset'], spec['task']
        if (dataset, task) in seen:
            raise ValueError(f'Duplicate task {dataset}/{task}; specify one run per task')
        seen.add((dataset, task))
        folder = Path(spec['directory'])
        missing = [seed for seed in seeds if not all((folder / f'seed_{seed}' / name).exists()
                   for name in ('COMPLETE.json', 'predictions.npz'))]
        inventory.append(dict(**spec, missing_seeds=missing, status='pending' if missing else 'ready'))
        if not missing:
            ready.append(spec)
    return ready, inventory


def truth_for(spec, data):
    default, idcol = default_manifest(spec['dataset'])
    manifest = Path(spec.get('manifest', default))
    idcol = spec.get('id_column', idcol)
    columns = pd.read_csv(manifest, nrows=0).columns
    portcol = spec.get('port_column', next((c for c in ('port', 'port_id') if c in columns), None))
    usecols = [idcol, 'class_id'] + ([portcol] if portcol else [])
    frame = pd.read_csv(manifest, usecols=usecols, keep_default_na=False).iloc[data['rows']]
    np.testing.assert_array_equal(frame[idcol].to_numpy(str), data['ids'])
    y = frame.class_id.to_numpy(int)
    np.testing.assert_array_equal(y[data['support']], data['support_labels'])
    ports = frame[portcol].to_numpy(str) if portcol else np.full(len(y), spec['task'])
    ports = np.where(ports == '', 'unknown', ports)
    return y, ports


def load_single(folder):
    data = archive(folder / 'predictions.npz')
    indices = [list(data['arms']).index(arm) for arm in ARMS]
    data['predictions'] = data['predictions'][:, indices]
    n = len(data['rows'])
    if data['predictions'].shape != (len(data['support']), 2, n):
        raise ValueError(f'Unexpected prediction axes in {folder}')
    if data['eval_masks'].dtype != np.bool_ or data['eval_masks'].shape != (len(data['support']), n):
        raise ValueError(f'Expected boolean draw/query masks in {folder}')
    for support, mask in zip(data['support'], data['eval_masks']):
        if len(np.unique(support)) != len(support) or mask[support].any():
            raise ValueError(f'Support/query overlap or repeated support in {folder}')
    data['_arm_indices'] = indices
    return data


def single_scores(folder, data):
    """Prefer explicit logits/probabilities; otherwise replay the original CPU head."""
    if 'margins' in data:
        gaps = data['margins'][:, data['_arm_indices']]
        if gaps.shape != data['predictions'].shape or not np.isfinite(gaps).all() or (gaps < 0).any():
            raise ValueError(f'Invalid original-score margins in {folder}')
        return None, None, 'raw_score_gap', 'stored_margins_no_replay'
    for key in ('logits', 'probabilities'):
        if key in data:
            values = data[key][:, data['_arm_indices']]
            if values.shape != data['predictions'].shape + (8,) or not np.isfinite(values).all():
                raise ValueError(f'Invalid {key} in {folder}')
            np.testing.assert_array_equal(values.argmax(-1), data['predictions'])
            if key == 'probabilities':
                if values.min() < 0 or values.max() > 1:
                    raise ValueError('Probabilities outside [0,1]')
                np.testing.assert_allclose(values.sum(-1), 1, atol=1e-5)
            return values, None, key, 'stored_' + key
    import torch
    from torch.nn import functional as F
    from aligned_residual_ops import residual_scores
    torch.set_num_threads(2)
    state = torch.load(folder / 'final_tail.pt', map_location='cpu', weights_only=False)
    np.testing.assert_array_equal(state['source_rows'], data['source_rows'])
    weights = [state[name]['weight'].detach().clone() for name in ('frozen_prior', 'prior')]
    del state
    shape = data['predictions'].shape + (8,)
    scores = np.empty(shape, np.float32)
    priors = np.empty((2, shape[2], 8), np.float32)
    with torch.no_grad():
        for ai, arm in enumerate(ARMS):
            phi = torch.from_numpy(np.load(folder / f'{arm}_phi.npy', mmap_mode='c'))
            if phi.dtype != torch.float32 or weights[ai].dtype != torch.float32:
                raise ValueError('Original CPU replay requires float32 phi and source head')
            prior = F.linear(phi, weights[ai])
            priors[ai] = prior.numpy()
            for draw, support in enumerate(data['support']):
                labels = torch.as_tensor(data['support_labels'][draw], dtype=torch.long)
                value = residual_scores(phi[support], prior[support], labels, phi, prior, strength=8.)
                scores[draw, ai] = value.numpy()
            del phi, prior
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite replay scores')
    np.testing.assert_array_equal(scores.argmax(-1), data['predictions'],
                                  err_msg='Original diagonal replay failed; use the original PyTorch environment')
    return scores, priors, 'logits', 'float32_cpu_lambda8_exact_argmax'


def confidence(values, kind):
    # Preserve the original score gap; no temperature or probability calibration.
    top = np.sort(values, axis=-1)[..., -2:]
    return top[..., 1] - top[..., 0]


def metrics(y, predictions):
    return dict(oa=100 * float(np.mean(y == predictions)),
                ba=100 * float(np.mean([np.mean(predictions[y == c] == c) for c in np.unique(y)])))


def gap_bridge(y, dual, full, frozen, trained):
    scores = {name: metrics(y, pred) for name, pred in
              [('dual', dual), ('full_R50only', full), ('frozen_compact', frozen), ('trained_compact', trained)]}
    result = []
    for metric in ('oa', 'ba'):
        d, r, f, t = [scores[name][metric] for name in ('dual', 'full_R50only', 'frozen_compact', 'trained_compact')]
        parts = [d - r, r - f, f - t]
        np.testing.assert_allclose(sum(parts), d - t, atol=1e-10)
        result.append(dict(metric=metric, dual=d, current_full_R50only=r, frozen_compact=f,
            task_trained_compact=t, dual_minus_current_full_R50only=parts[0],
            current_full_R50only_minus_frozen_compact=parts[1],
            frozen_compact_minus_task_trained_compact=parts[2], total_gap=d-t,
            closure_error=sum(parts)-(d-t)))
    return result


def common_hard(trained, frozen, full, mlp, dual, y, masks, rows, ids, r18=None):
    """Fixed measured set, including every requested neural seed; no target selection."""
    singles = np.concatenate([trained, frozen, full[None], mlp[None]], axis=0)
    if r18 is not None:
        singles = np.concatenate([singles, r18[None]], axis=0)
    correct = singles == y[None, None]
    tables = []
    for draw, mask in enumerate(masks):
        count = correct[:, draw, mask].sum(0)
        dual_correct = dual[draw, mask] == y[mask]
        tables.append(pd.DataFrame(dict(draw=draw, row=rows[mask], sample_id=ids[mask], class_id=y[mask],
            tested_single_entries=len(singles), correct_single_entries=count,
            all_tested_singles_wrong=count == 0, dual_rescues_commonhard=(count == 0) & dual_correct,
            all_tested_singles_and_dual_wrong=(count == 0) & ~dual_correct,
            dual_wrong_any_single_correct=~dual_correct & (count > 0),
            dual_wrong_all_singles_correct=~dual_correct & (count == len(singles)))))
    return pd.concat(tables, ignore_index=True)


def events(y, candidate, reference):
    new, old = candidate == y, reference == y
    return dict(rescue=new & ~old, harm=~new & old, both_correct=new & old, both_wrong=~new & ~old)


def flow(y, candidate, reference):
    n = len(y)
    ev = events(y, candidate, reference)
    counts = {name: int(value.sum()) for name, value in ev.items()}
    old, new = reference == y, candidate == y
    recall_deltas = [100 * (new[y == c].mean() - old[y == c].mean()) for c in np.unique(y)]
    net = counts['rescue'] - counts['harm']
    return dict(n=n, **counts, net_count=net,
                candidate_oa=100 * new.mean() if n else np.nan,
                reference_oa=100 * old.mean() if n else np.nan,
                net_oa_pp=100 * net / n if n else np.nan,
                net_ba_pp=float(np.mean(recall_deltas)) if n else np.nan,
                rescue_pct_of_reference_wrong=100 * counts['rescue'] / (~old).sum() if (~old).any() else np.nan,
                harm_pct_of_reference_correct=100 * counts['harm'] / old.sum() if old.any() else np.nan)


def margin_parts(parts, original, y):
    competitor = original.copy()
    competitor[np.arange(len(y)), y] = -np.inf
    rival = competitor.argmax(-1)
    return parts[:, np.arange(len(y)), y] - parts[:, np.arange(len(y)), rival]


def summarize_parts(records, context, parts, original, y, masks, names):
    margins = margin_parts(parts, original, y)
    for event, mask in masks.items():
        for name, value in zip(names, margins):
            records.append(dict(**context, event=event, component=name, n=int(mask.sum()),
                                margin_sum=float(value[mask].sum()),
                                positive=int((value[mask] > 0).sum()), negative=int((value[mask] < 0).sum())))


def consensus(predictions, dual, frozen, y, masks, rows, ids):
    """Each row is one paired query/draw; neural votes never multiply query counts."""
    result = []
    for contrast, candidates, references in (
        ('dual_vs_trained', np.broadcast_to(dual, predictions.shape), predictions),
        ('trained_vs_frozen', predictions, frozen),
    ):
        ev = events(y[None, None, :], candidates, references)
        for draw, mask in enumerate(masks):
            frame = pd.DataFrame(dict(contrast=contrast, draw=draw, row=rows[mask], sample_id=ids[mask],
                                      class_id=y[mask], neural_seeds=len(predictions)))
            for name, value in ev.items():
                frame[name + '_votes'] = value[:, draw, mask].sum(0)
                frame['all_seed_' + name] = frame[name + '_votes'] == len(predictions)
            frame['any_seed_rescue'] = frame.rescue_votes > 0
            frame['any_seed_harm'] = frame.harm_votes > 0
            frame['mixed_rescue_harm'] = frame.any_seed_rescue & frame.any_seed_harm
            result.append(frame)
    return pd.concat(result, ignore_index=True)


def explain_task(spec, seeds, out, dual_root, reconstruction_root, r18_root=None):
    folder = Path(spec['directory'])
    context = dict(dataset=spec['dataset'], task=spec['task'])
    dst = out / 'tasks' / spec['dataset'] / spec['task']
    dst.mkdir(parents=True, exist_ok=True)
    anchor = load_single(folder / f'seed_{seeds[0]}')
    ref_path = Path(spec.get('reference', ROOT / 'backbone_ablation_20261010/targets' /
                            spec['dataset'] / spec['task'] / 'predictions.npz'))
    ref = archive(ref_path)
    aligned(anchor, ref)
    dual = ref['predictions'][:, list(ref['modes']).index(spec.get('reference_mode', 'dual_refit_late'))]
    full = ref['predictions'][:, list(ref['modes']).index('r50_late')]
    reconstruction_path = reconstruction_root / spec['dataset'] / spec['task'] / 'predictions.npz'
    reconstruction = archive(reconstruction_path)
    aligned(anchor, reconstruction)
    mlp = reconstruction['predictions'][:, list(reconstruction['modes']).index('mlp_late')]
    r18 = None
    if r18_root is not None:
        r18_saved = archive(r18_root / spec['dataset'] / spec['task'] / 'predictions.npz')
        aligned(anchor, r18_saved)
        r18 = r18_saved['predictions'][:, list(r18_saved['modes']).index('r18_task_trained')]
    y, ports = truth_for(spec, anchor)
    evidence_dir = dual_root / spec['dataset'] / spec['task']
    evidence = None
    if (evidence_dir / 'predictions.npz').exists() and (evidence_dir / 'scores.npy').exists() \
            and (evidence_dir / 'expert_delta_parts.npy').exists():
        historical = archive(evidence_dir / 'predictions.npz')
        aligned(anchor, historical)
        hmodes = list(historical['modes'])
        np.testing.assert_array_equal(historical['predictions'][:, hmodes.index('DD_late')], dual)
        evidence = (np.load(evidence_dir / 'scores.npy', mmap_mode='r'),
                    np.load(evidence_dir / 'expert_delta_parts.npy', mmap_mode='r'), hmodes)
    records, class_rows, port_rows, strata, parts_rows, receipts, bridges = [], [], [], [], [], [], []
    trained, frozen = [], []
    for seed in seeds:
        seed_dir = folder / f'seed_{seed}'
        data = load_single(seed_dir)
        aligned(anchor, data, ALIGN + ('support_labels', 'source_rows'))
        values, priors, kind, replay = single_scores(seed_dir, data)
        gaps = data['margins'][:, data['_arm_indices'][1]] if kind == 'raw_score_gap' else confidence(values[:, 1], kind)
        gap_units = 'probability_gap' if kind == 'probabilities' else 'original_logit_gap'
        trained.append(data['predictions'][:, 1])
        frozen.append(data['predictions'][:, 0])
        receipts.append(dict(seed=seed, score_source=replay, single_prior_parts_available=priors is not None))
        for draw, mask in enumerate(data['eval_masks']):
            yy = y[mask]
            f, t, d = frozen[-1][draw, mask], trained[-1][draw, mask], dual[draw, mask]
            common = dict(**context, seed=seed, draw=draw)
            bridges.extend(dict(**common, **row) for row in gap_bridge(yy, d, full[draw, mask], f, t))
            for contrast, candidate, reference in (('dual_vs_trained', d, t), ('trained_vs_frozen', t, f)):
                ctx = dict(**common, contrast=contrast)
                record = flow(yy, candidate, reference)
                records.append(dict(**ctx, **record))
                class_contributions = []
                for c in np.unique(yy):
                    ix = yy == c
                    row = flow(yy[ix], candidate[ix], reference[ix])
                    contribution = row['net_oa_pp'] / len(np.unique(yy))
                    class_contributions.append(contribution)
                    class_rows.append(dict(**ctx, class_id=int(c), class_name=CLASSES[c],
                        support_count=int((data['support_labels'][draw] == c).sum()),
                        ba_contribution_pp=contribution, **row))
                np.testing.assert_allclose(sum(class_contributions), record['net_ba_pp'], atol=1e-10)
                np.testing.assert_allclose(record['net_oa_pp'], record['candidate_oa'] - record['reference_oa'], atol=1e-10)
                for port in np.unique(ports[mask]):
                    ix = ports[mask] == port
                    port_rows.append(dict(**ctx, port=port, **flow(yy[ix], candidate[ix], reference[ix])))
                bins = np.searchsorted([.05, .20], gaps[draw, mask], side='right')
                for bi, name in enumerate(STRATA):
                    ix = bins == bi
                    strata.append(dict(**ctx, stratum=name, gap_units=gap_units, confidence_model='trained_single',
                                       **flow(yy[ix], candidate[ix], reference[ix])))
            ev = events(yy, d, t)
            if priors is not None:
                ff, tt = values[draw, 0, mask], values[draw, 1, mask]
                pdiff = priors[1, mask] - priors[0, mask]
                rdiff = (tt - priors[1, mask]) - (ff - priors[0, mask])
                parts = np.stack([pdiff, rdiff])
                np.testing.assert_allclose(parts.sum(0), tt - ff, atol=2e-6, rtol=1e-5)
                summarize_parts(parts_rows, dict(**common, decomposition='trained_minus_frozen_single',
                    event_contrast='dual_vs_trained', rival_origin='frozen_single_strongest_wrong'),
                    parts, ff, yy, ev, ('source_prior_delta', 'support_correction_delta'))
            if evidence is not None:
                scores, components, modes = evidence
                rr = np.asarray(scores[draw, modes.index('RR_expert')][mask])
                dd = np.asarray(scores[draw, modes.index('DD_expert')][mask])
                parts = np.asarray(components[draw])[:, mask, :]
                np.testing.assert_allclose(parts.sum(0), dd - rr, atol=1e-9, rtol=1e-9)
                summarize_parts(parts_rows, dict(**common, decomposition='historical_DD_minus_RR_expert',
                    event_contrast='dual_vs_trained', rival_origin='historical_RR_expert_strongest_wrong'),
                    parts, rr, yy, ev, ('prior_delta', 'old_relation_reweight', 'kernel_direct'))
        print(f"EXPLAIN {spec['dataset']}/{spec['task']} seed={seed} replay={replay}", flush=True)
    tables = dict(rescue_harm_by_seed_draw=pd.DataFrame(records), per_class=pd.DataFrame(class_rows),
                  per_port=pd.DataFrame(port_rows), confidence_strata=pd.DataFrame(strata),
                  functional_parts=pd.DataFrame(parts_rows), gap_bridge_by_seed_draw=pd.DataFrame(bridges))
    votes = consensus(np.stack(trained), dual, np.stack(frozen), y, anchor['eval_masks'], anchor['rows'], anchor['ids'])
    votes.to_csv(dst / 'seed_consistency_queries.csv.gz', index=False)
    flags = ['all_seed_' + e for e in EVENTS] + ['any_seed_rescue', 'any_seed_harm', 'mixed_rescue_harm']
    tables['seed_consistency_by_draw_class'] = votes.groupby(['contrast', 'draw', 'class_id']).agg(
        n=('row', 'size'), **{flag: (flag, 'sum') for flag in flags}).reset_index().assign(**context, neural_seeds=len(seeds))
    unique = votes.groupby(['contrast', 'row', 'sample_id', 'class_id']).agg(
        eligible_draws=('draw', 'size'), all_seed_all_draw_rescue=('all_seed_rescue', 'all'),
        all_seed_all_draw_harm=('all_seed_harm', 'all'),
        any_seed_any_draw_rescue=('any_seed_rescue', 'any'),
        any_seed_any_draw_harm=('any_seed_harm', 'any')).reset_index()
    unique['mixed_rescue_harm_across_draws'] = unique.any_seed_any_draw_rescue & unique.any_seed_any_draw_harm
    unique.to_csv(dst / 'seed_consistency_unique_queries.csv.gz', index=False)
    flags_unique = [c for c in unique if c.startswith(('all_seed_', 'any_seed_', 'mixed_'))]
    tables['unique_query_summary'] = unique.groupby(['contrast', 'class_id']).agg(
        unique_queries=('row', 'size'), **{flag: (flag, 'sum') for flag in flags_unique}).reset_index().assign(**context)
    hard = common_hard(np.stack(trained), np.stack(frozen), full, mlp, dual, y,
                       anchor['eval_masks'], anchor['rows'], anchor['ids'], r18=r18)
    hard.to_csv(dst / 'commonhard_queries.csv.gz', index=False)
    hard_flags = ['all_tested_singles_wrong', 'dual_rescues_commonhard', 'all_tested_singles_and_dual_wrong',
                  'dual_wrong_any_single_correct', 'dual_wrong_all_singles_correct']
    tables['commonhard_by_draw_class'] = hard.groupby(['draw', 'class_id']).agg(
        n=('row', 'size'), **{flag: (flag, 'sum') for flag in hard_flags}).reset_index().assign(**context)
    hard_unique = hard.groupby(['row', 'sample_id', 'class_id']).agg(
        eligible_draws=('draw', 'size'),
        **{f'any_draw_{flag}': (flag, 'any') for flag in hard_flags},
        **{f'all_draw_{flag}': (flag, 'all') for flag in hard_flags}).reset_index()
    hard_unique.to_csv(dst / 'commonhard_unique_queries.csv.gz', index=False)
    # Reuse saved functional interventions only; never launch the old diagnosis runner.
    cross_path = folder / 'diagnosis/interactions_by_seed_draw.csv'
    cross_status = 'unavailable'
    if cross_path.exists():
        cross = pd.read_csv(cross_path)
        cross = cross[cross.seed.isin(seeds)]
        for seed in seeds:
            path = folder / f'diagnosis/seed_{seed}_predictions.npz'
            saved = archive(path)
            aligned(anchor, saved, ALIGN + ('support_labels', 'source_rows'))
            for mode, actual in [('frozen_prior_frozen_kernel', frozen[seeds.index(seed)]),
                                 ('trained_prior_trained_kernel', trained[seeds.index(seed)])]:
                np.testing.assert_array_equal(saved['predictions'][:, list(saved['modes']).index(mode)], actual)
        tables['existing_prior_kernel_interactions'] = cross.assign(**context)
        cross_status = 'existing_predictions_diagonals_aligned'
    for name, table in tables.items():
        table.to_csv(dst / f'{name}.csv', index=False)
    write_json(dst / 'RECEIPT.json', dict(**context, seeds=seeds, reference=str(ref_path),
        reference_mode=spec.get('reference_mode', 'dual_refit_late'), draws=len(anchor['support']),
        n_rows=len(y), score_replay=receipts, historical_parts_available=evidence is not None,
        crossover_status=cross_status, alignment_exact=True, device='cpu', neural_training=False,
        python=sys.version, torch_version=str(sys.modules['torch'].__version__) if 'torch' in sys.modules else None,
        commonhard_single_set=[f'{arm}:seed_{seed}' for arm in ARMS for seed in seeds] + ['current_full_R50only:r50_late', 'reconstruction:mlp_late'] + (['current_task_R18:single_readout'] if r18 is not None else []),
        r18_root=str(r18_root) if r18_root is not None else None,
        reconstruction_path=str(reconstruction_path),
        contrast='rescue favors candidate: dual_vs_trained or trained_vs_frozen',
        historical_parts_scope='DD-RR expert logits, NOT dual-minus-trained-single attribution'))
    return tables


def aggregate(all_tables, out, inventory, seeds):
    tables = {name: pd.concat([t[name] for t in all_tables if name in t], ignore_index=True)
              for name in set().union(*(t.keys() for t in all_tables))}
    for name, frame in tables.items():
        frame.to_csv(out / f'{name}.csv', index=False)
    flows = tables['rescue_harm_by_seed_draw']
    measures = ['candidate_oa', 'reference_oa', 'net_oa_pp', 'net_ba_pp', 'rescue', 'harm']
    task_seed = flows.groupby(['dataset', 'task', 'seed', 'contrast'], as_index=False)[measures].mean()
    task_seed.to_csv(out / 'task_seed_means.csv', index=False)
    task_seed.groupby(['dataset', 'task', 'contrast'], as_index=False)[measures].mean().to_csv(out / 'task_summary.csv', index=False)
    dataset_seed = task_seed.groupby(['dataset', 'seed', 'contrast'], as_index=False)[measures].mean()
    dataset_seed.to_csv(out / 'dataset_seed_means.csv', index=False)
    summary = dataset_seed.groupby(['dataset', 'contrast']).agg(
        net_oa_mean=('net_oa_pp', 'mean'), net_oa_seed_sd=('net_oa_pp', 'std'),
        net_ba_mean=('net_ba_pp', 'mean'), net_ba_seed_sd=('net_ba_pp', 'std'),
        neural_seeds=('seed', 'nunique')).reset_index()
    summary.to_csv(out / 'dataset_summary.csv', index=False)
    bridge = tables['gap_bridge_by_seed_draw']
    bridge_columns = [c for c in bridge if c not in ('dataset', 'task', 'seed', 'draw', 'metric')]
    bridge_task = bridge.groupby(['dataset', 'task', 'seed', 'metric'], as_index=False)[bridge_columns].mean()
    bridge_dataset = bridge_task.groupby(['dataset', 'seed', 'metric'], as_index=False)[bridge_columns].mean()
    bridge_summary = bridge_dataset.groupby(['dataset', 'metric'], as_index=False)[bridge_columns].mean()
    bridge_summary.to_csv(out / 'gap_bridge_summary.csv', index=False)
    st = tables['confidence_strata'].groupby(['dataset', 'contrast', 'gap_units', 'stratum'], as_index=False)[
        ['n', 'rescue', 'harm', 'both_correct', 'both_wrong']].sum()
    st['net_oa_pp'] = 100 * (st.rescue-st.harm) / st.n.replace(0, np.nan)
    st['rescue_pct_of_reference_wrong'] = 100 * st.rescue / (st.rescue+st.both_wrong).replace(0, np.nan)
    st['harm_pct_of_reference_correct'] = 100 * st.harm / (st.harm+st.both_correct).replace(0, np.nan)
    st.to_csv(out / 'confidence_exposure_summary.csv', index=False)
    cc = tables['per_class'].groupby(['dataset', 'task', 'seed', 'contrast', 'class_id'], as_index=False)[
        ['ba_contribution_pp']].sum()
    draws = flows.groupby(['dataset', 'task', 'seed', 'contrast']).size().rename('draw_count')
    cc = cc.merge(draws, on=['dataset', 'task', 'seed', 'contrast'])
    cc.ba_contribution_pp /= cc.draw_count
    cc = cc.groupby(['dataset', 'contrast', 'class_id'], as_index=False).ba_contribution_pp.sum()
    denom = task_seed.groupby(['dataset', 'contrast']).size().rename('task_seed_count')
    cc = cc.merge(denom, on=['dataset', 'contrast'])
    cc.ba_contribution_pp /= cc.task_seed_count
    cc.to_csv(out / 'class_macro_ba_contributions.csv', index=False)
    for row in summary.itertuples():
        selected = cc[(cc.dataset == row.dataset) & (cc.contrast == row.contrast)]
        np.testing.assert_allclose(selected.ba_contribution_pp.sum(), row.net_ba_mean, atol=1e-10)
    counts = flows.groupby('dataset').task.nunique().to_dict()
    write_json(out / 'RUN_RECEIPT.json', dict(analyzed_tasks=counts, requested_seeds=seeds,
        discovered_tasks=inventory, expected_full_scope=dict(self24ports=23, f0921=1, oss2=1),
        neural_training=False, gpu_used=False, query_label_selection=False,
        fixed_dual_fits_per_task=1, count_unit='query exposures within draw; unique-query table separately deduplicated',
        averaging='draw mean -> equal task mean within neural seed -> neural-seed mean/sample SD',
        confidence='Original score top1-top2, fixed .05/.20 thresholds; probability input is separately labeled; not calibrated'))
    numeric_report(out, tables, summary, bridge_summary, st, counts, seeds)
    print(summary.to_string(index=False), flush=True)


def numeric_report(out, tables, summary, bridge, strata, counts, seeds):
    lines = ['# Explanation Results', '', f'Analyzed tasks only: {counts}; neural seeds: {seeds}.',
        'One fixed current dual reference per task. No GPU, training, target selection or new crossover.', '',
        '## Paired Gaps', '', '| Dataset | Candidate - reference | OA pp | BA pp | BA neural-seed SD |',
        '| --- | --- | ---: | ---: | ---: |']
    for r in summary.itertuples():
        lines.append(f'| {r.dataset} | {r.contrast} | {r.net_oa_mean:+.5f} | {r.net_ba_mean:+.5f} | {r.net_ba_seed_sd:.5f} |')
    lines += ['', '## Exact Configuration Bridge', '',
        'D-T = (D-R) + (R-F) + (F-T). D: current dual; R: current full R50-only;',
        'F: frozen compact; T: task-trained compact. These are configuration differences, NOT independent causal effects.',
        'D-R is the existing matched second-route increment; R-F includes readout/geometry and other configuration differences.',
        'F-T is the negative of the observed task-training change. Do not attribute the whole gap to backbone count.', '',
        '| Dataset | Metric | D-R | R-F | F-T | Total D-T |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in bridge.itertuples():
        lines.append(f'| {r.dataset} | {r.metric.upper()} | {r.dual_minus_current_full_R50only:+.5f} | '
            f'{r.current_full_R50only_minus_frozen_compact:+.5f} | {r.frozen_compact_minus_task_trained_compact:+.5f} | {r.total_gap:+.5f} |')
    lines += ['', '## Neural-Seed Stability', '',
        'Counts below are query/draw exposures counted once across seeds, not independent samples.',
        '| Dataset | Contrast | Query/draw exposures | All-seed rescue | All-seed harm |',
        '| --- | --- | ---: | ---: | ---: |']
    stable = tables['seed_consistency_by_draw_class'].groupby(['dataset', 'contrast'])[
        ['n', 'all_seed_rescue', 'all_seed_harm']].sum().reset_index()
    for r in stable.itertuples():
        lines.append(f'| {r.dataset} | {r.contrast} | {r.n} | {r.all_seed_rescue} | {r.all_seed_harm} |')
    lines += ['', 'Sample IDs, seed votes and deduplicated all-eligible-draw events are exported in tasks/*/*/*.csv.gz.',
        'Fixed-dual correctness prevents rescue and harm from co-occurring across seeds for the same query/draw;',
        'across different support draws, dual correctness can change. Unanimity across seeds does not imply support stability.', '',
        '## Fixed Tested Single Set', '',
        'Includes all requested trained/frozen compact seeds, current full R50-only and the existing MLP reconstruction.',
        'The current CLI additionally requires the task-trained R18-only control; each task receipt records the exact fixed set.',
        'Repeated frozen predictions do not provide extra independent evidence; common-hard is a logical intersection.',
        '| Dataset | Exposures | All tested singles wrong | Dual rescues those | All including dual wrong | Dual wrong, any single correct |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    hard = tables['commonhard_by_draw_class'].groupby('dataset')[[
        'n', 'all_tested_singles_wrong', 'dual_rescues_commonhard', 'all_tested_singles_and_dual_wrong',
        'dual_wrong_any_single_correct']].sum().reset_index()
    for r in hard.itertuples():
        lines.append(f'| {r.dataset} | {r.n} | {r.all_tested_singles_wrong} | {r.dual_rescues_commonhard} | '
                     f'{r.all_tested_singles_and_dual_wrong} | {r.dual_wrong_any_single_correct} |')
    lines += ['', 'These are common failures of the measured configurations; they do not prove impossibility for a single backbone.', '',
        '## Original-Score Gap Strata', '',
        'Thresholds fixed at .05 and .20. Raw logit gaps are NOT historical softmax probability gaps or calibrated confidence.',
        '| Dataset | Contrast | Units | Stratum | Exposures | Rescue | Harm | Net OA pp |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in strata.itertuples():
        lines.append(f'| {r.dataset} | {r.contrast} | {r.gap_units} | {r.stratum} | {r.n} | {r.rescue} | {r.harm} | {r.net_oa_pp:+.5f} |')
    lines += ['', 'Stratum exposure sums repeat queries across draws/seeds; dataset OA/BA instead use task macro means.', '',
        '## Boundaries And Next Intervention', '',
        '- Historical expert parts explain DD-RR logits, conditioned on present dual-versus-trained events. They do NOT sum to D-T.',
        '- Where original phi/prior replay is available, trained-minus-frozen prior and support-correction deltas are exact score accounting.',
        '  The support-correction delta includes both changed relations and changed coefficients; no isolated kernel cause is claimed.',
        '- Saved four-cell results are reused only after original diagonal alignment. Missing crossover/decomposition is reported as unavailable.',
        '- No physical specialization, architecture necessity, equalized pretraining or untouched confirmation is established. F had historical unlabeled SSL exposure.',
        '- Read RUN_RECEIPT.json for completed coverage; source-selected controls and target estimates remain separate. Do not generalize a dataset-specific path effect to every dataset.',
        '- If source-versus-target kernel reversal persists, one controlled follow-up is source-only relation-preservation regularization during the same single-backbone task training.',
        '  Compare against the existing objective with the same seeds/budget and source-only selection; examine prior, kernel and class harm, not just mean BA.',
        '- If the full-readout bridge term dominates, first compare a single R50 with that complete readout under matched source fitting; do not add a second backbone to explain this term.', '']
    (out / 'NUMERIC_FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')


def self_test():
    # Asymmetric errors catch reversed rescue/harm; missing class catches BA denominator errors.
    y = np.array([0, 0, 0, 2])
    a, b = np.array([0, 1, 0, 2]), np.array([1, 0, 1, 2])
    result = flow(y, a, b)
    assert (result['rescue'], result['harm']) == (2, 1)
    np.testing.assert_allclose(result['net_oa_pp'], 25.)
    np.testing.assert_allclose(result['net_ba_pp'], 100 / 6)
    assert np.isnan(flow(y[:0], a[:0], b[:0])['net_ba_pp'])
    bridge = gap_bridge(y, a, b, y, b)
    assert all(abs(row['closure_error']) < 1e-10 for row in bridge)
    assert np.searchsorted([.05, .20], [.049, .05, .199, .20], side='right').tolist() == [0, 1, 1, 2]
    t = np.array([[[1, 0]], [[1, 1]], [[1, 1]]])
    f = np.array([[[0, 1]], [[0, 1]], [[0, 1]]])
    v = consensus(t, np.array([[0, 1]]), f, np.array([0, 1]), np.array([[True, True]]),
                  np.array([7, 9]), np.array(['a', 'b']))
    d = v[v.contrast == 'dual_vs_trained']
    assert d.all_seed_rescue.tolist() == [True, False]
    assert d.rescue_votes.tolist() == [3, 1]
    assert not d.mixed_rescue_harm.any()  # Fixed dual correctness forbids opposite events for the same query.
    try:
        aligned({'ids': np.array(['a', 'b'])}, {'ids': np.array(['b', 'a'])}, ('ids',))
    except AssertionError:
        pass
    else:
        raise AssertionError('Alignment accepted a permuted cohort')
    # Two unequal-class tasks exercise the new margins-only schema and macro aggregation.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix='strong_single_explain_') as temporary:
        root = Path(temporary)
        out = root / 'out'
        out.mkdir()
        specs = []
        for task, labels in [('A', np.array([0, 2, 0, 2, 0, 2])), ('B', np.zeros(6, dtype=int))]:
            folder = root / 'targets/self24ports' / task
            ids = np.array([f'{task}{i}' for i in range(6)])
            support = np.array([[0, 1]])
            masks = np.array([[False, False, True, True, True, True]])
            meta = dict(rows=np.arange(6), ids=ids, support=support, eval_masks=masks)
            full, frozen, trained = labels.copy(), labels.copy(), labels.copy()
            full[2], frozen[3], trained[4] = 7, 7, 7
            reference = root / f'{task}_reference.npz'
            np.savez(reference, **meta, modes=['dual_refit_late', 'r50_late'],
                     predictions=np.stack([labels, full])[None])
            recon = root / 'reconstruction/self24ports' / task
            recon.mkdir(parents=True)
            np.savez(recon / 'predictions.npz', **meta, modes=['mlp_late'], predictions=trained[None, None])
            for seed in (11, 12, 13):
                dst = folder / f'seed_{seed}'
                dst.mkdir(parents=True)
                pred = np.stack([frozen, trained])[None]
                np.savez(dst / 'predictions.npz', **meta, source_rows=[10, 20], arms=ARMS, predictions=pred,
                         support_labels=labels[support], margins=np.full(pred.shape, .1, np.float32))
                write_json(dst / 'COMPLETE.json', dict(dataset='self24ports', task=task))
            manifest = root / f'{task}_manifest.csv'
            pd.DataFrame(dict(instance_id=ids, class_id=labels, port=task)).to_csv(manifest, index=False)
            specs.append(dict(dataset='self24ports', task=task, directory=str(folder),
                              manifest=str(manifest), reference=str(reference)))
        # A training-only folder is pending, not mistaken for a finished task.
        (root / 'targets/self24ports/Pending/seed_11').mkdir(parents=True)
        ready, inventory = discover([root / 'targets'], None, [11, 12, 13])
        assert len(ready) == 2 and len(inventory) == 3
        outputs = [explain_task(s, [11, 12, 13], out, root / 'no_evidence', root / 'reconstruction') for s in specs]
        aggregate(outputs, out, inventory, [11, 12, 13])
        summary = pd.read_csv(out / 'dataset_summary.csv')
        measured = summary[summary.contrast == 'dual_vs_trained'].iloc[0]
        np.testing.assert_allclose([measured.net_oa_mean, measured.net_ba_mean], [25., 25.])
        hard = pd.read_csv(out / 'commonhard_by_draw_class.csv')
        assert hard.dual_rescues_commonhard.sum() == 0
        receipt = read_json(out / 'tasks/self24ports/A/RECEIPT.json')
        assert all(r['score_source'] == 'stored_margins_no_replay' for r in receipt['score_replay'])
    print('SELF_TEST passed: event direction, absent classes, empty strata, thresholds, seed votes, '
          'row alignment, bridge, margins-only multi-task I/O, pending coverage, common-hard')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', nargs='+', type=Path, default=[ROOT / 'strong_single_multidata_20261010'])
    parser.add_argument('--task-specs', type=Path, help='JSON list: dataset, task, directory; optional manifest/id_column/port_column/reference/reference_mode')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seeds', nargs='+', type=int, default=[20261010, 20261011, 20261012])
    parser.add_argument('--dual-evidence-root', type=Path, default=ROOT / 'dual_evidence_20261010/targets')
    parser.add_argument('--reconstruction-root', type=Path, default=ROOT / 'dual_reconstruction_20261010/targets')
    parser.add_argument('--r18-root', type=Path, default=ROOT / 'strong_single_multidata_20261010/r18_control/targets')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate neural seeds')
    specs, inventory = discover(args.root, args.task_specs, args.seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'INPUT_INVENTORY.json', inventory)
    if not specs:
        raise SystemExit('No tasks have all requested completed seeds; see INPUT_INVENTORY.json')
    tables = [explain_task(spec, args.seeds, args.out, args.dual_evidence_root, args.reconstruction_root, args.r18_root) for spec in specs]
    aggregate(tables, args.out, inventory, args.seeds)


if __name__ == '__main__':
    main()
