"""Matched 23-port M2 completion and source-only M3/joint continuation."""

import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from sar_joint_completion import SOURCE, WS, write_json


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
BASE = ROOT/'oof_base_adapt_results_20261007'
GATE = ROOT/'oof_context_results_20261007'
PARENT = ROOT/'oof_residual_expert_results_20261007_v2'
DEFAULT_OUT = ROOT/'module_completion_20261008/ports_projection'
ARMS = ('reference', 'complement_only')
MATCHED_ARMS = ('reference', 'complement_only', 'aligned_only', 'joint')


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def write_prepare_status(out, threads, state, current=''):
    ready = sorted(path.name for path in out.iterdir()
                   if path.is_dir() and (path/'EPISODES_COMPLETE.json').exists())
    pairs = len(list((out/'episode_geometry_cache').glob('*.npz')))
    text = '\n'.join(('# 24-Port CPU Prepare Status', '',
        f'- Updated: {datetime.now().isoformat(timespec="seconds")}',
        f'- State: {state}', f'- PID: {os.getpid()}', f'- CPU threads: {threads}',
        f'- Outer complete: {len(ready)}/23', f'- Cache pair files: {pairs}',
        f'- Current: {current or "none"}', '- GPU owner: root; this process is CPU-only',
        '', 'Ready outer ports: '+', '.join(ready), ''))
    (out/'PREPARE_STATUS.md').write_text(text, encoding='utf-8')


def extract_outer_targets(directory):
    """GPU stage: evaluate the original and both trained tails on outer rows only."""
    import torch
    from neural_complement import ComplementTail

    destination = directory/'matched_joint'
    destination.mkdir(exist_ok=True)
    if (destination/'FEATURES_COMPLETE.json').exists():
        return
    protocol = json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    source = np.load(protocol['source_prefix_path'], mmap_mode='r')
    with np.load(protocol['source_rows_path']) as saved:
        rows = saved['target_rows']
    old = archive(BASE/directory.name/'policies.npz')
    np.testing.assert_array_equal(rows, old['query_rows'])
    checkpoint = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    model = ComplementTail(checkpoint['backbone'])
    initial = {key: value.clone() for key, value in model.layer4.state_dict().items()}
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.cuda().eval()
    for arm in ('reference', 'aligned_only', 'joint'):
        state = initial if arm == 'reference' else torch.load(
            directory/arm/'layer4.pt', map_location='cpu', weights_only=True)
        model.layer4.load_state_dict(state)
        values = np.empty((len(rows), 512), dtype=np.float32)
        with torch.inference_mode():
            for lo in range(0, len(rows), 128):
                selected = rows[lo:lo+128]
                pixels = torch.from_numpy(np.array(source[selected])).cuda().float()
                values[lo:lo+len(selected)] = model.features(pixels).cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite target features in {directory.name}/{arm}')
        np.save(destination/f'{arm}_extra.npy', values)
        print(f'PORT EXTRACT {directory.name}/{arm}: {len(rows)} rows', flush=True)
    with np.load(WS/'b2_ssl_feats_38091.npz') as saved:
        previous = saved['F'][rows].astype(np.float32)
    reference = np.load(destination/'reference_extra.npy')
    relative = np.linalg.norm(reference-previous, axis=1)/np.maximum(np.linalg.norm(previous, axis=1), 1e-12)
    np.save(destination/'target_rows.npy', rows)
    write_json(destination/'FEATURES_COMPLETE.json', dict(target_rows=len(rows),
        arms=['reference', 'aligned_only', 'joint'], prefix_dtype=str(source.dtype),
        initial_max_abs_error_vs_old_ssl=float(np.max(np.abs(reference-previous))),
        initial_max_relative_l2_vs_old_ssl=float(relative.max()),
        reference='New initial-tail replay from the SAME float16 prefix as both trained arms',
        old_reference_preserved=True, target_labels_for_feature_extraction=0))
    del model
    torch.cuda.empty_cache()


def matched_views(xs, base_discarded, extra, model, stats, complement):
    """Keep cached R50 retained coordinates; only the actual R18 output changes."""
    from projection_residual_expert import discarded_coordinates

    z = model['pca_ssl'].transform(extra).astype(np.float32)
    joint = np.column_stack((xs, z))
    xe = (joint-model['expert_center'])/model['expert_scale']
    prior = model['expert'].decision_function(xe).astype(np.float32).astype(float)
    retained = (xe/np.sqrt(xe.shape[1])).astype(float)
    discarded = None
    if complement:
        split = base_discarded.shape[1]
        extra_discarded = ((discarded_coordinates(extra, model['pca_ssl'])-stats['discarded_center'][split:])/
                           stats['discarded_scale'][split:])
        discarded = np.column_stack((base_discarded, extra_discarded))
        discarded /= np.sqrt(discarded.shape[1])
    return prior, retained, discarded


def freeze_matched_port(port, out, base, labels):
    from projection_residual_expert import discarded_coordinates, freeze_kernel_draw

    directory = out/port
    destination = directory/'matched_joint'
    if (destination/'POLICIES_SAVED.json').exists():
        return
    if not (destination/'FEATURES_COMPLETE.json').exists():
        raise ValueError(f'{port}: extract the initial and trained tails before evaluation')
    old = archive(BASE/port/'policies.npz')
    q = old['query_rows']
    np.testing.assert_array_equal(q, np.load(destination/'target_rows.npy'))
    with (directory/'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    stats = archive(directory/'deployment_complement.npz')
    with np.load(ROOT/'pred_cache2'/f'{port}.npz') as saved:
        xs = saved['Xs'][q]
    bd = ((discarded_coordinates(base[q], model['pca'])-stats['discarded_center'][:base.shape[1]])/
          stats['discarded_scale'][:base.shape[1]])
    support_union = np.unique(np.searchsorted(q, old['support_rows'].ravel()))
    groups = [('reference', ('reference', 'complement_only')),
              ('aligned_only', ('aligned_only',)), ('joint', ('joint',))]
    for feature_arm, arms in groups:
        extra = np.load(destination/f'{feature_arm}_extra.npy')
        uses_complement = any(arm in ('joint', 'complement_only') for arm in arms)
        prior, retained, discarded = matched_views(xs, bd, extra, model, stats, uses_complement)
        kp = retained@retained[support_union].T
        kd = discarded@discarded[support_union].T if uses_complement else None
        for arm in arms:
            kernel = .5*kp+.5*kd+1. if arm in ('joint', 'complement_only') else kp+1.
            records = []
            for draw, support_global in enumerate(old['support_rows']):
                support = np.searchsorted(q, support_global)
                np.testing.assert_array_equal(q[support], support_global)
                result = freeze_kernel_draw(old['scores'][draw, 1], prior,
                    kernel[:, np.searchsorted(support_union, support)], support,
                    labels[support_global], model['gate'], q)
                np.testing.assert_array_equal(result['eval_mask'], old['eval_masks'][draw])
                np.testing.assert_array_equal(result['predictions'] != old['predictions'][draw, 2], result['selected'])
                if result['selected'][~result['eval_mask']].any() or result['cap'] != old['caps'][draw]:
                    raise ValueError('Matched arm changed support exclusion or budget')
                if result['selected'].sum() > result['cap']:
                    raise ValueError('Matched arm exceeded admission budget')
                records.append(result)
            np.savez_compressed(destination/f'{arm}_policies.npz',
                **{key: np.stack([r[key] for r in records]) for key in records[0]},
                query_rows=q, source_rows=old['source_rows'], support_rows=old['support_rows'],
                base_scores=old['scores'][:, 1])
    reference = archive(destination/'reference_policies.npz')
    previous = archive(directory/'reference_policies.npz')
    differences = []
    for draw in range(5):
        ev = reference['eval_mask'][draw]
        differences.append(dict(draw=draw,
            query_prediction_changes=int(np.sum(ev & (reference['predictions'][draw] != previous['predictions'][draw]))),
            query_admission_changes=int(np.sum(ev & (reference['selected'][draw] != previous['selected'][draw]))),
            expert_score_max_abs_difference=float(np.max(np.abs(reference['expert_scores'][draw]-previous['expert_scores'][draw])))))
    write_json(destination/'POLICIES_SAVED.json', dict(port=port, arms=MATCHED_ARMS,
        old_reference_path=str(directory/'reference_policies.npz'),
        reference='New matched float16-prefix initial-tail reference; NOT the historical original-cache reference',
        reference_differences_vs_old=differences, fixed_old_base64=True, fixed_old_gate=True,
        same_supports=True, target_labels_per_draw=64, draws=5, budget=.05,
        all_four_arms_saved_before_query_readout=True, target_encoder_labels=0))


def matched_metric_rows(port, destination, labels):
    from budgeted_residual_handoff import score_policy

    records = []
    for arm in MATCHED_ARMS:
        policy = archive(destination/f'{arm}_policies.npz')
        truth = labels[policy['query_rows']]
        for draw in range(5):
            base = policy['base_scores'][draw].argmax(1)
            pred, expert = policy['predictions'][draw], policy['expert_predictions'][draw]
            ev, selected = policy['eval_mask'][draw], policy['selected'][draw]
            full = score_policy(pred, base, truth, ev)
            em, bm = score_policy(expert, base, truth, ev), score_policy(base, base, truth, ev)
            records.append(dict(port=port, draw=draw, arm=arm, **full,
                expert_acc=em['acc'], expert_ba=em['ba'], base_acc=bm['acc'], base_ba=bm['ba'],
                selected=int(selected.sum()), cap=int(policy['cap'][draw]), evaluated=int(ev.sum()),
                rescued=int(np.sum(selected & (pred == truth) & (base != truth))),
                harmed=int(np.sum(selected & (pred != truth) & (base == truth)))))
    return records


def summarize_matched(records):
    import pandas as pd

    frame = pd.DataFrame(records)
    measures = ['acc', 'ba', 'expert_acc', 'expert_ba', 'base_acc', 'base_ba', 'selected', 'rescued', 'harmed']
    means = frame.groupby(['port', 'arm'])[measures].mean().groupby('arm').mean().reset_index().to_dict('records')
    contrasts = []
    for candidate, control in [('complement_only', 'reference'), ('aligned_only', 'reference'),
                               ('joint', 'reference'), ('joint', 'aligned_only'), ('joint', 'complement_only')]:
        a = frame[frame.arm == candidate].set_index(['port', 'draw'])
        b = frame[frame.arm == control].set_index(['port', 'draw'])
        delta = 100*(a[measures[:4]]-b[measures[:4]])
        port_delta = delta.groupby('port').mean()
        contrasts.append(dict(candidate=candidate, control=control,
            means_pp={m: float(port_delta[m].mean()) for m in measures[:4]},
            positive_ports={m: int((port_delta[m] > 1e-12).sum()) for m in measures[:4]},
            negative_ports={m: int((port_delta[m] < -1e-12).sum()) for m in measures[:4]},
            port_deltas_pp=port_delta.reset_index().to_dict('records')))
    return dict(n_ports=int(frame.port.nunique()), contexts=int(frame.port.nunique())*5,
        means=means, contrasts=contrasts, units='OA/BA fractions; contrasts percentage points',
        reference='New initial-tail float16-prefix replay; see separate old-reference comparisons',
        statistical_unit='Equal outer-port means over five shared64 support draws; one fixed training seed',
        independent_confirmation=False, fixed_old_base64=True, target_encoder_labels=0,
        parameters='Fixed six epochs and last checkpoint; no target tuning or port selection')


def evaluate_matched_ports(out, inventory, ports):
    import pandas as pd
    from build_oof_predictions import load_manifest

    missing = [item['port'] for item in inventory
               if not (out/item['port']/'matched_joint/FEATURES_COMPLETE.json').exists()]
    if missing:
        raise ValueError('Freeze all23 outer checkpoints/features before any matched query readout; missing: '+', '.join(missing))
    labels = load_manifest(SOURCE/'manifest.csv')['labels']
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    selected = [item['port'] for item in inventory if not ports or item['port'] in ports]
    with threadpool_limits(limits=2):
        for port in [item['port'] for item in inventory]:
            freeze_matched_port(port, out, base, labels)
        for port in selected:
            destination = out/port/'matched_joint'
            if (destination/'summary.json').exists():
                continue
            records = matched_metric_rows(port, destination, labels)
            write_json(destination/'metrics.json', records)
            pd.DataFrame([{k: v for k, v in r.items() if k != 'class_recall'} for r in records]).to_csv(
                destination/'metrics.csv', index=False)
            summary = summarize_matched(records)
            old_rows = json.loads((out/port/'metrics.json').read_text(encoding='utf-8'))
            reference_changes = {}
            for metric in ('acc', 'ba', 'expert_acc', 'expert_ba'):
                old_mean = np.mean([r[metric] for r in old_rows if r['arm'] == 'reference'])
                new_mean = np.mean([r[metric] for r in records if r['arm'] == 'reference'])
                reference_changes[metric] = float(100*(new_mean-old_mean))
            summary['new_reference_minus_old_reference_pp'] = reference_changes
            write_json(destination/'summary.json', summary)
            print(f'PORT MATCHED COMPLETE {port}: {json.dumps(summary["means"])}', flush=True)
    completed = [item['port'] for item in inventory if (out/item['port']/'matched_joint/summary.json').exists()]
    write_json(out/'MATCHED_JOINT_PROGRESS.json', dict(completed_ports=completed,
        expected_ports=[item['port'] for item in inventory], full_23_port_result=len(completed) == len(inventory)))
    if len(completed) == len(inventory):
        all_records = []
        old_reference_changes = []
        for port in completed:
            all_records.extend(json.loads((out/port/'matched_joint/metrics.json').read_text(encoding='utf-8')))
            own = json.loads((out/port/'matched_joint/summary.json').read_text(encoding='utf-8'))
            old_reference_changes.append(own['new_reference_minus_old_reference_pp'])
        summary = summarize_matched(all_records)
        summary.update(full_23_port_result=True, old_reference_summary_path=str(out/'summary.json'),
            new_reference_minus_old_reference_pp={m: float(np.mean([d[m] for d in old_reference_changes]))
                for m in ('acc', 'ba', 'expert_acc', 'expert_ba')})
        write_json(out/'joint_summary.json', summary)
        pd.DataFrame([{k: v for k, v in r.items() if k != 'class_recall'} for r in all_records]).to_csv(
            out/'matched_joint_metrics.csv', index=False)


def fit_episode_cache(path, base, extra, labels, train):
    """Cache a symmetric excluded-port pair; exact train rows govern reuse."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeClassifier
    from projection_residual_expert import fit_geometry_statistics
    from residual_expert import source_geometry

    if path.exists():
        cached = archive(path)
        np.testing.assert_array_equal(cached['train_rows'], train)
        return cached, True
    pca = PCA(n_components=128, random_state=0).fit(base[train])
    x = pca.transform(base[train])
    center, scale = x.mean(0), x.std(0)+1e-6
    xs = (x-center)/scale
    pca_ssl = PCA(n_components=64, random_state=0).fit(extra[train])
    joint = np.column_stack((xs, pca_ssl.transform(extra[train])))
    _, ec, es = source_geometry(joint, np.arange(len(train)))
    expert = RidgeClassifier(alpha=1., class_weight='balanced').fit((joint-ec)/es, labels[train])
    np.testing.assert_array_equal(expert.classes_, np.arange(8))
    stats = fit_geometry_statistics(base, extra, train, dict(pca=pca, pca_ssl=pca_ssl))
    values = dict(train_rows=train, base_mean=pca.mean_, base_components=pca.components_,
        base_center=center, base_scale=scale, ssl_mean=pca_ssl.mean_, ssl_components=pca_ssl.components_,
        expert_center=ec, expert_scale=es, expert_coef=expert.coef_, expert_intercept=expert.intercept_,
        discarded_center=stats['discarded_center'], discarded_scale=stats['discarded_scale'])
    np.savez_compressed(path, **values)
    return values, False


def materialize_episode(geometry, base, labels, held):
    from projection_residual_expert import discarded_coordinates

    # Match the helper's source-only geometry and deterministic 64-support recipe.
    x = base[held]@geometry['base_components'].T-geometry['base_mean']@geometry['base_components'].T
    xs = ((x-geometry['base_center'])/geometry['base_scale']).astype(np.float32)
    pca = SimpleNamespace(mean_=geometry['base_mean'], components_=geometry['base_components'])
    bd = ((discarded_coordinates(base[held], pca)-geometry['discarded_center'][:base.shape[1]])/
          geometry['discarded_scale'][:base.shape[1]])
    support = np.random.default_rng(0).choice(len(held), 64, replace=False)
    query = np.flatnonzero(~np.isin(np.arange(len(held)), support))
    keys = ('ssl_mean', 'ssl_components', 'expert_center', 'expert_scale', 'expert_coef',
            'expert_intercept', 'discarded_center', 'discarded_scale')
    return dict(rows=held, train_rows=geometry['train_rows'], support=support, query=query,
        labels=labels[held], xs=xs, base_discarded=bd.astype(np.float32),
        **{key: geometry[key].astype(np.float32) for key in keys})


def prepare_episodes(out, inventory, base, extra, manifest, ports=None, threads=2):
    from sar_joint_completion import PREFIX, TRAINING
    from build_oof_predictions import inner_split, outer_split

    cache = out/'episode_geometry_cache'
    cache.mkdir(exist_ok=True)
    write_prepare_status(out, threads, 'running')
    all_ports = list(np.unique(manifest['ports']))
    ready = []
    for item in inventory:
        port = item['port']
        if ports and port not in ports:
            continue
        directory = out/port
        if (directory/'EPISODES_COMPLETE.json').exists():
            ready.append(port)
            continue
        q, source = outer_split(port, manifest)
        np.savez(directory/'source_rows.npz', source_rows=source, labels=manifest['labels'], target_rows=q)
        protocol = dict(date='2026-10-08', dataset='24-port', target_port=port,
            fixed_training=TRAINING, trained_arms=['aligned_only', 'joint'],
            source_prefix_path=str(PREFIX), source_rows_path=str(directory/'source_rows.npz'),
            old_policy_path=str(BASE/port/'policies.npz'),
            deployment_retained_path=str(ROOT/'pred_cache2'/f'{port}.npz'),
            source_model_path=str(directory/'source_model.pkl'),
            scope='Strict outer target port and nonempty MMSI/product excluded before any supervised encoder training',
            inner_scope='Each held source port and all its identities excluded from episode prior/statistic fits',
            initialization='Original e206 layer4 only; never F0921-trained checkpoint',
            support='64 per source episode, seed0; target retains original five 64-label draws',
            target_encoder_labels=0, checkpoint='Final epoch6 only',
            episode_cache='Exact ordered train-row equality; deterministic geometry reused across symmetric port pairs',
            cpu_threads=threads, independent_confirmation=False)
        write_json(directory/'protocol.json', protocol)
        stats = archive(directory/'source_geometry.npz')
        np.savez_compressed(directory/'deployment_complement.npz', source_rows=source,
            discarded_center=stats['discarded_center'], discarded_scale=stats['discarded_scale'])
        folder = directory/'episodes'
        folder.mkdir(exist_ok=True)
        descriptors = []
        new_fits, reused = 0, 0
        for index, held_port in enumerate(np.unique(manifest['ports'][source])):
            train, held = inner_split(source, held_port, q, manifest)
            if len(held) <= 64:
                continue
            path = folder/f'{index:02d}.npz'
            started = time.monotonic()
            if not path.exists():
                pair = sorted((all_ports.index(port), all_ports.index(held_port)))
                geometry, hit = fit_episode_cache(cache/f'{pair[0]:02d}_{pair[1]:02d}.npz',
                                                   base, extra, manifest['labels'], train)
                arrays = materialize_episode(geometry, base, manifest['labels'], held)
                np.savez_compressed(path, **arrays)
                new_fits += not hit
                reused += hit
            else:
                with np.load(path) as saved:
                    np.testing.assert_array_equal(saved['train_rows'], train)
                    np.testing.assert_array_equal(saved['rows'], held)
            descriptors.append(dict(port=str(held_port), path=str(path), held=len(held),
                                    query=len(held)-64, source_fit_rows=len(train)))
            print(f'PORT PREP {port} / {held_port}: {time.monotonic()-started:.1f}s', flush=True)
            write_prepare_status(out, threads, 'running', f'{port} / {held_port}')
        write_json(directory/'EPISODES_COMPLETE.json', dict(episodes=descriptors,
            query_rows=sum(d['query'] for d in descriptors), source_only=True,
            outer_target_port=port, outer_exclusions_verified=True,
            new_geometry_fits_this_run=new_fits, reused_geometry_fits_this_run=reused))
        ready.append(port)
        write_json(out/'EPISODES_PREPARATION_STATUS.json', dict(ready_ports=ready,
            requested_ports=ports or [item['port'] for item in inventory], gpu_started=False,
            cpu_threads=threads))
        print(f'OUTER READY {port}: {len(descriptors)} episodes', flush=True)
        write_prepare_status(out, threads, 'running', f'{port} complete')
    write_json(out/'EPISODES_PREPARATION_STATUS.json', dict(ready_ports=ready,
        requested_ports=ports or [item['port'] for item in inventory], gpu_started=False))
    write_prepare_status(out, threads, 'complete')


@threadpool_limits.wrap(limits=1)
def refit_geometry(base, extra, source, cached, labels, gate):
    """Recover this outer fold, never an external-target or all-port geometry."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeClassifier
    from residual_expert import source_geometry

    pca = PCA(n_components=128, random_state=0).fit(base[source])
    x = pca.transform(base)
    center, scale = x[source].mean(0), x[source].std(0)+1e-6
    xs = ((x-center)/scale).astype(np.float32)
    pca_ssl = PCA(n_components=64, random_state=0).fit(extra[source])
    z = pca_ssl.transform(extra).astype(np.float32)
    errors = dict(xs=float(np.max(np.abs(xs-cached['Xs']))),
                  z=float(np.max(np.abs(z-cached['Z']))))
    for name, actual, expected in (('xs', xs, cached['Xs']), ('z', z, cached['Z'])):
        errors[name+'_relative_l2'] = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        if errors[name+'_relative_l2'] > 1e-3:
            raise ValueError(f'PCA geometry reconstruction differs: {errors}')
    print(f'PCA REPLAY {errors}', flush=True)
    # Preserve the cached retained branch exactly; PCA refits define its complement.
    joint = np.column_stack((cached['Xs'], cached['Z']))
    phi, ec, es = source_geometry(joint, source)
    _, bc, bs = source_geometry(cached['Xs'], source)
    model_b = RidgeClassifier(alpha=1., class_weight='balanced').fit(cached['Xs'][source], labels[source])
    model_e = RidgeClassifier(alpha=1., class_weight='balanced').fit(
        (joint[source]-ec)/es, labels[source])
    model = dict(pca=pca, pca_ssl=pca_ssl, center=center, scale=scale,
                 base=model_b, expert=model_e, expert_center=ec, expert_scale=es,
                 residual_center=bc, residual_scale=bs, gate=gate)
    replay = model_e.decision_function((joint-ec)/es)
    return model, phi, replay, errors


def freeze_port(port, out, base, extra, manifest):
    from build_oof_predictions import outer_split
    from projection_residual_expert import discarded_coordinates, fit_geometry_statistics, freeze_kernel_draw

    started = time.monotonic()
    directory = out/port
    if (directory/'POLICIES_SAVED.json').exists():
        return json.loads((directory/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    directory.mkdir(exist_ok=True)
    d = archive(ROOT/'pred_cache'/f'{port}.npz')
    cached = archive(ROOT/'pred_cache2'/f'{port}.npz')
    old = archive(BASE/port/'policies.npz')
    residual = archive(PARENT/port/'policies.npz')
    q, source = outer_split(port, manifest)
    for record in (d,):
        np.testing.assert_array_equal(q, record['qpos'])
        np.testing.assert_array_equal(source, record['spos'])
        np.testing.assert_array_equal(manifest['labels'], record['y_ids'])
    for record in (old, residual):
        np.testing.assert_array_equal(q, record['query_rows'])
        np.testing.assert_array_equal(source, record['source_rows'])
        np.testing.assert_array_equal(old['support_rows'], record['support_rows'])
    with (GATE/port/'source_model.pkl').open('rb') as stream:
        gate = pickle.load(stream)['prior']
    model, phi, replay, errors = refit_geometry(base, extra, source, cached, manifest['labels'], gate)
    head_error = float(np.max(np.abs(replay-d['expert_logits'])))
    if head_error > 1e-4:
        raise ValueError(f'{port}: source-head replay differs by {head_error}')
    stats = fit_geometry_statistics(base, extra, source, model)
    with (directory/'source_model.pkl').open('wb') as stream:
        pickle.dump(model, stream)
    np.savez_compressed(directory/'source_geometry.npz', source_rows=source, query_rows=q, **stats)
    discarded = np.column_stack((discarded_coordinates(base[q], model['pca']),
                                 discarded_coordinates(extra[q], model['pca_ssl'])))
    discarded = ((discarded-stats['discarded_center'])/stats['discarded_scale']/
                 np.sqrt(discarded.shape[1]))
    retained = phi[q, :-1]
    prior = d['expert_logits'][q].astype(float)
    parent_arm = list(old['arm_names']).index('visual_handoff')
    expert_arm = list(residual['score_names']).index('anchor_visual64')
    draws = {arm: [] for arm in ARMS}
    max_score_error = 0.
    for draw, support_global in enumerate(old['support_rows']):
        np.testing.assert_array_equal(support_global, np.random.default_rng(draw).choice(q, 64, replace=False))
        support = np.searchsorted(q, support_global)
        np.testing.assert_array_equal(q[support], support_global)
        kp, kd = retained@retained[support].T, discarded@discarded[support].T
        for arm, kernel in (('reference', kp+1.), ('complement_only', .5*kp+.5*kd+1.)):
            result = freeze_kernel_draw(old['scores'][draw, 1], prior, kernel, support,
                manifest['labels'][support_global], gate, q)
            np.testing.assert_array_equal(result['eval_mask'], old['eval_masks'][draw])
            np.testing.assert_array_equal(result['predictions'] != old['predictions'][draw, 2], result['selected'])
            if result['selected'][~result['eval_mask']].any() or result['cap'] != old['caps'][draw]:
                raise ValueError('support exclusion or cap differs')
            if result['selected'].sum() > result['cap']:
                raise ValueError('admission budget exceeded')
            if arm == 'reference':
                error = float(np.max(np.abs(result['expert_scores']-residual['scores'][draw, expert_arm])))
                max_score_error = max(max_score_error, error)
                np.testing.assert_allclose(result['expert_scores'], residual['scores'][draw, expert_arm], atol=1e-10, rtol=0)
                np.testing.assert_array_equal(result['predictions'], old['predictions'][draw, parent_arm])
                np.testing.assert_array_equal(result['selected'], old['selected'][draw, parent_arm])
            draws[arm].append(result)
    for arm, records in draws.items():
        np.savez_compressed(directory/f'{arm}_policies.npz',
            **{key: np.stack([r[key] for r in records]) for key in records[0]},
            query_rows=q, source_rows=source, support_rows=old['support_rows'],
            base_scores=old['scores'][:, 1])
    record = dict(port=port, target_rows=len(q), source_rows=len(source),
        pca_replay_max_errors=errors, source_head_max_error=head_error,
        reference_expert_score_max_error=max_score_error, reference_predictions_exact=True,
        reference_masks_exact=True, seconds=time.monotonic()-started)
    write_json(directory/'POLICIES_SAVED.json', record)
    print(f'FROZEN {port}: {record["seconds"]:.1f}s ref_error={max_score_error:.3g}', flush=True)
    return record


def evaluate(out, inventory, labels):
    import pandas as pd
    from budgeted_residual_handoff import score_policy

    records = []
    for item in inventory:
        port = item['port']
        block = []
        for arm in ARMS:
            policy = archive(out/port/f'{arm}_policies.npz')
            truth = labels[policy['query_rows']]
            for draw in range(5):
                prediction = policy['predictions'][draw]
                base = policy['base_scores'][draw].argmax(1)
                expert = policy['expert_predictions'][draw]
                ev, selected = policy['eval_mask'][draw], policy['selected'][draw]
                full = score_policy(prediction, base, truth, ev)
                bm, em = score_policy(base, base, truth, ev), score_policy(expert, base, truth, ev)
                block.append(dict(port=port, draw=draw, arm=arm, **full,
                    base_acc=bm['acc'], base_ba=bm['ba'], expert_acc=em['acc'], expert_ba=em['ba'],
                    selected=int(selected.sum()), evaluated=int(ev.sum()), cap=int(policy['cap'][draw]),
                    rescued=int(np.sum(selected & (prediction == truth) & (base != truth))),
                    harmed=int(np.sum(selected & (prediction != truth) & (base == truth)))))
        write_json(out/port/'metrics.json', block)
        records.extend(block)
    write_json(out/'metrics.json', records)
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != 'class_recall'} for r in records])
    frame.to_csv(out/'metrics.csv', index=False)
    measures = ['acc', 'ba', 'expert_acc', 'expert_ba', 'base_acc', 'base_ba', 'selected', 'rescued', 'harmed']
    port_means = frame.groupby(['port', 'arm'])[measures].mean().reset_index()
    port_means.to_csv(out/'port_means.csv', index=False)
    means = port_means.groupby('arm')[measures].mean().reset_index().to_dict('records')
    deltas = []
    for port in port_means.port.unique():
        pair = port_means[port_means.port == port].set_index('arm')
        deltas.append(dict(port=port, **{m: float(100*(pair.loc['complement_only', m]-pair.loc['reference', m]))
                                        for m in ('acc', 'ba', 'expert_acc', 'expert_ba')}))
    pd.DataFrame(deltas).to_csv(out/'port_deltas_pp.csv', index=False)
    delta_ba = np.array([d['ba'] for d in deltas])
    summary = dict(dataset='24-port self-built; 23 eligible outer targets', n_ports=len(inventory),
        draws_per_port=5, contexts=5*len(inventory), target_rows=sum(i['target_rows'] for i in inventory),
        units='OA/BA fractions; contrasts in percentage points', statistical_unit='Equal port mean over five support draws',
        means=means, delta_vs_reference_pp={m: float(np.mean([d[m] for d in deltas]))
                                           for m in ('acc', 'ba', 'expert_acc', 'expert_ba')},
        ba_positive_ports=int(np.sum(delta_ba > 1e-12)), ba_negative_ports=int(np.sum(delta_ba < -1e-12)),
        ba_tied_ports=int(np.sum(np.abs(delta_ba) <= 1e-12)), port_deltas_pp=deltas,
        reference_predictions_exact=True, reference_masks_exact=True,
        independent_confirmation=False, encoder_training=False, target_labels_per_port_per_draw=64,
        handoff_budget=.05, cost='Admission count only; both encoders are already evaluated',
        caveat='Frozen inherited OOF gate; M2 source-normalized complement, not M3/joint encoder training')
    write_json(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--stage', choices=('projection', 'prepare', 'train', 'evaluate'), default='projection')
    parser.add_argument('--ports', nargs='+', help='Schedule these outer ports; final summary requires all23')
    parser.add_argument('--threads', type=int, choices=(1, 2, 4, 8), default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'projection' and (args.out/'COMPLETE.json').exists():
        raise ValueError('Completed result already exists')
    if args.stage in ('train', 'evaluate'):
        inventory = json.loads((BASE/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
        if args.ports and not set(args.ports).issubset({item['port'] for item in inventory}):
            raise ValueError('Unknown requested outer port')
        if args.stage == 'train':
            from sar_joint_completion import train_arms
            for item in inventory:
                port = item['port']
                if args.ports and port not in args.ports:
                    continue
                directory = args.out/port
                if not (directory/'EPISODES_COMPLETE.json').exists():
                    raise ValueError(f'{port}: source episodes are not ready')
                train_arms(directory)
                extract_outer_targets(directory)
        else:
            evaluate_matched_ports(args.out, inventory, args.ports)
        return
    if args.stage == 'prepare':
        from build_oof_predictions import load_manifest

        if not (args.out/'COMPLETE.json').exists():
            raise ValueError('Complete matched M2 first to obtain verified per-port deployment geometries')
        inventory = json.loads((BASE/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
        if args.ports and not set(args.ports).issubset({item['port'] for item in inventory}):
            raise ValueError('Unknown requested outer port')
        manifest = load_manifest(SOURCE/'manifest.csv')
        base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
        with np.load(WS/'b2_ssl_feats_38091.npz') as saved:
            extra = saved['F'].astype(np.float32)
        with threadpool_limits(limits=args.threads):
            prepare_episodes(args.out, inventory, base, extra, manifest, args.ports, args.threads)
        return
    started = time.monotonic()
    from build_oof_predictions import load_manifest

    write_json(args.out/'PROTOCOL.json', dict(date='2026-10-08', arms=ARMS, cpu_threads=2,
        parent=str(BASE), source_features=str(SOURCE/'pooled_features.npy'),
        ssl_features=str(WS/'b2_ssl_feats_38091.npz'),
        source_scope='Per-outer-port, MMSI/product exclusions identical to cached original; no global external model',
        fixed='Exact cached base64 scores, source prior scores, source OOF gate, support rows and 5% cap',
        reconstruction='Refit per-port PCA128/64 on source only and check cached feature replay; retained features kept exact',
        new_arm='0.5 retained kernel + 0.5 source-standardized PCA-complement kernel + one bias',
        reference='Retained kernel + one bias; must reproduce parent scores, admissions and predictions',
        target_labels='Same 64 support labels shared across base/expert; never target-query fitting',
        independent_confirmation=False, no_gpu=True, freeze_all_before_query_readout=True))
    inventory = json.loads((BASE/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    if len(inventory) != 23:
        raise ValueError('Expected original 23-port inventory')
    manifest = load_manifest(SOURCE/'manifest.csv')
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    with np.load(WS/'b2_ssl_feats_38091.npz') as saved:
        extra = saved['F'].astype(np.float32)
    with threadpool_limits(limits=2):
        frozen = [freeze_port(item['port'], args.out, base, extra, manifest) for item in inventory]
        write_json(args.out/'POLICIES_SAVED.json', frozen)
        summary = evaluate(args.out, frozen, manifest['labels'])
    write_json(args.out/'COMPLETE.json', dict(status='complete', seconds=time.monotonic()-started,
        ports=len(frozen), contexts=5*len(frozen), policies=10*len(frozen), threads=2,
        maximum_reference_score_error=max(i['reference_expert_score_max_error'] for i in frozen),
        maximum_source_head_error=max(i['source_head_max_error'] for i in frozen)))
    print(json.dumps({k: v for k, v in summary.items() if k != 'port_deltas_pp'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
