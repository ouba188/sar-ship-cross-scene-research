"""Matched source calibration pairs for the frozen selective base and expert.

load_source_pairs(folder) returns (pairs, provenance). External linear heads
leave the episode port out; geometry and neural weights stay frozen. Missing
source features are cached only under compact_fusion_20261010/cache.
refit_joint=True matches the later joint-tail source-prior refit; the default
preserves the historical pre-tail head recipe.
These are calibration examples, not independently held-out model validation.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
EPISODES = ROOT/'module_completion_20261008/ports_projection'
EXTERNAL = dict(oss2=('opensarship2_transfer_20261007', 'module_completion_20261008/oss2'),
                f0921=('f0921_transfer_20261007', 'joint_residual_pilot_20261008'))
PRESERVE = ROOT/'frequency_preserve_20261010/source'
CACHE = ROOT/'compact_fusion_20261010/cache'


def load_npz(path):
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def _expert_inputs(raw, extra, model):
    # Identical casts/coordinates to run_opensarship2_transfer.target_geometry.
    xs = ((model['pca'].transform(raw)-model['center'])/model['scale']).astype(np.float32)
    joint = np.column_stack((xs, model['pca_ssl'].transform(extra).astype(np.float32)))
    return (joint-model['expert_center'])/model['expert_scale']


def _expert_fit_inputs(raw, extra, model):
    # Original fit_source recipe: pre-tail features, without deployment FP32 casts.
    xs = (model['pca'].transform(raw)-model['center'])/model['scale']
    joint = np.column_stack((xs, model['pca_ssl'].transform(extra)))
    return (joint-model['expert_center'])/model['expert_scale']


def _expert_kernel(raw, extra, xe, model, stats):
    # evaluate_joint_residual.build_kernels(complement=True), without its legacy imports.
    def views(rb, re, coordinates):
        discarded = []
        for values, pca in ((rb, model['pca']), (re, model['pca_ssl'])):
            centered = np.asarray(values, dtype=float)-pca.mean_
            components = np.asarray(pca.components_, dtype=float)
            discarded.append(centered-(centered@components.T)@components)
        residual = np.column_stack(discarded)
        residual = (residual-stats['discarded_center'])/stats['discarded_scale']/np.sqrt(residual.shape[1])
        return (coordinates/np.sqrt(192)).astype(float), residual

    # The reference implementation projects support separately, including FP32 rounding.
    support_xe = _expert_inputs(raw[:64], extra[:64], model)
    support_retained, support_residual = views(raw[:64], extra[:64], support_xe)
    retained, residual = views(raw, extra, xe)
    return .5*(retained@support_retained.T)+.5*(residual@support_residual.T)+1.


def _support_correction(kernel, labels, prior):
    # projection_residual_expert.fit_kernel_correction, fixed strength=8.
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    root = np.sqrt(len(labels)/(len(counts)*counts[inverse]))
    desired = 2*np.eye(prior.shape[1])[labels]-1.
    weighted = root[:, None]*kernel[:64]*root[None, :]
    dual = np.linalg.solve(weighted+8.*np.eye(len(labels)), root[:, None]*(desired-prior[:64]))
    return prior+kernel@(root[:, None]*dual)


def _episodes(folder, info):
    cached = PRESERVE/info['dataset']/info['task']
    paths = sorted(cached.glob('episode_*.npz'))
    if not paths:
        raise FileNotFoundError(f'No preserved source episodes: {cached}')
    result = []
    for path in paths:
        ei = int(path.stem.split('_')[-1])
        ep = load_npz(folder/'episodes'/f'{ei:02d}.npz')
        with np.load(path) as z:
            ix = z['indices'].astype(np.int64)
            base = z['expected'].astype(np.float64)
        np.testing.assert_array_equal(ix[:64], ep['support'])
        if len(np.unique(ix)) != len(ix) or not np.isin(ix[64:], ep['query']).all():
            raise ValueError(f'Non-matching source query indices: {path}')
        if base.shape != (len(ix)-64, 8):
            raise ValueError(f'Unexpected source score shape: {path}')
        result.append((ei, ep, ix, base))
    return result


def _external_features(joint, rows):
    joint = Path(joint)
    cached = joint/'source_extra.npy'
    if cached.exists():
        return np.asarray(np.load(cached, mmap_mode='r')[rows]), dict(
            source_features=str(cached), frozen_inference=False)

    cache = CACHE/f'{joint.parent.name}_source_extra.npz'
    if cache.exists():
        saved = load_npz(cache)
        np.testing.assert_array_equal(saved['rows'], rows)
        return saved['features'], dict(source_features=str(cache), frozen_inference=False)

    # OSS2 has only target features saved; apply its exact frozen joint tail.
    import torch
    from torchvision.models import resnet18

    protocol = json.loads((joint.parent/'protocol.json').read_text())
    prefix_path = Path(protocol['source_prefix_path'])
    prefix = np.load(prefix_path, mmap_mode='r')
    checkpoint = joint/'layer4.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tail = resnet18(weights=None).layer4
    tail.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    tail = tail.to(device).eval()
    values = np.empty((len(rows), 512), np.float32)
    print(f'FROZEN SOURCE FEATURES {joint.parent.name}: {len(rows)} rows on {device}', flush=True)
    with torch.inference_mode(), torch.backends.cudnn.flags(allow_tf32=False):
        for lo in range(0, len(rows), 128):
            pixels = torch.from_numpy(np.array(prefix[rows[lo:lo+128]])).to(device).float()
            values[lo:lo+len(pixels)] = tail(pixels).mean((2, 3)).cpu().numpy()
            if lo % 4096 == 0:
                print(f'FROZEN SOURCE FEATURES {lo+len(pixels)}/{len(rows)}', flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, rows=rows, features=values)
    return values, dict(source_prefix=str(prefix_path), checkpoint=str(checkpoint),
                        source_features=str(cache), frozen_inference=True, device=str(device))


def load_source_pairs(folder, refit_joint=False):
    """Return query-only pairs plus JSON-serializable calibration provenance.

    folder is a learned_frequency_residual_20261009/tasks task directory.
    Each pair has base/expert float64 [N,8], labels/rows int64 [N], episode
    int and held_port str. Experts use that episode's exact 64 support rows.
    Target archives and target query labels are never read.
    """
    folder = Path(folder)
    info = json.loads((folder/'PREPARED.json').read_text())
    ds, task = info['dataset'], info['task']
    episodes = _episodes(folder, info)
    provenance = dict(dataset=ds, task=task,
        base_cache=str(PRESERVE/ds/task), base_field='expected',
        support='Exact episode indices[:64], checked against episode support',
        query_plan='Frozen frequency_decision queries unused by selective neural fitting',
        target_query_labels_used=False, network_training=False,
        independent_validation=False,
        limitation='Representation training shares source ports; split pairs by held_port for fusion validation.')
    pairs = []
    if ds == 'self24ports':
        expert_path = ROOT/'directional_fusion_trial_20261009/ports'/task/'source_predictions.npz'
        scores = load_npz(expert_path)
        items = json.loads((EPISODES/task/'EPISODES_COMPLETE.json').read_text())['episodes']
        for ei, ep, ix, base in episodes:
            original = load_npz(Path(items[ei]['path']))
            np.testing.assert_array_equal(original['rows'], ep['rows'])
            np.testing.assert_array_equal(original['support'], ix[:64])
            if items[ei]['port'] != str(ep['held_port']):
                raise ValueError(f'Expert held-port mismatch: {task}/{ei}')
            mask = scores['episode'] == ei
            query_rows = ep['rows'][ix[64:]]
            positions = pd.Index(scores['rows'][mask]).get_indexer(query_rows)
            if (positions < 0).any():
                raise ValueError(f'Missing matched expert queries: {task}/{ei}')
            y = ep['labels'][ix[64:]].astype(np.int64)
            np.testing.assert_array_equal(scores['labels'][mask][positions], y)
            pairs.append(dict(base=base, expert=scores['expert'][mask][positions].astype(np.float64),
                labels=y, rows=query_rows.astype(np.int64), episode=ei, held_port=str(ep['held_port'])))
        provenance.update(expert_cache=str(expert_path), expert_field='expert',
            expert_head_inner_port_held=True, expert_geometry_inner_port_held=False,
            expert_encoder_inner_port_held=False,
            expert_recipe='Cached direct-condition expert, same source episode support; no new expert fit')
    elif ds in ('oss2', 'f0921'):
        from sklearn.base import clone
        from threadpoolctl import threadpool_limits

        source_dir, joint_dir = EXTERNAL[ds]
        joint = ROOT/joint_dir/'joint'
        model_path = joint.parent/'source_model.pkl'
        with model_path.open('rb') as stream:
            model = pickle.load(stream)
        stats_path = joint.parent/'deployment_complement.npz'
        stats = load_npz(stats_path)
        allowed = load_npz(ROOT/source_dir/'source_split.npz')['source_rows']
        rows = np.unique(np.concatenate([ep['rows'][ix] for _, ep, ix, _ in episodes]))
        if not np.isin(rows, allowed).all():
            raise ValueError(f'Expert calibration rows outside permitted {ds} source split')
        extra, feature_provenance = _external_features(joint, allowed)
        if refit_joint:
            fit_features_path = Path(feature_provenance['source_features'])
            fit_extra = extra
        elif ds == 'oss2':
            fit_features_path = SOURCE.parents[1]/'b2_ssl_feats_38091.npz'
            with np.load(fit_features_path) as saved:
                fit_extra = saved['F'].astype(np.float32)[allowed]
        else:
            fit_features_path = ROOT/'neural_complement_pilot_20261007/frozen_replay/source_extra.npy'
            fit_extra = np.asarray(np.load(fit_features_path, mmap_mode='r')[allowed])
        raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
        manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['port', 'class_id'])
        ports, labels = manifest.port.to_numpy()[allowed], manifest.class_id.to_numpy()[allowed]
        locations = pd.Index(allowed)
        head_fits = []
        with threadpool_limits(limits=2):
            input_fn = _expert_inputs if refit_joint else _expert_fit_inputs
            inputs = np.concatenate([input_fn(raw[allowed[lo:lo+2048]], fit_extra[lo:lo+2048], model)
                                     for lo in range(0, len(allowed), 2048)])
            for ei, ep, ix, base in episodes:
                held = str(ep['held_port'])
                train = ports != held
                episode_rows = ep['rows'][ix]
                pos = locations.get_indexer(episode_rows)
                if (pos < 0).any() or train[pos].any():
                    raise ValueError(f'Expert head train/episode overlap: {ds}/{ei}')
                head = clone(model['expert']).fit(inputs[train], labels[train])
                np.testing.assert_array_equal(head.classes_, np.arange(8))
                rb, re = raw[episode_rows], extra[pos]
                xe = _expert_inputs(rb, re, model)
                prior = head.decision_function(xe).astype(np.float32).astype(float)
                kernel = _expert_kernel(rb, re, xe, model, stats)
                sy = ep['labels'][ix[:64]].astype(np.int64)
                expert = _support_correction(kernel, sy, prior)
                pairs.append(dict(base=base, expert=expert[64:].astype(np.float64),
                    labels=ep['labels'][ix[64:]].astype(np.int64), rows=episode_rows[64:].astype(np.int64),
                    episode=ei, held_port=held))
                head_fits.append(dict(episode=ei, held_port=held, training_rows=int(train.sum()),
                                      held_port_training_rows=0))
        provenance.update(expert_model=str(model_path), expert_geometry=str(stats_path),
            expert_head_inner_port_held=True, expert_geometry_inner_port_held=False,
            expert_encoder_inner_port_held=False, **feature_provenance,
            expert_head_parameters=model['expert'].get_params(), expert_head_fits=head_fits,
            expert_head_fit_features=str(fit_features_path),
            expert_head_fit_representation='Original pre-tail source features in frozen deployment geometry',
            expert_inference_representation='Frozen joint-tail features; no post-tail head refit',
            expert_recipe='Held-port-out pre-tail linear expert prior applied to joint-tail coordinates; '
                          'frozen joint complement kernel; same64 support ridge8',
            limitation='Both linear priors leave the source episode port out; geometry and encoders '
                       'still share source data. This is not independent representation-held validation.')
        if refit_joint:
            provenance.update(expert_head_fit_representation='Frozen joint-tail deployment coordinates',
                expert_inference_representation='Frozen joint-tail deployment coordinates',
                expert_recipe='Held-port-out matched joint-tail ridge1 prior; same frozen joint kernel and support ridge8')
    else:
        raise ValueError(f'Unsupported compact comparison dataset: {ds}')
    provenance.update(episodes=len(pairs), queries=sum(len(p['labels']) for p in pairs))
    return pairs, provenance
