"""Complete matched SAR M2/M3 ablations using each target's purged source."""

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
WS = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline')
SOURCE = WS/'artifacts/eight_class_adaptive_20260916'
PREFIX = ROOT/'neural_complement_pilot_20261007/source_layer3.npy'
DATASETS = {'oss1': ROOT/'opensarship1_transfer_20261007',
            'oss2': ROOT/'opensarship2_transfer_20261007'}
ARMS = ('reference', 'complement_only', 'aligned_only', 'joint')
TRAINING = dict(epochs=6, query_batch_size=128, support_size=64, lr=1e-5,
                weight_decay=1e-4, gradient_clip=5., seed=20261008)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def fit_episode(base, extra, labels, train, held):
    """Fit only on the provided inner training rows; held rows form an episode."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeClassifier
    from projection_residual_expert import discarded_coordinates, fit_geometry_statistics
    from residual_expert import source_geometry

    if np.intersect1d(train, held).size:
        raise ValueError('source episode train/held overlap')
    pca = PCA(n_components=128, random_state=0).fit(base[train])
    x = pca.transform(base[train])
    center, scale = x.mean(0), x.std(0)+1e-6
    xs = (x-center)/scale
    held_xs = ((pca.transform(base[held])-center)/scale).astype(np.float32)
    pca_ssl = PCA(n_components=64, random_state=0).fit(extra[train])
    joint = np.column_stack((xs, pca_ssl.transform(extra[train])))
    _, ec, es = source_geometry(joint, np.arange(len(train)))
    expert = RidgeClassifier(alpha=1., class_weight='balanced').fit((joint-ec)/es, labels[train])
    np.testing.assert_array_equal(expert.classes_, np.arange(8))
    stats = fit_geometry_statistics(base, extra, train, dict(pca=pca, pca_ssl=pca_ssl))
    bd = ((discarded_coordinates(base[held], pca)-stats['discarded_center'][:base.shape[1]])/
          stats['discarded_scale'][:base.shape[1]])
    support = np.random.default_rng(0).choice(len(held), 64, replace=False)
    query = np.flatnonzero(~np.isin(np.arange(len(held)), support))
    floats = dict(xs=held_xs, ssl_mean=pca_ssl.mean_, ssl_components=pca_ssl.components_,
                  expert_center=ec, expert_scale=es, expert_coef=expert.coef_,
                  expert_intercept=expert.intercept_, base_discarded=bd,
                  discarded_center=stats['discarded_center'], discarded_scale=stats['discarded_scale'])
    return dict(rows=held, train_rows=train, support=support, query=query, labels=labels[held],
                **{key: value.astype(np.float32) for key, value in floats.items()})


def prepare(name, out):
    from build_oof_predictions import load_manifest
    from projection_residual_expert import fit_geometry_statistics
    from threadpoolctl import threadpool_limits

    out.mkdir(parents=True, exist_ok=True)
    if (out/'EPISODES_COMPLETE.json').exists():
        print(f'PREPARED {name}: reuse completed source episodes', flush=True)
        return
    parent = DATASETS[name]
    protocol = dict(date='2026-10-08', dataset=name, target=str(parent), arms=list(ARMS),
        trained_arms=['aligned_only', 'joint'], fixed_training=TRAINING,
        source_prefix_path=str(PREFIX), source_rows_path=str(out/'source_rows.npz'),
        source_scope='Existing dataset-specific identity/product-purged source, including encoder training',
        episode_scope='Held source port and overlapping identities excluded from its prior geometry fit',
        geometry='Original fixed source heads/statistics; retained192 and complement2560; one bias',
        matched_base='Exact saved Base64 scores; original shared64 supports and five draws',
        target_input='Original VH RGB Resize(128), aspect ratio retained, float32 inference',
        source_prefix='Existing deterministic e206 layer3 float16 cache; no target labels',
        budget_scope='5% handoff count, not feature compute', target_encoder_labels=0,
        no_target_checkpoint_selection=True, independent_confirmation=False)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise ValueError('changed completion protocol; use a fresh output')
    else:
        write_json(out/'protocol.json', protocol)
    manifest = load_manifest(SOURCE/'manifest.csv')
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    with np.load(WS/'b2_ssl_feats_38091.npz') as saved:
        extra = saved['F'].astype(np.float32)
    with np.load(parent/'source_split.npz') as saved:
        source_rows = saved['source_rows']
    np.savez(out/'source_rows.npz', source_rows=source_rows, labels=manifest['labels'])
    shutil.copyfile(parent/'source_model.pkl', out/'source_model.pkl')
    with (out/'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    (out/'episodes').mkdir(exist_ok=True)
    descriptors = []
    with threadpool_limits(limits=2):
        if not (out/'deployment_complement.npz').exists():
            stats = fit_geometry_statistics(base, extra, source_rows, model)
            np.savez_compressed(out/'deployment_complement.npz', source_rows=source_rows, **stats)
        ports = np.unique(manifest['ports'][source_rows])
        for index, port in enumerate(ports):
            started = time.monotonic()
            with np.load(parent/'source_oof'/f'{port}.npz') as saved:
                train, held = saved['train_rows'], saved['held_rows']
            if len(held) <= 64:
                continue
            if not np.isin(train, source_rows).all() or not np.isin(held, source_rows).all():
                raise ValueError('episode includes excluded external-target identities')
            path = out/'episodes'/f'{index:02d}.npz'
            if not path.exists():
                arrays = fit_episode(base, extra, manifest['labels'], train, held)
                np.savez_compressed(path, **arrays)
            descriptors.append(dict(port=str(port), path=str(path), held=len(held),
                query=len(held)-64, source_fit_rows=len(train)))
            print(f'PREP {name} {index+1}/{len(ports)} {port} {time.monotonic()-started:.1f}s', flush=True)
    write_json(out/'EPISODES_COMPLETE.json', dict(episodes=descriptors,
        query_rows=sum(d['query'] for d in descriptors), source_only=True))


def train_arms(out):
    """Reusable layer4-only trainer; protocol supplies source cache and row map."""
    import torch
    from torch.nn import functional as F
    from aligned_residual_ops import FixedExpertGeometry, residual_scores
    from joint_residual_ops import ComplementExpertGeometry
    from neural_complement import ComplementTail
    from train_joint_residual import check_operator

    if (out/'TAIL_TRAINING_COMPLETE.json').exists():
        print(f'TRAINED {out.name}: keep completed checkpoints', flush=True)
        return
    protocol = json.loads((out/'protocol.json').read_text())
    config = protocol['fixed_training']
    source = np.load(protocol['source_prefix_path'], mmap_mode='r')
    with np.load(protocol['source_rows_path']) as saved:
        counts = np.bincount(saved['labels'][saved['source_rows']], minlength=8)
    if (counts == 0).any():
        raise ValueError('all eight source classes required by this SAR recipe')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    device = torch.device('cuda')
    checkpoint = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    model = ComplementTail(checkpoint['backbone'])
    model.classifier.requires_grad_(False)
    model.utility.requires_grad_(False)
    initial = {key: value.clone() for key, value in model.layer4.state_dict().items()}
    model = model.to(device)
    # Keep the 1.25 GB frozen prefix on device: random mmap reads otherwise
    # dominate repeated support/query episodes on the 16 GB host.
    source_gpu = torch.empty(source.shape, dtype=torch.float16, device=device)
    for lo in range(0, len(source), 1024):
        hi = min(lo+1024, len(source))
        source_gpu[lo:hi].copy_(torch.from_numpy(np.array(source[lo:hi])))
    print(f'PREFIX GPU CACHE {len(source)} rows; {source_gpu.numel()*source_gpu.element_size()/1e9:.3f} GB', flush=True)
    weights = torch.as_tensor(counts.sum()/(8*counts), dtype=torch.float32, device=device)
    descriptors = json.loads((out/'EPISODES_COMPLETE.json').read_text())['episodes']
    episodes = []
    for descriptor in descriptors:
        with np.load(descriptor['path']) as saved:
            episodes.append({key: saved[key] for key in saved.files})
    errors = {}
    for arm in protocol['trained_arms']:
        folder = out/arm
        folder.mkdir(exist_ok=True)
        if (folder/'TRAINING_COMPLETE.json').exists():
            errors[arm] = json.loads((folder/'TRAINING_COMPLETE.json').read_text())['parity_error']
            continue
        complement = arm == 'joint'
        cls = ComplementExpertGeometry if complement else FixedExpertGeometry
        geometries = [cls(episode).to(device).float() for episode in episodes]
        model.layer4.load_state_dict(initial)
        model.eval()
        errors[arm] = check_operator(geometries[0], model, episodes[0], source, device, complement)
        optimizer = torch.optim.AdamW(model.layer4.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        history = []
        started = time.monotonic()
        for epoch in range(config['epochs']):
            rng = np.random.default_rng(config['seed']+epoch)
            total, count, steps = 0., 0, 0
            model.train()
            for ei in rng.permutation(len(episodes)):
                episode, geometry = episodes[ei], geometries[ei]
                support, query = episode['support'], rng.permutation(episode['query'])
                sy = torch.as_tensor(episode['labels'][support], dtype=torch.long, device=device)
                for lo in range(0, len(query), config['query_batch_size']):
                    q = query[lo:lo+config['query_batch_size']]
                    selected = np.r_[support, q]
                    row_index = torch.as_tensor(episode['rows'][selected], dtype=torch.long, device=device)
                    pixels = source_gpu[row_index].float()
                    xs = torch.as_tensor(episode['xs'][selected], dtype=torch.float32, device=device)
                    qy = torch.as_tensor(episode['labels'][q], dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        features = model.features(pixels)
                    if complement:
                        bd = torch.as_tensor(episode['base_discarded'][selected], dtype=torch.float32, device=device)
                        prior, phi = geometry(xs, features.float(), bd)
                    else:
                        prior, phi = geometry(xs, features.float())
                    adapted = residual_scores(phi[:64], prior[:64], sy, phi[64:], prior[64:])
                    loss = (F.cross_entropy(adapted, qy, reduction='none')*weights[qy]).sum()/weights[qy].sum()
                    if not torch.isfinite(loss):
                        raise ValueError(f'nonfinite loss in {arm}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.layer4.parameters(), config['gradient_clip'])
                    optimizer.step()
                    total += float(loss.detach())*len(q)
                    count += len(q)
                    steps += 1
            record = dict(epoch=epoch+1, classification=total/count, query_rows=count,
                steps=steps, elapsed_seconds=time.monotonic()-started)
            history.append(record)
            write_json(folder/'training_history.json', history)
            print(f'TRAIN {out.name}/{arm} {record}', flush=True)
        torch.save({key: value.detach().cpu() for key, value in model.layer4.state_dict().items()}, folder/'layer4.pt')
        write_json(folder/'TRAINING_COMPLETE.json', dict(epochs=config['epochs'], parity_error=errors[arm]))
        del geometries
    write_json(out/'TAIL_TRAINING_COMPLETE.json', dict(arms=protocol['trained_arms'],
        parity_max_abs_errors=errors, target_labels_for_encoder=0, epochs=config['epochs']))


def extract_targets(name, out):
    import pandas as pd
    import torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import resnet18
    from concurrent.futures import ThreadPoolExecutor

    if (out/'FEATURES_COMPLETE.json').exists():
        return
    frame = pd.read_csv(DATASETS[name]/'target_manifest.csv', keep_default_na=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    net = resnet18(weights=None)
    net.load_state_dict(checkpoint['backbone'], strict=False)
    prefix = torch.nn.Sequential(*list(net.children())[:-3]).cuda().eval()
    initial = {key: value.clone() for key, value in net.layer4.state_dict().items()}
    tail = net.layer4.cuda().eval()
    tf = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    def read(path):
        with Image.open(path) as im:
            return tf(im.convert('RGB'))

    values = {}
    for arm in ('reference', 'aligned_only', 'joint'):
        folder = out/arm
        folder.mkdir(exist_ok=True)
        state = initial if arm == 'reference' else torch.load(folder/'layer4.pt', map_location='cpu', weights_only=True)
        tail.load_state_dict(state)
        features = np.empty((len(frame), 512), dtype=np.float32)
        paths = frame.chip_path_vh.tolist()
        with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
            for lo in range(0, len(paths), 64):
                batch = list(pool.map(read, paths[lo:lo+64]))
                grouped = {}
                for i, pixels in enumerate(batch):
                    grouped.setdefault(tuple(pixels.shape), []).append(i)
                for indices in grouped.values():
                    pixels = torch.stack([batch[i] for i in indices]).cuda()
                    features[lo+np.asarray(indices)] = tail(prefix(pixels)).mean((2, 3)).cpu().numpy()
        np.save(folder/'target_extra.npy', features)
        values[arm] = dict(rows=len(frame), finite=bool(np.isfinite(features).all()))
        if not values[arm]['finite']:
            raise ValueError('nonfinite target features')
        print(f'EXTRACT {name}/{arm} {len(frame)}', flush=True)
    with np.load(DATASETS[name]/'features.npz') as saved:
        old = saved['vh_ssl']
        np.testing.assert_array_equal(saved['sample_ids'], frame.sample_id.to_numpy(str))
    replay = np.load(out/'reference/target_extra.npy')
    rel = np.linalg.norm(replay-old, axis=1)/np.maximum(np.linalg.norm(old, axis=1), 1e-12)
    if rel.max() > 1e-4:
        raise ValueError(f'initial target encoder differs from cached recipe: {rel.max()}')
    np.save(out/'target_sample_ids.npy', frame.sample_id.to_numpy(str))
    write_json(out/'FEATURES_COMPLETE.json', dict(arms=values,
        initial_max_abs_error=float(np.max(np.abs(replay-old))), initial_max_relative_l2=float(rel.max()),
        source='original source prior/statistics fixed before training', target_query_labels_read=False))


def evaluate(name, out):
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from budgeted_residual_handoff import score_policy
    from evaluate_joint_residual import build_kernels
    from projection_residual_expert import freeze_kernel_draw
    from run_projection_residual_pilot import load_target

    if (out/'summary.json').exists():
        print(f'EVALUATED {name}: preserve completed readout', flush=True)
        return
    if not (out/'FEATURES_COMPLETE.json').exists():
        raise ValueError('train and extract both arms first')
    frame, base, original_extra, old, model, _, _ = load_target(name)
    np.testing.assert_array_equal(np.load(out/'target_sample_ids.npy'), frame.sample_id.to_numpy(str))
    with np.load(out/'deployment_complement.npz') as saved:
        stats = {key: saved[key] for key in ('discarded_center', 'discarded_scale')}
    support_union = np.unique(old['support_rows'])
    policies = {}
    with threadpool_limits(limits=2):
        for extra, arms in [(original_extra, ['reference', 'complement_only']),
                            (np.load(out/'aligned_only/target_extra.npy'), ['aligned_only']),
                            (np.load(out/'joint/target_extra.npy'), ['joint'])]:
            prior, kernels = build_kernels(base, extra, model, stats, support_union,
                                           any(a in ('joint', 'complement_only') for a in arms))
            for arm in arms:
                draws = []
                key = 'complement' if arm in ('joint', 'complement_only') else 'retained'
                for i, support in enumerate(old['support_rows']):
                    result = freeze_kernel_draw(old['base_scores'][i], prior,
                        kernels[key][:, np.searchsorted(support_union, support)], support,
                        frame.iloc[support].class_id.to_numpy(int), model['gate'], frame.target_manifest_row.to_numpy())
                    np.testing.assert_array_equal(result['eval_mask'], old['eval_masks'][i])
                    np.testing.assert_array_equal(result['predictions'] != old['predictions'][i, 1], result['selected'])
                    if result['selected'][~result['eval_mask']].any() or result['selected'].sum() > old['caps'][i]:
                        raise ValueError('support/budget violation')
                    if arm == 'reference':
                        np.testing.assert_allclose(result['expert_scores'], old['expert_scores'][i], atol=3e-5, rtol=0)
                        np.testing.assert_array_equal(result['predictions'], old['predictions'][i, 2])
                        np.testing.assert_array_equal(result['selected'], old['selected'][i])
                    draws.append(result)
                folder = out/arm
                folder.mkdir(exist_ok=True)
                policies[arm] = draws
                np.savez_compressed(folder/'policies.npz',
                    **{key: np.stack([d[key] for d in draws]) for key in draws[0]},
                    support_rows=old['support_rows'], sample_ids=old['sample_ids'])
        write_json(out/'POLICIES_SAVED.json', dict(arms=ARMS, reference_predictions_and_admissions_exact=True,
            all_saved_before_query_readout=True, labels_per_draw=64, budget=.05))
        truth = frame.class_id.to_numpy(int)
        rows = []
        for arm in ARMS:
            for i, policy in enumerate(policies[arm]):
                ev, selected = policy['eval_mask'], policy['selected']
                baseline = old['predictions'][i, 1]
                pred, expert = policy['predictions'], policy['expert_predictions']
                full = score_policy(pred, baseline, truth, ev)
                em = score_policy(expert, baseline, truth, ev)
                rows.append(dict(dataset=name, arm=arm, draw=i, **full, expert_acc=em['acc'], expert_ba=em['ba'],
                    selected=int(selected.sum()), cap=int(policy['cap']), evaluated=int(ev.sum()),
                    rescued=int(np.sum(selected & (pred == truth) & (baseline != truth))),
                    harmed=int(np.sum(selected & (pred != truth) & (baseline == truth)))))
    measures = ('acc', 'ba', 'expert_acc', 'expert_ba', 'selected', 'rescued', 'harmed')
    means = [dict(arm=arm, **{m: float(np.mean([r[m] for r in rows if r['arm'] == arm]))
                            for m in measures}) for arm in ARMS]
    contrasts = []
    for arm in ARMS[1:]:
        delta = {m: [100*(a[m]-b[m]) for a, b in zip(
            [r for r in rows if r['arm'] == arm], [r for r in rows if r['arm'] == 'reference'])]
            for m in ('acc', 'ba')}
        contrasts.append(dict(candidate=arm, control='reference', draw_deltas_pp=delta,
            means_pp={m: float(np.mean(v)) for m, v in delta.items()},
            positive_draws={m: int(np.count_nonzero(np.asarray(v)>0)) for m, v in delta.items()}))
    write_json(out/'metrics.json', rows)
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]).to_csv(out/'metrics.csv', index=False)
    write_json(out/'summary.json', dict(dataset=name, means=means, contrasts=contrasts,
        source_encoder_target_excluded=True,
        exclusion_scope='New supervised tail training only; inherited SSL foundation pretraining is not newly isolated',
        target_labels_per_draw=64, draws=5,
        statistical_unit='five support draws, one fixed neural training seed', independent_confirmation=False))
    print(json.dumps(dict(dataset=name, means=means, contrasts=contrasts), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'train', 'evaluate'), required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.dataset, args.out)
    elif args.stage == 'train':
        train_arms(args.out)
        extract_targets(args.dataset, args.out)
    else:
        evaluate(args.dataset, args.out)


if __name__ == '__main__':
    main()
