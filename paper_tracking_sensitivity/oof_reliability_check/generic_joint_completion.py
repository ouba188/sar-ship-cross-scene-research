"""Matched M2 completion and separately retrained RGB M3 / joint outer folds.

Reuse the saved 64-support draws, base scores, source heads and gate. The only
changed path in the m2 stage is the expert's residual adaptation kernel.
Explicit RGB stages rebuild lost source checkpoints under a new reference.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
# Match generic_transfer_data without importing CPU-only dependencies on GPU.
DOMAINS = {'pacs': ('art_painting', 'cartoon', 'photo', 'sketch'),
           'officehome': ('Art', 'Clipart', 'Product', 'Real World')}
PREVIOUS = {name: ROOT/f'{name}_handoff_transfer_20261007' for name in DOMAINS}
ARMS = ('reference', 'complement_only')
MEASURES = ('acc', 'ba', 'expert_acc', 'expert_ba', 'base_acc', 'base_ba')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def freeze_pair(base_scores, prior_expert, pca_kernel, discarded_kernel,
                support_rows, support_labels, gate, row_ids):
    """A paired draw accepts support truth only; shared base is never refitted."""
    from projection_residual_expert import freeze_kernel_draw

    kernels = (pca_kernel+1., .5*pca_kernel+.5*discarded_kernel+1.)
    return {arm: freeze_kernel_draw(base_scores, prior_expert, kernel,
                                   support_rows, support_labels, gate, row_ids)
            for arm, kernel in zip(ARMS, kernels)}


def freeze_fold(dataset, domain, out):
    import pandas as pd
    from generic_transfer_data import load_generic_transfer
    from projection_residual_expert import build_support_kernels, fit_geometry_statistics

    started = time.monotonic()
    data = load_generic_transfer(dataset, domain)
    previous = PREVIOUS[dataset]/domain.replace(' ', '_')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'POLICIES_SAVED.json').exists():
        print(f'REUSE FROZEN {dataset}/{domain}', flush=True)
        return
    base, extra = data['base_features'], data['extra_features']
    target, source = data['target_rows'], data['source_rows']
    manifest = pd.read_csv(previous/'target_manifest.csv')
    with np.load(previous/'policies.npz') as saved:
        old = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(old['target_rows'], target)
    np.testing.assert_array_equal(manifest.row_id, target)
    np.testing.assert_array_equal(manifest.class_id, data['labels'][target])
    np.testing.assert_array_equal(old['seeds'], np.arange(5))
    with (previous/'source_model.pkl').open('rb') as handle:
        model = pickle.load(handle)
    write_json(out/'protocol.json', dict(
        dataset=dataset, target_domain=domain, previous=str(previous),
        provenance=data['provenance'], n_classes=data['n_classes'],
        source_rows=len(source), target_rows=len(target), labels_per_draw=64,
        budget=.05, seeds=list(range(5)), arms=ARMS,
        kernel_reference='K_retained + 1',
        kernel_complement='.5 K_retained + .5 K_discarded + 1',
        residual_ridge=8., source_only_geometry=True,
        original_base_scores_and_gate_frozen=True,
        inner_encoder_oof=False, independent_confirmation=False,
        M3='not_run: original outer-fold RGB R18 weights were not saved',
        joint='not_run: no layer4 training in this CPU completion'))
    stats = fit_geometry_statistics(base, extra, source, model)
    np.savez_compressed(out/'source_geometry.npz', source_rows=source, **stats)
    reference_rows = np.unique(old['support_rows'])
    kernels = build_support_kernels(base[target], extra[target], model, stats, reference_rows)
    draws = {arm: [] for arm in ARMS}
    error = 0.
    for draw, support in enumerate(old['support_rows']):
        columns = np.searchsorted(reference_rows, support)
        result = freeze_pair(old['base_scores'][draw], old['prior_expert_scores'],
                             kernels['pca'][:, columns], kernels['discarded'][:, columns],
                             support, data['labels'][target[support]], model['gate'], target)
        ref = result['reference']
        error = max(error, float(np.max(np.abs(ref['expert_scores']-old['expert_scores'][draw]))))
        np.testing.assert_allclose(ref['expert_scores'], old['expert_scores'][draw], atol=3e-5, rtol=0)
        np.testing.assert_array_equal(ref['predictions'], old['predictions'][draw, 2])
        np.testing.assert_array_equal(ref['selected'], old['selected'][draw])
        for arm, policy in result.items():
            np.testing.assert_array_equal(policy['eval_mask'], old['eval_masks'][draw])
            if policy['selected'][~policy['eval_mask']].any() or policy['selected'].sum() > old['caps'][draw]:
                raise ValueError('support inclusion or 5% budget violation')
            np.testing.assert_array_equal(policy['predictions'] != old['predictions'][draw, 1], policy['selected'])
            draws[arm].append(policy)
    for arm, policies in draws.items():
        np.savez_compressed(out/f'{arm}_policies.npz',
                            **{key: np.stack([p[key] for p in policies]) for key in policies[0]},
                            support_rows=old['support_rows'], target_rows=target)
    write_json(out/'POLICIES_SAVED.json', dict(
        dataset=dataset, domain=domain, draws=5, arms=ARMS,
        max_reference_score_error=error, reference_predictions_exact=True,
        policies_frozen_before_query_readout=True, seconds=time.monotonic()-started))
    print(f'FROZEN {dataset}/{domain}: 2 arms x 5 draws; replay maxerr={error:.3g}; '
          f'{time.monotonic()-started:.1f}s', flush=True)


def readout_fold(dataset, domain, out, arms=ARMS, protocol='cached_rgb_reference'):
    import pandas as pd
    from run_generic_transfer import metrics

    previous = PREVIOUS[dataset]/domain.replace(' ', '_')
    truth = pd.read_csv(previous/'target_manifest.csv').class_id.to_numpy(int)
    k = 7 if dataset == 'pacs' else 65
    with np.load(previous/'policies.npz') as old:
        base_predictions = old['predictions'][:, 1]
    rows = []
    for arm in arms:
        with np.load(out/f'{arm}_policies.npz') as policy:
            for draw in range(5):
                ev = policy['eval_mask'][draw]
                selected = policy['selected'][draw]
                prediction, expert = policy['predictions'][draw], policy['expert_predictions'][draw]
                base = base_predictions[draw]
                base_m = metrics(truth[ev], base[ev], k)
                expert_m = metrics(truth[ev], expert[ev], k)
                rows.append(dict(dataset=dataset, domain=domain, arm=arm, draw=draw, protocol=protocol,
                    **metrics(truth[ev], prediction[ev], k),
                    expert_acc=expert_m['acc'], expert_ba=expert_m['ba'],
                    base_acc=base_m['acc'], base_ba=base_m['ba'],
                    selected=int(selected.sum()), cap=int(policy['cap'][draw]),
                    query_rows=int(ev.sum()),
                    observed_support_classes=int(np.unique(truth[policy['support_rows'][draw]]).size),
                    rescued=int(np.sum(selected & (prediction == truth) & (base != truth))),
                    harmed=int(np.sum(selected & (prediction != truth) & (base == truth)))))
    write_json(out/'metrics.json', rows)
    pd.DataFrame([{k: v for k, v in r.items() if k != 'class_recall'} for r in rows]).to_csv(
        out/'metrics.csv', index=False)
    return rows


def summarize(rows, arms=ARMS):
    import pandas as pd

    frame = pd.DataFrame(rows)
    per_domain, macro = [], []
    for (dataset, domain), group in frame.groupby(['dataset', 'domain'], sort=False):
        reference = group[group.arm == 'reference'].set_index('draw')
        for arm in arms:
            own = group[group.arm == arm].set_index('draw')
            deltas = {key: (100*(own[key]-reference[key])).to_list() for key in MEASURES}
            per_domain.append(dict(dataset=dataset, domain=domain, arm=arm,
                **{key: float(own[key].mean()) for key in (*MEASURES, 'selected', 'rescued', 'harmed')},
                delta_vs_reference_pp={key: float(np.mean(value)) for key, value in deltas.items()},
                draw_deltas_pp=deltas,
                positive_ba_draws=int(np.count_nonzero(np.array(deltas['ba']) > 0))))
    for dataset in frame.dataset.unique():
        for arm in arms:
            selected = [r for r in per_domain if r['dataset'] == dataset and r['arm'] == arm]
            macro.append(dict(dataset=dataset, arm=arm,
                **{key: float(np.mean([r[key] for r in selected])) for key in MEASURES},
                delta_vs_reference_pp={key: float(np.mean([r['delta_vs_reference_pp'][key] for r in selected]))
                                       for key in MEASURES},
                positive_ba_domains=sum(r['delta_vs_reference_pp']['ba'] > 0 for r in selected),
                domains=len(selected)))
    return dict(domain_means=per_domain, domain_macro_means=macro,
                aggregation='equal-domain mean of five support-draw means; all metrics are fractions',
                M3='completed' if 'aligned_only' in arms else 'not_run',
                joint='completed' if 'joint' in arms else 'not_run', independent_confirmation=False)


RGB_ARMS = ('reference', 'complement_only', 'aligned_only', 'joint')
RGB_SEED = 20261008
RGB_RECIPE = dict(protocol='retrained_rgb_reference', epochs=12, batch_size=64,
                  input_size=224, storage='uint8 NHWC memmap after deterministic PIL bilinear Resize',
                  augmentation='GPU independent probability-.5 horizontal flip, then ImageNet normalization',
                  precision='bf16 autocast, float32 model parameters and Adam state',
                  optimizer='Adam lr=.001, all network parameters', target_labels_for_encoder=0)


def cache_rgb_images(dataset, root):
    """Stream encoded images out of parquet; do not retain a whole image table."""
    import pyarrow.parquet as pq
    from generic_transfer_data import load_generic_transfer

    data = load_generic_transfer(dataset, DOMAINS[dataset][0])
    directory = root/'rgb_images'/dataset
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(rows=len(data['labels']), parquet_files=data['provenance']['parquet_files'],
                    labels=data['labels'].tolist(), domains=data['domains'].tolist(),
                    n_classes=data['n_classes'])
    if (directory/'COMPLETE.json').exists():
        write_json(directory/'COMPLETE.json', metadata)
        return
    row = 0
    for path in data['provenance']['parquet_files']:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=['image'], use_threads=False):
            for item in batch.to_pydict()['image']:
                encoded = item['bytes'] if isinstance(item, dict) else item
                (directory/f'{row:08d}.image').write_bytes(encoded)
                row += 1
    if row != len(data['labels']):
        raise ValueError('image extraction order/length does not match feature rows')
    write_json(directory/'COMPLETE.json', metadata)
    print(f'RGB IMAGES READY {dataset}: {row}', flush=True)


def cache_resized_rgb(dataset, root):
    """Write a NumPy cache in small sequential batches, not a resident image table."""
    from PIL import Image

    directory = root/'rgb_images'/dataset
    metadata = json.loads((directory/'COMPLETE.json').read_text())
    if (directory/'RGB224_READY.json').exists():
        return
    n = metadata['rows']
    path = directory/'rgb224_uint8.npy'
    mapped = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=(n, 224, 224, 3))
    offset = mapped.offset
    del mapped
    started = time.monotonic()
    with path.open('r+b') as stream:
        stream.seek(offset)
        for lo in range(0, n, 32):
            values = []
            for row in range(lo, min(lo+32, n)):
                with Image.open(directory/f'{row:08d}.image') as image:
                    rgb = image.convert('RGB')
                    if dataset == 'pacs' and rgb.width != rgb.height:
                        raise ValueError('PACS non-square image would not match original Resize(224)')
                    values.append(np.asarray(rgb.resize((224, 224), Image.Resampling.BILINEAR)))
            stream.write(np.stack(values).tobytes())
            if lo % 2048 == 0:
                print(f'RGB UINT8 CACHE {dataset}: {min(lo+32, n)}/{n}', flush=True)
    write_json(directory/'RGB224_READY.json', dict(rows=n, path=str(path), shape=[n, 224, 224, 3],
        dtype='uint8', source_image_order='same parquet order', interpolation='PIL bilinear',
        elapsed_seconds=time.monotonic()-started))
    print(f'RGB UINT8 READY {dataset}: {time.monotonic()-started:.1f}s', flush=True)


def rgb_device_batch(cache, rows, device, *, augment):
    import torch

    images = torch.from_numpy(np.array(cache[rows])).to(device).permute(0, 3, 1, 2)
    if augment:
        flip = (torch.rand(len(rows), device=device) < .5)[:, None, None, None]
        images = torch.where(flip, images.flip(3), images)
    images = images.float().div_(255.)
    mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
    std = images.new_tensor([.229, .224, .225])[None, :, None, None]
    return images.sub_(mean).div_(std).contiguous(memory_format=torch.channels_last)


def load_rgb_metadata(dataset, domain, root):
    saved = json.loads((root/'rgb_images'/dataset/'COMPLETE.json').read_text())
    labels = np.asarray(saved['labels'], dtype=np.int64)
    domains = np.asarray(saved['domains'])
    return dict(labels=labels, domains=domains, n_classes=saved['n_classes'],
                source_rows=np.flatnonzero(domains != domain),
                target_rows=np.flatnonzero(domains == domain))


def bootstrap_rgb(dataset, domain, root, device_name, batch_size, stop_after_epoch=None):
    """Recreate the RGB source encoder recipe, not the lost old numerical state."""
    import torch
    from torch import nn
    from torchvision import models

    out = root/'rgb_retrained'/dataset/domain.replace(' ', '_')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'RGB_BACKBONE_READY.json').exists():
        return
    if batch_size != RGB_RECIPE['batch_size']:
        raise ValueError('all eight RGB folds use the frozen batch size 64')
    recipe_path = root/'RGB_RECIPE.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != RGB_RECIPE:
        raise ValueError('RGB throughput recipe changed; archive old partials before a new reference')
    write_json(recipe_path, RGB_RECIPE)
    directory = root/'rgb_images'/dataset
    if not (directory/'RGB224_READY.json').exists():
        raise FileNotFoundError('run --stage images and --stage resize for this dataset first')
    pixels = np.load(directory/'rgb224_uint8.npy', mmap_mode='r')
    data = load_rgb_metadata(dataset, domain, root)
    device = torch.device(device_name)
    seed = RGB_SEED+DOMAINS[dataset].index(domain)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, data['n_classes'])
    checkpoint = out/'rgb_source_r18.pt'
    if checkpoint.exists():
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    else:
        initial = Path('C:/Users/25791/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
        state = torch.load(initial, map_location='cpu', weights_only=True)
        del state['fc.weight'], state['fc.bias']
        net.load_state_dict(state, strict=False)
    net = net.to(device, memory_format=torch.channels_last)
    write_json(out/'RGB_PROTOCOL.json', dict(
        protocol='retrained_rgb_reference',
        dataset=dataset, target_domain=domain, seed=seed, source_rows=data['source_rows'].tolist(),
        target_rows=data['target_rows'].tolist(), backbone='ImageNet-V1 RGB ResNet18, source supervised',
        bootstrap_epochs=12, bootstrap_optimizer='Adam lr=.001, all network parameters',
        batch_size=batch_size, bootstrap_precision=RGB_RECIPE['precision'], resize='224x224',
        throughput_recipe=RGB_RECIPE,
        old_checkpoint_available=False, old_baseline_exact_replay=False,
        comparison='new source initialization; new matched reference/complement/aligned/joint',
        original_base64_preserved=True, outer_target_labels_for_encoder=0,
        inner_encoder_oof=False, independent_confirmation=False))
    if not checkpoint.exists():
        optimizer = torch.optim.Adam(net.parameters(), lr=.001)
        history = []
        resume_path = out/'bootstrap_resume.pt'
        first_epoch = 0
        if resume_path.exists():
            saved = torch.load(resume_path, map_location='cpu', weights_only=True)
            net.load_state_dict(saved['backbone'])
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['cpu_rng'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            first_epoch, history = saved['epoch'], saved['history']
            del saved
        started = time.monotonic()
        previous_seconds = history[-1]['seconds'] if history else 0.
        for epoch in range(first_epoch, 12):
            epoch_started = time.monotonic()
            net.train()
            order = data['source_rows'][torch.randperm(len(data['source_rows'])).numpy()]
            total, count = 0., 0
            for lo in range(0, len(order), batch_size):
                rows = order[lo:lo+batch_size]
                images = rgb_device_batch(pixels, rows, device, augment=True)
                labels = torch.as_tensor(data['labels'][rows], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    loss = nn.functional.cross_entropy(net(images), labels)
                loss.backward()
                optimizer.step()
                total += loss.detach()*len(rows)
                count += len(rows)
            record = dict(epoch=epoch+1, loss=float(total/count),
                          epoch_seconds=time.monotonic()-epoch_started,
                          seconds=previous_seconds+time.monotonic()-started)
            history.append(record)
            write_json(out/'bootstrap_history.json', history)
            torch.save(dict(epoch=epoch+1, backbone=net.state_dict(), optimizer=optimizer.state_dict(),
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                history=history), resume_path)
            print(f'RGB BOOTSTRAP {dataset}/{domain}: {record}', flush=True)
            if stop_after_epoch is not None and epoch+1 >= stop_after_epoch and epoch+1 < 12:
                print('BOOTSTRAP PAUSED: resume checkpoint saved; full recipe remains 12 epochs', flush=True)
                return
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, checkpoint)
        del optimizer
    net.eval()
    prefix = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                           net.layer1, net.layer2, net.layer3)
    feature_path, prefix_path = out/'initial_extra.npy', out/'rgb_layer3.npy'
    n = len(data['labels'])
    extra = np.lib.format.open_memmap(feature_path, mode='w+', dtype=np.float32, shape=(n, 512))
    spatial = None
    with torch.no_grad():
        for lo in range(0, n, batch_size):
            rows = np.arange(lo, min(lo+batch_size, n))
            images = rgb_device_batch(pixels, rows, device, augment=False)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                values = prefix(images).half()
            if spatial is None:
                spatial = np.lib.format.open_memmap(prefix_path, mode='w+', dtype=np.float16,
                    shape=(n, *values.shape[1:]))
            spatial[rows] = values.cpu().numpy()
            extra[rows] = net.layer4(values.float()).mean((2, 3)).cpu().numpy()
    extra.flush()
    spatial.flush()
    write_json(out/'RGB_BACKBONE_READY.json', dict(checkpoint=str(checkpoint),
        features=str(feature_path), prefix=str(prefix_path), prefix_shape=list(spatial.shape),
        original_checkpoint_reproduced=False, initial_features_from_same_float16_prefix=True))
    print(f'RGB BACKBONE/PREFIX READY {dataset}/{domain}', flush=True)


def geometry_arrays(model, stats):
    return dict(ssl_mean=model['pca_ssl'].mean_, ssl_components=model['pca_ssl'].components_,
                expert_center=model['expert_center'], expert_scale=model['expert_scale'],
                expert_coef=model['expert'].coef_, expert_intercept=model['expert'].intercept_,
                discarded_center=stats['discarded_center'], discarded_scale=stats['discarded_scale'])


def episode_rows(source_rows, domains, held_domain, seed):
    """Both the head-fitting rows and the support/query episode stay in source."""
    held = source_rows[domains[source_rows] == held_domain]
    train = source_rows[domains[source_rows] != held_domain]
    support = np.random.default_rng(seed).choice(len(held), 64, replace=False)
    query = np.setdiff1d(np.arange(len(held)), support)
    return train, held, support, query


def prepare_rgb(dataset, domain, root):
    from generic_transfer_data import load_generic_transfer
    from projection_residual_expert import discarded_coordinates, fit_geometry_statistics
    from run_generic_transfer import fit_gate, fit_heads
    from run_opensarship2_transfer import target_geometry

    out = root/'rgb_retrained'/dataset/domain.replace(' ', '_')
    if (out/'EPISODES_READY.json').exists():
        return
    data = load_generic_transfer(dataset, domain)
    base = data['base_features']
    extra = np.load(out/'initial_extra.npy', mmap_mode='r')
    source, labels, domains = data['source_rows'], data['labels'], data['domains']
    k = data['n_classes']
    oof_b, oof_e = np.empty((len(source), k)), np.empty((len(source), k))
    episodes = []
    (out/'episodes').mkdir(exist_ok=True)
    for index, held_domain in enumerate(np.unique(domains[source])):
        train, held, support, query = episode_rows(source, domains, held_domain, RGB_SEED+index)
        model = fit_heads(base, extra, labels, train, k)
        b, e, _, _ = target_geometry(model, base[held], extra[held])
        pos = np.searchsorted(source, held)
        oof_b[pos], oof_e[pos] = b, e
        stats = fit_geometry_statistics(base, extra, train, model)
        xs = ((model['pca'].transform(base[held])-model['center'])/model['scale']).astype(np.float32)
        bd = ((discarded_coordinates(base[held], model['pca'])-stats['discarded_center'][:2048])/
              stats['discarded_scale'][:2048]).astype(np.float32)
        path = out/'episodes'/f'{index:02d}.npz'
        np.savez(path, rows=held, train_rows=train, support=support, query=query,
                 labels=labels[held], xs=xs, base_discarded=bd, **geometry_arrays(model, stats))
        episodes.append(dict(domain=held_domain, path=str(path), query_rows=len(query)))
        print(f'RGB EPISODE {dataset}/{domain} held={held_domain}', flush=True)
    gate = fit_gate(oof_b, oof_e, labels[source], domains[source])
    model = fit_heads(base, extra, labels, source, k)
    model['gate'] = gate
    with (out/'source_model.pkl').open('wb') as handle:
        pickle.dump(model, handle)
    stats = fit_geometry_statistics(base, extra, source, model)
    np.savez(out/'deployment_geometry.npz', source_rows=source, **stats)
    np.savez(out/'source_oof.npz', rows=source, base=oof_b, expert=oof_e)
    write_json(out/'EPISODES_READY.json', dict(episodes=episodes, classes=k,
        protocol='retrained_rgb_reference',
        class_counts=np.bincount(labels[source], minlength=k).tolist(),
        outer_target_excluded=True, inner_encoder_oof=False, reference_gate_frozen_before_M3=True,
        training=dict(epochs=6, query_batch_size=128, lr=1e-5, weight_decay=1e-4,
                      gradient_clip=5., seed=RGB_SEED)))


def train_rgb(dataset, domain, root, device_name, batch_size):
    import torch
    from torch import nn
    from torchvision.models import resnet18
    from aligned_residual_ops import FixedExpertGeometry, residual_scores
    from joint_residual_ops import ComplementExpertGeometry

    out = root/'rgb_retrained'/dataset/domain.replace(' ', '_')
    if (out/'TRAINING_COMPLETE.json').exists():
        return
    descriptors = json.loads((out/'EPISODES_READY.json').read_text())
    device = torch.device(device_name)
    torch.set_num_threads(2)
    torch.manual_seed(RGB_SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(RGB_SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    net = resnet18(weights=None)
    net.fc = nn.Linear(512, descriptors['classes'])
    net.load_state_dict(torch.load(out/'rgb_source_r18.pt', map_location='cpu', weights_only=True))
    tail = net.layer4
    initial = {k: v.clone() for k, v in tail.state_dict().items()}
    del net
    tail = tail.to(device)
    prefix = np.load(out/'rgb_layer3.npy', mmap_mode='r')
    n_rows = len(prefix)
    device_prefix = None
    if device.type == 'cuda':
        device_prefix = torch.empty(prefix.shape, dtype=torch.float16, device=device)
        for lo in range(0, n_rows, 256):
            chunk = torch.from_numpy(np.array(prefix[lo:lo+256]))
            device_prefix[lo:lo+len(chunk)].copy_(chunk)
        del chunk
        prefix = None
        print(f'RGB PREFIX ON GPU {dataset}/{domain}: '
              f'{device_prefix.numel()*device_prefix.element_size()/1e9:.3f} GB', flush=True)
    counts = torch.tensor(descriptors['class_counts'], dtype=torch.float32, device=device)
    weights = counts.sum()/(len(counts)*counts)
    config = descriptors['training']
    for arm in ('aligned_only', 'joint'):
        directory = out/arm
        directory.mkdir(exist_ok=True)
        if (directory/'TRAINING_COMPLETE.json').exists():
            continue
        tail.load_state_dict(initial)
        optimizer = torch.optim.AdamW(tail.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        cls = ComplementExpertGeometry if arm == 'joint' else FixedExpertGeometry
        history = []
        started = time.monotonic()
        for epoch in range(config['epochs']):
            rng = np.random.default_rng(config['seed']+epoch)
            total, count = 0., 0
            # Cache-based tail learning keeps all BatchNorm running statistics fixed.
            tail.eval()
            for index in rng.permutation(len(descriptors['episodes'])):
                with np.load(descriptors['episodes'][index]['path']) as archive:
                    episode = {key: archive[key] for key in archive.files}
                geometry = cls(episode).to(device).float()
                support, query = episode['support'], rng.permutation(episode['query'])
                sy = torch.tensor(episode['labels'][support], dtype=torch.long, device=device)
                for lo in range(0, len(query), config['query_batch_size']):
                    q = query[lo:lo+config['query_batch_size']]
                    selected = np.r_[support, q]
                    rows = episode['rows'][selected]
                    if device_prefix is not None:
                        pixels = device_prefix[torch.as_tensor(rows, device=device)].float()
                    else:
                        pixels = torch.from_numpy(np.array(prefix[rows])).to(device).float()
                    xs = torch.as_tensor(episode['xs'][selected], device=device)
                    y = torch.tensor(episode['labels'][q], dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                        features = tail(pixels).mean((2, 3))
                    if arm == 'joint':
                        bd = torch.as_tensor(episode['base_discarded'][selected], device=device)
                        prior, phi = geometry(xs, features.float(), bd)
                    else:
                        prior, phi = geometry(xs, features.float())
                    scores = residual_scores(phi[:64], prior[:64], sy, phi[64:], prior[64:])
                    per_row = nn.functional.cross_entropy(scores, y, reduction='none')
                    loss = (per_row*weights[y]).sum()/weights[y].sum()
                    if not torch.isfinite(loss):
                        raise ValueError(f'nonfinite {dataset}/{domain}/{arm} loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(tail.parameters(), config['gradient_clip'])
                    optimizer.step()
                    total += float(loss.detach())*len(q)
                    count += len(q)
                del geometry, episode
            record = dict(epoch=epoch+1, loss=total/count, query_rows=count,
                          seconds=time.monotonic()-started)
            history.append(record)
            write_json(directory/'training_history.json', history)
            print(f'RGB TAIL {dataset}/{domain}/{arm}: {record}', flush=True)
        torch.save({k: v.cpu() for k, v in tail.state_dict().items()}, directory/'layer4.pt')
        features = np.lib.format.open_memmap(directory/'extra.npy', mode='w+',
                                              dtype=np.float32, shape=(n_rows, 512))
        tail.eval()
        with torch.no_grad():
            for lo in range(0, n_rows, batch_size):
                if device_prefix is not None:
                    pixels = device_prefix[lo:lo+batch_size].float()
                else:
                    pixels = torch.from_numpy(np.array(prefix[lo:lo+batch_size])).to(device).float()
                features[lo:lo+len(pixels)] = tail(pixels).mean((2, 3)).cpu().numpy()
        features.flush()
        del features
        write_json(directory/'TRAINING_COMPLETE.json', dict(epochs=config['epochs'],
            prefix_storage='GPU exact float16 cache' if device_prefix is not None else 'CPU cache',
            source_query_loss_only=True, target_labels_for_encoder=0,
            actual_support_solver_in_gradient=True, final_epoch_only=True))
    write_json(out/'TRAINING_COMPLETE.json', dict(arms=['aligned_only', 'joint'], rgb_only=True))


def evaluate_rgb(dataset, domain, root):
    from generic_transfer_data import load_generic_transfer
    from projection_residual_expert import build_support_kernels, freeze_kernel_draw
    from run_opensarship2_transfer import target_geometry

    out = root/'rgb_retrained'/dataset/domain.replace(' ', '_')
    data = load_generic_transfer(dataset, domain)
    target = data['target_rows']
    base = data['base_features'][target]
    with (out/'source_model.pkl').open('rb') as handle:
        model = pickle.load(handle)
    with np.load(out/'deployment_geometry.npz') as saved:
        stats = {key: saved[key] for key in saved.files}
    with np.load(PREVIOUS[dataset]/domain.replace(' ', '_')/'policies.npz') as saved:
        old = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(old['target_rows'], target)
    reference_rows = np.unique(old['support_rows'])
    for arm in RGB_ARMS:
        path = out/'initial_extra.npy' if arm in ARMS else out/arm/'extra.npy'
        extra = np.load(path, mmap_mode='r')[target]
        _, prior, _, _ = target_geometry(model, base, extra)
        kernels = build_support_kernels(base, extra, model, stats, reference_rows)
        policies = []
        for draw, support in enumerate(old['support_rows']):
            columns = np.searchsorted(reference_rows, support)
            kernel = kernels['pca'][:, columns]+1.
            if arm in ('complement_only', 'joint'):
                kernel = .5*kernels['pca'][:, columns]+.5*kernels['discarded'][:, columns]+1.
            policy = freeze_kernel_draw(old['base_scores'][draw], prior, kernel, support,
                data['labels'][target[support]], model['gate'], target)
            np.testing.assert_array_equal(policy['eval_mask'], old['eval_masks'][draw])
            if policy['selected'][~policy['eval_mask']].any() or policy['selected'].sum() > old['caps'][draw]:
                raise ValueError('RGB support/budget violation')
            policies.append(policy)
        np.savez_compressed(out/f'{arm}_policies.npz',
            **{key: np.stack([p[key] for p in policies]) for key in policies[0]},
            support_rows=old['support_rows'], target_rows=target)
    write_json(out/'POLICIES_SAVED.json', dict(arms=RGB_ARMS, draws=5,
        original_base64_preserved=True, old_expert_reference_reproduced=False,
        new_rgb_bootstrap_reference=True, all_policies_before_query_readout=True))
    rows = readout_fold(dataset, domain, out, RGB_ARMS, 'retrained_rgb_reference')
    summary = summarize(rows, RGB_ARMS)
    summary['new_rgb_bootstrap_reference'] = True
    summary['protocol'] = 'retrained_rgb_reference'
    summary['old_reference_exact_replay'] = False
    write_json(out/'summary.json', summary)
    summarize_rgb_dataset(dataset, root)
    print(json.dumps(summary['domain_macro_means']), flush=True)


def summarize_rgb_dataset(dataset, root):
    import pandas as pd

    directory = root/'rgb_retrained'/dataset
    rows, completed, timings = [], [], {}
    for domain in DOMAINS[dataset]:
        folder = directory/domain.replace(' ', '_')
        if not (folder/'summary.json').exists():
            continue
        own = json.loads((folder/'summary.json').read_text())
        if own['protocol'] != 'retrained_rgb_reference':
            raise ValueError('cannot mix old-reference and retrained-reference results')
        rows.extend(json.loads((folder/'metrics.json').read_text()))
        completed.append(domain)
        histories = {'bootstrap': folder/'bootstrap_history.json',
                     'aligned_only': folder/'aligned_only/training_history.json',
                     'joint': folder/'joint/training_history.json'}
        timings[domain] = {arm: json.loads(path.read_text())[-1]['seconds']
                          for arm, path in histories.items()}
    summary = summarize(rows, RGB_ARMS)
    summary.update(protocol='retrained_rgb_reference', completed_domains=completed,
        expected_domains=list(DOMAINS[dataset]), dataset_complete=len(completed) == len(DOMAINS[dataset]),
        old_reference_exact_replay=False, original_base64_preserved=True,
        logged_training_seconds=timings,
        timing_scope='Logged training duration only; excludes imports/preparation/prefix/extraction/readout')
    write_json(directory/'summary.json', summary)
    write_json(directory/'metrics.json', rows)
    pd.DataFrame([{k: v for k, v in r.items() if k != 'class_recall'} for r in rows]).to_csv(
        directory/'metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('all', *DOMAINS), default='all')
    parser.add_argument('--out', type=Path, default=ROOT/'module_completion_20261008/generic')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--stage', choices=('m2', 'images', 'resize', 'bootstrap', 'prepare', 'train', 'evaluate'), default='m2')
    parser.add_argument('--domain')
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--bootstrap-stop-after', type=int,
                        help='Pause after this epoch for GPU scheduling; resume still totals 12 epochs')
    args = parser.parse_args()
    if args.stage in ('m2', 'prepare', 'evaluate'):
        # Register CPU BLAS/OpenMP libraries before setting their thread limits.
        import run_generic_transfer

    if args.stage != 'm2':
        if args.dataset == 'all' or (args.stage not in ('images', 'resize') and args.domain not in DOMAINS[args.dataset]):
            parser.error('RGB stages require one --dataset and a valid --domain (except images)')
        if args.stage in ('bootstrap', 'train') and args.device is None:
            parser.error('GPU/resource coordination is external: explicitly choose --device')
        if args.stage in ('bootstrap', 'train') and args.device == 'cuda':
            (args.out/'GPU_RELEASED.txt').unlink(missing_ok=True)
        with threadpool_limits(limits=args.threads):
            if args.stage == 'images':
                cache_rgb_images(args.dataset, args.out)
            elif args.stage == 'resize':
                cache_resized_rgb(args.dataset, args.out)
            elif args.stage == 'bootstrap':
                bootstrap_rgb(args.dataset, args.domain, args.out, args.device, args.batch_size,
                              args.bootstrap_stop_after)
            elif args.stage == 'prepare':
                prepare_rgb(args.dataset, args.domain, args.out)
            elif args.stage == 'train':
                train_rgb(args.dataset, args.domain, args.out, args.device, args.batch_size)
            elif args.stage == 'evaluate':
                evaluate_rgb(args.dataset, args.domain, args.out)
        return
    datasets = tuple(DOMAINS) if args.dataset == 'all' else (args.dataset,)
    import pandas as pd

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'PROTOCOL.json', dict(date='2026-10-08', datasets=datasets,
        arms=ARMS, threads=args.threads, no_gpu=True, supports='exact saved five draws',
        source_models='exact previous source model, no head/encoder refit',
        all_domain_policies_before_readout=True, M3='not_run', joint='not_run'))
    with threadpool_limits(limits=args.threads):
        for dataset in datasets:
            for domain in DOMAINS[dataset]:
                freeze_fold(dataset, domain, args.out/dataset/domain.replace(' ', '_'))
        rows = []
        for dataset in datasets:
            for domain in DOMAINS[dataset]:
                rows.extend(readout_fold(dataset, domain, args.out/dataset/domain.replace(' ', '_')))
        summary = summarize(rows)
    write_json(args.out/'metrics.json', rows)
    write_json(args.out/'summary.json', summary)
    pd.DataFrame([{k: v for k, v in r.items() if k != 'class_recall'} for r in rows]).to_csv(
        args.out/'metrics.csv', index=False)
    for result in summary['domain_macro_means']:
        print(json.dumps(result), flush=True)
    print('GENERIC M2 COMPLETION DONE; M3 AND JOINT REMAIN NOT RUN', flush=True)


if __name__ == '__main__':
    main()
