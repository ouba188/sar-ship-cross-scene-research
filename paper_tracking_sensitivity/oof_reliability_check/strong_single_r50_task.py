"""Bounded source-selected R50 tail pilot; never starts CUDA in prepare/tests.

One shared encoder, C4 average, normalized embedding, source prior, support ridge.
No source prior or inherited geometry is fitted on held source validation ports.
"""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from aligned_residual_ops import residual_scores

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
SOURCE = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
OUT = ROOT/'strong_single_r50_task_20261010'
CACHE = Path('D:/SAR_completion_cache_20261010')
TASK = ROOT/'learned_frequency_residual_20261009/tasks/oss2'
CLASSES = 8
SPLIT_SEED = 20261010


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def valid_identity(series):
    text = series.fillna('').astype(str).str.strip()
    return text.mask(text.isin(['', '0', '0.0', 'nan', 'None']), '')


def grouped_split(frame, allowed_rows, seed=SPLIT_SEED, validation_ports=4):
    """Pick whole source ports, then purge shared vessels/acquisitions from train."""
    allowed_rows = np.asarray(allowed_rows, dtype=np.int64)
    counts = frame.iloc[allowed_rows].groupby('port').size()
    eligible = sorted(counts[counts >= 128].index.astype(str))
    if len(eligible) <= validation_ports:
        raise ValueError('Not enough eligible source ports for grouped selection')
    held = sorted(np.random.default_rng(seed).choice(eligible, validation_ports, replace=False))
    is_val = frame.iloc[allowed_rows].port.isin(held).to_numpy()
    val, train = allowed_rows[is_val], allowed_rows[~is_val]
    before = train.copy()
    audit = {}
    for column in ('mmsi', 'product'):
        if column not in frame:
            audit[column] = dict(available=False)
            continue
        values = valid_identity(frame[column])
        identities = set(values.iloc[val]) - {''}
        keep = ~values.iloc[train].isin(identities).to_numpy()
        audit[column] = dict(available=True, validation_identities=len(identities), removed=int((~keep).sum()))
        train = train[keep]
    if not len(train) or not len(val):
        raise ValueError('Empty inner split after identity purge')
    return train, val, dict(validation_ports=held, train_before_purge=len(before),
                            train_after_purge=len(train), validation_rows=len(val), identity_purge=audit)


def target_design(path=TASK/'target.npz'):
    # Loading this whitelist never accesses target query labels or expert scores.
    with np.load(path) as saved:
        data = {key: saved[key] for key in ('source_rows', 'rows', 'ids', 'support', 'support_labels', 'eval_masks')}
    if data['support'].shape != (5, 64) or data['support_labels'].shape != (5, 64):
        raise ValueError('Expected the established five 64-label support draws')
    for draw, support in enumerate(data['support']):
        if len(np.unique(support)) != 64 or data['eval_masks'][draw, support].any():
            raise ValueError('Duplicate support or support/query overlap')
    return data


def task_inputs(folder, frame, data):
    info = json.loads((folder/'PREPARED.json').read_text())
    dataset = info['dataset']
    if dataset == 'self24ports':
        pixels_path = CACHE/'source_inputs128.npy'
        ids = frame.instance_id.to_numpy(str)
        if np.intersect1d(data['source_rows'], data['rows']).size:
            raise ValueError('Outer source/query row overlap')
        if set(frame.iloc[data['source_rows']].port) & set(frame.iloc[data['rows']].port):
            raise ValueError('Held port is present in outer supervised source')
    else:
        source_dir = 'opensarship2_transfer_20261007' if dataset == 'oss2' else 'f0921_transfer_20261007'
        with np.load(ROOT/source_dir/'source_split.npz') as saved:
            np.testing.assert_array_equal(data['source_rows'], saved['source_rows'])
        target_dir = 'opensarship2_transfer_20261007' if dataset == 'oss2' else 'f0921_detector_recrop_20261007'
        ids = pd.read_csv(ROOT/target_dir/'target_manifest.csv', usecols=['sample_id']).sample_id.to_numpy(str)
        pixels_path = CACHE/'oss2_inputs128.npy' if dataset == 'oss2' else ROOT/target_dir/'inputs128.npy'
    np.testing.assert_array_equal(ids[data['rows']], data['ids'])
    pixels = np.load(pixels_path, mmap_mode='r')
    if pixels.shape != (len(ids), 2, 128, 128) or pixels.dtype != np.uint8:
        raise ValueError(f'Wrong paired pixel cache for {dataset}')
    return info, pixels


def prepare(out, folder=TASK):
    frame = pd.read_csv(SOURCE/'manifest.csv', keep_default_na=False,
                        usecols=['instance_id', 'port', 'class_id', 'mmsi', 'product'],
                        dtype={'mmsi': str, 'product': str})
    data = target_design(folder/'target.npz')
    info, _ = task_inputs(folder, frame, data)
    np.testing.assert_array_equal(np.load(CACHE/'source_ids.npy'), frame.instance_id.to_numpy(str))
    for name, count in [('source', len(frame))]:
        pixels = np.load(CACHE/f'{name}_inputs128.npy', mmap_mode='r')
        if pixels.shape != (count, 2, 128, 128) or pixels.dtype != np.uint8:
            raise ValueError(f'Wrong paired pixel cache for {name}')
    train, val, audit = grouped_split(frame, data['source_rows'])
    audit.update(dataset=info['dataset'], task=info['task'])
    audit['inner_train_class_counts'] = np.bincount(frame.iloc[train].class_id, minlength=8).tolist()
    audit['inner_val_class_counts'] = np.bincount(frame.iloc[val].class_id, minlength=8).tolist()
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'source_split.npz', allowed=data['source_rows'], inner_train=train, inner_val=val)
    write_json(out/'SPLIT.json', audit)
    print(json.dumps(audit), flush=True)
    return frame, data, train, val


def geometry(embedding):
    unit = F.normalize(embedding.float(), dim=1, eps=1e-12)
    return torch.cat((unit*np.sqrt(32.), unit.new_ones((len(unit), 1))), dim=1)


def support_scores(support_phi, labels, query_phi, strength=8., support_prior=None, query_prior=None):
    if support_prior is None:
        support_prior = support_phi.new_zeros((len(support_phi), CLASSES))
    if query_prior is None:
        query_prior = query_phi.new_zeros((len(query_phi), CLASSES))
    return residual_scores(support_phi, support_prior, labels, query_phi, query_prior, strength)


def fit_source_prior(raw, labels, rows, alpha=1.):
    """Balanced ridge, unpenalized intercept; only explicitly supplied source rows."""
    from scipy.linalg import cho_factor, cho_solve
    counts = np.bincount(labels[rows], minlength=CLASSES)
    if (counts == 0).any():
        raise ValueError('Source prior requires all eight source classes')
    weights = len(rows)/(CLASSES*counts)
    dimension = raw.shape[1]+1
    gram = np.zeros((dimension, dimension), dtype=np.float64)
    rhs = np.zeros((dimension, CLASSES), dtype=np.float64)
    for lo in range(0, len(rows), 512):
        batch = rows[lo:lo+512]
        phi = geometry(torch.from_numpy(np.array(raw[batch], dtype=np.float32))).numpy().astype(np.float64)
        weighted = phi*weights[labels[batch], None]
        gram += phi.T@weighted
        rhs += weighted.T@(2*np.eye(CLASSES)[labels[batch]]-1)
    gram.flat[::dimension+1] += np.r_[np.full(dimension-1, alpha), 0.]
    return cho_solve(cho_factor(gram, lower=True, check_finite=False), rhs, check_finite=False).T


def initialize_prior(net, raw, labels, rows):
    coef = fit_source_prior(raw, labels, rows)
    with torch.no_grad():
        net.prior.weight.copy_(torch.as_tensor(coef, dtype=net.prior.weight.dtype,
                                              device=net.prior.weight.device))


@torch.no_grad()
def cache_alignment(net, pixels, frozen_features, inner_rows, device, path):
    """Bounded source-only wiring check before any validation or optimization."""
    rows = np.asarray(inner_rows[:16], dtype=np.int64)
    net.eval()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        actual = net(pixel_tensor(pixels, rows, device)).cpu()
    cached = geometry(torch.from_numpy(np.array(frozen_features[rows], dtype=np.float32)))
    delta = actual[:, :-1]-cached[:, :-1]
    weight = net.prior.weight.detach().cpu()
    actual_logits, cached_logits = F.linear(actual, weight), F.linear(cached, weight)
    if not torch.isfinite(actual).all() or not torch.isfinite(actual_logits).all():
        raise RuntimeError('Nonfinite actual initialization geometry or source-prior scores')
    write_json(path, dict(source_rows=rows.tolist(), rows_used=len(rows),
        geometry_columns='sqrt32 normalized GAP2048; constant coordinate excluded from difference',
        geometry_relative_l2=float(delta.norm()/cached[:, :-1].norm().clamp_min(1e-12)),
        geometry_max_abs_difference=float(delta.abs().max()),
        source_prior_logit_rms_difference=float((actual_logits-cached_logits).square().mean().sqrt()),
        source_prior_argmax_agreement=float((actual_logits.argmax(1) == cached_logits.argmax(1)).float().mean()),
        actual_path='Current one-R50 paired pixels, C4, bf16 forward, float32 geometry',
        cached_path='Existing frozen C4 pooled2048 used only to initialize source prior',
        prior_fit_scope='Inner source training rows only', query_labels_used=False,
        interpretation='Measured source initialization alignment; no assumption that differences are only rounding'))


def trainable_parameters(net):
    return list(net.layer4.parameters())+list(net.prior.parameters())


class SingleR50(nn.Module):
    """Exactly one R50; frozen prefix, adapted layer4, no second frozen path."""

    def __init__(self, pretrained=True, polarization='sum'):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50
        if pretrained:
            expected = Path(torch.hub.get_dir())/'checkpoints'/Path(ResNet50_Weights.IMAGENET1K_V2.url).name
            if not expected.exists():
                raise FileNotFoundError(expected)
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        old = net.conv1
        net.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            common = old.weight.mean(1, keepdim=True)*1.5
            if polarization == 'sum':
                net.conv1.weight.copy_(common.repeat(1, 2, 1, 1))
            elif polarization == 'sum_difference':
                difference = .5*(old.weight[:, :1]-old.weight[:, 1:2])
                net.conv1.weight.copy_(torch.cat((common+difference, common-difference), dim=1))
            else:
                raise ValueError(f'Unknown polarization adapter: {polarization}')
        self.prefix = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                    net.layer1, net.layer2, net.layer3).requires_grad_(False)
        self.layer4 = net.layer4
        self.prior = nn.Linear(2049, CLASSES, bias=False)
        nn.init.zeros_(self.prior.weight)

    def train(self, mode=True):
        super().train(mode)
        self.prefix.eval()
        for module in self.layer4.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def encode(self, pixels, microbatch=16):
        outputs = []
        for block in pixels.split(microbatch):
            views = []
            for rotation in range(4):
                with torch.no_grad():
                    prefix = self.prefix(torch.rot90(block, rotation, (-2, -1)))
                if self.training and torch.is_grad_enabled():
                    # Retain gradients only through the trainable tail for four views.
                    z = checkpoint(self.layer4, prefix, use_reentrant=False)
                else:
                    z = self.layer4(prefix)
                views.append(z.float().mean((-2, -1)))
            outputs.append(torch.stack(views).mean(0))
        return torch.cat(outputs)

    def forward(self, pixels, microbatch=16):
        return geometry(self.encode(pixels, microbatch))


def port_pools(frame, rows, minimum=96):
    pools = {}
    for port, part in frame.iloc[rows].groupby('port'):
        if len(part) >= minimum:
            pools[str(port)] = part.index.to_numpy(np.int64)
    if not pools:
        raise ValueError('No source ports support disjoint 64-label episodes')
    return pools


def episode_rows(pool, rng, query=32):
    rows = rng.choice(pool, 64+query, replace=False)
    return rows[:64], rows[64:]


def pixel_tensor(pixels, rows, device):
    return (torch.as_tensor(np.array(pixels[rows]), device=device, dtype=torch.float32)/255.-.5)/.25


def balanced_loss(scores, labels):
    present, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    weights = len(labels)/(len(present)*counts[inverse].float())
    return (F.cross_entropy(scores*4., labels, reduction='none')*weights).mean()


def metrics(truth, prediction):
    classes = np.unique(truth)
    return dict(oa=100*float(np.mean(truth == prediction)),
                ba=100*float(np.mean([np.mean(prediction[truth == c] == c) for c in classes])))


def validation_episodes(frame, rows):
    rng = np.random.default_rng(SPLIT_SEED+1)
    episodes = []
    for port, pool in sorted(port_pools(frame, rows, minimum=128).items()):
        for draw in range(2):
            support, query = episode_rows(pool, rng, query=min(128, len(pool)-64))
            episodes.append(dict(port=port, draw=draw, support=support, query=query))
    return episodes


@torch.no_grad()
def validate(net, pixels, labels, episodes, device):
    net.eval()
    results = []
    for episode in episodes:
        support, query = episode['support'], episode['query']
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            phi = net(pixel_tensor(pixels, np.r_[support, query], device))
        prior = net.prior(phi)
        scores = support_scores(phi[:64], torch.as_tensor(labels[support], device=device), phi[64:],
                                support_prior=prior[:64], query_prior=prior[64:])
        results.append(dict(port=episode['port'], draw=episode['draw'], **metrics(labels[query], scores.argmax(1).cpu().numpy())))
    return float(np.mean([row['ba'] for row in results])), results


def train_epoch(net, pixels, labels, pools, optimizer, rng, device, episodes):
    net.train()
    names = sorted(pools)
    losses = []
    start = time.perf_counter()
    for step in range(episodes):
        port = names[int(rng.integers(len(names)))]
        support, query = episode_rows(pools[port], rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            phi = net(pixel_tensor(pixels, np.r_[support, query], device))
        prior = net.prior(phi)
        scores = support_scores(phi[:64], torch.as_tensor(labels[support], device=device), phi[64:],
                                support_prior=prior[:64], query_prior=prior[64:])
        query_labels = torch.as_tensor(labels[query], device=device)
        loss = balanced_loss(scores, query_labels)+.25*balanced_loss(prior[64:], query_labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite episodic loss')
        loss.backward()
        grad = nn.utils.clip_grad_norm_(trainable_parameters(net), 5., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 8 == 0:
            print(f'episode {step+1}/{episodes} loss={losses[-1]:.5f} grad={float(grad):.4f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    return float(np.mean(losses))


def train(args, seed):
    out = args.out/f'seed_{seed}'
    if (out/'TRAINING_COMPLETE.json').exists():
        raise ValueError('Completed run already exists; use another output for another protocol')
    folder = getattr(args, 'task', TASK)
    frame, data, inner_train, inner_val = prepare(args.out, folder)
    info = json.loads((folder/'PREPARED.json').read_text())
    config = dict(seed=seed, source_split_seed=SPLIT_SEED, dataset=info['dataset'], task=info['task'], epochs=args.epochs,
                  episodes_per_epoch=args.episodes, lr=1e-4, weight_decay=1e-4,
                  backbone='ImageNetV2 R50, paired VV/VH, layer4 only, BN statistics fixed',
                  readout='sqrt(32)*L2(C4 mean GAP2048) + intercept; balanced residual ridge lambda8; trainable source prior',
                  source_prior='Linear2049x8, initialized balanced source ridge alpha1/unpenalized intercept from existing frozen C4 pooled2048; allowed fit rows only',
                  objective='balanced residual-query CE + 0.25 balanced prior-query CE; both fixed score multiplier4; 64 support + 32 query per source-port episode',
                  source_validation='4 whole source ports; shared MMSI/product removed from inner train; 2 fixed draws/port',
                  selection='highest source-validation macro port/draw BA; ties earliest including epoch0',
                  refit='restart pretrained layer4 and balanced prior fitted on all allowed source rows; source-selected episode count; same LR schedule',
                  inference='one paired-channel R50, four rotation forwards, no second R50 or SSL model',
                  target_support_draws=5, support_labels_per_draw=64,
                  query_labels_used_for_selection=False, pilot_only=True,
                  limitations=['tail adaptation not full optimization', 'no SAR SSL pretraining parity',
                               'compact source-prior/residual operator, not full current dual modules',
                               'cache/current source geometry alignment measured before validation; mismatch cause not assumed',
                               'bounded source validation, not untouched confirmation'])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'PROTOCOL.json', config)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA; prepare and tests do not')
    torch.cuda.manual_seed_all(seed)
    labels = frame.class_id.to_numpy(np.int64)
    pixels = np.load(CACHE/'source_inputs128.npy', mmap_mode='r')
    net = SingleR50().to(device)
    initial = {key: value.detach().cpu().clone() for key, value in net.layer4.state_dict().items()}
    frozen_features = np.load(ROOT/'c4_base_trial_20261009/pooled_c4.npy', mmap_mode='r')
    if len(frozen_features) != len(frame):
        raise ValueError('Source initialization feature rows mismatch')
    initialize_prior(net, frozen_features, labels, inner_train)
    cache_alignment(net, pixels, frozen_features, inner_train, device, out/'CACHE_ALIGNMENT.json')
    episodes = validation_episodes(frame, inner_val)
    np.savez(out/'validation_episodes.npz', **{f'{i}_{k}': e[k] for i, e in enumerate(episodes) for k in ('support', 'query')})
    pools = port_pools(frame, inner_train)
    optimizer = torch.optim.AdamW(trainable_parameters(net), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    score, detail = validate(net, pixels, labels, episodes, device)
    history = [dict(epoch=0, source_validation_ba=score, validation=detail)]
    best, best_epoch = score, 0
    torch.save(dict(layer4=initial, prior={k: v.detach().cpu() for k, v in net.prior.state_dict().items()}, epoch=0), out/'selected_inner.pt')
    write_json(out/'curve.json', history)
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(net, pixels, labels, pools, optimizer, rng, device, args.episodes)
        scheduler.step()
        score, detail = validate(net, pixels, labels, episodes, device)
        history.append(dict(epoch=epoch, source_loss=loss, source_validation_ba=score, validation=detail))
        if score > best+1e-10:
            best, best_epoch = score, epoch
            torch.save(dict(layer4={k: v.detach().cpu() for k, v in net.layer4.state_dict().items()},
                            prior={k: v.detach().cpu() for k, v in net.prior.state_dict().items()}, epoch=epoch), out/'selected_inner.pt')
        write_json(out/'curve.json', history)
        print(json.dumps({k: v for k, v in history[-1].items() if k != 'validation'}), flush=True)
    # Refit only after the source-selected epoch is frozen; no target image/label access.
    write_json(out/'SELECTION.json', dict(epoch=best_epoch, source_validation_ba=best,
               best_at_budget_boundary=best_epoch == args.epochs, query_labels_used=False))
    inner_change = sum(float((net.layer4.state_dict()[k].detach().cpu()-v).float().square().sum()) for k, v in initial.items())**.5
    if inner_change == 0:
        raise RuntimeError('Source episode training did not update the tail')
    net.layer4.load_state_dict(initial)
    initialize_prior(net, frozen_features, labels, data['source_rows'])
    frozen_prior = {k: v.detach().cpu().clone() for k, v in net.prior.state_dict().items()}
    optimizer = torch.optim.AdamW(trainable_parameters(net), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    rng = np.random.default_rng(seed)
    final_curve = []
    for epoch in range(1, best_epoch+1):
        loss = train_epoch(net, pixels, labels, port_pools(frame, data['source_rows']), optimizer, rng, device, args.episodes)
        scheduler.step()
        final_curve.append(dict(epoch=epoch, source_loss=loss))
        write_json(out/'refit_curve.json', final_curve)
    change = sum(float((net.layer4.state_dict()[k].detach().cpu()-v).float().square().sum()) for k, v in initial.items())**.5
    prior_change = float((net.prior.weight.detach().cpu()-frozen_prior['weight']).norm())
    if best_epoch > 0 and change == 0:
        raise RuntimeError('Selected adapted tail did not change')
    torch.save(dict(layer4={k: v.detach().cpu() for k, v in net.layer4.state_dict().items()},
                    prior={k: v.detach().cpu() for k, v in net.prior.state_dict().items()}, frozen_prior=frozen_prior,
                    selected_epoch=best_epoch, source_rows=data['source_rows'], seed=seed), out/'final_tail.pt')
    write_json(out/'TRAINING_COMPLETE.json', dict(seconds=time.perf_counter()-started, selected_epoch=best_epoch,
        source_validation_ba=best, update_l2=change, prior_update_l2=prior_change,
        inner_update_l2=inner_change, seeds_completed_here=1,
        frozen_prefix_parameters=sum(p.numel() for p in net.prefix.parameters()),
        trainable_parameters=sum(p.numel() for p in trainable_parameters(net)),
        peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20, query_labels_used=False,
        convergence_not_established=True, best_at_budget_boundary=best_epoch == args.epochs))


@torch.no_grad()
def evaluate(args, seed):
    out = args.out/f'seed_{seed}'
    if not (out/'TRAINING_COMPLETE.json').exists():
        raise ValueError('Complete source selection/refit before target evaluation')
    torch.set_num_threads(2)
    folder = getattr(args, 'task', TASK)
    data = target_design(folder/'target.npz')
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'port'])
    info, pixels = task_inputs(folder, frame, data)
    state = torch.load(out/'final_tail.pt', map_location='cpu', weights_only=False)
    np.testing.assert_array_equal(state['source_rows'], data['source_rows'])
    n = len(data['ids'])
    device = torch.device('cuda')
    net = SingleR50().to(device).eval()
    net.prior.load_state_dict(state['frozen_prior'])
    arms = ('frozen_single_same_readout', 'task_adapted_single_same_readout')
    predictions = np.empty((5, 2, n), np.int8)
    margins = np.empty((5, 2, n), np.float32)
    times = []
    for ai, arm in enumerate(arms):
        if ai:
            net.layer4.load_state_dict(state['layer4'])
            net.prior.load_state_dict(state['prior'])
        shared_frozen = args.out/'shared_frozen_phi.npy'
        torch.cuda.synchronize()
        started = time.perf_counter()
        reused = ai == 0 and shared_frozen.exists()
        if reused:
            phi = torch.from_numpy(np.load(shared_frozen, mmap_mode='c'))
            if phi.shape != (n, 2049) or phi.dtype != torch.float32:
                raise ValueError('Shared frozen geometry has wrong shape or precision')
        else:
            phi = torch.empty((n, 2049), dtype=torch.float32)
            for lo in range(0, n, 32):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = net(pixel_tensor(pixels, data['rows'][lo:min(lo+32, n)], device))
                phi[lo:lo+len(z)] = z.cpu()
            if ai == 0:
                np.save(shared_frozen, phi.numpy())
        torch.cuda.synchronize()
        times.append(dict(arm=arm, extraction_seconds=time.perf_counter()-started,
                          reused_frozen_features=reused, actual_forwards_per_image=0 if reused else 4))
        prior = F.linear(phi, net.prior.weight.detach().cpu())
        for draw, support in enumerate(data['support']):
            for lo in range(0, n, 1024):
                scores = support_scores(phi[support], torch.as_tensor(data['support_labels'][draw]).long(), phi[lo:lo+1024],
                                        support_prior=prior[support], query_prior=prior[lo:lo+1024])
                predictions[draw, ai, lo:lo+len(scores)] = scores.argmax(1).numpy()
                top = scores.topk(2, dim=1).values
                margins[draw, ai, lo:lo+len(scores)] = (top[:, 0]-top[:, 1]).numpy()
        np.save(out/f'{arm}_phi.npy', phi.numpy())
    np.savez_compressed(out/'predictions.npz', predictions=predictions, margins=margins, arms=np.array(arms), **data)
    write_json(out/'extraction_time.json', times)
    # Query labels become visible only after source selection and target predictions are persisted.
    if info['dataset'] == 'self24ports':
        truth_frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'class_id']).iloc[data['rows']]
        np.testing.assert_array_equal(truth_frame.instance_id.to_numpy(str), data['ids'])
    else:
        target_dir = 'opensarship2_transfer_20261007' if info['dataset'] == 'oss2' else 'f0921_detector_recrop_20261007'
        truth_frame = pd.read_csv(ROOT/target_dir/'target_manifest.csv', usecols=['sample_id', 'class_id']).iloc[data['rows']]
        np.testing.assert_array_equal(truth_frame.sample_id.to_numpy(str), data['ids'])
    truth = truth_frame.class_id.to_numpy()
    records, classes, changes = [], [], []
    for draw, mask in enumerate(data['eval_masks']):
        y = truth[mask]
        for ai, arm in enumerate(arms):
            p = predictions[draw, ai, mask]
            records.append(dict(seed=seed, draw=draw, arm=arm, n=len(y), **metrics(y, p)))
            for c in np.unique(y):
                classes.append(dict(seed=seed, draw=draw, arm=arm, class_id=int(c), n=int((y == c).sum()), recall=100*float(np.mean(p[y == c] == c))))
        old, new = predictions[draw, 0, mask] == y, predictions[draw, 1, mask] == y
        for c in np.unique(y):
            subset = y == c
            changes.append(dict(seed=seed, draw=draw, class_id=int(c), n=int(subset.sum()),
                                rescue=int((~old & new & subset).sum()), harm=int((old & ~new & subset).sum())))
    table = pd.DataFrame(records)
    table.to_csv(out/'metrics.csv', index=False)
    table.groupby('arm')[['oa', 'ba']].mean().to_csv(out/'summary.csv')
    pd.DataFrame(classes).to_csv(out/'per_class.csv', index=False)
    pd.DataFrame(changes).to_csv(out/'rescues_harms.csv', index=False)
    write_json(out/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], seed=seed, network_seeds=1, support_draws=5,
        selected_epoch=int(state['selected_epoch']), same_operator_frozen_control=True,
        strong_single_qualification_complete=False, other_datasets_completed=False))
    print(table.groupby('arm')[['oa', 'ba']].mean().to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'train', 'evaluate', 'all'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--task', type=Path, default=TASK)
    parser.add_argument('--seeds', nargs='+', type=int, default=[20261010])
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--episodes', type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.episodes < 1:
        parser.error('epochs and episodes must be positive')
    if args.stage == 'prepare':
        prepare(args.out, args.task)
        return
    for seed in args.seeds:
        if args.stage in ('train', 'all'):
            train(args, seed)
        if args.stage in ('evaluate', 'all'):
            evaluate(args, seed)


if __name__ == '__main__':
    main()
