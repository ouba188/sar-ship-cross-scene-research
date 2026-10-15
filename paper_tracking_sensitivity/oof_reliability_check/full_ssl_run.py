"""Rebuild complete external readout operators over the matched G/D encoders."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import gc
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from conditional_direction_residual import rank_conditions
from general_domain_ssl import (OUT as PILOT, CACHE as OLD_CACHE, PIXELS, SIZE,
                                backbone, normalize, rgb, SOURCE, SEED, encode)
from general_domain_readout import TARGETS
from learned_frequency_model import FrequencyResidual, support_scores
from unified_head_core import coefficients, late


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT / 'general_domain_full_readout_20261011' / 'consistent_replay'
CACHE = Path('C:/Users/25791/AppData/Local/Temp/sar_general_domain_full_20261011/consistent_replay')
EPOCHS = 6
ETA = .25
BETA = .25
MODES = ('base_raw', 'base_c4', 'base_condition', 'base_full',
         'expert_single', 'expert_dual', 'late_single', 'late_dual',
         'unified_single', 'unified_dual', 'late_single_no_frequency', 'late_dual_no_frequency')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def source_frame():
    return pd.read_csv(SOURCE / 'manifest.csv', keep_default_na=False,
                      usecols=['instance_id', 'class_id', 'class_name', 'port', 'mmsi', 'product'],
                      dtype={'mmsi': str, 'product': str})


def prepare():
    if (OUT / 'PROTOCOL.json').exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    allowed = archive(PILOT / 'source_split.npz')['allowed']
    frame = source_frame()
    rng = np.random.default_rng(SEED)
    arrays = dict(source_rows=allowed)
    ports, map_rows, details = [], [], []
    for port in sorted(frame.port.iloc[allowed].unique()):
        rows = allowed[frame.port.to_numpy()[allowed] == port]
        if len(rows) < 128:
            continue
        support = rng.choice(len(rows), 64, replace=False)
        remaining = np.setdiff1d(np.arange(len(rows)), support)
        queries = np.stack([rng.choice(remaining, min(128, len(remaining)), replace=False) for _ in range(EPOCHS)])
        train = allowed[frame.port.to_numpy()[allowed] != port]
        for column in ('mmsi', 'product'):
            values = frame[column].to_numpy(str)
            identities = np.unique(values[rows])
            identities = identities[~np.isin(identities, ['', 'nan', 'None', '0', '-1'])]
            train = train[~np.isin(values[train], identities)]
        index = len(ports)
        ports.append(port)
        arrays.update({f'{index}_rows': rows, f'{index}_support': support,
                       f'{index}_queries': queries, f'{index}_train': train})
        used = rows[np.unique(np.r_[support, queries.ravel()])]
        map_rows.append(used)
        details.append(dict(port=port, cohort_n=len(rows), prior_train_n=len(train),
                            source_map_n=len(used), queries_per_epoch=queries.shape[1]))
    arrays.update(ports=np.asarray(ports), map_rows=np.unique(np.concatenate(map_rows)))
    np.savez(OUT / 'episode_plan.npz', **arrays)
    for ds, paths in TARGETS.items():
        old = archive(paths['design'])
        np.savez(OUT / f'{ds}_design.npz', **{k: old[k] for k in
                 ('rows', 'ids', 'support', 'support_labels', 'eval_masks')})
    write(OUT / 'SOURCE_PLAN.json', dict(episodes=details, source_rows=len(allowed),
                                      map_rows=len(arrays['map_rows']), seed=SEED))
    write(OUT / 'PROTOCOL.json', dict(date='2026-10-11', phase='Complete external readout integration',
        factors=['R50 G or source-SAR D', 'single backbone or same fixed D18 second backbone'],
        primary='Fixed equal-softmax fusion: late_single and late_dual, for each G/D R50',
        secondary='Unified half-kernel/half-prior head; staged base and expert diagnostics',
        modes=list(MODES), source_rows=len(allowed), source_split=str(PILOT / 'source_split.npz'),
        encoders='Exact previous pilot G50, D50 epoch6, D18 epoch2; all frozen, no legacy F-pretrained R18',
        input='112px RGB[VV,VH,mean], same ImageNet normalization; original GAP and C4 mean GAP',
        base='Refit original-view source PCA128/scales; C4 source-balanced ridge1; support ridge8',
        conditional='Same descriptor cohort ranks, eta.25, RBF exp(-squared_distance/.5)',
        frequency='New source-trained selective FrequencyResidual on native512x14x14 maps; Haar four7x7 bands, beta.25',
        frequency_training='Six epochs, balanced query CE through deployed base64 residual solve, AdamW.003/decay.001, finalepoch fixed',
        frequency_source='One fixed64 support per permitted source port; up to128 queries/epoch; held-port/identity-purged prior; PCA sees all allowed source',
        expert='Original R50 PCA128 plus optional D18 PCA64; half kept plus half source-standardized discarded coordinates, one bias',
        single_fairness='Single keeps both heads, C4, conditional and frequency operators; uses native128/2048 dimension normalization',
        target='Same five64-label draws on F0921/OSS2; all predictions frozen before query readout; no new target tuning',
        training_boundary='Full readout operator rebuild, not replication of legacy supervised R18 layer4 training; no backbone task-tail refit this round',
        unchanged_scope='No Qs/Qd/incidence additions, no adaptive handoff, no generic or FUSAR/OSS1 datasets',
        comparison_scope='Matched new pretraining/input/sourcepool; historical full scores are context only, not a controlled delta',
        numerical_revision='Fresh original/C4 R50 and D18 caches; cudnn deterministic=True, benchmark=False, tf32=False; previous attempted artifacts preserved in parent directory',
        seed=SEED, neural_seeds=1, independent_confirmation=False, old_outputs_unchanged=True))


def load_r50(domain):
    net, _ = backbone('r50')
    if domain == 'D':
        saved = torch.load(OLD_CACHE / 'models/r50_best.pt', map_location='cpu', weights_only=False)
        net.load_state_dict(saved['backbone'], strict=True)
    return net.cuda().eval()


def views(net, pair):
    """Preserve native layer2 layout for Haar; C4 acts on the input, not its map."""
    pair = F.interpolate(pair, size=(SIZE, SIZE), mode='bilinear', align_corners=False, antialias=True)
    x = normalize(rgb(pair))
    pooled, spatial = [], None
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for turn in range(4):
            value = net.maxpool(net.relu(net.bn1(net.conv1(torch.rot90(x, turn, (-2, -1))))))
            value = net.layer2(net.layer1(value))
            if turn == 0:
                spatial = value
            value = net.layer4(net.layer3(value))
            pooled.append(net.avgpool(value).flatten(1).float())
    return pooled[0], torch.stack(pooled).mean(0), spatial


def live_probe(net, pair, zero):
    resized = F.interpolate(pair, size=(SIZE, SIZE), mode='bilinear', align_corners=False, antialias=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        direct = net(normalize(rgb(resized))).float()
    torch.testing.assert_close(zero, direct, atol=0., rtol=0.)


def frequency_embedding(net, spatial, conditions):
    # Cached source maps are promoted to float32 before Haar during training.
    return net(spatial.float(), conditions)


@torch.inference_mode()
def extract_extra():
    if (OUT / 'D18_FEATURES.json').exists():
        return
    net, width = backbone('r18')
    net.load_state_dict(torch.load(OLD_CACHE / 'models/r18_best.pt', map_location='cpu', weights_only=False)['backbone'])
    net = net.cuda().eval()
    records = {}
    for ds in ('source', 'oss2', 'f0921'):
        pixels = np.load(PIXELS[ds], mmap_mode='r')
        rows = archive(OUT / 'episode_plan.npz')['source_rows'] if ds == 'source' else np.arange(len(pixels))
        records[ds] = encode(net, pixels, rows, CACHE / f'{ds}_D18.npy', width)
    write(OUT / 'D18_FEATURES.json', dict(seconds=records, source_checkpoint_unchanged=True,
          cudnn_deterministic=True, benchmark=False, tf32=False, target_labels_read=False))
    del net
    torch.cuda.empty_cache()


@torch.inference_mode()
def extract_source(domain):
    directory = CACHE / domain
    directory.mkdir(exist_ok=True)
    if (OUT / domain / 'SOURCE_FEATURES.json').exists():
        return
    plan = archive(OUT / 'episode_plan.npz')
    rows, used = plan['source_rows'], plan['map_rows']
    pixels = np.load(PIXELS['source'], mmap_mode='r')
    original = np.load(OLD_CACHE / 'features' / f'source_{domain}50.npy', mmap_mode='r')
    net = load_r50(domain)
    averaged = np.lib.format.open_memmap(directory / 'source_c4.npy', mode='w+', dtype=np.float32,
                                         shape=(len(pixels), 2048))
    averaged[:] = np.nan
    fresh_raw = np.lib.format.open_memmap(directory / 'source_raw.npy', mode='w+', dtype=np.float32,
                                         shape=(len(pixels), 2048))
    fresh_raw[:] = np.nan
    maps = np.lib.format.open_memmap(directory / 'source_maps.npy', mode='w+', dtype=np.float16,
                                     shape=(len(used), 512, 14, 14))
    lookup = np.full(len(pixels), -1, dtype=np.int64)
    lookup[used] = np.arange(len(used))
    start, probe_error = time.perf_counter(), 0.
    for lo in range(0, len(rows), 64):
        ix = rows[lo:lo+64]
        pair = torch.from_numpy(np.array(pixels[ix], copy=True)).cuda().float()/255.
        zero, c4, spatial = views(net, pair)
        raw = zero.cpu().numpy()
        if lo == 0:
            probe_error = float(np.abs(raw-original[ix]).max())
            live_probe(net, pair, zero)
        fresh_raw[ix] = raw
        averaged[ix] = c4.cpu().numpy()
        keep = lookup[ix] >= 0
        if keep.any():
            maps[lookup[ix[keep]]] = spatial[torch.as_tensor(keep, device='cuda')].half().cpu().numpy()
        if lo % 8192 == 0:
            print('SOURCE C4', domain, min(lo+64, len(rows)), '/', len(rows), flush=True)
    averaged.flush()
    fresh_raw.flush()
    maps.flush()
    write(OUT / domain / 'SOURCE_FEATURES.json', dict(source_n=len(rows), map_n=len(used),
          map_shape=[512, 14, 14], old_cache_probe_max_error=probe_error, live_forward_exact=True,
          fresh_original_and_c4=True, seconds=time.perf_counter()-start))
    del net, maps, averaged, fresh_raw
    gc.collect()
    torch.cuda.empty_cache()


@threadpool_limits.wrap(limits=2, user_api='blas')
def prepare_geometry(domain):
    from full_ssl_geometry import fit_geometry, base_parts
    directory = CACHE / domain
    if (OUT / domain / 'GEOMETRY.json').exists():
        return
    frame = source_frame()
    plan = archive(OUT / 'episode_plan.npz')
    labels = frame.class_id.to_numpy(int)
    raw = np.load(directory / 'source_raw.npy', mmap_mode='r')
    extra = np.load(CACHE / 'source_D18.npy', mmap_mode='r')
    c4 = np.load(directory / 'source_c4.npy', mmap_mode='r')
    state = fit_geometry(raw, c4, extra, labels, plan['source_rows'])
    with (directory / 'geometry.pkl').open('wb') as f:
        pickle.dump(state, f)
    descriptors = pd.read_csv(ROOT / 'c4_condition_diagnosis_20261009/descriptors/self24ports.csv').set_index('row_id')
    np.testing.assert_array_equal(descriptors.loc[plan['source_rows'], 'instance_id'].to_numpy(str),
                                  frame.instance_id.to_numpy(str)[plan['source_rows']])
    all_parts = base_parts(c4[plan['source_rows']], state)
    lookup = np.full(len(frame), -1, dtype=np.int64)
    lookup[plan['source_rows']] = np.arange(len(plan['source_rows']))
    epdir = directory / 'episodes'
    epdir.mkdir(exist_ok=True)
    for i, port in enumerate(plan['ports']):
        er, train = plan[f'{i}_rows'], plan[f'{i}_train']
        local, tr = lookup[er], lookup[train]
        head = RidgeClassifier(alpha=1., class_weight='balanced').fit(all_parts['xs'][tr], labels[train])
        np.testing.assert_array_equal(head.classes_, np.arange(8))
        np.savez(epdir / f'{i:02d}.npz', rows=er, phi=all_parts['phi'][local],
                 prior=head.decision_function(all_parts['xs'][local]), mean=all_parts['mean'][local],
                 conditions=rank_conditions(descriptors.loc[er]), labels=labels[er],
                 support=plan[f'{i}_support'], queries=plan[f'{i}_queries'])
    write(OUT / domain / 'GEOMETRY.json', dict(source_n=len(plan['source_rows']), episodes=len(plan['ports']),
          fresh_source_PCA_scales_priors=True, episode_prior_port_and_identity_purged=True,
          episode_PCA_is_shared_source_training_geometry=True))
    print('GEOMETRY', domain, 'ready', flush=True)


def train_frequency(domain):
    directory = CACHE / domain
    if (OUT / domain / 'FREQUENCY_TRAINED.json').exists():
        return
    plan = archive(OUT / 'episode_plan.npz')
    maps = np.load(directory / 'source_maps.npy', mmap_mode='r')
    used = plan['map_rows']
    lookup = np.full(len(source_frame()), -1, dtype=np.int64)
    lookup[used] = np.arange(len(used))
    episodes = [archive(p) for p in sorted((directory / 'episodes').glob('*.npz'))]
    torch.manual_seed(SEED)
    net = FrequencyResidual('selective').cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.001)
    records = []
    rng = np.random.default_rng(SEED)
    for epoch in range(EPOCHS):
        losses, norms, start = [], [], time.perf_counter()
        for ei in rng.permutation(len(episodes)):
            ep = episodes[ei]
            ix = np.r_[ep['support'], ep['queries'][epoch]]
            selected = lookup[ep['rows'][ix]]
            if (selected < 0).any():
                raise ValueError('Missing source map in episode')
            x = torch.from_numpy(np.array(maps[selected], dtype=np.float32)).cuda()
            args = [torch.as_tensor(ep[k][ix], dtype=torch.float32, device='cuda')
                    for k in ('phi', 'prior', 'mean', 'conditions')]
            labels = torch.as_tensor(ep['labels'][ix], dtype=torch.long, device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                embedding, weights = net(x, args[-1])
            scores = support_scores(*args, embedding, labels[:64], ETA, beta=BETA)
            counts = torch.bincount(labels[64:], minlength=8).clamp_min(1)
            balance = 1./counts[labels[64:]].float()
            loss = (F.cross_entropy(scores[64:], labels[64:], reduction='none')*balance).sum()/balance.sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite source frequency loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            norms.append(float(grad))
        records.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), gradient_norm=float(np.mean(norms)),
                            seconds=time.perf_counter()-start, last_gate_mean=weights.detach().mean(0).cpu().tolist()))
        print('FREQUENCY', domain, epoch+1, records[-1]['loss'], flush=True)
    torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, epochs=EPOCHS, seed=SEED,
                    source_rows=plan['source_rows']), directory / 'frequency.pt')
    write(OUT / domain / 'FREQUENCY_TRAINED.json', dict(records=records, epochs=EPOCHS,
          parameters=sum(p.numel() for p in net.parameters()), frozen_R50=True, target_rows_trained=0))
    del net, optimizer
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def extract_targets(domain):
    directory = CACHE / domain
    net = load_r50(domain)
    frequency = FrequencyResidual('selective').cuda().eval()
    frequency.load_state_dict(torch.load(directory / 'frequency.pt', map_location='cpu', weights_only=False)['state'])
    for ds in ('oss2', 'f0921'):
        receipt = OUT / domain / f'{ds}_FEATURES.json'
        if receipt.exists():
            continue
        data = archive(OUT / f'{ds}_design.npz')
        conditions = archive(TARGETS[ds]['design'])['conditions']
        pixels = np.load(PIXELS[ds], mmap_mode='r')
        np.testing.assert_array_equal(data['rows'], np.arange(len(pixels)))
        old = np.load(OLD_CACHE / 'features' / f'{ds}_{domain}50.npy', mmap_mode='r')
        c4 = np.lib.format.open_memmap(directory / f'{ds}_c4.npy', mode='w+', dtype=np.float32, shape=(len(pixels), 2048))
        fresh_raw = np.lib.format.open_memmap(directory / f'{ds}_raw.npy', mode='w+', dtype=np.float32, shape=(len(pixels), 2048))
        embedding = np.lib.format.open_memmap(directory / f'{ds}_frequency.npy', mode='w+', dtype=np.float32, shape=(len(pixels), 256))
        gates = np.lib.format.open_memmap(directory / f'{ds}_gates.npy', mode='w+', dtype=np.float32, shape=(len(pixels), 4))
        start, error = time.perf_counter(), 0.
        for lo in range(0, len(pixels), 64):
            hi = min(lo+64, len(pixels))
            pair = torch.from_numpy(np.array(pixels[lo:hi], copy=True)).cuda().float()/255.
            zero, pooled, spatial = views(net, pair)
            if lo == 0:
                error = float(np.abs(zero.cpu().numpy()-old[lo:hi]).max())
                live_probe(net, pair, zero)
            fresh_raw[lo:hi] = zero.cpu().numpy()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z, w = frequency_embedding(frequency, spatial, torch.as_tensor(conditions[lo:hi], dtype=torch.float32, device='cuda'))
            c4[lo:hi], embedding[lo:hi], gates[lo:hi] = pooled.cpu().numpy(), z.cpu().numpy(), w.cpu().numpy()
            if lo % 16384 == 0:
                print('TARGET C4+FREQ', domain, ds, min(lo+64, len(pixels)), '/', len(pixels), flush=True)
        for a in (c4, embedding, gates, fresh_raw):
            a.flush()
        write(receipt, dict(rows=len(pixels), old_cache_probe_max_error=error, live_forward_exact=True,
                            fresh_original_and_c4=True, seconds=time.perf_counter()-start,
                            target_maps_streamed_not_saved=True, target_query_labels_read=False))
        del c4, embedding, gates, fresh_raw
    del net, frequency
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'extra', 'source', 'geometry', 'frequency', 'target', 'all'), default='all')
    parser.add_argument('--domain', choices=('G', 'D'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepare()
    if args.stage in ('extra', 'all'):
        extract_extra()
    for domain in ([args.domain] if args.domain else ['G', 'D']):
        if args.stage in ('source', 'all'):
            extract_source(domain)
        if args.stage in ('geometry', 'all'):
            prepare_geometry(domain)
        if args.stage in ('frequency', 'all'):
            train_frequency(domain)
        if args.stage in ('target', 'all'):
            extract_targets(domain)


if __name__ == '__main__':
    main()
