"""Source-only episodic training; targets receive no gradient or neural refit."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from learned_frequency_model import FrequencyResidual, support_scores

OUT = Path('E:/Hermes/paper_tracking/sensitivity/learned_frequency_residual_20261009')
MODES = ('unsplit', 'fixed', 'selective')


def archive(path):
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def train_task(folder, maps, epochs, modes):
    info = json.loads((folder/'PREPARED.json').read_text())
    eps = [archive(p) for p in sorted((folder/'episodes').glob('*.npz'))]
    if not eps:
        raise ValueError('No source episodes')
    target = archive(folder/'target.npz')
    allowed = target['source_rows']
    if any(np.setdiff1d(ep['rows'], allowed).size for ep in eps):
        raise ValueError('Source map training includes excluded rows')
    if all((folder/f'{mode}.pt').exists() for mode in modes):
        print('REUSE TRAIN '+folder.name, flush=True)
        return
    rng = np.random.default_rng(20261009)
    plan = []
    for _ in range(epochs):
        batches = []
        for ei in rng.permutation(len(eps)):
            ep = eps[ei]
            q = rng.choice(ep['query'], min(128, len(ep['query'])), replace=False)
            batches.append((ei, np.r_[ep['support'], q]))
        plan.append(batches)
    used = np.unique(np.concatenate([eps[ei]['rows'][ix] for batches in plan for ei, ix in batches]))
    lookup = np.full(len(maps), -1, dtype=np.int64)
    lookup[used] = np.arange(len(used))
    cached = torch.empty((len(used), 512, 16, 16), dtype=torch.float16, device='cuda')
    for lo in range(0, len(used), 256):
        ix = used[lo:lo+256]
        cached[lo:lo+len(ix)].copy_(torch.from_numpy(np.array(maps[ix], copy=True)))
    # Catch cache index/order errors before using any source labels to fit.
    np.testing.assert_array_equal(cached[:2].cpu().numpy(), maps[used[:2]])
    eps_gpu = [{key: torch.as_tensor(ep[key], device='cuda',
                  dtype=torch.long if key == 'labels' else torch.float32)
                for key in ('phi', 'prior', 'mean', 'conditions', 'labels')} for ep in eps]
    print(f'GPU SOURCE CACHE {folder.name} {len(used)} rows {cached.numel()*2/2**30:.2f} GiB', flush=True)
    for mode in modes:
        checkpoint = folder/f'{mode}.pt'
        if checkpoint.exists():
            print('REUSE TRAIN '+folder.name+' '+mode, flush=True)
            continue
        torch.manual_seed(20261009)
        model = FrequencyResidual(mode).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
        started = time.perf_counter()
        log = []
        first_gradient, max_gradient = None, 0.
        for epoch, batches in enumerate(plan):
            records = []
            for ei, ix in batches:
                ep = eps[ei]
                local = torch.as_tensor(ix, device='cuda')
                cached_rows = torch.as_tensor(lookup[ep['rows'][ix]], device='cuda')
                x = cached[cached_rows].float()
                args = [eps_gpu[ei][key][local] for key in ('phi', 'prior', 'mean', 'conditions')]
                sy, yq = eps_gpu[ei]['labels'][local[:64]], eps_gpu[ei]['labels'][local[64:]]
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    embedding, weights = model(x, args[-1])
                scores = support_scores(*args, embedding, sy, info['eta'])
                counts = torch.bincount(yq, minlength=8).clamp_min(1)
                balance = 1./counts[yq].float()
                loss = (F.cross_entropy(scores[64:], yq, reduction='none')*balance).sum()/balance.sum()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite source loss')
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5))
                if first_gradient is None:
                    first_gradient = norm
                max_gradient = max(norm, max_gradient)
                opt.step()
                records.append(float(loss.detach()))
            row = dict(epoch=epoch+1, query_ce=float(np.mean(records)), seconds=time.perf_counter()-started,
                       gate_mean=weights.detach().mean(0).cpu().tolist())
            log.append(row)
            print(f'TRAIN {folder.name} {mode} {epoch+1}/{epochs} CE={row["query_ce"]:.6f} {row["seconds"]:.1f}s', flush=True)
        model.eval()
        torch.save(dict(mode=mode, state=model.state_dict(), epochs=epochs, seed=20261009,
                        precision='bf16 convolution, float32 normalization and support solve',
                        eta=info['eta'], beta=.25, source_rows=allowed.tolist()), checkpoint)
        (folder/f'{mode}_training.json').write_text(json.dumps(dict(
            source_only=True, final_epoch_used=True, target_checkpoint_selection=False,
            parameters=sum(p.numel() for p in model.parameters()), first_gradient_norm=first_gradient,
            maximum_gradient_norm=max_gradient, epochs=log), indent=2))
        del model, opt
        torch.cuda.empty_cache()
    del cached, eps_gpu


def encode_task(folder, map_root, modes, batch_size):
    info = json.loads((folder/'PREPARED.json').read_text())
    dataset = info['dataset']
    if not (map_root/dataset/'COMPLETE.json').exists():
        raise FileNotFoundError('Target feature maps incomplete: '+dataset)
    maps = np.load(map_root/dataset/'maps.npy', mmap_mode='r')
    target = archive(folder/'target.npz')
    rows = target['rows']
    np.testing.assert_array_equal(np.load(map_root/dataset/'sample_ids.npy')[rows], target['ids'])
    pending, models, vectors, gates = [], {}, {}, {}
    for mode in modes:
        if (folder/f'{mode}_embedding.npz').exists():
            print('REUSE ENCODE '+folder.name+' '+mode, flush=True)
            continue
        checkpoint = torch.load(folder/f'{mode}.pt', map_location='cpu', weights_only=True)
        model = FrequencyResidual(mode).cuda().eval()
        model.load_state_dict(checkpoint['state'])
        pending.append(mode)
        models[mode] = model
        vectors[mode] = np.empty((len(rows), 256), dtype=np.float32)
        gates[mode] = np.empty((len(rows), 4), dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for lo in range(0, len(rows) if pending else 0, batch_size):
            ix = rows[lo:lo+batch_size]
            x = torch.from_numpy(np.asarray(maps[ix], dtype=np.float32)).cuda()
            c = torch.as_tensor(target['conditions'][lo:lo+batch_size], dtype=torch.float32, device='cuda')
            for mode in pending:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    z, weights = models[mode](x, c)
                vectors[mode][lo:lo+len(ix)] = z.cpu().numpy()
                gates[mode][lo:lo+len(ix)] = weights.cpu().numpy()
            if lo % 8192 == 0 or lo+len(ix) == len(rows):
                print(f'ENCODE {folder.name} {lo+len(ix)}/{len(rows)} {time.perf_counter()-started:.1f}s', flush=True)
    for mode in pending:
        output = folder/f'{mode}_embedding.npz'
        np.savez_compressed(output, embedding=vectors[mode], weights=gates[mode],
                            rows=rows, ids=target['ids'])
        if mode == 'selective':
            joined, weights = vectors[mode], gates[mode]
            uniform = (joined.reshape(len(rows), 4, 64)/np.sqrt(weights[:, :, None])*.5).reshape(len(rows), -1)
            np.savez_compressed(folder/'selective_uniform_embedding.npz', embedding=uniform,
                                weights=np.full_like(weights, .25), rows=rows, ids=target['ids'])
        print('ENCODED '+folder.name+' '+mode, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['train', 'encode', 'all'], default='all')
    parser.add_argument('--datasets', nargs='+', default=['self24ports', 'oss2', 'f0921'])
    parser.add_argument('--ports', nargs='+')
    parser.add_argument('--modes', nargs='+', default=list(MODES), choices=MODES)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    maps = None
    if args.stage != 'encode':
        if not (OUT/'maps/self24ports/COMPLETE.json').exists():
            raise FileNotFoundError('Source maps incomplete')
        maps = np.load(OUT/'maps/self24ports/maps.npy', mmap_mode='r')
    folders = []
    if 'self24ports' in args.datasets:
        folders += [p for p in sorted((OUT/'tasks/self24ports').iterdir())
                    if (not args.ports or p.name in args.ports)]
    folders += [OUT/'tasks'/d for d in args.datasets if d != 'self24ports']
    for folder in folders:
        if args.stage != 'encode':
            train_task(folder, maps, args.epochs, args.modes)
        if args.stage != 'train':
            encode_task(folder, OUT/'maps', args.modes, args.batch_size)


if __name__ == '__main__':
    main()
