"""Matched source continuation and shared-I/O encoding of branch specialists."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch

from prepare_learned_frequency import ROOT, OUT as EXP, load_npz
from learned_frequency_model import FrequencyResidual, support_scores
from frequency_specialist_core import per_band_scores, rescue_responsibility, branch_objective, balanced_ce

OUT = ROOT/'frequency_specialists_20261009'
ARMS = ('continued', 'independent', 'specialist')


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def model_from(path, mode='fixed'):
    model = FrequencyResidual(mode).cuda().eval()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state'])
    return model


def task_paths(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    dst = OUT/'tasks'/info['dataset']/info['task']
    dst.mkdir(parents=True, exist_ok=True)
    return info, dst


def encode(models, maps, rows, conditions):
    chunks = {key: [] for key in models}
    with torch.inference_mode():
        for lo in range(0, len(rows), 128):
            x = torch.from_numpy(np.asarray(maps[rows[lo:lo+128]], dtype=np.float32)).cuda()
            c = torch.as_tensor(conditions[lo:lo+128], dtype=torch.float32, device='cuda')
            for key, model in models.items():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    z, _ = model(x, c)
                chunks[key].append(z.cpu().numpy())
    return {key: np.concatenate(value) for key, value in chunks.items()}


def train_task(folder, maps, arms=None, auxiliary=None):
    arms = ARMS if arms is None else arms
    info, dst = task_paths(folder)
    if all((dst/f'{a}_training.json').exists() for a in arms):
        return
    episodes = [load_npz(p) for p in sorted((folder/'episodes').glob('*.npz'))]
    original = torch.load(folder/'fixed.pt', map_location='cpu', weights_only=True)
    allowed = np.asarray(original['source_rows'])
    if any(np.setdiff1d(ep['rows'], allowed).size for ep in episodes):
        raise ValueError('Source episode includes excluded target rows')
    rng, plan = np.random.default_rng(20261009), []
    for epoch in range(3):
        batch = []
        for ei in rng.permutation(len(episodes)):
            ep = episodes[ei]
            q = rng.choice(ep['query'], min(128, len(ep['query'])), replace=False)
            batch.append((ei, np.r_[ep['support'], q]))
        plan.append(batch)
    used = np.unique(np.concatenate([episodes[ei]['rows'][ix] for batch in plan for ei, ix in batch]))
    val = load_npz(ROOT/'frequency_decision_20261009/source'/info['dataset']/info['task']/'examples.npz')
    if np.intersect1d(used, val['rows']).size:
        raise ValueError('Training reused frozen source validation queries')
    lookup = np.full(len(maps), -1, np.int64)
    lookup[used] = np.arange(len(used))
    cache = torch.empty((len(used), 512, 16, 16), dtype=torch.float16, device='cuda')
    for lo in range(0, len(used), 256):
        cache[lo:lo+256].copy_(torch.from_numpy(np.array(maps[used[lo:lo+256]], copy=True)))
    ep_gpu = [{k: torch.as_tensor(ep[k], dtype=torch.long if k == 'labels' else torch.float32, device='cuda')
        for k in ('phi', 'prior', 'mean', 'conditions', 'labels')} for ep in episodes]
    print(f'SPECIALIST CACHE {info["task"]}: {len(used)} rows', flush=True)
    teacher, initial = model_from(folder/'selective.pt', 'selective'), model_from(folder/'fixed.pt')
    anchors = {}
    # Freeze supervision before any branch updates, using source labels only.
    with torch.no_grad():
        for epoch, batches in enumerate(plan):
            for bi, (ei, ix) in enumerate(batches):
                local = torch.as_tensor(ix, device='cuda')
                x = cache[torch.as_tensor(lookup[episodes[ei]['rows'][ix]], device='cuda')].float()
                args = [ep_gpu[ei][k][local] for k in ('phi', 'prior', 'mean', 'conditions')]
                y = ep_gpu[ei]['labels'][local]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    ref_z, _ = teacher(x, args[-1])
                    init_z, _ = initial(x, args[-1])
                reference = support_scores(*args, ref_z, y[:64], info['eta'])[64:]
                single = per_band_scores(args, init_z, y[:64], info['eta'])[64:]
                anchors[epoch, bi] = (reference.detach(), rescue_responsibility(single, reference, y[64:]))
    del teacher, initial
    for arm in arms:
        if (dst/f'{arm}_training.json').exists():
            continue
        torch.manual_seed(20261009)
        model = model_from(folder/'fixed.pt').train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
        log, first_grad, started = [], None, time.perf_counter()
        auxiliary_gradient = None
        for epoch, batches in enumerate(plan):
            rec = []
            for bi, (ei, ix) in enumerate(batches):
                local = torch.as_tensor(ix, device='cuda')
                x = cache[torch.as_tensor(lookup[episodes[ei]['rows'][ix]], device='cuda')].float()
                args = [ep_gpu[ei][k][local] for k in ('phi', 'prior', 'mean', 'conditions')]
                y = ep_gpu[ei]['labels'][local]
                reference, allocation = anchors[epoch, bi]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    z, _ = model(x, args[-1])
                if arm == 'continued':
                    scores = support_scores(*args, z, y[:64], info['eta'])[64:]
                    loss = ce = balanced_ce(scores, y[64:])
                    protection = loss*0
                else:
                    scores = per_band_scores(args, z, y[:64], info['eta'])[64:]
                    loss, ce, protection = branch_objective(scores, reference, y[64:],
                        allocation if arm == 'specialist' else None)
                extra = loss*0 if auxiliary is None else auxiliary(z, args[-1], y[:64], reference, y[64:], arm)
                if auxiliary is not None and first_grad is None:
                    params = list(model.parameters())
                    base_grad = torch.cat([g.flatten() for g in torch.autograd.grad(loss, params, retain_graph=True)])
                    aux_grad = torch.cat([g.flatten() for g in torch.autograd.grad(.25*extra, params, retain_graph=True)])
                    auxiliary_gradient = dict(base_norm=float(base_grad.norm()), weighted_aux_norm=float(aux_grad.norm()),
                        cosine=float(torch.nn.functional.cosine_similarity(base_grad[None], aux_grad[None])))
                loss = loss+.25*extra
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite specialist loss')
                loss.backward()
                if first_grad is None:
                    first_grad = [float(torch.sqrt(sum(p.grad.square().sum() for p in block.parameters())))
                        for block in model.blocks]
                    if min(first_grad) <= 0:
                        raise ValueError('A branch received no gradient')
                grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5))
                optimizer.step()
                rec.append([float(ce.detach()), float(protection.detach()), grad, float(extra.detach())])
            row = dict(epoch=epoch+1, ce=float(np.mean([r[0] for r in rec])),
                protection=float(np.mean([r[1] for r in rec])), max_gradient=max(r[2] for r in rec),
                auxiliary=float(np.mean([r[3] for r in rec])),
                seconds=time.perf_counter()-started)
            log.append(row)
            print(f'SPECIALIST TRAIN {info["task"]} {arm} {epoch+1}/3 CE={row["ce"]:.5f} protection={row["protection"]:.5f}', flush=True)
        changes = [float(sum((v.detach().cpu()-original['state'][k]).square().sum()
            for k, v in model.state_dict().items() if k.startswith(f'blocks.{b}.')).sqrt()) for b in range(4)]
        torch.save(dict(state=model.state_dict(), arm=arm, epochs=3, source_only=True), dst/f'{arm}.pt')
        alloc = torch.cat([x[1] for x in anchors.values()]).cpu().numpy()
        write(dst/f'{arm}_training.json', dict(source_only=True, epochs=log,
            first_branch_gradients=first_grad, branch_parameter_l2_change=changes,
            auxiliary_first_gradient=auxiliary_gradient,
            allocation_mean=alloc.mean(0).tolist(), allocation_min=float(alloc.min()), allocation_max=float(alloc.max()),
            source_validation_query_overlap=0, trainable_parameters=sum(p.numel() for p in model.parameters()),
            training_rows=len(used)))
        del model, optimizer
    del cache, ep_gpu, anchors
    torch.cuda.empty_cache()


def encode_source(folder, maps):
    info, dst = task_paths(folder)
    out = OUT/'source'/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'ENCODED.json').exists():
        return
    d = load_npz(ROOT/'frequency_decision_20261009/source'/info['dataset']/info['task']/'examples.npz')
    eps = sorted((folder/'episodes').glob('*.npz'))
    models = {a: model_from(dst/f'{a}.pt') for a in ARMS}
    models.update(old=model_from(folder/'fixed.pt'), learned=model_from(folder/'selective.pt', 'selective'))
    for ei in np.unique(d['episode']):
        path = out/f'episode_{ei:02d}.npz'
        if path.exists():
            continue
        ep = load_npz(eps[ei])
        use = d['episode'] == ei
        q = pd.Index(ep['rows']).get_indexer(d['rows'][use])
        if (q < 0).any() or np.intersect1d(q, ep['support']).size:
            raise ValueError('Source encoding row mapping mismatch')
        ix = np.r_[ep['support'], q]
        z = encode(models, maps, ep['rows'][ix], ep['conditions'][ix])
        np.savez_compressed(path, **z, indices=ix, expected=d['new'][use])
    write(out/'ENCODED.json', dict(dataset=info['dataset'], task=info['task'], queries=len(d['labels'])))
    print(f'SPECIALIST SOURCE ENCODE {info["task"]}', flush=True)


def encode_target(folder):
    info, dst = task_paths(folder)
    data = load_npz(folder/'target.npz')
    missing = [a for a in ARMS if not (dst/f'{a}_embedding.npz').exists()]
    if not missing:
        return
    maps = np.load(EXP/'maps'/info['dataset']/'maps.npy', mmap_mode='r')
    np.testing.assert_array_equal(np.load(EXP/'maps'/info['dataset']/'sample_ids.npy')[data['rows']], data['ids'])
    models = {a: model_from(dst/f'{a}.pt') for a in missing}
    z = encode(models, maps, data['rows'], data['conditions'])
    for a in missing:
        np.savez_compressed(dst/f'{a}_embedding.npz', embedding=z[a], rows=data['rows'])
    print(f'SPECIALIST TARGET ENCODE {info["task"]}: {len(data["rows"])} rows', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['train', 'source', 'target', 'all'], default='all')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    maps = np.load(EXP/'maps/self24ports/maps.npy', mmap_mode='r')
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    for folder in folders:
        if args.stage in ('train', 'all'):
            train_task(folder, maps)
        if args.stage in ('source', 'all'):
            encode_source(folder, maps)
        if args.stage in ('target', 'all'):
            encode_target(folder)
