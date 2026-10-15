"""Matched CE continuation versus comparator-supervised frequency representation."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from prepare_learned_frequency import ROOT, OUT as EXP, load_npz
from learned_frequency_model import FrequencyResidual, support_scores
from frequency_joint_core import visual_scores, comparison_loss
from frequency_reliability_core import kernels, coefficients
from frequency_reliability_run import router, statistics
from frequency_decision_run import report

OUT = ROOT/'frequency_joint_20261009'
ARMS = ('continued_ce', 'joint_kernel')
MODES = ('baseline', 'learned', *ARMS, 'joint_compare', 'source_selected')


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def paths(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    dst = OUT/'tasks'/info['dataset']/info['task']
    dst.mkdir(parents=True, exist_ok=True)
    return info, dst


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = FrequencyResidual('selective').cuda().eval()
    model.load_state_dict(checkpoint['state'])
    return model


def encode_rows(model, maps, rows, conditions):
    chunks = []
    with torch.inference_mode():
        for lo in range(0, len(rows), 128):
            x = torch.from_numpy(np.asarray(maps[rows[lo:lo+128]], dtype=np.float32)).cuda()
            c = torch.as_tensor(conditions[lo:lo+128], dtype=torch.float32, device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                embedding, _ = model(x, c)
            chunks.append(embedding.cpu().numpy())
    return np.concatenate(chunks)


def compare(embedding, conditions, support, rows, sy):
    chunks = []
    with torch.inference_mode():
        for lo in range(0, len(rows), 256):
            ix = np.r_[support, rows[lo:lo+256]]
            visual, _ = visual_scores(torch.as_tensor(embedding[ix], device='cuda'),
                torch.as_tensor(conditions[ix], dtype=torch.float32, device='cuda'),
                torch.as_tensor(sy, dtype=torch.long, device='cuda'))
            chunks.append(visual.cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, 8), np.float32)


def choose(old, new, visual, support_labels, rows):
    result = new.copy()
    a, b = old[rows].argmax(1), new[rows].argmax(1)
    counts = np.bincount(support_labels, minlength=8)
    margin = visual[np.arange(len(rows)), b]-visual[np.arange(len(rows)), a]
    reject = (a != b) & (counts[a] >= 2) & (counts[b] >= 2) & (margin < 0)
    result[rows[reject]] = old[rows[reject]]
    return result, reject, margin


def train_task(folder, maps):
    info, dst = paths(folder)
    if all((dst/f'{arm}_training.json').exists() for arm in ARMS):
        return
    episodes = [load_npz(p) for p in sorted((folder/'episodes').glob('*.npz'))]
    original = torch.load(folder/'selective.pt', map_location='cpu', weights_only=True)
    allowed = np.asarray(original['source_rows'])
    if any(np.setdiff1d(ep['rows'], allowed).size for ep in episodes):
        raise ValueError('Episode contains excluded outer rows')
    rng, plan = np.random.default_rng(20261009), []
    for epoch in range(3):
        batches = []
        for ei in rng.permutation(len(episodes)):
            ep = episodes[ei]
            q = rng.choice(ep['query'], min(128, len(ep['query'])), replace=False)
            batches.append((ei, np.r_[ep['support'], q]))
        plan.append(batches)
    used = np.unique(np.concatenate([episodes[ei]['rows'][ix] for batches in plan for ei, ix in batches]))
    validation = load_npz(ROOT/'frequency_decision_20261009/source'/info['dataset']/info['task']/'examples.npz')
    if np.intersect1d(validation['rows'], used).size:
        raise ValueError('Continuation training reused validation query rows')
    lookup = np.full(len(maps), -1, np.int64)
    lookup[used] = np.arange(len(used))
    cached = torch.empty((len(used), 512, 16, 16), dtype=torch.float16, device='cuda')
    for lo in range(0, len(used), 256):
        cached[lo:lo+256].copy_(torch.from_numpy(np.array(maps[used[lo:lo+256]], copy=True)))
    ep_gpu = [{key: torch.as_tensor(ep[key], device='cuda',
        dtype=torch.long if key == 'labels' else torch.float32)
        for key in ('phi', 'prior', 'mean', 'conditions', 'labels')} for ep in episodes]
    print(f'JOINT CACHE {info["task"]}: {len(used)} rows {cached.numel()*2/2**30:.2f}GiB', flush=True)
    for arm in ARMS:
        if (dst/f'{arm}_training.json').exists():
            continue
        torch.manual_seed(20261009)
        model = load_model(folder/'selective.pt').train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
        log, first, started = [], None, time.perf_counter()
        for epoch, batches in enumerate(plan):
            records = []
            for ei, ix in batches:
                local = torch.as_tensor(ix, device='cuda')
                x = cached[torch.as_tensor(lookup[episodes[ei]['rows'][ix]], device='cuda')].float()
                args = [ep_gpu[ei][key][local] for key in ('phi', 'prior', 'mean', 'conditions')]
                labels = ep_gpu[ei]['labels'][local]
                sy, yq = labels[:64], labels[64:]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    embedding, gate = model(x, args[-1])
                scores = support_scores(*args, embedding, sy, info['eta'])
                per_class = torch.bincount(yq, minlength=8).clamp_min(1)
                weight = 1./per_class[yq].float()
                ce = (F.cross_entropy(scores[64:], yq, reduction='none')*weight).sum()/weight.sum()
                visual, counts = visual_scores(embedding, args[-1], sy)
                pair, valid, negative = comparison_loss(visual, counts, scores[64:], yq)
                if first is None:
                    gc = torch.autograd.grad(ce, embedding, retain_graph=True)[0]
                    gp = torch.autograd.grad(pair, embedding, retain_graph=True)[0]
                    first = dict(ce_embedding_grad=float(gc.norm()), pair_embedding_grad=float(gp.norm()),
                        gradient_cosine=float(F.cosine_similarity(gc.flatten(), gp.flatten(), dim=0)))
                loss = ce+(.1*pair if arm == 'joint_kernel' else 0.)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite joint training loss')
                loss.backward()
                grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5))
                optimizer.step()
                rr = torch.arange(len(yq), device='cuda')
                pa = float((visual[rr, yq][valid] > visual[rr, negative][valid]).float().mean()) if valid.any() else None
                records.append(dict(ce=float(ce.detach()), pair=float(pair.detach()), gradient=grad,
                    valid_pairs=int(valid.sum()), pair_accuracy=pa, gate=gate.detach().mean(0).cpu().tolist()))
            row = dict(epoch=epoch+1, ce=float(np.mean([r['ce'] for r in records])),
                pair=float(np.mean([r['pair'] for r in records])),
                valid_pairs=sum(r['valid_pairs'] for r in records),
                pair_accuracy=float(np.mean([r['pair_accuracy'] for r in records if r['pair_accuracy'] is not None])),
                gate_mean=np.mean([r['gate'] for r in records], axis=0).tolist(),
                max_gradient=max(r['gradient'] for r in records), seconds=time.perf_counter()-started)
            log.append(row)
            print(f'JOINT TRAIN {info["task"]} {arm} epoch{epoch+1}: CE{row["ce"]:.5f} pair{row["pair"]:.5f}', flush=True)
        drift = float(sum((v.detach().cpu()-original['state'][k]).square().sum() for k, v in model.state_dict().items()).sqrt())
        torch.save(dict(state=model.state_dict(), arm=arm, epochs=3, source_only=True), dst/f'{arm}.pt')
        write(dst/f'{arm}_training.json', dict(source_only=True, epochs=log, initial_gradient=first,
            parameter_l2_change=drift, reused_validation_queries=0, training_rows=len(used),
            trainable_parameters=sum(p.numel() for p in model.parameters())))
        del model, optimizer
    del cached, ep_gpu
    torch.cuda.empty_cache()


def source_validate(folder, maps):
    info, dst = paths(folder)
    out = OUT/'source'/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'SELECTION.json').exists():
        return
    d = load_npz(ROOT/'frequency_decision_20261009/source'/info['dataset']/info['task']/'examples.npz')
    episodes = sorted((folder/'episodes').glob('*.npz'))
    models = {arm: load_model(dst/f'{arm}.pt') for arm in ARMS}
    records = []
    for ei in np.unique(d['episode']):
        ep = load_npz(episodes[ei])
        valid = d['episode'] == ei
        q = pd.Index(ep['rows']).get_indexer(d['rows'][valid])
        if (q < 0).any() or np.intersect1d(q, ep['support']).size:
            raise ValueError('Source validation row mapping mismatch')
        ix = np.r_[ep['support'], q]
        data = {key: ep[key][ix] for key in ('phi', 'prior', 'mean', 'conditions')}
        support, rows = np.arange(64), ep['rows'][ix]
        sy, y = ep['labels'][ep['support']], d['labels'][valid]
        np.testing.assert_array_equal(ep['labels'][q], y)
        scores = dict(learned=d['new'][valid])
        for arm in ARMS:
            embedding = encode_rows(models[arm], maps, rows, data['conditions'])
            _, k = kernels(data, embedding.astype(float), support, info['eta'])
            score = data['prior']+k@coefficients(k[:64], sy, data['prior'][:64])
            scores[arm] = score[64:]
            if arm == 'joint_kernel':
                visual = compare(embedding, data['conditions'], support, np.arange(64, len(rows)), sy)
                scores['joint_compare'], _, _ = choose(scores['learned'], score[64:], visual, sy, np.arange(len(y)))
        _, old_ba, _ = statistics(y, scores['learned'].argmax(1))
        for arm, score in scores.items():
            oa, ba, _ = statistics(y, score.argmax(1))
            records.append(dict(episode=int(ei), port=str(ep['held_port']), mode=arm, n=len(y), oa=oa,
                ba=ba, delta_ba=ba-old_ba))
    result = pd.DataFrame(records)
    gains = result.groupby('mode').delta_ba.mean().to_dict()
    best = max(('learned', *ARMS, 'joint_compare'), key=lambda m: gains[m])
    result.to_csv(out/'validation.csv', index=False)
    a, b, y = d['old'].argmax(1), d['new'].argmax(1), d['labels']
    write(out/'SELECTION.json', dict(dataset=info['dataset'], task=info['task'], rule=best,
        source_delta_ba=gains, source_queries=len(y), source_conflicts=int(np.sum(a != b)),
        source_rescues=int(np.sum((b == y) & (a != y))), source_harms=int(np.sum((a == y) & (b != y))),
        validation='Unused source query rows; source ports not representation-held', no_target_labels=True))
    print(f'JOINT SELECT {info["task"]}: {best} {gains}', flush=True)
    del models


def encode_target(folder):
    info, dst = paths(folder)
    data = load_npz(folder/'target.npz')
    maps = np.load(EXP/'maps'/info['dataset']/'maps.npy', mmap_mode='r')
    np.testing.assert_array_equal(np.load(EXP/'maps'/info['dataset']/'sample_ids.npy')[data['rows']], data['ids'])
    for arm in ARMS:
        path = dst/f'{arm}_embedding.npz'
        if path.exists():
            continue
        model = load_model(dst/f'{arm}.pt')
        embedding = encode_rows(model, maps, data['rows'], data['conditions'])
        np.savez_compressed(path, embedding=embedding, rows=data['rows'])
        print(f'JOINT ENCODE {info["task"]} {arm}: {len(embedding)}', flush=True)
        del model


def target_predict(folder):
    info, dst = paths(folder)
    out = OUT/'targets'/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return
    selection = json.loads((OUT/'source'/info['dataset']/info['task']/'SELECTION.json').read_text())
    data = load_npz(folder/'target.npz')
    embeddings = {'learned': load_npz(folder/'selective_embedding.npz')}
    embeddings.update({arm: load_npz(dst/f'{arm}_embedding.npz') for arm in ARMS})
    for z in embeddings.values():
        np.testing.assert_array_equal(z['rows'], data['rows'])
    old = load_npz(folder/'policies.npz')
    reference = list(old['modes']).index('selective')
    routing = router(info, data)
    shape = (5, len(MODES), len(data['rows']))
    base, full, frozen = [np.empty(shape, np.int8) for _ in range(3)]
    selected = np.zeros(shape, bool)
    choices = {}
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        if len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('Support/query budget mismatch')
        scores = {}
        for arm, z in embeddings.items():
            k0, k = kernels(data, z['embedding'].astype(float), support, info['eta'])
            scores[arm] = data['prior']+k@coefficients(k[support], sy, data['prior'][support])
            if arm == 'learned':
                scores['baseline'] = data['prior']+k0@coefficients(k0[support], sy, data['prior'][support])
        rows = np.flatnonzero(ev & (scores['learned'].argmax(1) != scores['joint_kernel'].argmax(1)))
        visual = compare(embeddings['joint_kernel']['embedding'], data['conditions'], support, rows, sy)
        scores['joint_compare'], reject, margin = choose(scores['learned'], scores['joint_kernel'], visual, sy, rows)
        choices[f'rows_{draw}'], choices[f'visual_{draw}'] = rows, visual
        choices[f'margin_{draw}'], choices[f'reject_{draw}'] = margin, reject
        scores['source_selected'] = scores[selection['rule']]
        for j, mode in enumerate(MODES):
            bp = scores[mode].argmax(1)
            pred, sel = routing(scores[mode], data['expert'][draw], sy, ev)
            if sel[~ev].any() or sel.sum() > int(.05*ev.sum()):
                raise ValueError('Expert handoff exceeded')
            if mode == 'learned':
                for got, expected in zip((bp, pred, sel), (old['base_predictions'][draw, reference],
                    old['predictions'][draw, reference], old['selected'][draw, reference])):
                    np.testing.assert_array_equal(got, expected)
            if mode == 'baseline':
                for got, expected in zip((bp, pred, sel), (data['expected_base'][draw], data['expected_full'][draw], data['expected_selected'][draw])):
                    np.testing.assert_array_equal(got, expected)
            base[draw, j], full[draw, j], selected[draw, j] = bp, pred, sel
            frozen[draw, j] = np.where(old['selected'][draw, reference], data['expert'][draw].argmax(1), bp)
        print(f'JOINT TARGET {info["task"]} draw{draw}: {len(rows)} conflicts, {reject.sum()} reverted', flush=True)
    np.savez_compressed(out/'policies.npz', modes=np.array(MODES), base_predictions=base,
        predictions=full, frozen_predictions=frozen, selected=selected, rows=data['rows'],
        support=data['support'], eval_masks=data['eval_masks'])
    np.savez_compressed(out/'comparisons.npz', **choices)
    write(out/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], rule=selection['rule'],
        draws=5, target_labels_per_draw=64, handoff_fraction=.05, target_eval_labels_read=False,
        baseline_replayed=True, learned_replayed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'train', 'target', 'report'], default='all')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    maps = np.load(EXP/'maps/self24ports/maps.npy', mmap_mode='r') if args.stage in ('all', 'train') else None
    for folder in folders:
        if args.stage in ('all', 'train'):
            train_task(folder, maps)
            source_validate(folder, maps)
        if args.stage in ('all', 'target'):
            encode_target(folder)
            target_predict(folder)
    if args.stage in ('all', 'report'):
        report(folders, output=OUT, candidates=())
