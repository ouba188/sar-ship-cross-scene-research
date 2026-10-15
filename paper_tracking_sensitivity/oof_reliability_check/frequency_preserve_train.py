"""Matched preservation trial, sharing cached maps between fit and source encoding."""

import json
import time
import numpy as np
import torch
from prepare_learned_frequency import ROOT, OUT as EXP, load_npz
from learned_frequency_model import support_scores
from frequency_specialist_train import model_from, write, encode
from frequency_specialist_core import per_band_scores, branch_objective
from frequency_standalone_core import auxiliary_objective
from frequency_preserve_core import unique_anchors, retention_loss

OUT = ROOT/'frequency_preserve_20261010'
PREVIOUS = ROOT/'frequency_standalone_20261009'


def train_and_encode_source(folder, maps):
    info = json.loads((folder/'PREPARED.json').read_text())
    ds, task = info['dataset'], info['task']
    dst, src = OUT/'tasks'/ds/task, OUT/'source'/ds/task
    src.mkdir(parents=True, exist_ok=True)
    if (src/'ENCODED.json').exists():
        return
    episodes = [load_npz(p) for p in sorted((folder/'episodes').glob('*.npz'))]
    original = torch.load(folder/'fixed.pt', map_location='cpu', weights_only=True)
    if any(np.setdiff1d(ep['rows'], original['source_rows']).size for ep in episodes):
        raise ValueError('Source rows outside original allowed split')
    rng, plan = np.random.default_rng(20261009), []
    for epoch in range(3):
        plan.append([(ei, np.r_[episodes[ei]['support'], rng.choice(episodes[ei]['query'],
            min(128, len(episodes[ei]['query'])), replace=False)]) for ei in rng.permutation(len(episodes))])
    used = np.unique(np.concatenate([episodes[ei]['rows'][ix] for batch in plan for ei, ix in batch]))
    val = load_npz(ROOT/'frequency_decision_20261009/source'/ds/task/'examples.npz')
    if np.intersect1d(used, val['rows']).size:
        raise ValueError('Training/source validation query overlap')
    encoded = [(int(p.stem.split('_')[-1]), load_npz(p)) for p in
        sorted((PREVIOUS/'source'/ds/task).glob('episode_*.npz'))]
    all_rows = np.unique(np.r_[used, np.concatenate([episodes[ei]['rows'][z['indices']] for ei, z in encoded])])
    lookup = np.full(len(maps), -1, np.int64)
    lookup[all_rows] = np.arange(len(all_rows))
    cache = torch.empty((len(all_rows), 512, 16, 16), dtype=torch.float16, device='cuda')
    for lo in range(0, len(all_rows), 256):
        cache[lo:lo+256].copy_(torch.from_numpy(np.array(maps[all_rows[lo:lo+256]], copy=True)))
    ep_gpu = [{k: torch.as_tensor(ep[k], dtype=torch.long if k == 'labels' else torch.float32, device='cuda')
        for k in ('phi', 'prior', 'mean', 'conditions', 'labels')} for ep in episodes]

    def batch(ei, ix):
        local = torch.as_tensor(ix, device='cuda')
        x = cache[torch.as_tensor(lookup[episodes[ei]['rows'][ix]], device='cuda')].float()
        args = [ep_gpu[ei][k][local] for k in ('phi', 'prior', 'mean', 'conditions')]
        return x, args, ep_gpu[ei]['labels'][local]

    if not (dst/'preserved_training.json').exists():
        teacher = model_from(dst/'independent.pt')
        main = model_from(folder/'selective.pt', 'selective')
        standalone = model_from(dst/'standalone.pt')
        anchors, archive = {}, []
        with torch.no_grad():
            for epoch, batches in enumerate(plan):
                for bi, (ei, ix) in enumerate(batches):
                    x, args, y = batch(ei, ix)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        tz, _ = teacher(x, args[-1])
                        mz, _ = main(x, args[-1])
                        sz, _ = standalone(x, args[-1])
                    reference = support_scores(*args, mz, y[:64], info['eta'])[64:]
                    scores = per_band_scores(args, tz, y[:64], info['eta'])[64:]
                    old = per_band_scores(args, sz, y[:64], info['eta'])[64:]
                    mask, margin = unique_anchors(scores, reference, y[64:])
                    anchors[epoch, bi] = (reference.detach(), mask, margin)
                    archive.append(dict(epoch=epoch, episode=ei, rows=episodes[ei]['rows'][ix[64:]],
                        labels=y[64:].cpu().numpy(), mask=mask.cpu().numpy(),
                        margin=margin.cpu().numpy(), standalone=old.argmax(-1).cpu().numpy()))
        del teacher, main, standalone
        torch.manual_seed(20261009)
        model = model_from(folder/'fixed.pt').train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
        logs, grad_diagnostic, started = [], None, time.perf_counter()
        print(f'PRESERVE CACHE {task}: train={len(used)} cached={len(all_rows)} anchor_events={sum(int(x[1].sum()) for x in anchors.values())}', flush=True)
        for epoch, batches in enumerate(plan):
            rec = []
            for bi, (ei, ix) in enumerate(batches):
                x, args, y = batch(ei, ix)
                reference, mask, margin = anchors[epoch, bi]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    z, _ = model(x, args[-1])
                scores = per_band_scores(args, z, y[:64], info['eta'])[64:]
                common, ce, _ = branch_objective(scores, reference, y[64:])
                aux = auxiliary_objective(z, args[-1], y[:64], reference, y[64:], 'standalone')
                prior_loss = common+.25*aux
                preserve = retention_loss(scores, y[64:], mask, margin)
                if grad_diagnostic is None and preserve.detach() > 0:
                    params = list(model.parameters())
                    g = torch.cat([v.flatten() for v in torch.autograd.grad(prior_loss, params, retain_graph=True)])
                    h = torch.cat([v.flatten() for v in torch.autograd.grad(.25*preserve, params, retain_graph=True)])
                    grad_diagnostic = dict(original_norm=float(g.norm()), protection_norm=float(h.norm()),
                        cosine=float(torch.nn.functional.cosine_similarity(g[None], h[None])))
                loss = prior_loss+.25*preserve
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite preservation loss')
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5))
                optimizer.step()
                rec.append([float(ce.detach()), float(aux.detach()), float(preserve.detach()), norm])
            values = np.mean(rec, axis=0)
            logs.append(dict(epoch=epoch+1, ce=float(values[0]), auxiliary=float(values[1]),
                preservation=float(values[2]), max_gradient=max(v[3] for v in rec)))
            print(f'PRESERVE TRAIN {task} {epoch+1}/3 CE={values[0]:.5f} preserve={values[2]:.5f}', flush=True)
        model.eval()
        predictions = []
        with torch.no_grad():
            for batches in plan:
                for ei, ix in batches:
                    x, args, y = batch(ei, ix)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        z, _ = model(x, args[-1])
                    predictions.append(per_band_scores(args, z, y[:64], info['eta'])[64:].argmax(-1).cpu().numpy())
        fields = {k: np.concatenate([a[k] for a in archive]) for k in ('rows', 'labels', 'mask', 'margin', 'standalone')}
        fields.update(preserved=np.concatenate(predictions), epoch=np.concatenate([np.full(len(a['rows']), a['epoch']) for a in archive]),
            episode=np.concatenate([np.full(len(a['rows']), a['episode']) for a in archive]))
        np.savez_compressed(dst/'training_anchors.npz', **fields)
        mask, labels = fields['mask'], fields['labels'][:, None]
        torch.save(dict(state=model.state_dict(), source_only=True, epochs=3), dst/'preserved.pt')
        write(dst/'preserved_training.json', dict(source_only=True, source_validation_query_overlap=0, epochs=logs,
            trainable_parameters=sum(p.numel() for p in model.parameters()), training_rows=len(used),
            cached_rows=len(all_rows), anchor_events=int(mask.sum()), anchor_per_band=mask.sum(0).tolist(),
            anchor_distinct_source_rows=int(len(np.unique(fields['rows'][mask.any(1)]))),
            standalone_anchor_correct=int(np.sum(mask & (fields['standalone'] == labels))),
            preserved_anchor_correct=int(np.sum(mask & (fields['preserved'] == labels))),
            first_active_gradient=grad_diagnostic, seconds=time.perf_counter()-started))
        del model, optimizer, anchors, ep_gpu
    model = model_from(dst/'preserved.pt')
    with torch.inference_mode():
        for ei, old in encoded:
            path = src/f'episode_{ei:02d}.npz'
            if path.exists():
                continue
            ix, chunks = old['indices'], []
            for lo in range(0, len(ix), 128):
                rows = episodes[ei]['rows'][ix[lo:lo+128]]
                x = cache[torch.as_tensor(lookup[rows], device='cuda')].float()
                c = torch.as_tensor(episodes[ei]['conditions'][ix[lo:lo+128]], dtype=torch.float32, device='cuda')
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    z, _ = model(x, c)
                chunks.append(z.cpu().numpy())
            np.savez_compressed(path, **{k: old[k] for k in ('old', 'learned', 'independent', 'standalone', 'indices', 'expected')},
                preserved=np.concatenate(chunks))
    write(src/'ENCODED.json', dict(dataset=ds, task=task, queries=len(val['labels']), train_query_overlap=0))
    del cache, model
    torch.cuda.empty_cache()
    print(f'PRESERVE SOURCE ENCODE {task}', flush=True)


def encode_target(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    dst = OUT/'tasks'/info['dataset']/info['task']
    if (dst/'preserved_embedding.npz').exists():
        return
    data = load_npz(folder/'target.npz')
    maps = np.load(EXP/'maps'/info['dataset']/'maps.npy', mmap_mode='r')
    np.testing.assert_array_equal(np.load(EXP/'maps'/info['dataset']/'sample_ids.npy')[data['rows']], data['ids'])
    z = encode({'preserved': model_from(dst/'preserved.pt')}, maps, data['rows'], data['conditions'])['preserved']
    np.savez_compressed(dst/'preserved_embedding.npz', embedding=z, rows=data['rows'])
    print(f'PRESERVE TARGET ENCODE {info["task"]}', flush=True)
