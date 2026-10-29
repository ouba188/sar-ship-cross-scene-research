"""Paired fixed-endpoint source fitting of legacy and spatial M1/M2."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
PACKAGE = WORKSPACE/'sar_m123_standalone'
RELEASE = WORKSPACE/'sar_m123_current_20261027'
sys.path.insert(0, str(PACKAGE))
from sar_m123.current.data import read_manifest, prepare_record, batch, pair_mask
from sar_m123.current.model import load_model, save_model, setup, training_policy, LIVE
from sar_m123.current.spatial_m12 import from_legacy
from sar_m123.current.training import class_weights, fitting_loss
from sar_m123.input import prepare
from sar_m123.regions import soft_regions

spec = importlib.util.spec_from_file_location('spatial_refit_cache', HERE.parent/'selective_m123_20261027/run.py')
cached = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cached)
ARMS = ('legacy', 'spatial')
SEEDS = (17, 29, 43)
EPOCHS = 8
BATCH = 24


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def model(arm):
    net = load_model(RELEASE/'weights/init.pt')
    if arm == 'spatial':
        net = from_legacy(net)
    return net.cuda().eval()


def gpu_cache(arrays, indices):
    ids = np.unique(np.concatenate(list(indices.values())))
    mapping = dict(prefix='prefix', masks='masks', valid='valid', available='available',
        slot_masks='slots', frequency='energy', local_slot_masks='local_slots',
        local_frequency='local_energy', field2='field2', field3='field3')
    values = {}
    tick = time.perf_counter()
    for name, source in mapping.items():
        array = arrays[source]
        values[name] = torch.empty((len(ids), *array.shape[1:]), device='cuda', dtype=torch.float32)
        for first in range(0, len(ids), 16):
            values[name][first:first+16].copy_(torch.as_tensor(
                np.array(array[ids[first:first+16]], dtype=np.float32), device='cuda'))
    torch.cuda.synchronize()
    arrays['_gpu'] = values
    arrays['_lookup'] = {int(i): j for j, i in enumerate(ids)}
    print('GPU_INPUT_CACHE', len(ids), round(torch.cuda.memory_allocated()/2**30, 3),
          round(time.perf_counter()-tick, 1), flush=True)


def get_batch(arrays, ids):
    ix = torch.tensor([arrays['_lookup'][int(i)] for i in ids], device='cuda')
    data = {k: v.index_select(0, ix) for k, v in arrays['_gpu'].items()}
    return ([data[k] for k in ('prefix', 'masks', 'valid', 'available')],
            {k: v for k, v in data.items() if k not in ('prefix', 'masks', 'valid', 'available')})


@torch.no_grad()
def precise_prefix(rows, arrays, indices):
    root = HERE/'cache'
    root.mkdir(exist_ok=True)
    # Sparse original indexes preserve the existing validated geometry caches.
    targets = {int(i): r for key in rows for i, r in zip(indices[key], rows[key])}
    receipt_path = root/'complete.json'
    if not receipt_path.exists():
        net = model('legacy')
        prefix = np.lib.format.open_memmap(root/'prefix.npy', mode='w+', dtype='float32',
            shape=arrays['prefix'].shape)
        masks = np.lib.format.open_memmap(root/'masks.npy', mode='w+', dtype='float32',
            shape=arrays['masks'].shape)
        keys = sorted(targets)
        tick = time.perf_counter()
        for first in range(0, len(keys), 16):
            ix = keys[first:first+16]
            images = np.zeros((len(ix), 2, 1, 160, 160), np.float32)
            native_masks = np.zeros((len(ix), 2, 3, 160, 160), np.float32)
            for j, idx in enumerate(ix):
                for p, view in enumerate(prepare(targets[idx])):
                    if view is not None and view['valid_target']:
                        images[j, p, 0] = view['pixels'].astype(np.float32)/255
                        native_masks[j, p] = soft_regions(view['geometry'])*view['footprint']
            prefix[ix] = net.prefix(torch.as_tensor(images, device='cuda'), stop=2).cpu().numpy()
            masks[ix] = native_masks
            if first % 160 == 0:
                print('FLOAT32_PREFIX', first, len(keys), round(time.perf_counter()-tick, 1), flush=True)
        prefix.flush()
        masks.flush()
        write(receipt_path, dict(count=len(keys), original_indices=keys, dtype='float32',
            description='Raw native chips; no quantized backbone prefix; no learnable module cached.'))
        del net, prefix, masks
    else:
        assert json.loads(receipt_path.read_text(encoding='utf-8'))['original_indices'] == sorted(targets)
    arrays['prefix'] = np.load(root/'prefix.npy', mmap_mode='r')
    arrays['masks'] = np.load(root/'masks.npy', mmap_mode='r')


def load():
    rows = {k: read_manifest(RELEASE/f'manifests/{v}.json') for k, v in (
        ('fit', 'source_fit'), ('development', 'development'), ('references', 'references'))}
    table, arrays = cached.load()
    lookup = {str(int(v)): i for i, v in enumerate(table.obs_id)}
    indices = {k: np.array([lookup[str(r['id'])] for r in group]) for k, group in rows.items()}
    for key, group in rows.items():
        assert len({r['id'] for r in group}) == len(group)
        for r, idx in zip(group, indices[key]):
            assert int(table.iloc[idx].class_id) == r['label']
            assert str(table.iloc[idx].mmsi) == str(r['mmsi'])
    for key in ('id', 'mmsi', 'pass_group'):
        assert not ({str(r[key]) for r in rows['fit']} &
                    {str(r[key]) for r in rows['development']}), key
    precise_prefix(rows, arrays, indices)
    gpu_cache(arrays, indices)
    values, extra = get_batch(arrays, indices['references'])
    bank = dict(zip(('prefix', 'masks', 'valid', 'available'), values), **extra)
    write(HERE/'cohorts.json', rows)
    return rows, arrays, indices, bank


def score(net, arrays, ids, rows, references, bank, details=False):
    values, extra = get_batch(arrays, ids)
    return net.forward_from(*values, **extra, reference=bank,
        pair_valid=pair_mask(rows, references, 'cuda'), return_details=details)


@torch.no_grad()
def predict(net, arrays, ids, rows, references, bank):
    return np.concatenate([score(net, arrays, ids[i:i+BATCH], rows[i:i+BATCH],
        references, bank).cpu().numpy() for i in range(0, len(ids), BATCH)])


def save_scores(name, scores, rows):
    (HERE/'predictions').mkdir(exist_ok=True)
    np.savez(HERE/'predictions'/f'{name}.npz', score=scores,
        ids=np.array([str(r['id']) for r in rows]), y=np.array([r['label'] for r in rows]))


def check(rows, arrays, indices, bank):
    net = model('legacy')
    native_rows = [rows['fit'][0], rows['fit'][-1], rows['development'][0]]
    native_ids = np.array([indices['fit'][0], indices['fit'][-1], indices['development'][0]])
    values = batch([prepare_record(r) for r in native_rows], 'cuda')
    refs = batch([prepare_record(r) for r in rows['references']], 'cuda')
    with torch.no_grad():
        refs['prefix'] = net.prefix(refs['images'], stop=2)
        native_prefix = net.prefix(values['images'], stop=2)
        original = net.forward_from(native_prefix,
            **{k: v for k, v in values.items() if k != 'images'}, reference=refs,
            pair_valid=pair_mask(native_rows, rows['references'], 'cuda'))
        replay = score(net, arrays, native_ids, native_rows, rows['references'], bank)
    error = (original-replay).abs().max().item()
    old_values, old_extra = get_batch(arrays, native_ids)
    input_difference = {k: (v-old_extra[k]).abs().max().item() for k, v in values.items()
                        if k in old_extra}
    input_difference.update({k: (values[k]-old_values[i]).abs().max().item()
                             for i, k in enumerate(('prefix', 'masks', 'valid', 'available')) if i > 0})
    input_difference['prefix'] = (native_prefix-old_values[0]).abs().max().item()
    input_difference['reference_prefix'] = (refs['prefix']-bank['prefix']).abs().max().item()
    write(HERE/'native_cache_diagnosis.json', dict(logit_error=error, inputs=input_difference))
    assert error < 1e-4, ('native_vs_cached', error)
    del net, original, replay, refs, values
    initial = {}
    checks = dict(native_cache_max_error=error, frozen_prefix_only=True, arms={})
    for arm in ARMS:
        net = model(arm)
        initial[arm] = predict(net, arrays, indices['development'], rows['development'], rows['references'], bank)
        save_scores(f'initial_{arm}', initial[arm], rows['development'])
        training_policy(net, 'fit')
        chosen = np.arange(3)
        qrows = [rows['fit'][i] for i in chosen]
        logits, detail = score(net, arrays, indices['fit'][chosen], qrows, rows['references'], bank, True)
        loss, _ = fitting_loss(logits, detail, torch.tensor([r['label'] for r in qrows], device='cuda'),
            torch.ones(3, device='cuda'), torch.tensor([r['label'] for r in rows['references']], device='cuda'))
        loss.backward()
        gradients = {n: p.grad.norm().item() for n, p in net.named_parameters() if p.grad is not None}
        for prefix in ('internal_m1.', 'internal_m2.', 'm3.', 'head.'):
            assert any(n.startswith(prefix) and v > 0 for n, v in gradients.items()), (arm, prefix)
        assert all(gradients[n] > 0 for n in LIVE)
        assert all(p.grad is None for p in net.parameters() if not p.requires_grad)
        checks['arms'][arm] = dict(trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
            gradients=gradients)
        del net, logits, detail, loss
    checks['initial_arms_max_error'] = float(abs(initial['legacy']-initial['spatial']).max())
    assert checks['initial_arms_max_error'] < 1e-5
    write(HERE/'checks.json', checks)
    print('CHECK_PASS', error, checks['initial_arms_max_error'], flush=True)


def train(rows, arrays, indices, bank, arms=ARMS, seeds=SEEDS):
    weights = class_weights(rows['fit'], 'fit')
    reference_labels = torch.tensor([r['label'] for r in rows['references']], device='cuda')
    for seed in seeds:
        for arm in arms:
            out = HERE/'training'/f'{arm}_s{seed}'
            if (out/'training.json').exists():
                raise RuntimeError(f'Completed run already exists: {out}')
            net = model(arm)
            training_policy(net, 'fit')
            frozen = {n: p.detach().cpu().clone() for n, p in net.state_dict().items()
                      if n not in {k for k, v in net.named_parameters() if v.requires_grad}}
            params = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW([
                dict(params=[p for n, p in params if n in LIVE], lr=3e-5),
                dict(params=[p for n, p in params if n not in LIVE], lr=.001)], weight_decay=.01)
            rng = np.random.default_rng(seed)
            curve, steps = [], 0
            tick = time.perf_counter()
            for epoch in range(1, EPOCHS+1):
                terms_all = []
                for ix in np.array_split(rng.permutation(len(rows['fit'])),
                                         range(BATCH, len(rows['fit']), BATCH)):
                    qrows = [rows['fit'][i] for i in ix]
                    optimizer.zero_grad(set_to_none=True)
                    logits, detail = score(net, arrays, indices['fit'][ix], qrows,
                        rows['references'], bank, True)
                    loss, terms = fitting_loss(logits, detail,
                        torch.tensor([r['label'] for r in qrows], device='cuda'),
                        torch.tensor(weights[ix], device='cuda'), reference_labels)
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                    steps += 1
                    terms_all.append({k: float(v.detach()) for k, v in terms.items()})
                entry = dict(epoch=epoch, steps=steps, **{k: float(np.mean([t[k] for t in terms_all]))
                             for k in terms_all[0]})
                curve.append(entry)
                print(arm, seed, json.dumps(entry), round(time.perf_counter()-tick, 1), flush=True)
            assert all(torch.equal(net.state_dict()[n].cpu(), value) for n, value in frozen.items())
            del frozen, optimizer, logits, detail, loss
            net.requires_grad_(False)
            predictions = predict(net, arrays, indices['development'], rows['development'], rows['references'], bank)
            save_scores(f'{arm}_s{seed}', predictions, rows['development'])
            receipt = dict(arm=arm, seed=seed, stage='source_fit_only', source_count=len(rows['fit']),
                reference_count=len(rows['references']), development_count=len(rows['development']),
                completed_epochs=EPOCHS, steps=steps, curve=curve, validation_selection=False,
                trainable_parameters=sum(p.numel() for _, p in params), seconds=time.perf_counter()-tick,
                frozen_unchanged=True, selection_and_adaptation_not_run=True)
            save_model(net, out/'model.pt', **receipt)
            del net, params
            restored = load_model(out/'model.pt', 'cuda')
            check_scores = predict(restored, arrays, indices['development'][:24],
                rows['development'][:24], rows['references'], bank)
            receipt['save_replay_max_error'] = float(abs(check_scores-predictions[:24]).max())
            assert receipt['save_replay_max_error'] < 1e-5
            write(out/'training.json', receipt)
            print('COMPLETE', arm, seed, receipt['seconds'], flush=True)
            del restored
            torch.cuda.empty_cache()


def diagnose(rows, arrays, indices, bank):
    """Same trained legacy weights with M1/M2 spectral paths removed, no refit."""
    records = []
    for seed in SEEDS:
        old = load_model(HERE/'training'/f'legacy_s{seed}'/'model.pt')
        net = from_legacy(old).cuda().eval().requires_grad_(False)
        del old
        result = predict(net, arrays, indices['development'], rows['development'], rows['references'], bank)
        save_scores(f'pruned_s{seed}', result, rows['development'])
        records.append(dict(seed=seed, intervention='same trained legacy weights; remove both M1/M2 spectral paths',
            trained=False, m3_unchanged=True, scope='path dependency, not a third independently trained contender'))
        del net
        torch.cuda.empty_cache()
    write(HERE/'posthoc_pruning.json', records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('check', 'train', 'diagnose'))
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = parser.parse_args()
    setup('cuda')
    rows, arrays, indices, bank = load()
    if args.command == 'check':
        check(rows, arrays, indices, bank)
    elif args.command == 'train':
        assert (HERE/'checks.json').exists()
        train(rows, arrays, indices, bank, args.arms, args.seeds)
    else:
        diagnose(rows, arrays, indices, bank)


if __name__ == '__main__':
    main()
