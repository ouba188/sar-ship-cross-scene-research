"""Independent OSS2 runner. Import/--help are torch-free; GPU commands are deferred."""
import argparse
import importlib.util
import os
import sys
import time

from protocol import (HERE, PACKAGE, INIT, MODEL, CLASS_INDICES, CLASSES, ARMS, SEEDS,
                      BUDGET, COEFFICIENT, COUNTS, read, write, cohorts, checkpoint,
                      assert_rows, class_weights4)

# Imported shared modules must not write __pycache__ outside this task directory.
sys.dont_write_bytecode = True


def runtime(device):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    global np, torch, api, data, model_api, training, interventions
    import numpy as np
    import torch
    sys.path.insert(0, str(PACKAGE))
    from sar_m123.current import data, model as model_api, training, interventions
    spec = importlib.util.spec_from_file_location('oss2_shared_ablation_model', MODEL)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    model_api.setup(device)


def fresh(arm, device):
    return api.fresh(arm, INIT, device, class_indices=CLASS_INDICES)


def make_cache(device):
    """One native float32 cache; only frozen layer2 and fixed input fields are stored."""
    out = HERE/'cache'
    if (out/'complete.json').exists():
        Cache(device)
        print('CACHE_ALREADY_COMPLETE', flush=True)
        return
    out.mkdir(exist_ok=True)
    records = read(HERE/'manifests/all.json')
    net = fresh('full', device)
    arrays = {}
    started = time.perf_counter()
    with torch.no_grad():
        for first in range(0, len(records), 8):
            rows = records[first:first+8]
            values = data.batch([data.prepare_record(r) for r in rows], device)
            prefix = net.prefix(values.pop('images'), stop=2)
            assert prefix.shape[1:] == (2, 512, 20, 20)
            values['prefix'] = prefix
            for key, value in values.items():
                part = value.detach().cpu().numpy()
                assert part.dtype == np.float32 and np.isfinite(part).all()
                if key not in arrays:
                    arrays[key] = np.lib.format.open_memmap(out/f'{key}.npy', mode='w+',
                        dtype=np.float32, shape=(len(records), *part.shape[1:]))
                arrays[key][first:first+len(rows)] = part
            expected = np.array([[r[p] is not None for p in ('vv', 'vh')] for r in rows])
            assert np.array_equal(values['available'].cpu().numpy(), expected)
            if first % 64 == 0:
                print('CACHE', first+len(rows), len(records), flush=True)
    for array in arrays.values():
        array.flush()
    write(out/'complete.json', dict(count=len(records), ids=[r['id'] for r in records],
        labels=[r['label'] for r in records], init=str(INIT), class_indices=list(CLASS_INDICES),
        native_size=160, dtype='float32', stop=2, single_orientation=True,
        shapes={k: list(v.shape) for k, v in arrays.items()},
        seconds=time.perf_counter()-started, learned_features_cached=False))
    del arrays, net


class Cache:
    """Memory-map on disk; transfer only the requested batch, never the full pool."""
    def __init__(self, device):
        self.device = device
        self.records = read(HERE/'manifests/all.json')
        receipt = read(HERE/'cache/complete.json')
        assert receipt['ids'] == [r['id'] for r in self.records]
        assert receipt['labels'] == [r['label'] for r in self.records]
        assert receipt['class_indices'] == list(CLASS_INDICES)
        assert receipt['init'] == str(INIT) and receipt['dtype'] == 'float32'
        assert receipt['native_size'] == 160 and receipt['stop'] == 2
        self.lookup = {r['id']: i for i, r in enumerate(self.records)}
        self.arrays = {k: np.load(HERE/'cache'/f'{k}.npy', mmap_mode='r') for k in receipt['shapes']}
        assert set(self.arrays) == (set(data.SHAPES)-{'images'}) | {'prefix'}
        for key, array in self.arrays.items():
            expected = (2, 512, 20, 20) if key == 'prefix' else data.SHAPES[key]
            assert array.dtype == np.float32 and array.shape == (len(self.records), *expected)

    def batch(self, rows):
        ids = [self.lookup[r['id']] for r in rows]
        return {key: torch.as_tensor(np.array(array[ids], copy=True), device=self.device)
                for key, array in self.arrays.items()}

    def bank(self, refs):
        return self.batch(refs)

    def score(self, net, rows, refs, bank, details=False):
        return net.forward_from(**self.batch(rows), reference=bank,
            pair_valid=data.pair_mask(rows, refs, self.device), return_details=details)


def sample_rows(rows):
    """Bounded diagnostic batch: all classes, both single-pol cases, large/raw-swapped OBB."""
    candidates = rows['source_fit']+rows['source_adapt']+rows['evaluation']
    selected = {}
    conditions = [lambda r, y=y: r['label'] == y for y in range(4)]
    conditions += [lambda r: r['vv'] is not None and r['vh'] is not None,
                   lambda r: r['vv'] is None, lambda r: r['vh'] is None,
                   lambda r: max(r['native_size']) > 160,
                   lambda r: any(r[p] and r[p]['raw_xywhr'][3] > r[p]['raw_xywhr'][2]
                                 for p in ('vv', 'vh'))]
    for condition in conditions:
        record = next((r for r in candidates if condition(r)), None)
        if record is not None:
            selected[record['id']] = record
    return list(selected.values())


def native_error(net, cache, queries, refs, bank):
    native_bank = model_api.reference_bank(net, refs, cache.device)
    cached_parts, native_parts = [], []
    with torch.no_grad():
        for start in range(0, len(queries), 3):
            group = queries[start:start+3]
            values = data.batch([data.prepare_record(r) for r in group], cache.device)
            native_parts.append(model_api.forward_batch(net, values, native_bank,
                data.pair_mask(group, refs, cache.device)))
            cached_parts.append(cache.score(net, group, refs, bank))
    error = float((torch.cat(native_parts)-torch.cat(cached_parts)).abs().max())
    assert error < 1e-4, ('native_cached_logits', net.arm, error)
    return error


def train_one(arm, seed, stage, rows, cache, device):
    split = 'source_fit' if stage == 'fit' else 'source_adapt'
    records, refs = rows[split], rows['references']
    assert_rows(records, split)
    output = checkpoint(arm, seed, stage)
    if output.exists():
        raise FileExistsError(output)
    net = fresh(arm, device) if stage == 'fit' else api.load(checkpoint(arm, seed, 'fit'), device)
    assert net.arm == arm and net.head.out_features == 4
    model_api.training_policy(net, stage)
    params = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    trainable = {n for n, _ in params}
    if stage == 'adapt':
        assert trainable == set(model_api.LIVE)
    frozen = {n: p.detach().cpu().clone() for n, p in net.state_dict().items() if n not in trainable}
    live_before = {n: net.state_dict()[n].detach().cpu().clone() for n in model_api.LIVE}
    groups = [dict(params=[p for n, p in params if n in model_api.LIVE], lr=3e-5)]
    if stage == 'fit':
        groups.append(dict(params=[p for n, p in params if n not in model_api.LIVE], lr=.001))
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    weights = np.asarray(class_weights4(records, stage), dtype=np.float32)
    bank = cache.bank(refs)
    ref_labels = torch.tensor([r['label'] for r in refs], device=device)
    targets = None
    if stage == 'adapt':
        targets = read(HERE/'teachers'/f'{arm}_s{seed}.json')
        assert [(r['id'], r['label']) for r in targets] == [(r['id'], r['label']) for r in records]
        assert all(len(r['original']) == 4 for r in targets)
        if arm == 'plain':
            assert not any(r['useful'] for r in targets)
    coefficient = COEFFICIENT if arm == 'full' else 0.
    epochs, batch_size, expected_steps = BUDGET[stage]
    rng = np.random.default_rng(seed)
    curve, steps = [], 0
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        epoch_terms = []
        for ix in np.array_split(rng.permutation(len(records)), range(batch_size, len(records), batch_size)):
            queries = [records[int(i)] for i in ix]
            labels = torch.tensor([r['label'] for r in queries], device=device)
            weight = torch.as_tensor(weights[ix], device=device)
            optimizer.zero_grad(set_to_none=True)
            if stage == 'fit':
                score, detail = cache.score(net, queries, refs, bank, True)
                loss, terms = api.fitting_loss(score, detail, labels, weight, ref_labels)
            else:
                score = cache.score(net, queries, refs, bank)
                teacher = torch.tensor([targets[int(i)]['original'] for i in ix], device=device)
                useful = torch.tensor([targets[int(i)]['useful'] for i in ix], device=device)
                if steps == 0:
                    assert float((score.detach()-teacher).abs().max()) < 1e-4
                loss, terms = training.adaptation_loss(score, labels, teacher, weight, useful, coefficient)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite OSS2 training loss')
            loss.backward()
            optimizer.step()
            steps += 1
            epoch_terms.append({k: float(v.detach()) for k, v in terms.items()})
        entry = dict(epoch=epoch, steps=steps,
            **{k: float(np.mean([t[k] for t in epoch_terms])) for k in epoch_terms[0]})
        curve.append(entry)
        print(arm, seed, stage, entry, flush=True)
    assert steps == expected_steps
    assert all(torch.equal(net.state_dict()[n].cpu(), value) for n, value in frozen.items())
    updates = {n: float((net.state_dict()[n].cpu()-value).norm()) for n, value in live_before.items()}
    assert all(v > 0 for v in updates.values())
    receipt = dict(arm=arm, stage=stage, seed=seed, shared_init=str(INIT),
        classes=list(CLASSES), class_indices=list(CLASS_INDICES), source_count=len(records),
        reference_count=len(refs), steps=steps, completed_epochs=epochs, batch_size=batch_size,
        coefficient=coefficient if stage == 'adapt' else None, curve=curve,
        selected_count=None if targets is None else sum(r['useful'] for r in targets),
        validation_selection=False, frozen_unchanged=True, live_updates=updates,
        trainable_parameters=sum(p.numel() for _, p in params), seconds=time.perf_counter()-started)
    api.save(net, output, **receipt)
    restored = api.load(output, device)
    queries = sample_rows(rows)
    with torch.no_grad():
        error = max(float((cache.score(net, queries[i:i+3], refs, bank)-
                           cache.score(restored, queries[i:i+3], refs, bank)).abs().max())
                    for i in range(0, len(queries), 3))
    assert error < 1e-5 and restored.arm == arm and restored.head.out_features == 4
    receipt.update(save_replay_error=error, native_cache_error=native_error(net, cache, queries, refs, bank))
    write(output.parent/'training.json', receipt)
    print('TRAIN_COMPLETE', arm, seed, stage, flush=True)


def select_one(arm, seed, rows, cache, device):
    output = HERE/'teachers'/f'{arm}_s{seed}.json'
    if output.exists():
        raise FileExistsError(output)
    records, refs = rows['source_adapt'], rows['references']
    assert_rows(records, 'source_adapt')
    net = api.load(checkpoint(arm, seed, 'fit'), device)
    assert net.arm == arm and net.head.out_features == 4
    bank = cache.bank(refs)
    with torch.no_grad():
        if arm == 'full':
            def progress(done, total, record):
                if done % 16 == 0 or done == total:
                    print('SELECT', arm, seed, done, total, flush=True)
            targets = interventions.select_records(net, records, refs, device, progress=progress)
        else:
            targets = []
            for first in range(0, len(records), 16):
                queries = records[first:first+16]
                score = cache.score(net, queries, refs, bank).cpu().tolist()
                targets.extend(dict(id=r['id'], label=r['label'], original=s, useful=False)
                               for r, s in zip(queries, score))
        errors = []
        for first in range(0, len(records), 16):
            score = cache.score(net, records[first:first+16], refs, bank)
            teacher = torch.tensor([r['original'] for r in targets[first:first+16]], device=device)
            errors.append(float((score-teacher).abs().max()))
    assert max(errors) < 1e-4, ('own_teacher_cached_original', max(errors))
    write(output, targets)
    write(output.with_suffix('.receipt.json'), dict(arm=arm, seed=seed, count=len(targets),
        own_teacher=str(checkpoint(arm, seed, 'fit')), selected=sum(r['useful'] for r in targets),
        original_cache_error=max(errors), implementation='current.select_records' if arm == 'full' else 'own_original_only'))


def evaluate(arm, seed, rows, cache, device):
    refs = rows['references']
    bank = cache.bank(refs)
    for stage in ('fit', 'adapt'):
        net = api.load(checkpoint(arm, seed, stage), device)
        assert net.arm == arm and net.head.out_features == 4
        with torch.no_grad():
            for scope in ('evaluation',):
                records = rows[scope]
                parts = [cache.score(net, records[i:i+16], refs, bank).cpu().numpy()
                         for i in range(0, len(records), 16)]
                scores = np.concatenate(parts)
                assert scores.shape == (COUNTS[scope], 4) and np.isfinite(scores).all()
                output = HERE/'predictions'/f'{scope}_{arm}_{stage}_s{seed}.npz'
                output.parent.mkdir(exist_ok=True)
                np.savez_compressed(output, score=scores, ids=np.array([r['id'] for r in records]),
                    y=np.array([r['label'] for r in records]), classes=np.array(CLASSES))
        write(HERE/'evaluation'/f'{arm}_{stage}_s{seed}.json', dict(arm=arm, seed=seed, stage=stage,
            primary='evaluation378', available_only=['validation228', 'selection194'],
            checkpoint_selection=False, native_cache_error=native_error(net, cache, sample_rows(rows), refs, bank)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('cache', 'check', 'fit', 'select', 'adapt', 'evaluate', 'all'))
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    from checkreadiness import check
    check()
    runtime(args.device)
    if args.command in ('cache', 'all'):
        make_cache(args.device)
        if args.command == 'cache':
            return
    rows, cache = cohorts(), Cache(args.device)
    if args.command in ('check', 'all'):
        from tests_gpu import check_gpu
        check_gpu(sys.modules[__name__], rows, cache, args.device)
        if args.command == 'check':
            return
    checks = read(HERE/'gpu_checks.json')
    assert checks['status'] == 'passed' and checks['class_indices'] == list(CLASS_INDICES)
    for seed in args.seeds:
        for arm in args.arms:
            for step in ('fit', 'select', 'adapt', 'evaluate'):
                if args.command not in (step, 'all'):
                    continue
                if step in ('fit', 'adapt'):
                    train_one(arm, seed, step, rows, cache, args.device)
                elif step == 'select':
                    select_one(arm, seed, rows, cache, args.device)
                else:
                    evaluate(arm, seed, rows, cache, args.device)
                if args.device.startswith('cuda'):
                    torch.cuda.empty_cache()
    print('COMPLETE', args.command, flush=True)


if __name__ == '__main__':
    main()
