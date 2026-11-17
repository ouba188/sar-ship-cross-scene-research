"""Conditional-modulation experiments; importing this file never loads torch.

Main process examples: --phase tune; --phase tune --arms head_only;
--test-gradients --arms affine_head. Gradient checks use disposable models,
not training checkpoints. Tune/confirm never load evaluation pixels.
Model owns requires_grad and branch disabling; this runner never refreezes it.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
ARMS = ('head_only', 'broadcast_head', 'affine_frozen', 'affine_head')
SEEDS = (17, 29, 43)
EPOCHS = 4
CONFIGS = {
    'base': dict(reader=1e-5, exchange=1e-3, head=1e-5),
    'slow': dict(reader=1e-5, exchange=3e-4, head=1e-5),
    'head_small': dict(reader=1e-5, exchange=1e-3, head=1e-6),
}
EVALUATION_NOTE = ('Historically exposed evaluation378; held-out frozen confirmation, '
                   'not a new blind test; no evaluation-based selection.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def schedule_factor(epoch):
    return (.5, 1., .55, .1)[epoch]


def effective_config(arm, config):
    if (arm == 'head_only' and config == 'slow') or (arm == 'affine_frozen' and config == 'head_small'):
        return 'base'
    return config


def destination(arm, seed, config):
    effective = effective_config(arm, config)
    parent = ROOT / 'tuning' / effective if seed == 17 else ROOT / 'confirm'
    return parent / f'{arm}_s{seed}'


def parameter_groups(net, arm, config):
    buckets = {key: [] for key in ('reader', 'exchange', 'head')}
    names = {key: [] for key in buckets}
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('net.guide.head.'):
            key = 'head'
        elif name.startswith('ccrel.value.'):
            key = 'reader'
        elif name.startswith('exchange.'):
            key = 'exchange'
        else:
            raise ValueError('Unexpected trainable parameter; check Model scope: ' + name)
        buckets[key].append(parameter)
        names[key].append(name)
    expected = {'head'} if arm == 'head_only' else {'reader', 'exchange'}
    if arm in ('broadcast_head', 'affine_head'):
        expected.add('head')
    assert {key for key, values in buckets.items() if values} == expected, names
    if arm == 'head_only':
        assert net.exchange.disabled, 'Model must disable the branch for head_only'
    groups, description = [], []
    for key, parameters in buckets.items():
        if not parameters:
            continue
        rate = CONFIGS[config][key]
        groups.append(dict(name=key, params=parameters, lr=rate, base_lr=rate))
        description.append(dict(name=key, base_lr=rate,
                                parameter_names=names[key],
                                parameter_count=sum(p.numel() for p in parameters),
                                epoch_learning_rates=[rate*schedule_factor(e) for e in range(EPOCHS)]))
    return groups, description


def runtime():
    model = load('conditional_modulation_model', ROOT / 'model.py')
    assert set(ARMS).issubset(model.ARMS)
    inputs = model.full.Inputs()
    for split in ('small', 'validation'):
        inputs.prepare_geometry(split)
    assert tuple(len(inputs.groups[k]) for k in ('small', 'validation', 'references')) == (719, 228, 32)
    assert set(inputs.groups['references']).issubset(inputs.groups['small'])
    # Inputs supplies the existing ship/pass/id split-disjointness assertions.
    return model, inputs


def validation_key(metrics):
    return [metrics['ba'], metrics['oa'], -metrics['nll']]


def replay(model, ids, score, path):
    np = model.np
    with np.load(path, allow_pickle=False) as original:
        assert np.array_equal(ids, original['ids'])
        error = float(abs(score-original['score']).max())
        changes = int((score.argmax(1) != original['score'].argmax(1)).sum())
    assert error < 1e-3 and changes == 0, (str(path), error, changes)
    return dict(max_abs_error=error, prediction_diff=changes)


def describe(arm, seed, config):
    dest = destination(arm, seed, config)
    receipt = read(dest / 'receipt.json')
    effective = effective_config(arm, config)
    assert (receipt['arm'], receipt['seed'], receipt['config']) == (arm, seed, effective)
    assert receipt['steps'] == 120 and receipt['epochs'] == 4
    best = receipt['best']
    return dict(arm=arm, seed=seed, config=config, effective_fit_config=effective,
                reused_identical_fit=effective != config,
                reuse_reason='Changed LR belongs only to a frozen parameter group' if effective != config else None,
                learning_rates=CONFIGS[config], epoch=best['epoch'],
                metrics=best['metrics'], key=validation_key(best['metrics']),
                checkpoint=str((dest / 'best.pt').resolve()),
                validation_predictions=str((dest / 'validation.npz').resolve()),
                receipt=str((dest / 'receipt.json').resolve()))


def fit(model, inputs, arm, seed, config):
    dest = destination(arm, seed, config)
    if (dest / 'receipt.json').exists():
        return describe(arm, seed, config)
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(f'Incomplete run exists at {dest}; do not overwrite an active fit')
    torch, np, full = model.torch, model.np, model.full
    effective = effective_config(arm, config)
    torch.manual_seed(seed)
    net = model.Model(seed, arm)
    model.calibrate(net, inputs, seed)
    net.accelerate()
    groups, group_info = parameter_groups(net, arm, effective)
    params = [p for group in groups for p in group['params']]
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    ids, val = inputs.groups['small'], inputs.groups['validation']
    score = full.predict(net, inputs, val, 'validation')
    initial_replay = replay(model, val, score, full.ROOT / f'selected_s{seed}/validation.npz')
    initial = full.metrics(inputs.y[val], score)
    best = dict(epoch=0, metrics=initial, key=validation_key(initial))
    cfg = dict(arm=arm, seed=seed, config=effective, source=719, validation=228,
               references=32, epochs=EPOCHS, expected_steps=120, batch=24, microbatch=12,
               optimizer='AdamW', weight_decay=.01, clip_grad_norm=5.,
               learning_rates=CONFIGS[effective], parameter_groups=group_info,
               trainable_scope='Model requires_grad preserved; named reader/exchange/head groups only',
               schedule=[schedule_factor(e) for e in range(EPOCHS)],
               order_rng_seed=seed+20261116, augmentation_rng_seed=seed+20261117,
               loss='unchanged class/vessel weighted CE + .02 M1 auxiliary CE',
               augmentation='unchanged paired rot90/flip and native geometry',
               calibration='model.calibrate(net, inputs, seed); source719 only, seed-specific bank',
               selection='validation BA/OA/-NLL; epoch0 eligible; ties retain earlier epoch',
               initial=initial, initial_replay=initial_replay,
               head_only_reuse='base and slow share the base fit; head_small is a separate head-rate fit' if arm == 'head_only' else None,
               evaluation_loaded=False, fine_residual=False,
               split_note='existing validation, not independent test')
    dest.mkdir(parents=True, exist_ok=True)
    write(dest / 'config.json', cfg)
    torch.save(net.state_dict(), dest / 'best.pt')
    np.savez_compressed(dest / 'validation.npz', ids=val, label=inputs.y[val], score=score)
    rng = np.random.default_rng(seed+20261116)
    aug = np.random.default_rng(seed+20261117)
    curve, steps = [], 0
    tick = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        factor = schedule_factor(epoch)
        for group in opt.param_groups:
            group['lr'] = group['base_lr']*factor
        net.train()
        order = rng.permutation(ids)
        total = correct = 0
        for first in range(0, len(order), 24):
            batch = order[first:first+24]
            opt.zero_grad(set_to_none=True)
            code = int(aug.integers(4))+4*int(aug.random() < .5)
            for start in range(0, len(batch), 12):
                part = batch[start:start+12]
                loss, logits = full.loss_for(net, inputs, part, code)
                assert torch.isfinite(loss)
                (loss*len(part)/len(batch)).backward()
                total += float(loss.detach())*len(part)
                correct += int((logits.argmax(1) == torch.as_tensor(inputs.y[part], device=logits.device)).sum())
            torch.nn.utils.clip_grad_norm_(params, 5.)
            opt.step()
            steps += 1
        score = full.predict(net, inputs, val, 'validation')
        metrics = full.metrics(inputs.y[val], score)
        curve.append(dict(epoch=epoch+1, validation=metrics, train_loss=total/len(ids),
                          train_oa=100.*correct/len(ids), seconds=time.perf_counter()-tick,
                          learning_rates={g['name']: g['lr'] for g in opt.param_groups}))
        if validation_key(metrics) > best['key']:
            best = dict(epoch=epoch+1, metrics=metrics, key=validation_key(metrics))
            torch.save(net.state_dict(), dest / 'best.pt')
            np.savez_compressed(dest / 'validation.npz', ids=val, label=inputs.y[val], score=score)
        write(dest / 'curve.json', curve)
        print('VAL', arm, seed, effective, epoch+1, metrics, 'best', best['epoch'], flush=True)
    assert steps == 120
    torch.save(net.state_dict(), dest / 'last.pt')
    np.savez_compressed(dest / 'lastvalidation.npz', ids=val, label=inputs.y[val], score=score)
    write(dest / 'receipt.json', dict(**cfg, best=best, last=curve[-1]['validation'],
                                     steps=steps, seconds=time.perf_counter()-tick,
                                     trainable=sum(p.numel() for p in params),
                                     peak_gib=torch.cuda.max_memory_allocated()/1024**3))
    del net, opt, groups, params, loss, logits
    gc.collect()
    torch.cuda.empty_cache()
    return describe(arm, seed, config)


def test_gradients(model, inputs, arm, seed=17, config='base'):
    """Two disposable backward passes with one update; caller opts into GPU use."""
    torch, full = model.torch, model.full
    torch.manual_seed(seed)
    net = model.Model(seed, arm)
    model.calibrate(net, inputs, seed)
    net.accelerate()
    groups, info = parameter_groups(net, arm, config)
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    ids = inputs.groups['small'][:12]
    passes = []
    for stage in range(2):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss, _ = full.loss_for(net, inputs, ids, 0)
        assert torch.isfinite(loss)
        loss.backward()
        stats = {}
        for group in groups:
            grads = [p.grad for p in group['params'] if p.grad is not None]
            assert all(bool(torch.isfinite(g).all()) for g in grads), group['name']
            stats[group['name']] = dict(tensors_with_grad=len(grads),
                                       nonzero_tensors=sum(bool(g.abs().max() > 0) for g in grads),
                                       norm=float(sum(float(g.detach().square().sum()) for g in grads)**.5))
        assert all(p.grad is None for p in net.parameters() if not p.requires_grad)
        passes.append(dict(stage='initial' if stage == 0 else 'after_one_update',
                           loss=float(loss.detach()), groups=stats))
        if stage == 0:
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 5.)
            opt.step()
    report = dict(arm=arm, seed=seed, config=config, parameter_groups=info, passes=passes,
                  note='Disposable one-step check; no checkpoint changed. Initial zero-gated gradients may be zero.')
    report['passed'] = all(passes[-1]['groups'][g['name']]['nonzero_tensors'] > 0 for g in groups)
    write(ROOT / f'gradients_{arm}_s{seed}_{config}.json', report)
    del net, opt, groups, loss
    gc.collect()
    torch.cuda.empty_cache()
    assert report['passed'], f'Group disconnected or still zero after one update: {arm}; inspect gradient report'
    return report


def tune(model, inputs, arms):
    path = ROOT / 'selection.json'
    selection = read(path) if path.exists() else dict(
        status='partial', selection_seed=17, selection_split='validation228',
        rule='BA/OA/-NLL; epoch0 eligible; config ties choose base',
        configs=CONFIGS, arms={}, candidates={}, evaluation_loaded=False)
    for arm in arms:
        if arm in selection['arms'] and {r['config'] for r in selection['candidates'][arm]} == set(CONFIGS):
            print('SKIP_FROZEN_ARM', arm, flush=True)
            continue
        candidates = [fit(model, inputs, arm, 17, config) for config in CONFIGS]
        selection['configs'] = CONFIGS
        selection['candidates'][arm] = candidates
        selection['arms'][arm] = max(candidates, key=lambda row: row['key'])
        selection['status'] = 'frozen' if set(selection['arms']) == set(ARMS) else 'partial'
        write(path, selection)
    update_selected(selection)


def update_selected(selection):
    selected = {'17': selection['arms']}
    for seed in (29, 43):
        rows = {}
        for arm, entry in selection['arms'].items():
            if (destination(arm, seed, entry['config']) / 'receipt.json').exists():
                rows[arm] = describe(arm, seed, entry['config'])
        if rows:
            selected[str(seed)] = rows
    write(ROOT / 'selected_checkpoints.json', selected)
    return selected


def confirm(model, inputs, arms):
    selection = read(ROOT / 'selection.json')
    for arm in arms:
        if arm not in selection['arms']:
            raise ValueError(f'Finish both validation configurations for {arm} first')
        for seed in (29, 43):
            fit(model, inputs, arm, seed, selection['arms'][arm]['config'])
            update_selected(selection)


def evaluate(model, inputs, arms):
    selection = read(ROOT / 'selection.json')
    assert selection['status'] == 'frozen', 'Freeze all four arms before evaluation'
    selected = update_selected(selection)
    for seed in SEEDS:
        assert all(arm in selected.get(str(seed), {}) for arm in arms), 'Finish requested confirmation first'
    np, torch, full = model.np, model.torch, model.full
    inputs.load_split('evaluation')
    inputs.prepare_geometry('evaluation')
    ids = inputs.groups['evaluation']
    assert len(ids) == 378
    summaries = []
    for seed in SEEDS:
        with np.load(full.ROOT / f'selected_s{seed}/evaluation.npz', allow_pickle=False) as initial:
            assert np.array_equal(ids, initial['ids'])
            baseline = initial['score'].copy()
        labels = inputs.y[ids]
        for arm in arms:
            entry = selected[str(seed)][arm]
            dest = ROOT / 'evaluation' / f'{arm}_s{seed}'
            if (dest / 'receipt.json').exists():
                report = read(dest / 'receipt.json')
                assert report['selected'] == entry
                summaries.append(report)
                continue
            net = model.Model(seed, arm)
            net.load_state_dict(torch.load(entry['checkpoint'], map_location='cuda', weights_only=False))
            net.accelerate()
            val = inputs.groups['validation']
            check = replay(model, val, full.predict(net, inputs, val, 'validation'), entry['validation_predictions'])
            score = full.predict(net, inputs, ids, 'evaluation')
            old_good, new_good = baseline.argmax(1) == labels, score.argmax(1) == labels
            rescue, harm = int((~old_good & new_good).sum()), int((old_good & ~new_good).sum())
            report = dict(seed=seed, arm=arm, selected=entry, note=EVALUATION_NOTE,
                          metrics=full.metrics(labels, score), initial=full.metrics(labels, baseline),
                          rescue=rescue, harm=harm, net=rescue-harm, validation_replay=check,
                          baseline_note='Original Full cache; trained head branch-off is NOT original Full')
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / 'predictions.npz', ids=ids, label=labels,
                                initial_full=baseline, score=score)
            write(dest / 'receipt.json', report)
            summaries.append(report)
            del net
            gc.collect()
            torch.cuda.empty_cache()
    write(ROOT / 'evaluation_summary.json', dict(note=EVALUATION_NOTE, requested_arms=arms, rows=summaries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('tune', 'confirm', 'evaluate', 'all'), default='tune')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--test-gradients', action='store_true', help='Disposable gradient checks only; no fit')
    args = parser.parse_args()
    model, inputs = runtime()
    if args.test_gradients:
        for arm in args.arms:
            print('GRADIENT_CHECK', test_gradients(model, inputs, arm), flush=True)
        return
    if args.phase in ('tune', 'all'):
        tune(model, inputs, args.arms)
    if args.phase in ('confirm', 'all'):
        confirm(model, inputs, args.arms)
    if args.phase in ('evaluate', 'all'):
        evaluate(model, inputs, args.arms)


if __name__ == '__main__':
    main()
