"""Matched PCEA continuation with paired classification and margin supervision."""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import math
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'pcea_class_local_20261117'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


previous = load('paired_previous_recipe', OLD/'run.py')
m = previous.m
torch, np, F = m.torch, m.np, m.F
read, write = previous.read, previous.write
SEEDS, STEPS, INTERVAL = previous.SEEDS, previous.STEPS, previous.INTERVAL
ARMS = ('paired_ce', 'guided')
TAU, IMPROVEMENT, LOSS_WEIGHT = .2, .1, .2


def margin(z, y):
    wrong = z.masked_fill(F.one_hot(y, z.shape[-1]).bool(), -torch.inf)
    return z.gather(1, y[:, None]).squeeze(1)-wrong.max(1).values


def correction(z, off, teacher, y, weights, active=None):
    before, fixed = margin(off.detach(), y), margin(teacher.detach(), y)
    after = margin(z, y)
    if active is None:
        active = torch.ones_like(y, dtype=torch.bool)
    hard, safe = (fixed < TAU) & active, (fixed > 0) & active
    target = before+(TAU-before).clamp(0, IMPROVEMENT)
    improve = F.relu(target-after)*hard
    protect = F.relu(fixed.clamp(max=TAU)-after)*safe
    loss = LOSS_WEIGHT*((improve+protect)*weights).mean()
    stats = dict(improve=float((improve*weights).mean().detach()),
                 protect=float((protect*weights).mean().detach()),
                 hard=float(hard.float().mean()), safe=float(safe.float().mean()),
                 eligible=float(active.float().mean()),
                 delta_margin=float((after-before).mean().detach()))
    return loss, stats


def runtime():
    # Reuse the exact source split and input handling without rewriting old reports.
    previous.ROOT = ROOT
    inputs, fit, inner = previous.runtime()
    saved = read(OLD/'split_local.json')
    assert np.array_equal(fit, saved['fit']) and np.array_equal(inner, saved['inner'])
    for seed in SEEDS:
        order = np.random.default_rng(seed+20261117).permutation(fit)[:STEPS*24]
        for arm in ('full', 'generic'):
            dest = OLD/'fits'/f'{arm}_s{seed}'
            conf = read(dest/'config.json')
            assert conf['rates'] == previous.RATES and conf['steps'] == STEPS
            assert conf['selection'] == 'source-inner BA/OA/-NLL, step0 eligible'
            assert np.array_equal(order, np.load(dest/'training_ids_local.npy'))
    scope = read(ROOT/'scope.json')
    scope.update(new_arms=list(ARMS), reused_arms=['full', 'generic'],
                 teacher='fixed formal selected_s17; same augmentation as source query',
                 tau=TAU, improvement=IMPROVEMENT, margin_weight=LOSS_WEIGHT,
                 equal_updates_not_equal_compute=True, new_gates=False)
    write(ROOT/'scope.json', scope)
    return inputs, fit, inner


def teacher_model():
    net = m.Model('full')
    net.requires_grad_(False)
    net.accelerate()
    return net.eval()


def paired_backward(net, teacher, inputs, ids, code, guided, multiplier):
    # Sequential backward keeps one training graph resident on the 8GB device.
    net.exchange.disabled = True
    off_loss, off = m.full.loss_for(net, inputs, ids, code)
    detached = off.detach()
    (.5*multiplier*off_loss).backward()
    off_scalar = float(off_loss.detach())
    del off_loss, off
    anchor = None
    if guided:
        with torch.no_grad():
            _, anchor = m.full.loss_for(teacher, inputs, ids, code)
    net.exchange.disabled = False
    on_loss, z = m.full.loss_for(net, inputs, ids, code)
    extra, stats = z.new_zeros(()), {}
    if guided:
        y = torch.as_tensor(inputs.y[ids], device='cuda')
        weights = torch.as_tensor(inputs.weights[ids], device='cuda')
        active = net.exchange.trace['active'].any(1)
        extra, stats = correction(z, detached, anchor, y, weights, active)
    loss = .5*on_loss+extra
    (multiplier*loss).backward()
    stats.update(loss=.5*off_scalar+float(loss.detach()),
                 off_ce_aux=off_scalar, on_ce_aux=float(on_loss.detach()),
                 extra=float(extra.detach()))
    return z.detach(), stats


def smoke(inputs, fit):
    y = torch.zeros(3, dtype=torch.long, device='cuda')
    off = torch.tensor([[.1, 0.], [0., 1.], [.1, 0.]], device='cuda', requires_grad=True)
    anchor = torch.tensor([[1., 0.], [0., 1.], [.1, 0.]], device='cuda', requires_grad=True)
    z = torch.tensor([[-.1, 0.], [0., 1.], [.1, 0.]], device='cuda', requires_grad=True)
    loss, stats = correction(z, off, anchor, y, torch.ones(3, device='cuda'))
    loss.backward()
    assert off.grad is None and anchor.grad is None
    assert bool((z.grad[:, 0] < 0).all())
    assert abs(float(loss)-1/30) < 1e-6
    inactive, _ = correction(z, off, anchor, y, torch.ones(3, device='cuda'),
                             torch.zeros(3, dtype=torch.bool, device='cuda'))
    assert float(inactive) == 0
    teacher = teacher_model()
    net = m.Model('generic')
    net.accelerate()
    ids = fit[:12]
    on = m.full.predict(net, inputs, ids, 'small')
    net.exchange.disabled = True
    off0 = m.full.predict(net, inputs, ids, 'small')
    base = m.full.predict(teacher, inputs, ids, 'small')
    error = max(float(abs(on-off0).max()), float(abs(on-base).max()))
    assert error < 1e-5, error
    groups = previous.groups_for(net)
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    reports = []
    for step in range(2):
        net.train()
        opt.zero_grad(set_to_none=True)
        _, record = paired_backward(net, teacher, inputs, ids, 1, True, 1.)
        norms = {g['name']: sum(float(p.grad.square().sum()) for p in g['params']
                 if p.grad is not None)**.5 for g in groups}
        assert all(math.isfinite(x) for x in norms.values())
        if step == 1:
            assert all(x > 0 for x in norms.values()), norms
        assert all(p.grad is None for p in teacher.parameters())
        assert not net.exchange.disabled
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 5.)
        opt.step()
        reports.append(dict(**record, gradients=norms))
    write(ROOT/'smoke.json', dict(passed=True, initial_max_error=error,
          synthetic_loss=1/30, detached_off_teacher=True, steps=reports,
          max_cuda_gib=torch.cuda.max_memory_allocated()/1024**3))
    print('SMOKE', json.dumps(read(ROOT/'smoke.json')), flush=True)


def fit_one(inputs, fit, inner, arm, seed):
    dest = ROOT/'fits'/f'{arm}_s{seed}'
    if (dest/'receipt.json').exists():
        return read(dest/'receipt.json')
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Inspect incomplete run before resuming: '+str(dest))
    net = m.Model('generic')
    net.accelerate()
    groups = previous.groups_for(net)
    params = [p for g in groups for p in g['params']]
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    teacher = teacher_model() if arm == 'guided' else None
    order = np.random.default_rng(seed+20261117).permutation(fit)[:STEPS*24]
    aug = np.random.default_rng(seed+20261118)
    score = m.full.predict(net, inputs, inner, 'small')
    initial = m.full.metrics(inputs.y[inner], score)
    best = dict(step=0, metrics=initial)
    dest.mkdir(parents=True)
    write(dest/'config.json', dict(arm=arm, continuation_seed=seed, checkpoint_seed=17,
          rates=previous.RATES, steps=STEPS, batch=24, microbatch=12,
          selection='source-inner BA/OA/-NLL, step0 eligible',
          objective='.5 on CE/aux + .5 off CE/aux; guided adds paired improve/protect',
          tau=TAU, improvement=IMPROVEMENT, margin_weight=LOSS_WEIGHT,
          original_teacher_fixed=True, backbone_trainable=True, bn_running_fixed=True,
          scale=1, formal_model_changed=False))
    np.save(dest/'training_ids_local.npy', order)
    torch.save(net.state_dict(), dest/'best.pt')
    np.savez_compressed(dest/'inner.npz', ids=inner, label=inputs.y[inner], score=score)
    curve, tick = [], time.perf_counter()
    for block in range(STEPS//INTERVAL):
        factor = .1+.9*.5*(1+math.cos(math.pi*block/(STEPS//INTERVAL-1)))
        for group in groups:
            group['lr'] = group['base_lr']*factor
        sums = {}
        net.train()
        for j in range(INTERVAL):
            step = block*INTERVAL+j
            batch = order[step*24:(step+1)*24]
            code = int(aug.integers(4))+4*int(aug.random()<.5)
            opt.zero_grad(set_to_none=True)
            for offset in (0, 12):
                z, record = paired_backward(net, teacher, inputs, batch[offset:offset+12],
                                             code, arm == 'guided', .5)
                assert all(math.isfinite(x) for x in record.values()), record
                for key, value in record.items():
                    sums[key] = sums.get(key, 0.)+value/(INTERVAL*2)
            torch.nn.utils.clip_grad_norm_(params, 5.)
            opt.step()
        score = m.full.predict(net, inputs, inner, 'small')
        metrics = m.full.metrics(inputs.y[inner], score)
        row = dict(step=(block+1)*INTERVAL, inner=metrics, training=sums,
                   seconds=time.perf_counter()-tick)
        curve.append(row)
        if previous.key(metrics) > previous.key(best['metrics']):
            best = dict(step=row['step'], metrics=metrics)
            torch.save(net.state_dict(), dest/'best.pt')
            np.savez_compressed(dest/'inner.npz', ids=inner, label=inputs.y[inner], score=score)
        write(dest/'curve.json', curve)
        write(ROOT/'status.json', dict(phase='training', arm=arm, seed=seed, step=row['step'], best=best))
        print('TRAIN', arm, seed, row['step'], 'OA/BA', metrics['oa'], metrics['ba'],
              'best', best['step'], 'sec', round(row['seconds'], 1), flush=True)
    torch.save(net.state_dict(), dest/'last.pt')
    net.load_state_dict(torch.load(dest/'best.pt', map_location='cuda', weights_only=True), strict=True)
    net.exchange.disabled = True
    off = m.full.predict(net, inputs, inner, 'small')
    np.savez_compressed(dest/'inner_off.npz', ids=inner, label=inputs.y[inner], score=off)
    receipt = dict(arm=arm, seed=seed, initial=initial, best=best, last=metrics,
                   seconds=time.perf_counter()-tick, source_only=True)
    write(dest/'receipt.json', receipt)
    del net, teacher, opt, groups, params, z
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def evaluate(inputs):
    assert all((ROOT/'fits'/f'{a}_s{s}/receipt.json').exists() for a in ARMS for s in SEEDS)
    extension = load('paired_existing_expansion', ROOT.parent/'pcea_class_local_expansion_20261118/evaluate.py')
    extra = extension.Inputs()
    report = []
    for seed in SEEDS:
        for arm in ARMS:
            dest = ROOT/'fits'/f'{arm}_s{seed}'
            net = m.Model('generic')
            net.load_state_dict(torch.load(dest/'best.pt', map_location='cuda', weights_only=True), strict=True)
            net.requires_grad_(False)
            net.accelerate()
            for split, data in [('validation', inputs), ('expansion', extra)]:
                ids = data.groups['validation']
                for mode in ('on', 'off'):
                    net.exchange.disabled = mode == 'off'
                    z = m.full.predict(net, data, ids, 'validation')
                    np.savez_compressed(dest/f'{split}_{mode}.npz', ids=ids, label=data.y[ids], score=z)
                    item = dict(arm=arm, seed=seed, split=split, mode=mode, **m.full.metrics(data.y[ids], z))
                    report.append(item)
                    write(ROOT/'results.json', report)
                    print('EVAL', arm, seed, split, mode, item['oa'], item['ba'], flush=True)
            del net
            gc.collect()
            torch.cuda.empty_cache()
    write(ROOT/'status.json', dict(phase='complete', new_fits=6, reused_fits=6,
                                  external_conditions=24, test_accessed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('smoke', 'train', 'evaluate'), default='smoke')
    args = parser.parse_args()
    inputs, fit, inner = runtime()
    if args.phase == 'smoke':
        smoke(inputs, fit)
    elif args.phase == 'evaluate':
        evaluate(inputs)
    else:
        assert read(ROOT/'smoke.json')['passed']
        for seed in SEEDS:
            for arm in ARMS:
                fit_one(inputs, fit, inner, arm, seed)
        evaluate(inputs)
