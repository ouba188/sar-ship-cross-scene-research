"""Route paired correction to CCREL/PCEA; classification still trains all groups."""
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
PAIRED = ROOT.parent/'pcea_paired_correction_20261118'
spec = importlib.util.spec_from_file_location('routed_paired_recipe', PAIRED/'run.py')
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)
m, previous = b.m, b.previous
torch, np = b.torch, b.np
read, write = b.read, b.write
SEEDS, STEPS, INTERVAL = b.SEEDS, b.STEPS, b.INTERVAL


def runtime():
    b.ROOT = ROOT
    inputs, fit, inner = b.runtime()
    for seed in SEEDS:
        order = np.random.default_rng(seed+20261117).permutation(fit)[:STEPS*24]
        for arm in ('paired_ce', 'guided'):
            dest = PAIRED/'fits'/f'{arm}_s{seed}'
            c = read(dest/'config.json')
            assert c['rates'] == previous.RATES and c['steps'] == STEPS
            assert c['scale'] == 1 and read(dest/'receipt.json')['source_only']
            assert np.array_equal(order, np.load(dest/'training_ids_local.npy'))
    scope = read(ROOT/'scope.json')
    scope.update(new_arms=['routed'], reused_arms=['full', 'generic', 'paired_ce', 'guided'],
                 correction_gradient_targets=['ccrel', 'exchange'],
                 classification_gradient_targets='all trainable parameters',
                 only_change='extra-loss gradient routing; same losses, data, scale and budget')
    write(ROOT/'scope.json', scope)
    return inputs, fit, inner


def paired_backward(net, teacher, inputs, ids, code, multiplier, audit=False):
    net.exchange.disabled = True
    off_loss, off = m.full.loss_for(net, inputs, ids, code)
    detached = off.detach()
    (.5*multiplier*off_loss).backward()
    off_scalar = float(off_loss.detach())
    del off_loss, off
    with torch.no_grad():
        _, anchor = m.full.loss_for(teacher, inputs, ids, code)
    net.exchange.disabled = False
    on_loss, z = m.full.loss_for(net, inputs, ids, code)
    y = torch.as_tensor(inputs.y[ids], device='cuda')
    weights = torch.as_tensor(inputs.weights[ids], device='cuda')
    extra, stats = b.correction(z, detached, anchor, y, weights,
                                net.exchange.trace['active'].any(1))
    targets = [(n, p) for n, p in net.named_parameters()
               if p.requires_grad and n.startswith(('ccrel.', 'exchange.'))]
    # autograd.grad does not accumulate into non-target .grad buffers.
    auxiliary = torch.autograd.grad(multiplier*extra, [p for _, p in targets],
                                    retain_graph=True, allow_unused=True)
    (.5*multiplier*on_loss).backward()
    before = {n: p.grad.detach().clone() if p.grad is not None else None
              for n, p in net.named_parameters() if p.requires_grad} if audit else None
    for (_, p), g in zip(targets, auxiliary):
        if g is not None:
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad.add_(g.detach())
    if audit:
        aux_by_name = {n: g for (n, _), g in zip(targets, auxiliary)}
        errors, aux_norm = {}, dict(reader=0., exchange=0.)
        for n, p in net.named_parameters():
            if not p.requires_grad:
                continue
            old, g = before[n], aux_by_name.get(n)
            expected = old if g is None else g if old is None else old+g
            if expected is None:
                assert p.grad is None, n
                error = 0.
            else:
                error = float((p.grad-expected).abs().max())
                assert torch.isfinite(p.grad).all(), n
            assert error == 0., (n, error)
            group = 'target' if n in aux_by_name else 'non_target'
            errors[group] = max(errors.get(group, 0.), error)
            if g is not None:
                key = 'reader' if n.startswith('ccrel.') else 'exchange'
                aux_norm[key] += float(g.square().sum())
        stats.update(routing_max_error=errors,
                     auxiliary_gradient_norm={k: v**.5 for k, v in aux_norm.items()})
    stats.update(loss=.5*off_scalar+.5*float(on_loss.detach())+float(extra.detach()),
                 off_ce_aux=off_scalar, on_ce_aux=float(on_loss.detach()),
                 extra=float(extra.detach()))
    return z.detach(), stats


def smoke(inputs, fit):
    teacher = b.teacher_model()
    net = m.Model('generic')
    net.accelerate()
    ids = fit[:12]
    on = m.full.predict(net, inputs, ids, 'small')
    net.exchange.disabled = True
    off = m.full.predict(net, inputs, ids, 'small')
    base = m.full.predict(teacher, inputs, ids, 'small')
    error = max(float(abs(on-off).max()), float(abs(on-base).max()))
    assert error < 1e-5
    groups = previous.groups_for(net)
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    reports = []
    for step in range(2):
        net.train()
        opt.zero_grad(set_to_none=True)
        _, record = paired_backward(net, teacher, inputs, ids, 1, .5, audit=True)
        norms = {g['name']: sum(float(p.grad.square().sum()) for p in g['params']
                 if p.grad is not None)**.5 for g in groups}
        assert all(math.isfinite(x) for x in norms.values())
        assert norms['backbone'] > 0 and norms['modules'] > 0
        assert record['auxiliary_gradient_norm']['exchange'] > 0
        if step == 1:
            assert record['auxiliary_gradient_norm']['reader'] > 0
        assert all(p.grad is None for p in teacher.parameters())
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 5.)
        opt.step()
        reports.append(dict(**record, total_gradients=norms))
    y = torch.zeros(3, dtype=torch.long, device='cuda')
    z = torch.tensor([[-.1, 0.], [0., 1.], [.1, 0.]], device='cuda', requires_grad=True)
    off = torch.tensor([[.1, 0.], [0., 1.], [.1, 0.]], device='cuda', requires_grad=True)
    anchor = torch.tensor([[1., 0.], [0., 1.], [.1, 0.]], device='cuda', requires_grad=True)
    loss, _ = b.correction(z, off, anchor, y, torch.ones(3, device='cuda'))
    loss.backward()
    assert off.grad is None and anchor.grad is None
    assert bool((z.grad[:, 0] < 0).all()) and abs(float(loss)-1/30) < 1e-6
    inactive, _ = b.correction(z, off, anchor, y, torch.ones(3, device='cuda'),
                               torch.zeros(3, dtype=torch.bool, device='cuda'))
    assert float(inactive) == 0
    write(ROOT/'smoke.json', dict(passed=True, initial_max_error=error,
          non_target_gradients_exactly_ce_only=True, target_gradients_exact_sum=True,
          detached_off_teacher=True, inactive_loss=0, steps=reports,
          max_cuda_gib=torch.cuda.max_memory_allocated()/1024**3))
    print('SMOKE', json.dumps(read(ROOT/'smoke.json')), flush=True)


def fit_one(inputs, fit, inner, seed):
    dest = ROOT/'fits'/f'routed_s{seed}'
    if (dest/'receipt.json').exists():
        return read(dest/'receipt.json')
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Inspect incomplete run before resuming: '+str(dest))
    net = m.Model('generic')
    net.accelerate()
    groups = previous.groups_for(net)
    params = [p for g in groups for p in g['params']]
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    teacher = b.teacher_model()
    order = np.random.default_rng(seed+20261117).permutation(fit)[:STEPS*24]
    aug = np.random.default_rng(seed+20261118)
    score = m.full.predict(net, inputs, inner, 'small')
    initial = m.full.metrics(inputs.y[inner], score)
    best = dict(step=0, metrics=initial)
    dest.mkdir(parents=True)
    config = read(PAIRED/'fits'/f'guided_s{seed}/config.json')
    config.update(arm='routed', correction_gradient_targets=['ccrel', 'exchange'],
                  classification_gradient_targets='all trainable parameters')
    write(dest/'config.json', config)
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
                z, record = paired_backward(net, teacher, inputs, batch[offset:offset+12], code, .5)
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
        write(ROOT/'status.json', dict(phase='training', arm='routed', seed=seed, step=row['step'], best=best))
        print('TRAIN routed', seed, row['step'], 'OA/BA', metrics['oa'], metrics['ba'],
              'best', best['step'], 'sec', round(row['seconds'], 1), flush=True)
    torch.save(net.state_dict(), dest/'last.pt')
    net.load_state_dict(torch.load(dest/'best.pt', map_location='cuda', weights_only=True), strict=True)
    net.exchange.disabled = True
    off = m.full.predict(net, inputs, inner, 'small')
    np.savez_compressed(dest/'inner_off.npz', ids=inner, label=inputs.y[inner], score=off)
    receipt = dict(arm='routed', seed=seed, initial=initial, best=best, last=metrics,
                   seconds=time.perf_counter()-tick, source_only=True)
    write(dest/'receipt.json', receipt)
    del net, teacher, opt, groups, params, z
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def evaluate(inputs):
    b.ROOT, b.ARMS = ROOT, ('routed',)
    b.evaluate(inputs)
    write(ROOT/'status.json', dict(phase='complete', new_fits=3, reused_fits=12,
                                  external_conditions=12, test_accessed=False))


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
            fit_one(inputs, fit, inner, seed)
        evaluate(inputs)
