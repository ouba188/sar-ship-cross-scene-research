"""Finite 2x2 continuation experiment; validation selection, no test-driven tuning."""
from pathlib import Path
import argparse
import gc
import importlib.util
import math
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('startpoint_helpers',
    ROOT.parent/'pcea_conditional_modulation_20261117/run.py')
h = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h
spec.loader.exec_module(h)
read, write = h.read, h.write
ARMS = ('full', 'cross_mean')
STRENGTHS = (0., .1)
EPOCHS = 8
RATES = dict(backbone=1e-5, modules=1e-5, reader=1e-4, exchange=3e-4)


def runtime():
    model = h.load('startpoint_model', ROOT/'model.py')
    inputs = model.full.Inputs()
    for split in ('small', 'validation'):
        inputs.prepare_geometry(split)
    return model, inputs


def groups_for(net):
    buckets = {k: [] for k in RATES}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('net.base.backbone.'):
            key = 'backbone'
        elif name.startswith('net.'):
            key = 'modules'
        elif name.startswith('ccrel.'):
            key = 'reader'
        elif name.startswith('exchange.'):
            key = 'exchange'
        else:
            raise ValueError(name)
        buckets[key].append(p)
    return [dict(name=k, params=v, lr=RATES[k], base_lr=RATES[k])
            for k, v in buckets.items() if v]


def destination(arm, seed, strength):
    return ROOT/'fits'/f'{arm}_s{seed}_sp{strength:g}'


def smoke(model, inputs):
    torch, full, np = model.torch, model.full, model.np
    results = {}
    val = inputs.groups['validation']
    initial = {}
    for arm in ARMS:
        net = model.Model(17, arm)
        net.accelerate()
        score = full.predict(net, inputs, val, 'validation')
        initial[arm] = score
        replay = h.replay(model, val, score, full.ROOT/'selected_s17/validation.npz')
        anchor = model.StartingPoint(net)
        assert float(anchor.penalty()) == 0.
        groups = groups_for(net)
        opt = torch.optim.AdamW(groups, weight_decay=.01)
        passes = []
        for step in range(2):
            net.train()
            opt.zero_grad(set_to_none=True)
            task, _ = full.loss_for(net, inputs, inputs.groups['small'][:12], 1)
            penalty = .1*anchor.penalty()
            (task+penalty).backward()
            grad = {g['name']: sum(float(p.grad.square().sum()) for p in g['params']
                                    if p.grad is not None)**.5 for g in groups}
            assert all(math.isfinite(v) for v in grad.values())
            if step == 1:
                assert all(v > 0 for v in grad.values()), grad
                assert float(penalty.detach()) > 0.
            passes.append(dict(task=float(task.detach()), penalty=float(penalty.detach()), grad=grad))
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 5.)
            opt.step()
        delta_grad = torch.autograd.grad(anchor.penalty(), [p for _, p in anchor.params])
        error = max(float((g-(p-q)).abs().max())
                    for g, (_, p), q in zip(delta_grad, anchor.params, anchor.origin))
        assert error < 1e-7, error
        results[arm] = dict(replay=replay, passes=passes, drift=anchor.drift(),
                            anchor_gradient_error=error)
        print('SMOKE', arm, results[arm], flush=True)
        del net, anchor, opt, groups, task, penalty, delta_grad
        gc.collect(); torch.cuda.empty_cache()
    error = float(abs(initial['full']-initial['cross_mean']).max())
    assert error < 1e-5
    write(ROOT/'smoke.json', dict(passed=True, arms=results, initial_max_difference=error))


def fit(model, inputs, arm, seed, strength):
    dest = destination(arm, seed, strength)
    if (dest/'receipt.json').exists():
        return read(dest/'receipt.json')
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Inspect incomplete fit before retry: '+str(dest))
    torch, np, full = model.torch, model.np, model.full
    net = model.Model(seed, arm)
    net.accelerate()
    anchor = model.StartingPoint(net)
    groups = groups_for(net)
    params = [p for g in groups for p in g['params']]
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    train, val = inputs.groups['small'], inputs.groups['validation']
    score = full.predict(net, inputs, val, 'validation')
    replay = h.replay(model, val, score, full.ROOT/f'selected_s{seed}/validation.npz')
    initial = full.metrics(inputs.y[val], score)
    cfg = dict(arm=arm, seed=seed, strength=strength,
        start_checkpoint=str(net.start_checkpoint), initial=initial, replay=replay,
        learning_rates=RATES, epochs=EPOCHS, steps_expected=240, batch=24, microbatch=12,
        source=len(train), validation=len(val), references=32, evaluation_loaded=False,
        scope='Full backbone and original used modules trainable; BN stats fixed, affine trainable',
        objective='weighted CE + .02 M1 auxiliary CE + lambda/2 * SUM old trainable parameter drift squared',
        new_parameters='reader/exchange ordinary AdamW decay, not constrained to initial zero writer',
        optimizer='AdamW wd .01, clip5, cosine floor .1',
        checkpoint_rule='validation BA/OA/-NLL, epoch0 eligible, tie keeps earlier',
        groups={g['name']:sum(p.numel() for p in g['params']) for g in groups},
        native_input='160 crop/pad, same paired rot90/flip, no resize, no OBB transfer',
        intervention='retained trained Full history; no new teacher selection')
    dest.mkdir(parents=True)
    write(dest/'config.json', cfg)
    best = dict(epoch=0, metrics=initial, key=h.validation_key(initial))
    torch.save(net.state_dict(), dest/'best.pt')
    np.savez_compressed(dest/'validation.npz', ids=val, label=inputs.y[val], score=score)
    rng, aug = np.random.default_rng(seed+20261116), np.random.default_rng(seed+20261117)
    tick = time.perf_counter(); curve=[]
    for epoch in range(EPOCHS):
        factor = .1+.9*.5*(1+math.cos(math.pi*epoch/(EPOCHS-1)))
        for group in groups:
            group['lr'] = group['base_lr']*factor
        net.train()
        order = rng.permutation(train)
        total = correct = regularization = 0.
        diagnostics = []
        for first in range(0, len(order), 24):
            batch = order[first:first+24]
            opt.zero_grad(set_to_none=True)
            code = int(aug.integers(4))+4*int(aug.random()<.5)
            for offset in range(0, len(batch), 12):
                ids = batch[offset:offset+12]
                task, logits = full.loss_for(net, inputs, ids, code)
                assert torch.isfinite(task)
                (task*len(ids)/len(batch)).backward()
                total += float(task.detach())*len(ids)
                correct += int((logits.argmax(1)==torch.as_tensor(inputs.y[ids], device='cuda')).sum())
            if first in (0, 24*29):
                task_norm = sum(float(p.grad.square().sum()) for _,p in anchor.params
                                if p.grad is not None)**.5
                restoring_norm = strength*(2*float(anchor.penalty().detach()))**.5
                diagnostics.append(dict(batch=first//24, old_task_gradient=task_norm,
                    anchor_gradient=restoring_norm,
                    anchor_to_task_ratio=restoring_norm/max(task_norm, 1e-12)))
            if strength:
                reg = strength*anchor.penalty()
                assert torch.isfinite(reg)
                reg.backward()
                regularization += float(reg.detach())
            torch.nn.utils.clip_grad_norm_(params, 5.)
            opt.step()
        score = full.predict(net, inputs, val, 'validation')
        metrics = full.metrics(inputs.y[val], score)
        curve.append(dict(epoch=epoch+1, validation=metrics, train_loss=total/len(train),
            train_oa=100*correct/len(train), mean_regularization=regularization/30,
            drift=anchor.drift(), gradient_diagnostics=diagnostics,
            seconds=time.perf_counter()-tick))
        if h.validation_key(metrics)>best['key']:
            best = dict(epoch=epoch+1, metrics=metrics, key=h.validation_key(metrics))
            torch.save(net.state_dict(), dest/'best.pt')
            np.savez_compressed(dest/'validation.npz', ids=val, label=inputs.y[val], score=score)
        write(dest/'curve.json', curve)
        print('VAL', arm, seed, strength, epoch+1, metrics['oa'], metrics['ba'],
              'best', best['epoch'], 'train', curve[-1]['train_oa'], flush=True)
    torch.save(net.state_dict(), dest/'last.pt')
    report = dict(**cfg, best=best, last=metrics, final_drift=anchor.drift(),
        seconds=time.perf_counter()-tick, checkpoint=str(dest/'best.pt'),
        validation_predictions=str(dest/'validation.npz'))
    write(dest/'receipt.json', report)
    del net, anchor, opt, groups, params, task, logits
    gc.collect(); torch.cuda.empty_cache()
    return report


def run(model, inputs, seeds):
    reports=[]
    for seed in seeds:
        for strength in STRENGTHS:
            for arm in ARMS:
                reports.append(fit(model, inputs, arm, seed, strength))
                write(ROOT/'progress.json', dict(rows=reports, seeds=seeds,
                    complete=len(reports)==4*len(seeds)))
    write(ROOT/('pilot.json' if seeds==[17] else 'confirmation.json'), dict(rows=reports, complete=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('smoke', 'pilot', 'confirm'), default='pilot')
    args=parser.parse_args()
    model, inputs=runtime()
    if args.phase=='smoke':
        smoke(model, inputs)
    elif args.phase=='pilot':
        assert read(ROOT/'smoke.json')['passed']
        run(model, inputs, [17])
    else:
        assert read(ROOT/'decision.json')['confirm']
        run(model, inputs, [29,43])

