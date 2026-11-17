"""Paired, bounded full joint training from one task-untrained initialization."""
from pathlib import Path
import argparse
import gc
import importlib.util
import math
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('joint_full_runner_helper',
    ROOT.parent/'pcea_conditional_modulation_20261117/run.py')
h = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h
spec.loader.exec_module(h)
read, write = h.read, h.write
ARMS = ('full', 'cross_mean', 'cross_local')
CONFIGS = {'slow': dict(backbone=1e-5, modules=1e-3, reader=1e-3, exchange=1e-3),
           'base': dict(backbone=1e-4, modules=1e-3, reader=1e-3, exchange=1e-3)}
EPOCHS, MIN_EPOCHS, PATIENCE = 30, 16, 8


def runtime():
    model = h.load('joint_full_model', ROOT/'model.py')
    inputs = model.full.Inputs()
    for split in ('small', 'validation'):
        inputs.prepare_geometry(split)
    return model, inputs


def parameter_groups(net, config):
    buckets = {k: [] for k in CONFIGS[config]}
    names = {k: [] for k in buckets}
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
        names[key].append(name)
    groups = [dict(name=k, params=v, lr=CONFIGS[config][k], base_lr=CONFIGS[config][k])
              for k, v in buckets.items() if v]
    return groups, {k: dict(names=names[k], count=sum(p.numel() for p in v))
                    for k, v in buckets.items() if v}


def smoke(model, inputs):
    torch, full, np = model.torch, model.full, model.np
    initial, reports = {}, {}
    for arm in ARMS:
        net = model.Model(17, arm)
        net.accelerate()
        ids = inputs.groups['validation'][:12]
        initial[arm] = full.predict(net, inputs, ids, 'validation')
        groups, info = parameter_groups(net, 'slow')
        optimizer = torch.optim.AdamW(groups, weight_decay=.01)
        bb = net.net.base.backbone.model
        weights0 = {k: v.detach().clone() for k, v in
                    [('stem', bb.conv1.weight), ('layer2', bb.layer2[3].conv3.weight)]}
        with torch.no_grad():
            ref0 = net.reference(inputs.data(inputs.groups['references'], 'small'))['desc'].clone()
            raw0 = None if arm == 'full' else net.exchange.source_raw.clone()
        passes = []
        for stage in range(2):
            net.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = full.loss_for(net, inputs, inputs.groups['small'][:12], 1)
            assert torch.isfinite(loss)
            loss.backward()
            stats = {g['name']: dict(nonzero=sum(p.grad is not None and bool(p.grad.abs().max()>0)
                              for p in g['params']), norm=sum(float(p.grad.square().sum())
                              for p in g['params'] if p.grad is not None)**.5) for g in groups}
            for name, p in net.named_parameters():
                if p.grad is not None:
                    assert bool(torch.isfinite(p.grad).all()), name
            for prefix in ('net.base.backbone.model.conv1.', 'net.base.backbone.model.layer1.',
                           'net.base.backbone.model.layer2.', 'net.base.backbone.model.layer3.',
                           'net.base.backbone.model.layer4.', 'net.m1.', 'net.base.internal_m2.',
                           'net.matcher.', 'net.guide.'):
                assert any(n.startswith(prefix) and p.grad is not None and bool(p.grad.abs().max()>0)
                           for n,p in net.named_parameters()), prefix
            passes.append(dict(loss=float(loss.detach()), groups=stats))
            if stage == 0:
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 5.)
                optimizer.step()
        assert all(v['nonzero'] > 0 for v in stats.values())
        net.eval()
        with torch.no_grad():
            ref1 = net.reference(inputs.data(inputs.groups['references'], 'small'))['desc']
        changes = dict(stem=float((bb.conv1.weight-weights0['stem']).abs().max()),
                       layer2=float((bb.layer2[3].conv3.weight-weights0['layer2']).abs().max()),
                       reference=float((ref1-ref0).abs().max()))
        if arm != 'full':
            changes['source_raw'] = float((net.exchange.source_raw-raw0).abs().max())
        assert all(v > 0 for v in changes.values())
        reports[arm] = dict(groups=info, passes=passes, changes=changes,
                            peak_gib=torch.cuda.max_memory_allocated()/1024**3)
        print('SMOKE', arm, changes, reports[arm]['peak_gib'], flush=True)
        del net, optimizer, groups, loss, ref0, ref1
        gc.collect(); torch.cuda.empty_cache()
    errors = {a: float(abs(z-initial['full']).max()) for a,z in initial.items()}
    assert max(errors.values()) < 1e-5
    write(ROOT/'smoke.json', dict(initial_errors=errors, arms=reports, passed=True))


def factor(epoch):
    return (epoch+1)/2 if epoch < 2 else .05+.95*.5*(1+math.cos(math.pi*(epoch-2)/(EPOCHS-3)))


def fit(model, inputs, arm, seed, config):
    dest = ROOT/'fits'/f'{arm}_s{seed}_{config}'
    if (dest/'receipt.json').exists():
        return read(dest/'receipt.json')
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Incomplete fit must be inspected before restarting: '+str(dest))
    dest.mkdir(parents=True, exist_ok=True)
    torch, np, full = model.torch, model.np, model.full
    net = model.Model(seed, arm)
    net.accelerate()
    groups, info = parameter_groups(net, config)
    params = [p for g in groups for p in g['params']]
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    train, val = inputs.groups['small'], inputs.groups['validation']
    score = full.predict(net, inputs, val, 'validation')
    m = full.metrics(inputs.y[val], score)
    cfg = dict(arm=arm, seed=seed, config=config, initialization='same ImageNet R50 + task-untrained Full modules',
        source=len(train), validation=len(val), references=32, full_backbone=True,
        original_modules_trainable=True, bn='fixed running stats; affine trainable',
        source_normalization='current fixed32 source references; equal observation weights, active regions only; differentiable live prefix and reader',
        source_raw_cache=False, restored_task_checkpoint=False, epochs_max=EPOCHS,
        minimum_epochs=MIN_EPOCHS, patience=PATIENCE, batch=24, microbatch=12,
        learning_rates=CONFIGS[config], groups=info, loss='source class/vessel CE + .02 M1 auxiliary CE',
        selection='validation BA/OA/-NLL; epoch0 eligible', evaluation_loaded=False,
        parameter_count=sum(p.numel() for p in params), initial=m)
    write(dest/'config.json', cfg)
    best = dict(epoch=0, metrics=m, key=h.validation_key(m))
    torch.save(net.state_dict(), dest/'best.pt')
    np.savez_compressed(dest/'validation.npz', ids=val, label=inputs.y[val], score=score)
    rng, aug = np.random.default_rng(seed+20261116), np.random.default_rng(seed+20261117)
    curve, steps, tick = [], 0, time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        for g in groups:
            g['lr'] = g['base_lr']*factor(epoch)
        net.train(); order = rng.permutation(train)
        total = correct = 0
        for first in range(0, len(order), 24):
            batch = order[first:first+24]
            opt.zero_grad(set_to_none=True)
            code = int(aug.integers(4))+4*int(aug.random()<.5)
            for start in range(0, len(batch), 12):
                ids = batch[start:start+12]
                loss, logits = full.loss_for(net, inputs, ids, code)
                assert torch.isfinite(loss)
                (loss*len(ids)/len(batch)).backward()
                total += float(loss.detach())*len(ids)
                correct += int((logits.argmax(1)==torch.as_tensor(inputs.y[ids], device='cuda')).sum())
            torch.nn.utils.clip_grad_norm_(params, 5.)
            opt.step(); steps += 1
        score = full.predict(net, inputs, val, 'validation')
        m = full.metrics(inputs.y[val], score)
        curve.append(dict(epoch=epoch+1, validation=m, train_loss=total/len(train),
                          train_oa=100*correct/len(train), seconds=time.perf_counter()-tick,
                          learning_rates={g['name']: g['lr'] for g in groups}))
        if h.validation_key(m) > best['key']:
            best = dict(epoch=epoch+1, metrics=m, key=h.validation_key(m))
            torch.save(net.state_dict(), dest/'best.pt')
            np.savez_compressed(dest/'validation.npz', ids=val, label=inputs.y[val], score=score)
        write(dest/'curve.json', curve)
        print('VAL', arm, seed, config, epoch+1, 'OA/BA',round(m['oa'],3),round(m['ba'],3),
              'train',round(curve[-1]['train_oa'],2),'best',best['epoch'],flush=True)
        if epoch+1 >= MIN_EPOCHS and epoch+1-best['epoch'] >= PATIENCE:
            break
    torch.save(net.state_dict(), dest/'last.pt')
    np.savez_compressed(dest/'lastvalidation.npz', ids=val, label=inputs.y[val], score=score)
    report = dict(**cfg, best=best, last=m, epochs=len(curve), steps=steps,
                  seconds=time.perf_counter()-tick, peak_gib=torch.cuda.max_memory_allocated()/1024**3,
                  checkpoint=str(dest/'best.pt'), validation_predictions=str(dest/'validation.npz'))
    write(dest/'receipt.json', report)
    del net, opt, groups, params, loss, logits
    gc.collect(); torch.cuda.empty_cache()
    return report


def tune(model, inputs):
    rows=[]
    for config in CONFIGS:
        for arm in ARMS:
            rows.append(fit(model, inputs, arm, 17, config))
            write(ROOT/'progress.json', dict(completed=len(rows), planned=6, rows=rows))
    selected={arm: max([r for r in rows if r['arm']==arm], key=lambda r:r['best']['key']) for arm in ARMS}
    base=selected['full']['best']['metrics']; mean=selected['cross_mean']['best']['metrics']; local=selected['cross_local']['best']['metrics']
    # A net accuracy gain is required, not just a lower NLL on identical predictions.
    worth_confirming = any(r['best']['metrics']['ba'] >= base['ba'] and r['best']['metrics']['oa'] > base['oa']
                          or r['best']['metrics']['ba'] > base['ba'] and r['best']['metrics']['oa'] >= base['oa']
                          for a,r in selected.items() if a!='full')
    write(ROOT/'selection.json', dict(status='frozen', rows=rows, selected=selected,
        worth_confirming=worth_confirming, local_ba_minus_mean=local['ba']-mean['ba'],
        rule='validation BA/OA/-NLL; two learning-rate opportunities per arm; no evaluation selection'))
    print('SELECTED', {a:r['best'] for a,r in selected.items()},'CONFIRM',worth_confirming,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--phase', choices=('smoke','tune','confirm'), default='tune')
    args=p.parse_args()
    model, inputs=runtime()
    if args.phase=='smoke': smoke(model, inputs)
    elif args.phase=='tune': tune(model, inputs)
    else:
        selection=read(ROOT/'selection.json')
        assert selection['worth_confirming']
        rows=[]
        for seed in (29,43):
            for arm in ARMS:
                rows.append(fit(model,inputs,arm,seed,selection['selected'][arm]['config']))
                write(ROOT/'confirmation.json',dict(rows=rows,complete=len(rows)==6))
