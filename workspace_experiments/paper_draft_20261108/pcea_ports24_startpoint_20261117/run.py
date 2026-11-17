"""Fixed-update 24-port 2x2 pilot; no test images or test-based selection."""
from pathlib import Path
import argparse
import gc
import importlib.util
import math
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ports24_startpoint_runner', ROOT.parent/'pcea_startpoint_20261117/run.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
old.ROOT = ROOT
h = old.h
write, read = h.write, h.read


def runtime():
    model = h.load('ports24_startpoint_model', ROOT/'model.py')
    inputs = model.Inputs()
    np = model.np
    # One permutation: the 240 updates see 5,760 DISTINCT source observations.
    selected = np.random.default_rng(17+20261116).permutation(inputs.groups['small'])[:5760]
    def describe(ids):
        available = np.array(inputs.arrays['available'][ids], copy=True)
        for j, idx in enumerate(ids):
            if int(idx) in inputs.restored:
                available[j, inputs.overlay[int(idx)]['p']] = 1
        return dict(n=len(ids), classes=np.bincount(inputs.y[ids], minlength=8).tolist(),
            dual=int(available.astype(bool).all(1).sum()),
            single=int((available.sum(1)==1).sum()),
            ports=len({inputs.rows[i]['port'] for i in ids}),
            vessels=len({inputs.rows[i]['mmsi'] for i in ids}),
            passes=len({inputs.rows[i]['pass_group'] for i in ids}),
            recovered=sum(int(i) in inputs.restored for i in ids))
    write(ROOT/'data_scope.json', dict(training_pool=describe(inputs.groups['small']),
        pilot=describe(selected), validation=describe(inputs.groups['validation']),
        references=describe(inputs.groups['references']),
        test_pixels_loaded=False, test_for_selection=False,
        seed=17, updates_per_arm=240, validation_every=30,
        native160=True, recovery='existing geospatial rule, training/validation/references only'))
    write(ROOT/'training_ids.json', selected.tolist())
    return model, inputs, selected


def fit(model, inputs, selected, arm, strength):
    seed = 17
    dest = old.destination(arm, seed, strength)
    if (dest/'receipt.json').exists():
        return read(dest/'receipt.json')
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Inspect incomplete fit before retry: '+str(dest))
    torch, np, full = model.torch, model.np, model.full
    net = model.Model(seed, arm)
    net.accelerate()
    anchor = model.StartingPoint(net)
    groups = old.groups_for(net)
    params = [p for g in groups for p in g['params']]
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    val = inputs.groups['validation']
    score = full.predict(net, inputs, val, 'validation')
    replay = h.replay(model, val, score, full.ROOT/'selected_s17/validation.npz')
    initial = full.metrics(inputs.y[val], score)
    cfg = dict(arm=arm, strength=strength, seed=seed, dataset='24ports-eight-classes',
        start_checkpoint=str(net.start_checkpoint), initial=initial, replay=replay,
        learning_rates=old.RATES, steps=240, rounds=8, batch=24, microbatch=12,
        unique_source_observations=len(selected), full_training_pool=20653,
        validation=len(val), references=16, test_pixels_loaded=False,
        scope='Full backbone and used original modules trainable; BN stats fixed',
        objective='weighted CE + .02 M1 auxiliary CE + lambda/2 SUM old parameter drift squared',
        selection='validation BA/OA/-NLL; starting checkpoint eligible',
        groups={g['name']:sum(p.numel() for p in g['params']) for g in groups},
        recovery='existing geospatial rule; equal across four arms',
        note='Fixed-budget continuation pilot, NOT eight full-data epochs',
        new_teacher_selection=False)
    dest.mkdir(parents=True)
    write(dest/'config.json', cfg)
    best = dict(step=0, metrics=initial, key=h.validation_key(initial))
    torch.save(net.state_dict(), dest/'best.pt')
    np.savez_compressed(dest/'validation.npz', ids=val, label=inputs.y[val], score=score)
    aug = np.random.default_rng(seed+20261117)
    tick = time.perf_counter()
    curve = []
    for block in range(8):
        factor = .1+.9*.5*(1+math.cos(math.pi*block/7))
        for group in groups:
            group['lr'] = group['base_lr']*factor
        net.train()
        total = correct = regularization = 0.
        for j in range(30):
            step = block*30+j
            batch = selected[step*24:(step+1)*24]
            opt.zero_grad(set_to_none=True)
            code = int(aug.integers(4))+4*int(aug.random()<.5)
            for offset in (0,12):
                ids = batch[offset:offset+12]
                task, logits = full.loss_for(net, inputs, ids, code)
                assert torch.isfinite(task)
                (task*.5).backward()
                total += float(task.detach())*12
                correct += int((logits.argmax(1)==torch.as_tensor(inputs.y[ids],device='cuda')).sum())
            if strength:
                reg = strength*anchor.penalty()
                assert torch.isfinite(reg)
                reg.backward()
                regularization += float(reg.detach())
            torch.nn.utils.clip_grad_norm_(params, 5.)
            opt.step()
        score = full.predict(net, inputs, val, 'validation')
        metrics = full.metrics(inputs.y[val], score)
        curve.append(dict(step=(block+1)*30, validation=metrics, train_loss=total/720,
            train_oa=100*correct/720, mean_regularization=regularization/30,
            drift=anchor.drift(), seconds=time.perf_counter()-tick))
        if h.validation_key(metrics)>best['key']:
            best = dict(step=(block+1)*30, metrics=metrics, key=h.validation_key(metrics))
            torch.save(net.state_dict(), dest/'best.pt')
            np.savez_compressed(dest/'validation.npz', ids=val, label=inputs.y[val], score=score)
        write(dest/'curve.json', curve)
        write(ROOT/'status.json', dict(arm=arm, strength=strength, step=(block+1)*30,
            complete=False, best=best, seconds=time.perf_counter()-tick))
        print('VAL',arm,strength,(block+1)*30,metrics['oa'],metrics['ba'],
              'best',best['step'],'seconds',round(time.perf_counter()-tick,1),flush=True)
    torch.save(net.state_dict(), dest/'last.pt')
    np.savez_compressed(dest/'lastvalidation.npz', ids=val,label=inputs.y[val],score=score)
    if arm != 'full':
        net.exchange.disabled = True
        off = full.predict(net,inputs,val,'validation')
        np.savez_compressed(dest/'lastvalidation_off.npz',ids=val,label=inputs.y[val],score=off)
    report = dict(**cfg,best=best,last=metrics,final_drift=anchor.drift(),seconds=time.perf_counter()-tick)
    write(dest/'receipt.json',report)
    del net,anchor,opt,groups,params,task,logits
    gc.collect()
    torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('smoke','pilot'),default='smoke')
    args=parser.parse_args()
    model,inputs,selected=runtime()
    if args.phase=='smoke':
        old.smoke(model,inputs)
    else:
        assert read(ROOT/'smoke.json')['passed']
        reports=[]
        for strength in old.STRENGTHS:
            for arm in old.ARMS:
                reports.append(fit(model,inputs,selected,arm,strength))
                write(ROOT/'progress.json',dict(rows=reports,complete=len(reports)==4))
        write(ROOT/'status.json',dict(complete=True,arms=4,total_steps=960))
