"""Bounded reader-LR repair; independent loop, unchanged model and source calibration."""
import argparse
import gc
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT/'reader_lr1e5'
ARMS = ('joint', 'joint_fine', 'independent_fine')
SEED, EPOCHS = 17, 4
READER_LR, OTHER_LR = 1e-5, 1e-3


def schedule_factor(epoch):
    return .5 if epoch == 0 else .1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(EPOCHS-2, 1)))


def parameter_groups(net):
    buckets = dict(ccrel_value=[], other_trainable=[])
    names = {key: [] for key in buckets}
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        key = 'ccrel_value' if name.startswith('ccrel.value.') else 'other_trainable'
        buckets[key].append(parameter)
        names[key].append(name)
    if not all(buckets.values()):
        raise ValueError('Both reader and other-trainable optimizer groups must be nonempty')
    groups, description = [], []
    for key, rate in (('ccrel_value', READER_LR), ('other_trainable', OTHER_LR)):
        groups.append(dict(name=key, params=buckets[key], lr=rate, base_lr=rate))
        description.append(dict(name=key, base_lr=rate, weight_decay=.01,
                                parameter_count=sum(p.numel() for p in buckets[key]),
                                parameter_names=names[key],
                                epoch_learning_rates=[rate*schedule_factor(e) for e in range(EPOCHS)]))
    return groups, description


def fit(model, inputs, arm):
    torch, np, full = model.torch, model.np, model.full
    dest = RUN_ROOT/f'{arm}_s{SEED}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'receipt.json').exists():
        print('SKIP_COMPLETE', arm, flush=True)
        return
    net = model.Model(SEED, arm)
    model.calibrate(net, inputs, SEED)
    net.accelerate()
    groups, group_config = parameter_groups(net)
    params = [p for group in groups for p in group['params']]
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    ids, val = inputs.groups['small'], inputs.groups['validation']
    z = full.predict(net, inputs, val, 'validation')
    with np.load(full.ROOT/f'selected_s{SEED}/validation.npz', allow_pickle=False) as expected:
        assert np.array_equal(val, expected['ids'])
        error = float(abs(z-expected['score']).max())
        identical = bool(np.array_equal(z.argmax(1), expected['score'].argmax(1)))
    assert error < 1e-3 and identical, (error, identical)
    initial = full.metrics(inputs.y[val], z)
    best = dict(epoch=0, metrics=initial, key=[initial['ba'], initial['oa'], -initial['nll']])
    torch.save(net.state_dict(), dest/'best.pt')
    np.savez_compressed(dest/'validation.npz', ids=val, score=z, label=inputs.y[val])
    cfg = dict(arm=arm, seed=SEED, source=len(ids), validation=len(val), epochs=EPOCHS,
               optimizer='AdamW', parameter_groups=group_config,
               learning_rates=dict(ccrel_value=READER_LR, other_trainable=OTHER_LR),
               weight_decay=.01, clip_grad_norm=5., batch=24, microbatch=12, references=32,
               schedule='same previous 4-epoch multipliers: 0.5, 1.0, 0.55, 0.1',
               order_rng_seed=SEED+20261116, augmentation_rng_seed=SEED+20261117,
               trainable_scope='ccrel.value at 1e-5; ALL remaining trainable parameters at 1e-3, including new ccrel.fine_value; original network frozen by Model',
               selection='validation BA/OA/-NLL; epoch0 eligible', initial=initial,
               initial_logit_error=error, initial_predictions_identical=identical, test_read=False,
               source_statistics='unchanged model.calibrate: previous identical initial reader source-only moments, fixed during training',
               calibration_root=str(model.old.ROOT),
               evidence_interface='CCREL [B,2,3,5,32] passed to PCEA without detach',
               fine_residual='native layer1 residual before pooling' if arm != 'joint' else 'none',
               augmentation='paired rot90/flip with native geometry; same RNG and per-batch draws as previous fit',
               loss='same class/vessel weighted CE + .02 frozen M1 auxiliary CE',
               run_root=str(RUN_ROOT))
    full.write(dest/'config.json', cfg)
    rng = np.random.default_rng(SEED+20261116)
    aug = np.random.default_rng(SEED+20261117)
    tick = time.perf_counter(); curve = []; steps = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        factor = schedule_factor(epoch)
        for group in opt.param_groups:
            group['lr'] = group['base_lr']*factor
        net.train(); order = rng.permutation(ids); total = correct = 0
        for first in range(0, len(order), 24):
            batch = order[first:first+24]; opt.zero_grad(set_to_none=True)
            code = int(aug.integers(4))+4*int(aug.random() < .5)
            for start in range(0, len(batch), 12):
                part = batch[start:start+12]
                loss, score = full.loss_for(net, inputs, part, code)
                assert torch.isfinite(loss)
                (loss*len(part)/len(batch)).backward()
                total += float(loss.detach())*len(part)
                correct += int((score.argmax(1) == torch.as_tensor(inputs.y[part], device='cuda')).sum())
            torch.nn.utils.clip_grad_norm_(params, 5.)
            opt.step()
            steps += 1
        z = full.predict(net, inputs, val, 'validation')
        m = full.metrics(inputs.y[val], z)
        row = dict(epoch=epoch+1, train_loss=total/len(ids), train_oa=100*correct/len(ids),
                   validation=m, seconds=time.perf_counter()-tick, lr_factor=factor,
                   learning_rates={g['name']: g['lr'] for g in opt.param_groups})
        curve.append(row)
        key = [m['ba'], m['oa'], -m['nll']]
        if key > best['key']:
            best = dict(epoch=epoch+1, metrics=m, key=key)
            torch.save(net.state_dict(), dest/'best.pt')
            np.savez_compressed(dest/'validation.npz', ids=val, score=z, label=inputs.y[val])
        full.write(dest/'curve.json', curve)
        full.write(RUN_ROOT/'status.json', dict(arm=arm, epoch=epoch+1, validation=m, best=best,
                                              learning_rates=row['learning_rates']))
        print('VAL', arm, epoch+1, m, 'best', best['epoch'], 'lr', row['learning_rates'], flush=True)
    torch.save(net.state_dict(), dest/'last.pt')
    np.savez_compressed(dest/'last_validation.npz', ids=val, score=z, label=inputs.y[val])
    full.write(dest/'receipt.json', dict(**cfg, best=best, last=curve[-1]['validation'], steps=steps,
               seconds=time.perf_counter()-tick, trainable=sum(p.numel() for p in params),
               peak_gib=torch.cuda.max_memory_allocated()/1024**3))
    del net, opt, params, groups
    gc.collect(); torch.cuda.empty_cache()


def self_test():
    class Parameter:
        def __init__(self, size, trainable=True):
            self.size, self.requires_grad = size, trainable
        def numel(self): return self.size
    parameters = [('ccrel.value.0.weight', Parameter(10)), ('ccrel.value.1.weight', Parameter(11)),
                  ('ccrel.fine_value.0.weight', Parameter(12)), ('exchange.output.0.weight', Parameter(13)),
                  ('exchange.gate.0.weight', Parameter(14)), ('net.guide.value.0.weight', Parameter(15, False))]
    class Net:
        def named_parameters(self): return iter(parameters)
    groups, config = parameter_groups(Net())
    assert config[0]['parameter_names'] == ['ccrel.value.0.weight', 'ccrel.value.1.weight']
    assert config[1]['parameter_names'] == [name for name, _ in parameters[2:5]]
    assert config[0]['parameter_count'] == 21 and config[1]['parameter_count'] == 39
    assert groups[0]['lr'] == 1e-5 and groups[1]['lr'] == 1e-3
    for epoch, expected in enumerate((.5, 1., .55, .1)):
        assert math.isclose(schedule_factor(epoch), expected)
        assert math.isclose(config[0]['epoch_learning_rates'][epoch], 1e-5*expected)
        assert math.isclose(config[1]['epoch_learning_rates'][epoch], 1e-3*expected)
    assert len({id(p) for group in groups for p in group['params']}) == 5
    assert 'torch' not in sys.modules
    print('SCALED_SELF_TEST_OK: exact group membership, fine projection at other LR, frozen exclusion, schedule; no torch/GPU/file writes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--seed', type=int, choices=(SEED,), default=SEED)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    import model
    assert (ROOT/'smoke.json').exists(), 'Existing model smoke must be complete'
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    model.full.write(RUN_ROOT/'run_config.json', dict(planned_arms=list(ARMS), seed=SEED, epochs=EPOCHS,
                     model_root=str(ROOT), learning_rates=dict(ccrel_value=READER_LR, other_trainable=OTHER_LR),
                     source_calibration='unchanged model.calibrate', test_read=False))
    inputs = model.full.Inputs()
    for split in ('small', 'validation'):
        inputs.prepare_geometry(split)
    for arm in args.arms:
        fit(model, inputs, arm)
    completed = [a for a in ARMS if (RUN_ROOT/f'{a}_s{SEED}/receipt.json').exists()]
    model.full.write(RUN_ROOT/'status.json', dict(state='complete' if len(completed) == len(ARMS) else 'requested_arms_complete',
                     completed_arms=completed, requested_arms=args.arms, seed=SEED, test_read=False))


if __name__ == '__main__':
    main()
