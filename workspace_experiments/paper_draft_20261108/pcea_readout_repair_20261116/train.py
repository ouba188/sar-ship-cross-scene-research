"""Six paired branch-learning probes, same source and validation as the preceding pilot."""
import argparse
import gc
import math
import time
from model import ROOT, ARMS, Model, calibrate, full, torch, np


def smoke(inputs):
    records = {}
    ids = inputs.groups['small'][:12]
    for arm in ('coarse_raw', 'fine_center_gate', 'fine_center_product', 'fine_independent'):
        net = Model(17, arm)
        calibrate(net, inputs, 17)
        val = inputs.groups['validation'][:12]
        before = full.predict(net, inputs, val, 'validation')
        net.exchange.disabled = True
        original = full.predict(net, inputs, val, 'validation')
        net.exchange.disabled = False
        assert float(abs(before-original).max()) < 1e-5
        params = {n: p for n, p in net.named_parameters() if p.requires_grad}
        assert all(n.startswith('exchange.') and not n.startswith('exchange.value.') for n in params)
        opt = torch.optim.AdamW(params.values(), lr=1e-3, weight_decay=.01)
        steps = []
        for step in range(2):
            net.train(); opt.zero_grad(set_to_none=True)
            loss, _ = full.loss_for(net, inputs, ids, 1)
            loss.backward()
            grads = {n: float(p.grad.norm()) for n, p in params.items() if p.grad is not None}
            assert np.isfinite(list(grads.values())).all()
            assert grads['exchange.output.0.weight'] > 0
            if step:
                assert grads['exchange.project.0.weight'] > 0
                if arm != 'fine_independent':
                    assert grads['exchange.gate.0.weight'] > 0
            torch.nn.utils.clip_grad_norm_(list(params.values()), 5.)
            opt.step()
            steps.append(dict(loss=float(loss.detach()), gradients=grads))
        if arm.startswith('fine_center'):
            u = torch.randn(2, 3, 5, 16, device='cuda')
            zero = torch.zeros_like(u)
            assert float(net.exchange.interaction(u, zero)[0].abs().max()) < 1e-7
            assert float(net.exchange.interaction(zero, u)[0].abs().max()) < 1e-7
        records[arm] = dict(steps=steps, initial_error=float(abs(before-original).max()))
        print('SMOKE', arm, 'OK', flush=True)
        del net, opt, params; gc.collect(); torch.cuda.empty_cache()
    full.write(ROOT/'smoke.json', records)


def fit(inputs, arm, seed, epochs=4, rate=1e-3):
    dest = ROOT/f'{arm}_s{seed}'
    dest.mkdir(exist_ok=True)
    if (dest/'receipt.json').exists():
        return
    net = Model(seed, arm)
    calibrate(net, inputs, seed)
    net.accelerate()
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rate, weight_decay=.01)
    ids, val = inputs.groups['small'], inputs.groups['validation']
    z = full.predict(net, inputs, val, 'validation')
    expected = np.load(full.ROOT/f'selected_s{seed}/validation.npz')
    assert np.array_equal(val, expected['ids'])
    error = float(abs(z-expected['score']).max())
    assert error < 1e-3, error
    initial = full.metrics(inputs.y[val], z)
    best = dict(epoch=0, metrics=initial, key=[initial['ba'], initial['oa'], -initial['nll']])
    torch.save(net.state_dict(), dest/'best.pt')
    np.savez_compressed(dest/'validation.npz', ids=val, score=z, label=inputs.y[val])
    cfg = dict(arm=arm, seed=seed, source=len(ids), validation=len(val), epochs=epochs,
               learning_rate=rate, batch=24, microbatch=12, references=32,
               trainable_scope='four source-trained class offsets only' if arm == 'bias_control' else 'new projection/gate/output only; old network and shallow reader fixed',
               selection='validation BA/OA/-NLL; epoch0 eligible', initial=initial,
               initial_logit_error=error, test_read=False,
               source_statistics='all source code0, dual-valid regional class/vessel-weighted moments; no label-specific slots',
               augmentation='paired rot90/flip with native geometry',
               loss='same class/vessel weighted CE + .02 frozen M1 auxiliary CE')
    full.write(dest/'config.json', cfg)
    rng = np.random.default_rng(seed+20261116)
    aug = np.random.default_rng(seed+20261117)
    tick = time.perf_counter(); curve = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        factor = .5 if epoch == 0 else .1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(epochs-2, 1)))
        for group in opt.param_groups:
            group['lr'] = rate*factor
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
        z = full.predict(net, inputs, val, 'validation')
        m = full.metrics(inputs.y[val], z)
        row = dict(epoch=epoch+1, train_loss=total/len(ids), train_oa=100*correct/len(ids),
                   validation=m, seconds=time.perf_counter()-tick)
        curve.append(row)
        key = [m['ba'], m['oa'], -m['nll']]
        if key > best['key']:
            best = dict(epoch=epoch+1, metrics=m, key=key)
            torch.save(net.state_dict(), dest/'best.pt')
            np.savez_compressed(dest/'validation.npz', ids=val, score=z, label=inputs.y[val])
        full.write(dest/'curve.json', curve)
        full.write(ROOT/'status.json', dict(arm=arm, epoch=epoch+1, validation=m, best=best))
        print('VAL', arm, epoch+1, m, 'best', best['epoch'], 'sec', round(row['seconds'], 1), flush=True)
    torch.save(net.state_dict(), dest/'last.pt')
    np.savez_compressed(dest/'last_validation.npz', ids=val, score=z, label=inputs.y[val])
    full.write(dest/'receipt.json', dict(**cfg, best=best, last=curve[-1]['validation'],
               seconds=time.perf_counter()-tick, trainable=sum(p.numel() for p in params),
               peak_gib=torch.cuda.max_memory_allocated()/1024**3))
    del net, opt, params; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    inputs = full.Inputs()
    for split in ('small', 'validation'):
        inputs.prepare_geometry(split)
    if args.smoke:
        smoke(inputs)
    else:
        assert (ROOT/'smoke.json').exists()
        for arm in args.arms:
            fit(inputs, arm, args.seed)
        full.write(ROOT/'status.json', dict(state='complete', arms=args.arms, seed=args.seed, test_read=False))
