"""Matched continuation, source-inner selection, external evaluation after all fits."""
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
spec = importlib.util.spec_from_file_location('class_detail_model', ROOT/'model.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
torch, np = m.torch, m.np
SEEDS = (17, 29, 43)
RATES = dict(backbone=1e-6, modules=1e-6, reader=1e-5, exchange=3e-4)
STEPS, INTERVAL = 240, 30


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def key(metrics):
    return (metrics['ba'], metrics['oa'], -metrics['nll'])


def runtime():
    inputs = m.Inputs()
    split = read(ROOT.parent/'pcea_readout_probe_20261117/source_split_local.json')
    meta = read(ROOT.parent/'pcea_readout_probe_20261117/metadata_local.json')
    inner = np.asarray([meta['train'][i]['index'] for i in split['inner']], dtype=np.int64)
    def passage(i):
        return '_'.join(inputs.rows[i]['pass_group'].split('_')[:2])
    vessels = {inputs.rows[i]['mmsi'] for i in inner}
    passes = {passage(i) for i in inner}
    fit = np.array([i for i in inputs.groups['small'] if inputs.rows[i]['mmsi'] not in vessels
                    and passage(i) not in passes], dtype=np.int64)
    for field in (lambda i: inputs.rows[i]['mmsi'], passage):
        assert not {field(i) for i in fit} & {field(i) for i in inner}
        assert not {field(i) for i in inputs.groups['references']} & {field(i) for i in inner}
    inputs.weights = m.old.t.e.weights(inputs.rows, fit)
    def describe(ids):
        av = np.array(inputs.arrays['available'][ids], copy=True)
        for j, i in enumerate(ids):
            if int(i) in inputs.restored:
                av[j, inputs.overlay[int(i)]['p']] = 1
        return dict(n=len(ids), classes=np.bincount(inputs.y[ids], minlength=8).tolist(),
                    dual=int(av.astype(bool).all(1).sum()),
                    ports=len({inputs.rows[i]['port'] for i in ids}),
                    passes=len({passage(i) for i in ids}),
                    vessels=len({inputs.rows[i]['mmsi'] for i in ids}))
    write(ROOT/'split_local.json', dict(fit=fit.tolist(), inner=inner.tolist()))
    write(ROOT/'scope.json', dict(fit=describe(fit), inner=describe(inner),
          validation=describe(inputs.groups['validation']), references=describe(inputs.groups['references']),
          selection='source-inner BA/OA/-NLL; encoder has previously seen these source observations',
          external_validation_only_after_all_fits=True, test_accessed=False,
          all_arms_start='formal selected_s17/best.pt', seeds='continuation order/augmentation replicates'))
    return inputs, fit, inner


def groups_for(net):
    buckets = {k: [] for k in RATES}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('net.base.backbone.'):
            group = 'backbone'
        elif name.startswith('net.'):
            group = 'modules'
        elif name.startswith('ccrel.'):
            group = 'reader'
        elif name.startswith('exchange.'):
            group = 'exchange'
        else:
            raise ValueError(name)
        buckets[group].append(p)
    return [dict(name=k, params=v, lr=RATES[k], base_lr=RATES[k]) for k, v in buckets.items() if v]


def smoke(inputs, fit):
    ids = fit[:12]
    scores, report = {}, {}
    for arm in m.ARMS:
        net = m.Model(arm)
        net.accelerate()
        scores[arm] = m.full.predict(net, inputs, ids, 'small')
        groups = groups_for(net)
        opt = torch.optim.AdamW(groups, weight_decay=.01)
        gradient = []
        for step in range(2):
            net.train()
            opt.zero_grad(set_to_none=True)
            loss, z = m.full.loss_for(net, inputs, ids, 0)
            loss.backward()
            norms = {g['name']: sum(float(p.grad.square().sum()) for p in g['params'] if p.grad is not None)**.5 for g in groups}
            assert all(math.isfinite(x) for x in norms.values())
            if step == 1:
                assert all(x > 0 for x in norms.values()), norms
            gradient.append(norms)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 5.)
            opt.step()
        extra = {}
        if arm == 'class_detail':
            net.eval()
            on = m.full.predict(net, inputs, ids, 'small')
            gain = net.ccrel.detail_gain
            mass = net.ccrel.detail_mass
            nonempty = mass.sum(-1) > 0
            mean_error = float((((gain*mass[:, :, :, None]).sum(-1)-1)*nonempty[..., None]).abs().max())
            assert mean_error < 1e-5
            assert float((gain[:, :, :, 0]-1).abs().max()) == 0
            assert float(gain[:, :, :, 1:].std(3).max()) > 0
            probe = (gain[..., ::7]**2).sum()
            assert not probe.requires_grad  # predict is inference-only
            net.exchange.uniform_detail = True
            uniform = m.full.predict(net, inputs, ids, 'small')
            net.exchange.uniform_detail = False
            net.exchange.use_class_detail = False
            generic = m.full.predict(net, inputs, ids, 'small')
            assert float(abs(uniform-generic).max()) == 0
            extra = dict(gain_mean_error=mean_error, gain_min=float(gain.min()), gain_max=float(gain.max()),
                         class_to_uniform_logit_max=float(abs(on-uniform).max()), uniform_generic_error=0.)
            # Direct gradient to class weights verifies the new path independently of old summary supervision.
            net.train()
            net.exchange.use_class_detail = True
            loss, z = m.full.loss_for(net, inputs, ids, 0)
            grad = torch.autograd.grad(loss, net.ccrel.detail_gain, retain_graph=False)[0]
            assert float(grad.abs().sum()) > 0
            extra['detail_gain_loss_gradient_l1'] = float(grad.abs().sum())
        report[arm] = dict(gradients=gradient, params=sum(p.numel() for p in net.parameters() if p.requires_grad), **extra)
        del net, opt, groups, loss, z
        gc.collect()
        torch.cuda.empty_cache()
    diff = max(float(abs(scores[a]-scores['full']).max()) for a in m.ARMS)
    assert diff < 1e-5, diff
    assert report['generic']['params'] == report['class_detail']['params']
    write(ROOT/'smoke.json', dict(passed=True, initial_max_difference=diff, arms=report))
    print('SMOKE PASSED', json.dumps(report), flush=True)


def fit_one(inputs, fit, inner, arm, seed):
    dest = ROOT/'fits'/f'{arm}_s{seed}'
    if (dest/'receipt.json').exists():
        return read(dest/'receipt.json')
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Incomplete fit needs inspection: '+str(dest))
    net = m.Model(arm)
    net.accelerate()
    groups = groups_for(net)
    params = [p for g in groups for p in g['params']]
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    order = np.random.default_rng(seed+20261117).permutation(fit)[:STEPS*24]
    assert len(order) == STEPS*24
    aug = np.random.default_rng(seed+20261118)
    score = m.full.predict(net, inputs, inner, 'small')
    initial = m.full.metrics(inputs.y[inner], score)
    best = dict(step=0, metrics=initial)
    dest.mkdir(parents=True)
    write(dest/'config.json', dict(arm=arm, continuation_seed=seed, checkpoint_seed=17,
          rates=RATES, steps=STEPS, batch=24, microbatch=12, samples=len(order),
          selection='source-inner BA/OA/-NLL, step0 eligible',
          objective='source class/vessel weighted CE + .02 original M1 auxiliary CE',
          initialization=str(net.start_checkpoint), full_backbone_trainable=True,
          bn_running_fixed=True, intervention='inherited checkpoint history; no new intervention teacher',
          params={g['name']:sum(p.numel() for p in g['params']) for g in groups}))
    np.save(dest/'training_ids_local.npy', order)
    torch.save(net.state_dict(), dest/'best.pt')
    np.savez_compressed(dest/'inner.npz', ids=inner, label=inputs.y[inner], score=score)
    curve, tick = [], time.perf_counter()
    for block in range(STEPS//INTERVAL):
        factor = .1+.9*.5*(1+math.cos(math.pi*block/(STEPS//INTERVAL-1)))
        for group in groups:
            group['lr'] = group['base_lr']*factor
        net.train()
        loss_sum, correct = 0., 0
        for j in range(INTERVAL):
            step = block*INTERVAL+j
            batch = order[step*24:(step+1)*24]
            code = int(aug.integers(4))+4*int(aug.random()<.5)
            opt.zero_grad(set_to_none=True)
            for offset in (0, 12):
                ids = batch[offset:offset+12]
                loss, z = m.full.loss_for(net, inputs, ids, code)
                assert torch.isfinite(loss)
                (loss*.5).backward()
                loss_sum += float(loss.detach())*12
                correct += int((z.argmax(1)==torch.as_tensor(inputs.y[ids], device='cuda')).sum())
            torch.nn.utils.clip_grad_norm_(params, 5.)
            opt.step()
        score = m.full.predict(net, inputs, inner, 'small')
        metrics = m.full.metrics(inputs.y[inner], score)
        row = dict(step=(block+1)*INTERVAL, inner=metrics, training_loss=loss_sum/(INTERVAL*24),
                   training_oa=100*correct/(INTERVAL*24), seconds=time.perf_counter()-tick)
        curve.append(row)
        if key(metrics) > key(best['metrics']):
            best = dict(step=row['step'], metrics=metrics)
            torch.save(net.state_dict(), dest/'best.pt')
            np.savez_compressed(dest/'inner.npz', ids=inner, label=inputs.y[inner], score=score)
        write(dest/'curve.json', curve)
        write(ROOT/'status.json', dict(phase='training', arm=arm, seed=seed, step=row['step'], best=best))
        print('TRAIN', arm, seed, row['step'], 'inner OA/BA', metrics['oa'], metrics['ba'],
              'best', best['step'], 'sec', round(row['seconds'], 1), flush=True)
    torch.save(net.state_dict(), dest/'last.pt')
    report = dict(arm=arm, seed=seed, initial=initial, best=best, last=metrics, seconds=time.perf_counter()-tick)
    write(dest/'receipt.json', report)
    del net, opt, groups, params, loss, z
    gc.collect()
    torch.cuda.empty_cache()
    return report


def evaluate(inputs):
    assert all((ROOT/'fits'/f'{a}_s{s}/receipt.json').exists() for a in m.ARMS for s in SEEDS)
    ids = inputs.groups['validation']
    reports = []
    for seed in SEEDS:
        for arm in m.ARMS:
            dest = ROOT/'fits'/f'{arm}_s{seed}'
            net = m.Model(arm)
            net.load_state_dict(torch.load(dest/'best.pt', map_location='cuda', weights_only=True), strict=True)
            net.accelerate()
            variants = ['on'] if arm == 'full' else ['on', 'off']
            if arm == 'class_detail':
                variants += ['uniform', 'permuted']
            entry = dict(arm=arm, seed=seed, best=read(dest/'receipt.json')['best'])
            for variant in variants:
                if arm != 'full':
                    net.exchange.disabled = variant == 'off'
                    net.exchange.uniform_detail = variant == 'uniform'
                    net.exchange.permute_detail = variant == 'permuted'
                score = m.full.predict(net, inputs, ids, 'validation')
                np.savez_compressed(dest/f'validation_{variant}.npz', ids=ids, label=inputs.y[ids], score=score)
                entry[variant] = m.full.metrics(inputs.y[ids], score)
            reports.append(entry)
            write(ROOT/'results.json', reports)
            print('EVAL', arm, seed, entry['on']['oa'], entry['on']['ba'], flush=True)
            del net
            gc.collect()
            torch.cuda.empty_cache()
    write(ROOT/'status.json', dict(phase='complete', fits=9, test_accessed=False))


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
            for arm in m.ARMS:
                fit_one(inputs, fit, inner, arm, seed)
        evaluate(inputs)
