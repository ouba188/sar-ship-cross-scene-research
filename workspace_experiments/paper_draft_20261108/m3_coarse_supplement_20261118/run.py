"""Source-selected paired training followed by bounded development evaluation."""
from pathlib import Path
import argparse
import gc
import json
import math
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import model as m

torch, np = m.torch, m.np
recipe = m.load('coarse_plus_training_recipe', ROOT.parent/'pcea_class_local_20261117/run.py')
SEEDS = (17, 29, 43)
RATES = dict(backbone=1e-6, modules=1e-6, reader=1e-5, exchange=1e-5, supplement=3e-4)


def write(path, value):
    recipe.write(path, value)


def groups_for(net):
    buckets = {k: [] for k in RATES}
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('net.base.backbone.'):
            group = 'backbone'
        elif name.startswith('net.'):
            group = 'modules'
        else:
            group = {'ccrel': 'reader', 'exchange': 'exchange', 'supplement': 'supplement'}[name.split('.')[0]]
        buckets[group].append(param)
    return [dict(name=k, params=v, lr=RATES[k], base_lr=RATES[k]) for k, v in buckets.items() if v]


def runtime():
    recipe.ROOT = ROOT
    recipe.m.Inputs = m.Inputs
    inputs, fit, inner = recipe.runtime()
    scope = recipe.read(ROOT/'scope.json')
    scope.update(all_arms_start=str(m.START), steps=120, arms=list(m.ARMS), rates=RATES,
                 fine_slots_preserved=20, new_intervention_teacher=False,
                 seeds='same selected routed_s17 start; three continuation order/augmentation replicates')
    write(ROOT/'scope.json', scope)
    recipe.m = m
    recipe.RATES, recipe.STEPS, recipe.INTERVAL = RATES, 120, 30
    recipe.groups_for = groups_for
    return inputs, fit, inner


def smoke(inputs, fit):
    ids = fit[:12]
    scores, report = {}, {}
    for arm in m.ARMS:
        net = m.Model(arm)
        net.accelerate()
        scores[arm] = m.full.predict(net, inputs, ids, 'small')
        groups = groups_for(net)
        opt = torch.optim.AdamW(groups, weight_decay=.01)
        gradients = []
        for step in range(2):
            net.train()
            opt.zero_grad(set_to_none=True)
            loss, z = m.full.loss_for(net, inputs, ids, step)
            loss.backward()
            norms = {g['name']: sum(float(p.grad.square().sum()) for p in g['params'] if p.grad is not None)**.5
                     for g in groups}
            assert all(math.isfinite(v) and v > 0 for v in norms.values()), norms
            gradients.append(norms)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 5.)
            opt.step()
        extra = {}
        if hasattr(net, 'supplement'):
            for p in range(2):
                assert float(net.supplement.down[p].weight.grad.abs().sum()) > 0
                assert float(net.supplement.up[p].weight.grad.abs().sum()) > 0
            net.eval()
            on = m.full.predict(net, inputs, ids, 'small')
            net.supplement.disabled = True
            off = m.full.predict(net, inputs, ids, 'small')
            extra['trained_on_off_max'] = float(abs(on-off).max())
            assert extra['trained_on_off_max'] > 1e-7
            net.supplement.disabled = False
            with torch.no_grad():
                data = inputs.data(ids, 'small')
                a, _ = m.old.Model.extract(net, data, True)
                for p in range(2):
                    assert (data['coarse_adjacency'][:, p].diagonal(dim1=-2, dim2=-1)[a['mass'][:, p]>0] == 1).all()
                output = net.supplement(a['desc'], a['mass'], data['coarse_adjacency'])
                for lo, hi in ((0, 256), (256, 768)):
                    x = a['desc'][..., lo:hi]
                    scale = x.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
                    assert ((output[..., lo:hi]-x).abs() <= .10001*scale+1e-6).all()
                assert (output[a['mass'] == 0] == 0).all()
            extra['supplement_params'] = sum(p.numel() for p in net.supplement.parameters())
        report[arm] = dict(gradients=gradients, **extra)
        del net, opt, groups, loss, z
        gc.collect()
        torch.cuda.empty_cache()
    difference = max(float(abs(scores[a]-scores['continue']).max()) for a in m.ARMS)
    assert difference < 1e-5, difference
    assert report['pointwise']['supplement_params'] == report['coarse_plus']['supplement_params']
    write(ROOT/'smoke.json', dict(passed=True, initial_max_difference=difference, arms=report))
    print('SMOKE PASSED', difference, flush=True)


def evaluate():
    assert all((ROOT/'fits'/f'{a}_s{s}/receipt.json').exists() for a in m.ARMS for s in SEEDS)
    compact = m.geometry
    compact.m, compact.torch, compact.np = m.old, torch, np
    compact.ROOT = ROOT/'evaluation_inputs'
    compact.ROOT.mkdir(exist_ok=True)
    inputs = compact.Inputs()
    matrix = m.adjacency(inputs.rows)
    def data(ids):
        return dict(inputs.data(ids), coarse_adjacency=matrix[ids])
    records = []
    for seed in SEEDS:
        for arm in m.ARMS:
            dest = ROOT/'fits'/f'{arm}_s{seed}'
            net = m.Model(arm)
            net.load_state_dict(torch.load(dest/'best.pt', map_location='cuda', weights_only=True), strict=True)
            net.requires_grad_(False).eval()
            net.accelerate()
            for variant in (('on',) if arm == 'continue' else ('on', 'off')):
                if hasattr(net, 'supplement'):
                    net.supplement.disabled = variant == 'off'
                with torch.no_grad():
                    ref = net.reference(data(inputs.refs))
                    for split, ids in inputs.splits.items():
                        score = []
                        for first in range(0, len(ids), 12):
                            part = ids[first:first+12]
                            z, _ = net(data(part), [inputs.rows[i] for i in part], ref, inputs.rows[358:])
                            score.append(z.cpu().numpy())
                        score = np.concatenate(score)
                        np.savez_compressed(dest/f'{split}_{variant}_local.npz', ids=ids, label=inputs.y[ids], score=score)
                        metrics = m.full.metrics(inputs.y[ids], score)
                        records.append(dict(seed=seed, arm=arm, variant=variant, split=split,
                                            selected_step=recipe.read(dest/'receipt.json')['best']['step'], **metrics))
                        write(ROOT/'results.json', records)
                        print('EVAL', seed, arm, variant, split, metrics['oa'], metrics['ba'], flush=True)
            del net
            gc.collect()
            torch.cuda.empty_cache()
    write(ROOT/'status.json', dict(phase='complete', fits=9, evaluations=len(records), test3075_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('smoke', 'train', 'evaluate'), default='smoke')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase == 'evaluate':
        evaluate()
    else:
        inputs, fit, inner = runtime()
        if args.phase == 'smoke':
            smoke(inputs, fit)
        else:
            assert recipe.read(ROOT/'smoke.json')['passed']
            for seed in SEEDS:
                for arm in m.ARMS:
                    recipe.fit_one(inputs, fit, inner, arm, seed)
            del inputs
            gc.collect()
            evaluate()
