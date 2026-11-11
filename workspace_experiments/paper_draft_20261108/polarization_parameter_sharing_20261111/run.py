"""Paired polarization parameter tying with exactly matched initial functions."""
from pathlib import Path
import argparse
import gc
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'staged_ablation_20261109'
sys.path.insert(0, str(SOURCE))
import staged as e

torch = e.torch
SEEDS = (17, 29, 43)
ARMS = ('specific', 'shared')
PAIRS = (
    'm1', 'base.internal_m2', 'base.deep_output',
    'guide.value', 'guide.output', 'guide.scorer',
    'matcher.shallow', 'matcher.shallow_norm', 'matcher.deep', 'matcher.deep_norm',
    'matcher.key', 'matcher.key_norm', 'matcher.value', 'matcher.value_norm', 'matcher.output',
)


def get_module(net, path):
    obj = net
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj


def install(net, arm):
    # Both arms start from the same two functions. Only parameter identity differs.
    for path in PAIRS:
        pair = get_module(net, path)
        assert len(pair) == 2
        pair[1].load_state_dict(pair[0].state_dict(), strict=True)
        if arm == 'shared':
            pair[1] = pair[0]


def audit(net, arm):
    result = {}
    for path in PAIRS:
        pair = get_module(net, path)
        a, b = dict(pair[0].named_parameters()), dict(pair[1].named_parameters())
        same_object = all(a[k] is b[k] for k in a)
        assert same_object == (arm == 'shared'), path
        result[path] = dict(tied=same_object, parameters_per_pol=sum(p.numel() for p in a.values()))
    return result


def create(arm, seed, protocol):
    net = e.Network('full', seed, protocol['pretrain'])
    install(net, arm)
    return net


def smoke(protocol, rows, groups, a, c):
    outputs, checks = {}, {}
    ids = groups['small'][:4]
    for arm in ARMS:
        net = create(arm, 17, protocol)
        checks[arm] = dict(parameters=sum(p.numel() for p in net.parameters()), tying=audit(net, arm))
        outputs[arm] = {}
        for phase in (0, 1, 2, 3):
            e.configure(net, phase)
            z, aux = net(a, c, rows, ids, groups['references'])
            outputs[arm][phase] = z.detach().cpu().numpy()
            loss = e.F.cross_entropy(z, torch.tensor([rows[i]['label'] for i in ids], device=e.task.DEVICE))
            if aux is not None:
                loss = loss + .02 * e.F.cross_entropy(aux.flatten(0,1), torch.tensor([rows[i]['label'] for i in ids], device=e.task.DEVICE).repeat_interleave(2))
            loss.backward()
            grads = {n: float(p.grad.norm()) for n,p in net.named_parameters() if p.requires_grad and p.grad is not None}
            assert grads and all(np.isfinite(list(grads.values())))
            checks[arm][f'phase{phase}_gradient'] = grads
            net.zero_grad(set_to_none=True)
        # Serialization must not break aliases.
        audit(net, arm)
        del net, z, aux, loss
        gc.collect(); torch.cuda.empty_cache()
    errors = {str(phase):float(abs(outputs['specific'][phase]-outputs['shared'][phase]).max()) for phase in outputs['shared']}
    assert max(errors.values()) < 1e-6, errors
    checks['initial_forward_errors'] = errors
    e.task.write(ROOT/'smoke.json', checks)
    print('SMOKE_OK', errors, flush=True)


def fit(arm, seed, protocol, rows, groups, a, c):
    folder = ROOT/f'{arm}_s{seed}'
    if (folder/'complete.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    net = create(arm, seed, protocol)
    initial_counts = sum(p.numel() for p in net.parameters())
    for cfg in protocol['phases']:
        dest = folder/cfg['name']
        e.task.write(ROOT/'status.json', dict(arm=arm, seed=seed, phase=cfg['name'], complete=False))
        if (dest/'receipt.json').exists():
            net.load_state_dict(torch.load(dest/'final.pt', map_location='cpu', weights_only=True))
            e.configure(net, cfg['phase'])
        else:
            e.train_phase(net, cfg, seed, folder, rows, groups, a, c, scale=1.)
        expected = json.loads((SOURCE/f'full_s{seed}_lr1'/cfg['name']/'orders.json').read_text())
        assert json.loads((dest/'orders.json').read_text()) == expected
        audit(net, arm)
        if cfg['phase'] == 1:
            e.freeze_delta(net, a, rows, groups['references'])
    e.configure(net, 3)
    net.requires_grad_(False)
    metrics = {}
    for split in ('validation', 'evaluation'):
        ids = groups[split]
        score = e.predict(net, a, c, rows, ids, groups['references'])
        np.savez_compressed(folder/f'{split}.npz', ids=ids, score=score)
        metrics[split] = e.task.metrics(np.array([rows[i]['label'] for i in ids]), score)
    divergence = {}
    for path in PAIRS:
        pair = get_module(net, path)
        diffs = [(p-q).detach().square().sum() for p,q in zip(pair[0].parameters(),pair[1].parameters())]
        divergence[path] = float(torch.stack(diffs).sum().sqrt()) if diffs else 0.
    if arm == 'shared':
        assert max(divergence.values()) == 0
    else:
        assert any(divergence[k] > 0 for k in ('m1','base.internal_m2','matcher.output'))
    e.task.write(folder/'complete.json', dict(arm=arm, seed=seed, metrics=metrics, parameters=initial_counts,
        orders_identical=True, parameter_divergence=divergence, formal_model_changed=False))
    print('COMPLETE', arm, seed, metrics, flush=True)
    del net
    gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = parser.parse_args()
    e.task.current.setup(e.task.DEVICE)
    protocol = json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    # Same ImageNet V2 file restored by the concurrent input-polarization experiment.
    protocol['pretrain'] = str(ROOT.parent/'oss2_polarization_training_20261111/resnet50-11ad3fa6.pth')
    assert Path(protocol['pretrain']).exists()
    rows, groups = e.prev.records()
    a, c = e.prev.arrays()
    a['write'] = np.load(e.PREVIOUS/'common_init/write.npy', mmap_mode='r')
    e.task.write(ROOT/'protocol.json', dict(parent=protocol, arms=ARMS, seeds=SEEDS, paired_module_paths=PAIRS,
        data={k:len(v) for k,v in groups.items()},
        initialization='Copy each VV operator to VH in BOTH arms; shared then aliases VV/VH objects. Exact same initial functions.',
        scope='Same full five-stage schedule, data, references, orders, optimizer rates and backbone; no input channel ablation.',
        teacher='Recompute selection and source-hard weights per arm under unchanged rules.',
        unchanged='Backbone weights/BN policy, geometry, masks, cached M3 descriptors, shared metric, final classifier architecture.',
        selection='Fixed scale1 inherited from current Full; no test tuning; current historical Full reported separately.'))
    e.resident(a,c)
    smoke(protocol,rows,groups,a,c)
    if not args.smoke_only:
        for seed in args.seeds:
            for arm in ARMS:
                fit(arm,seed,protocol,rows,groups,a,c)
        e.task.write(ROOT/'status.json', dict(complete=True, arms=ARMS, seeds=list(args.seeds)))


if __name__ == '__main__':
    main()
