"""Reuse the paired pilot schedule, but train layer3/layer4 and the classifier."""
from pathlib import Path
import argparse
import importlib.util
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'unfrozen_runner_previous', ROOT.parent/'pcea_conditional_modulation_20261117/run.py')
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
runner.ROOT = ROOT
runner.ARMS = ('full', 'cross_mean', 'cross_local')
runner.CONFIGS = {
    'base': dict(backbone=1e-5, head=1e-5, reader=1e-5, exchange=1e-3),
    'slow': dict(backbone=1e-6, head=1e-6, reader=1e-5, exchange=1e-3),
}
original_write = runner.write


def write(path, value):
    if isinstance(value, dict) and 'trainable_scope' in value:
        value = dict(value, trainable_scope='shared layer3 + layer4, classifier; added reader/exchange only in PCEA arms',
            bn='running statistics fixed; layer3/layer4 affine trainable',
            frozen_prefix='stem, layer1, layer2, original modules and anchor projection',
            calibration='source719 raw bank remains valid: pre-reader prefix and class-response projection fixed; current reader projection recomputes moments',
            references_policy='32 source references re-extracted by the existing live path per microbatch; prefix fixed',
            scope_note='late-backbone fine-tuning, NOT full-backbone unfreezing',
            hypothesis='does matched downstream co-adaptation enable incremental local-interaction utility?')
    original_write(path, value)


def parameter_groups(net, arm, config):
    buckets = {k: [] for k in ('backbone', 'head', 'reader', 'exchange')}
    names = {k: [] for k in buckets}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(('net.base.backbone.model.layer3.', 'net.base.backbone.model.layer4.')):
            key = 'backbone'
        elif name.startswith('net.guide.head.'):
            key = 'head'
        elif name.startswith('ccrel.value.'):
            key = 'reader'
        elif name.startswith('exchange.'):
            key = 'exchange'
        else:
            raise ValueError('Unplanned trainable parameter: '+name)
        buckets[key].append(p)
        names[key].append(name)
    expected = {'backbone', 'head'} if arm == 'full' else set(buckets)
    assert {k for k, v in buckets.items() if v} == expected
    groups, info = [], []
    for key, params in buckets.items():
        if not params:
            continue
        lr = runner.CONFIGS[config][key]
        groups.append(dict(name=key, params=params, lr=lr, base_lr=lr))
        info.append(dict(name=key, base_lr=lr, parameter_names=names[key],
            parameter_count=sum(p.numel() for p in params),
            epoch_learning_rates=[lr*runner.schedule_factor(e) for e in range(runner.EPOCHS)]))
    return groups, info


runner.parameter_groups = parameter_groups
runner.write = write


def smoke(model, inputs):
    torch, full = model.torch, model.full
    for arm in runner.ARMS:
        report = runner.test_gradients(model, inputs, arm)
        print('GRADIENTS', arm, report['passed'], flush=True)
    # Detect stale source coordinates, stopped backbone gradients, or BN drift.
    net = model.Model(17, 'cross_local')
    model.calibrate(net, inputs, 17)
    net.accelerate()
    groups, _ = parameter_groups(net, 'cross_local', 'base')
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    bb = net.net.base.backbone.model
    before = bb.layer3[5].conv2.weight.detach().clone()
    prefix = bb.layer2[3].conv3.weight.detach().clone()
    bn = {n: b.detach().clone() for n, b in net.named_buffers() if 'running_' in n}
    net.eval()
    with torch.no_grad():
        ref0 = net.reference(inputs.data(inputs.groups['references'], 'small'))['desc'].clone()
        full.loss_for(net, inputs, inputs.groups['small'][:12], 0)
        raw0 = net.ccrel.raw.clone()
    net.train()
    loss, _ = full.loss_for(net, inputs, inputs.groups['small'][:12], 0)
    loss.backward()
    assert bb.layer3[5].conv2.weight.grad.abs().sum() > 0
    assert bb.layer4[2].conv3.weight.grad.abs().sum() > 0
    opt.step()
    with torch.no_grad():
        ref1 = net.reference(inputs.data(inputs.groups['references'], 'small'))['desc']
        full.loss_for(net, inputs, inputs.groups['small'][:12], 0)
        raw1 = net.ccrel.raw
    check = dict(reference_change=float((ref1-ref0).abs().max()),
        source_raw_change=float((raw1-raw0).abs().max()),
        prefix_change=float((bb.layer2[3].conv3.weight-prefix).abs().max()),
        receiver_change=float((bb.layer3[5].conv2.weight-before).abs().max()),
        bn_running_change=max(float((b-bn[n]).abs().max()) for n,b in net.named_buffers() if n in bn))
    assert check['receiver_change'] > 0
    assert all(check[k] == 0 for k in ('reference_change', 'source_raw_change', 'prefix_change', 'bn_running_change'))
    write(ROOT/'scope_check.json', check)
    print('SCOPE_CHECK', check, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('smoke', 'tune', 'confirm', 'evaluate'), default='tune')
    p.add_argument('--arms', nargs='+', choices=runner.ARMS, default=list(runner.ARMS))
    args = p.parse_args()
    model, inputs = runner.runtime()
    if args.phase == 'smoke':
        smoke(model, inputs)
    else:
        {'tune': runner.tune, 'confirm': runner.confirm, 'evaluate': runner.evaluate}[args.phase](model, inputs, args.arms)
