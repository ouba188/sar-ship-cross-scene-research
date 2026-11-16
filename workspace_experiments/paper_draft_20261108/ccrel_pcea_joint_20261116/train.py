"""Paired joint-reader and shallow-residual development experiments."""
import argparse
import gc
import importlib.util
import sys
from model import ROOT, ARMS, Model, calibrate, full, torch, np

spec = importlib.util.spec_from_file_location('paired_fit_previous', ROOT.parent/'pcea_readout_repair_20261116/train.py')
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
runner.ROOT, runner.Model, runner.calibrate = ROOT, Model, calibrate


def smoke(inputs):
    records = {}
    ids = inputs.groups['small'][:12]
    for arm in ARMS:
        net = Model(17, arm)
        calibrate(net, inputs, 17)
        val = inputs.groups['validation'][:12]
        before = full.predict(net, inputs, val, 'validation')
        net.exchange.disabled = True
        original = full.predict(net, inputs, val, 'validation')
        net.exchange.disabled = False
        assert float(abs(before-original).max()) < 1e-5
        params = {n: p for n, p in net.named_parameters() if p.requires_grad}
        assert all(n.startswith(('exchange.', 'ccrel.')) for n in params)
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
                assert grads['ccrel.value.0.weight'] > 0
                assert grads['exchange.project.0.weight'] > 0
                if arm != 'joint':
                    assert grads['ccrel.fine_value.0.weight'] > 0
                if arm != 'independent_fine':
                    assert grads['exchange.gate.0.weight'] > 0
            torch.nn.utils.clip_grad_norm_(list(params.values()), 5.)
            opt.step()
            steps.append(dict(loss=float(loss.detach()), gradients=grads))
        actual = full.predict(net, inputs, val, 'validation')
        net.exchange.disabled = True
        withdrawn = full.predict(net, inputs, val, 'validation')
        assert float(abs(withdrawn-original).max()) < 1e-5
        records[arm] = dict(steps=steps, initial_error=float(abs(before-original).max()),
            learned_change=float(abs(actual-original).max()), withdrawal_error=float(abs(withdrawn-original).max()))
        print('SMOKE', arm, 'OK', flush=True)
        del net, opt, params; gc.collect(); torch.cuda.empty_cache()
    full.write(ROOT/'smoke.json', records)


def fit(inputs, arm, seed):
    runner.fit(inputs, arm, seed)
    # The shared runner owns numerical training only; record this experiment's true scope.
    import json
    for filename in ('config.json', 'receipt.json'):
        path = ROOT/f'{arm}_s{seed}'/filename
        content = json.loads(path.read_text(encoding='utf-8'))
        content['trainable_scope'] = 'CCREL value projection + optional layer1 residual projection + PCEA; original backbone/class response/reference/classifier frozen'
        content['evidence_interface'] = 'CCREL [B,2,3,5,32] passed directly to PCEA without detach'
        content['fine_residual'] = 'native layer1 256x40x40 -> LN/1x1/tanh -> area alignment20 -> bounded addition BEFORE regional pooling' if arm != 'joint' else 'none'
        content['source_statistics'] = 'previous identical initial reader source-only calibration, fixed units during joint reader learning'
        full.write(path, content)


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
