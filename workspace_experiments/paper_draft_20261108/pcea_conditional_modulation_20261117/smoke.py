"""Check actual insertion, zero-init replay, and reachable trainable paths."""
import gc
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import model

ROOT = Path(__file__).resolve().parent
torch, np, full = model.torch, model.np, model.full


def main():
    inputs = full.Inputs()
    for split in ('small', 'validation'):
        inputs.prepare_geometry(split)
    val = inputs.groups['validation'][:12]
    ids = inputs.groups['small'][:12]
    with np.load(full.ROOT/'selected_s17/validation.npz') as saved:
        expected = saved['score'][:len(val)]
    result = {}
    for arm in ('head_only', 'broadcast_head', 'affine_frozen', 'affine_head'):
        net = model.Model(17, arm)
        model.calibrate(net, inputs, 17)
        net.accelerate()
        z = full.predict(net, inputs, val, 'validation')
        error = float(abs(z-expected).max())
        assert error < 1e-3 and np.array_equal(z.argmax(1), expected.argmax(1))
        names = [n for n, p in net.named_parameters() if p.requires_grad]
        assert all(n.startswith(('ccrel.', 'exchange.', 'net.guide.head.')) for n in names)
        assert any(n.startswith('net.guide.head.') for n in names) == (arm != 'affine_frozen')
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4)
        steps = []
        for step in range(2):
            net.train()
            opt.zero_grad(set_to_none=True)
            loss, _ = full.loss_for(net, inputs, ids, 0)
            loss.backward()
            norms = {}
            for prefix in ('net.guide.head.', 'ccrel.value.', 'exchange.output.', 'exchange.scale_output.'):
                terms = [p.grad.square().sum() for n, p in net.named_parameters()
                         if n.startswith(prefix) and p.grad is not None]
                norms[prefix] = float(torch.stack(terms).sum().sqrt()) if terms else 0.
            assert torch.isfinite(loss) and all(np.isfinite(v) for v in norms.values())
            if arm != 'affine_frozen':
                assert norms['net.guide.head.'] > 0
            if arm != 'head_only':
                assert norms['exchange.output.'] > 0
                if step:
                    assert norms['ccrel.value.'] > 0
            if arm.startswith('affine_'):
                assert norms['exchange.scale_output.'] > 0
            steps.append(dict(loss=float(loss.detach()), gradient_norms=norms))
            opt.step()
        result[arm] = dict(zero_init_max_error=error, trainable=sum(p.numel() for p in params),
                           parameter_names=names, steps=steps)
        print('SMOKE', arm, result[arm]['trainable'], error, steps[-1], flush=True)
        del net, opt, params
        gc.collect()
        torch.cuda.empty_cache()
    (ROOT/'smoke.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
