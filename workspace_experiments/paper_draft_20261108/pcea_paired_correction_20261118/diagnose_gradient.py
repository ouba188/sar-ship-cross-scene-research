"""Source-only gradient routing check; no optimization or model selection."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('paired_routing_recipe', ROOT/'run.py')
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)
torch, np = r.torch, r.np


def norm(grads):
    return sum(float(g.detach().square().sum()) for g in grads if g is not None)**.5


def main():
    inputs, fit, _ = r.runtime()
    net = r.m.Model('generic')
    net.accelerate()
    net.train()
    groups = r.previous.groups_for(net)
    params = [p for g in groups for p in g['params']]
    records = []
    for start in (0, 6):
        ids = fit[start:start+6]
        net.exchange.disabled = True
        with torch.no_grad():
            _, off = r.m.full.loss_for(net, inputs, ids, 1)
        net.exchange.disabled = False
        _, on = r.m.full.loss_for(net, inputs, ids, 1)
        y = torch.as_tensor(inputs.y[ids], device='cuda')
        w = torch.as_tensor(inputs.weights[ids], device='cuda')
        active = net.exchange.trace['active'].any(1)
        # At zero-initialized writeout, off is exactly the fixed starting teacher.
        loss, stats = r.correction(on, off, off.detach(), y, w, active)
        extra_grad = torch.autograd.grad(loss, params, allow_unused=True)
        assert all(bool(torch.isfinite(g).all()) for g in extra_grad if g is not None)
        offset, report = 0, {}
        for group in groups:
            count = len(group['params'])
            report[group['name']] = dict(extra_loss_gradient_norm=norm(extra_grad[offset:offset+count]))
            offset += count
        records.append(dict(source_offset=start, n=len(ids), stats=stats,
                            output_max_abs_difference=float((on-off).detach().abs().max()), gradients=report))
        del off, on, loss, extra_grad
    assert all(x['output_max_abs_difference'] < 1e-5 for x in records)
    assert any(x['gradients']['backbone']['extra_loss_gradient_norm'] > 0 for x in records)
    r.write(ROOT/'gradient_routing.json', dict(records=records, updates=0,
        interpretation='At zero writeout on/off predictions are equal, yet the actual detached-off auxiliary loss sends gradients to common parameters. This establishes a routing path, not the unique cause of external failure; raw norm does not measure eventual contribution.',
        test_accessed=False, external_predictions_used=False))
    print(r.read(ROOT/'gradient_routing.json'))


if __name__ == '__main__':
    main()
