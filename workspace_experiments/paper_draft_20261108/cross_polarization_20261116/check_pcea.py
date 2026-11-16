"""Check operator interfaces, real cross-polar influence, and zero-init optimization."""
import json
from pathlib import Path
import torch
from torch import nn
from pcea import PolarizationConditionedEvidenceArbitration, regional_writeback


def main():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    net = PolarizationConditionedEvidenceArbitration()
    x = torch.randn(4, 2, 3, 5, 32)
    av = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]])
    region_valid = torch.ones(4, 2, 3, dtype=torch.bool)
    region_valid[0, 0, 2] = False
    out, trace = net(x, av, region_valid)
    assert torch.equal(out, x), 'zero initialization must reproduce the original tokens'
    target = torch.randn_like(x)
    optimizer = torch.optim.Adam(net.parameters(), lr=.01)
    gradient = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        out, trace = net(x, av, region_valid)
        (out - target).square().mean().backward()
        g = sum(float(p.grad.abs().sum()) for p in net.gate.parameters() if p.grad is not None)
        gradient.append(g)
        optimizer.step()
    assert gradient[1] > 0, 'gate must receive gradient after the zero output projection opens'
    out, trace = net(x, av, region_valid)
    assert torch.equal(out[1:], x[1:]), 'missing-polarization samples must retain their own evidence'
    assert torch.equal(out[0, :, 2], x[0, :, 2]), 'empty region cannot create cross evidence'
    changed = x.clone()
    changed[0, 1, 0] *= .5
    changed_out, _ = net(changed, av, region_valid)
    cross_effect = float((changed_out[0, 0, 0] - out[0, 0, 0]).abs().max().detach())
    assert cross_effect > 1e-7, 'VH evidence must be able to change VV update'
    # With constant gates the only value-combination distinction is the explicit product.
    with_product, _ = net(x, av, region_valid, constant_gate=True)
    without_product, _ = net(x, av, region_valid, product=False, constant_gate=True)
    assert (with_product - without_product).abs().max() > 1e-7
    masks = torch.zeros(4, 2, 3, 10, 10)
    masks[:, :, 0, 3:7, 3:7] = 1
    masks[:, :, 1, 1:3, 3:7] = 1
    masks[:, :, 2, 8:10, 3:7] = 1
    projection = nn.ModuleList(nn.Linear(160, 256, bias=False) for _ in range(2))
    update = regional_writeback(trace['delta'], masks, projection)
    assert update.shape == (4, 2, 256, 10, 10)
    assert torch.equal(update[:, :, :, 0], torch.zeros_like(update[:, :, :, 0]))
    assert torch.isfinite(update).all()
    assert update[0, :, :, 3:7, 3:7].abs().max() > 1e-7
    assert torch.equal(update[1:], torch.zeros_like(update[1:]))
    assert torch.equal(update[0, :, :, 8:10, 3:7], torch.zeros_like(update[0, :, :, 8:10, 3:7]))
    uncovered = (masks.sum(2) == 0).unsqueeze(2).expand_as(update)
    assert torch.equal(update[uncovered], torch.zeros_like(update[uncovered]))
    receipt = dict(parameters=sum(p.numel() for p in net.parameters()), identity_at_init=True,
        missing_polarization_fallback=True, empty_region_fallback=True,
        gate_gradient_step1_step2=gradient, measured_cross_polar_effect=cross_effect,
        spatial_write_shape=list(update.shape), active_region_nonzero=True,
        inactive_region_write_zero=True, missing_polarization_write_zero=True,
        formal_model_integrated=False,
        sar_training_performed=False, test='synthetic interface/gradient check, not classification efficacy')
    Path(__file__).with_name('operator_checks.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
