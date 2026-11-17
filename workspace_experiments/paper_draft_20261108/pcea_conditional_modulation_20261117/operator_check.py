"""Nonzero-weight equivalence, update bounds and conditioning isolation."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

sys.dont_write_bytecode = True
import model

torch = model.torch


def main():
    torch.manual_seed(17)
    reader = model.nn.ModuleList([model.nn.Conv2d(512, 32, 1, bias=False) for _ in range(2)])
    original = model.old.PCEA('coordinated', reader)
    affine = model.RegionalModulation(reader, 'affine_head')
    with torch.no_grad():
        for layer in original.output:
            layer.weight.normal_(0, .03)
    missing = affine.load_state_dict(original.state_dict(), strict=False)
    assert missing.missing_keys == ['scale_output.0.weight', 'scale_output.1.weight']
    source = torch.randn(8, 2, 3, 5, 32)
    for module in (original, affine):
        module.source_raw = source
        module.source_weight = torch.ones(8, 3)
        object.__setattr__(module, 'reader', SimpleNamespace(project_raw=lambda raw: raw))
    x = torch.randn(3, 2, 256, 5, 5)
    masks = torch.rand(3, 2, 3, 5, 5)
    evidence = torch.randn(3, 2, 3, 5, 32)
    available = torch.tensor([[1, 1], [1, 1], [1, 0]])
    args = (x, masks, evidence, torch.ones(3, 2, 3), masks, available)
    with torch.no_grad():
        target = original(*args)
        zero_scale_error = float((target-affine(*args)).abs().max())
        assert zero_scale_error < 1e-6
        for layer in affine.scale_output:
            layer.weight.normal_(0, .05)
        actual = affine(*args)
        assert float((actual-target).abs().max()) > 1e-5
        assert torch.equal(actual[2], x[2])
        bounds = []
        writer = masks/masks.sum(2, keepdim=True).clamp_min(1)
        for p in range(2):
            w = masks[:, p].flatten(2)
            w = w/w.sum(-1, keepdim=True)
            regional = torch.einsum('bru,bcu->brc', w, x[:, p].flatten(2))
            scale = regional.square().mean(-1).sqrt()
            bounds.append(.1*(scale[..., None, None]*writer[:, p]).sum(1))
        bound = torch.stack(bounds, 1)[:, :, None]
        assert bool(((actual-x).abs() <= bound+1e-6).all())
        affine.no_scale = True
        no_scale_error = float((target-affine(*args)).abs().max())
        assert no_scale_error < 1e-6
        independent = model.RegionalModulation(reader, 'affine_independent')
        independent.load_state_dict(affine.state_dict())
        t = torch.randn(3, 2, 3, 5, 32)
        before = independent.coefficients(t)
        altered = t.clone()
        altered[:, 1] += torch.randn_like(altered[:, 1])
        after = independent.coefficients(altered)
        assert torch.equal(before[0][0], after[0][0])
        assert torch.equal(before[1][0], after[1][0])
        assert not torch.equal(before[0][1], after[0][1])
    result = dict(zero_scale_broadcast_max_error=zero_scale_error,
        no_scale_broadcast_max_error=no_scale_error,
        same_regional_update_bound=True, single_polar_no_update=True,
        independent_other_polarization_isolation=True,
        note='CPU synthetic operator check only; not SAR evidence or training.')
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
