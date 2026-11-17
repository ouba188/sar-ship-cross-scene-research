"""Same-weight pooled equivalence and genuine local dependence; no SAR claims."""
import json
from pathlib import Path
from types import SimpleNamespace
import model

torch = model.torch
torch.manual_seed(17)
reader = model.nn.ModuleList(model.nn.Conv2d(512, 32, 1, bias=False) for _ in range(2))
baseline = model.old.RegionalModulation(reader, 'affine_frozen')
candidate = model.Modulation(reader, 'both')
with torch.no_grad():
    for layer in list(baseline.output)+list(baseline.scale_output):
        layer.weight.normal_(0, .02)
candidate.load_state_dict(baseline.state_dict())
source = torch.randn(12, 2, 3, 5, 32)
for net in (baseline, candidate):
    net.source_raw = source
    net.source_weight = torch.ones(12, 3)
    object.__setattr__(net, 'reader', SimpleNamespace(project_raw=lambda x: x))
x = torch.randn(3, 2, 256, 10, 10)
mask = torch.rand(3, 2, 3, 10, 10)
evidence = torch.randn(3, 2, 3, 5, 32)
available = torch.tensor([[1., 1.], [1., 1.], [1., 0.]])
args = x, mask, evidence, torch.ones(3, 2, 3), mask, available
candidate.local_value = torch.randn(3, 2, 32, 10, 10)
candidate.teacher_score = torch.tensor([[1., 2., 3., 4.]]).expand(3, -1)
candidate.no_detail = candidate.no_competition = True
with torch.no_grad():
    expected = baseline(*args)
    actual = candidate(*args)
    error = float((actual-expected).abs().max())
    assert error < 1e-6, error
    candidate.no_detail = False
    local = candidate(*args)
    detail_change = float((local-actual).abs().max())
    assert detail_change > 1e-5
    candidate.no_competition = False
    joint = candidate(*args)
    competition_change = float((joint-local).abs().max())
    assert competition_change > 1e-5
    assert torch.equal(joint[2], x[2])
report = dict(pooled_equivalence_max_error=error, local_changes_output=detail_change,
              predicted_top2_changes_output=competition_change, single_polar_no_update=True,
              trainable_equal=sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in baseline.parameters()),
              note='Synthetic operator checks, not classification evidence.')
Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(report)
