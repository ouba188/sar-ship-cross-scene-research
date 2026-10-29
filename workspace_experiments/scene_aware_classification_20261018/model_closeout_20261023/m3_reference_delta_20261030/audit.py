"""Check reference-specific subtraction against the unchanged original writer."""
import importlib.util
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'m3_relation_repair_20261029/audit'))
import cpu_gradient_audit as base
import model
torch = base.torch


def run():
    torch.manual_seed(1030)
    old = base.SelectiveReferenceMatch().double().requires_grad_(False)
    with torch.no_grad():
        old.output.weight.normal_(std=.02)
    new = model.DeltaReferenceMatch('delta').double()
    new.load_state_dict(old.state_dict(),strict=False)
    q,r = base.inputs(2),base.inputs(4)
    q['features'] = torch.randn(2,1024,4,4,dtype=base.DTYPE)
    r['features'] = torch.randn(4,1024,4,4,dtype=base.DTYPE)
    legal = torch.ones(2,4,dtype=torch.bool)
    expected,_ = base.invoke(old,q,r,legal)
    initial,d = base.invoke(new,q,r,legal)
    identity_error = float((initial-expected).abs().max())
    assert identity_error == 0
    loss = (initial*torch.randn_like(initial)).sum()
    grad, = torch.autograd.grad(loss,new.extra_output.weight)
    assert grad.norm() > 0
    with torch.no_grad():
        new.extra_output.weight.normal_(std=.02)
    actual,d = base.invoke(new,q,r,legal)
    perm = torch.tensor([2,0,3,1])
    rp = {k:v[perm] for k,v in r.items()}
    shuffled,_ = base.invoke(new,q,rp,legal[:,perm])
    permutation_error = float((actual-shuffled).abs().max())
    assert permutation_error < 1e-10
    new.temperature = 1e30
    _,d = base.invoke(new,q,r,legal)
    uniform_update = float(d['extra_update'].abs().max())
    assert uniform_update < 1e-10
    new.temperature = .1
    q['valid'][0] = 0
    legal[1] = False
    _,d = base.invoke(new,q,r,legal)
    missing_update = float(d['extra_update'].abs().max())
    assert missing_update == 0
    result = dict(identity_error=identity_error,output_gradient_norm=float(grad.norm()),
        reference_permutation_error=permutation_error,uniform_reference_delta_update=uniform_update,
        missing_query_or_no_reference_update=missing_update,
        note='Synthetic CPU operator checks, not scientific benefit evidence.')
    (HERE/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    run()
