"""Small independent explicit attention oracle and gradient/masking checks."""
from pathlib import Path
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('checked_local', ROOT/'model.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
t = m.torch
t.set_num_threads(2)
t.manual_seed(17)
q = t.randn(2, 3, 5, 7, 16, requires_grad=True)
v = t.randn_like(q, requires_grad=True)
mask = t.rand(2, 3, 5, 7)
mask[..., -2:] = 0
actual = m.attend(q, v, mask)
logit = 4*t.matmul(m.F.normalize(q, dim=-1), m.F.normalize(v, dim=-1).transpose(-2, -1))
expected = t.matmul((logit+mask.clamp_min(1e-12).log()[..., None, :]).masked_fill(mask[..., None, :] == 0, -1e4).softmax(-1), v)
error = float((actual-expected).abs().max())
g1 = t.autograd.grad(actual.square().sum(), (q, v), retain_graph=True)
g2 = t.autograd.grad(expected.square().sum(), (q, v))
gradient_error = max(float((a-b).abs().max()) for a, b in zip(g1, g2))
assert error < 2e-6 and gradient_error < 2e-5, (error, gradient_error)
changed = v.detach().clone()
changed[..., -2:, :] = 1000
mask_error = float((m.attend(q.detach(), changed, mask)-actual.detach()).abs().max())
assert mask_error < 2e-6
mean = (v.detach()*mask[..., None]).sum(-2, keepdim=True)/mask.sum(-1)[..., None, None]
uniform_error = float((m.attend(q.detach(), v.detach(), mask, True)-mean).abs().max())
constant = mean.expand_as(v).clone()
constant_error = float((m.attend(q.detach(), constant, mask)-constant).abs().max())
assert uniform_error < 1e-6 and constant_error < 1e-6
report = dict(passed=True, explicit_error=error, gradient_error=gradient_error,
              outside_region_error=mask_error, uniform_error=uniform_error,
              constant_error=constant_error,
              scope='CPU operator checks; full-model replay/gradients checked separately')
reader=m.old.old.CCREL(m.nn.ModuleList(m.nn.Conv2d(512,32,1,bias=False) for _ in range(2)))
exchange=m.LocalExchange(reader.value,'cross_local')
exchange.source_raw=t.randn(7,2,3,5,512)
exchange.source_weight=t.ones(7,3)
object.__setattr__(exchange,'reader',reader)
for layer in list(exchange.output)+list(exchange.scale_output):
    m.nn.init.normal_(layer.weight,std=.02)
exchange.local_value=t.randn(1,2,32,20,20)
x=t.randn(1,2,256,10,10)
rm=t.rand(1,2,3,20,20)
wm=m.F.avg_pool2d(rm.flatten(0,1),2).reshape(1,2,3,10,10)
e=reader.project_raw(t.randn(1,2,3,5,512))
args=(x,wm,e,t.ones(1,2,3),rm,t.ones(1,2))
with t.no_grad():
    actual=exchange(*args)
    exchange.uniform_attention=True
    uniform=exchange(*args)
    exchange.uniform_attention=False
    exchange.mode='cross_mean'
    mean_out=exchange(*args)
    writer_equivalence=float((uniform-mean_out).abs().max())
    assert writer_equivalence<1e-6
    exchange.mode='cross_local'
    exchange.other_mean=True
    collapsed=exchange(*args)
    exchange.other_mean=False
    collapse_equivalence=float((collapsed-mean_out).abs().max())
    assert collapse_equivalence<2e-6
    exchange.mode='independent'
    before=exchange(*args)
    exchange.local_value[:,1]+=3
    after=exchange(*args)
    independent_error=float((before[:,0]-after[:,0]).abs().max())
    assert independent_error==0
    single=exchange(*args[:-1],t.tensor([[1.,0.]]))
    assert t.equal(single,x)
report.update(uniform_writer_equals_mean=writer_equivalence,
    collapsed_writer_equals_mean=collapse_equivalence,independent_cross_leak=independent_error,
    single_polar_identity=True,nonzero_local_effect=float((actual-mean_out).abs().max()))
assert report['nonzero_local_effect']>1e-6
overlap=m.origin0_overlap()
assert t.allclose(overlap.sum(1),t.ones(10),atol=1e-7)
# Unit impulses at fine-cell indices 1 and 2 both overlap coarse cell1;
# ordinary nonoverlapping 2x2 grouping incorrectly excludes index1 there.
assert overlap[1,1]>0 and overlap[1,2]>0 and overlap[1,3]>0
assert overlap[1,0]==0 and overlap[1,4]==0
mask=t.ones(1,1,20,20)
assert t.allclose(m.regional_reduce(t.full_like(mask,3.),mask,overlap),t.full((1,1,10,10),3.))
report.update(origin0_overlap_rows_sum_one=True,origin0_coarse_cell1_fine_support=t.where(overlap[1]>0)[0].tolist())
(ROOT/'operator_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
