"""Independent CPU checks of the experiment-only metric, without importing run.py."""
import copy
import importlib.util
import json
from pathlib import Path
import cpu_gradient_audit as audit

torch, nn, F = audit.torch, audit.nn, audit.F
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('audit_experiment_metric', HERE.parent/'model.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def run():
    torch.manual_seed(1030)
    old = audit.SelectiveReferenceMatch().double()
    with torch.no_grad():
        old.output.weight.normal_(std=.02)
        old.selector[-1].weight.normal_(std=.01)
    metric = candidate.MetricReferenceMatch().double()
    missing, unexpected = metric.load_state_dict(old.state_dict(), strict=False)
    assert set(missing) == {'metric_deep', 'metric_local'} and not unexpected
    q, r = audit.inputs(2), audit.inputs(4)
    q['features'] = torch.randn(2, 1024, 4, 4, dtype=audit.DTYPE)
    r['features'] = torch.randn(4, 1024, 4, 4, dtype=audit.DTYPE)
    legal = torch.ones(2, 4, dtype=torch.bool)
    labels, refs = torch.tensor([0, 1]), torch.tensor([0, 0, 1, 1])
    yo, do = audit.invoke(old, q, r, legal)
    ym, dm = audit.invoke(metric, q, r, legal)
    identity = dict(output=float((yo-ym).abs().max()),
                    relation=float((do['reference_logits']-dm['reference_logits']).abs().max()),
                    attention=float((do['reference_weights']-dm['reference_weights']).abs().max()))
    assert max(identity.values()) < 1e-12
    rel = audit.relation_loss(dm['reference_logits'], dm['reference_valid'], labels, refs)
    head = torch.randn(1024, 2, dtype=audit.DTYPE)
    ce = F.cross_entropy(ym.mean((-2, -1)) @ head, labels)
    gradients = {}
    for name, loss in [('relation', rel), ('ce', ce)]:
        gd, gl = torch.autograd.grad(loss, (metric.metric_deep, metric.metric_local), retain_graph=True)
        gradients[name] = dict(deep=float(gd.norm()), local=float(gl.norm()))
        assert gd.norm() > 0 and gl.norm() > 0
    analytic, = torch.autograd.grad(rel, metric.metric_deep)
    coordinate = int(analytic.abs().argmax())
    values = []
    h = 1e-4
    for delta in (h, -h):
        with torch.no_grad():
            metric.metric_deep[coordinate] = delta
        _, d = audit.invoke(metric, q, r, legal)
        values.append(float(audit.relation_loss(d['reference_logits'], d['reference_valid'], labels, refs)))
    with torch.no_grad():
        metric.metric_deep[coordinate] = 0
    finite_error = abs((values[0]-values[1])/(2*h)-float(analytic[coordinate]))
    assert finite_error < 1e-8
    with torch.no_grad():
        metric.metric_deep.copy_(torch.linspace(-.3, .3, 1024, dtype=audit.DTYPE))
        metric.metric_local.copy_(torch.linspace(.2, -.2, 512, dtype=audit.DTYPE))
    args = lambda d: (d['anchor'], d['local'], d['slots'], d['local_slots'], d['valid'], d['energy'], d['local_energy'])
    qs, _ = metric.descriptors(*args(q))
    rs, _ = metric.descriptors(*args(r))
    qraw, _ = old.descriptors(*args(q))
    rraw, _ = old.descriptors(*args(r))
    shared_errors = []
    for branch in range(5):
        scale = (metric.metric_deep if branch == 0 else metric.metric_local).tanh().exp()
        actual = torch.einsum('bid,kjd->bkij', qs[branch], rs[branch])
        expected = torch.einsum('bid,kjd->bkij', F.normalize(qraw[branch]*scale, dim=-1),
                                F.normalize(rraw[branch]*scale, dim=-1))
        shared_errors.append(float((actual-expected).abs().max()))
    assert max(shared_errors) < 1e-12

    q['valid'][0] = 0
    r['valid'][1] = 0
    legal[:, 2] = False
    r['features'].requires_grad_()
    y, d = audit.invoke(metric, q, r, legal)
    assert not d['reference_valid'][0].any()
    assert not d['reference_valid'][:, 1:3].any()
    assert float(d['update'][0].norm()) == 0
    assert float(d['reference_weights'][0].norm()) == 0
    assert float(d['reference_weights'][:, 1:3].norm()) == 0
    gref, = torch.autograd.grad((y*torch.randn_like(y)).sum(), r['features'], retain_graph=True)
    assert float(gref[1:3].norm()) == 0
    assert float(gref[[0, 3]].norm()) > 0
    rel = audit.relation_loss(d['reference_logits'], d['reference_valid'], labels, refs)
    gd, gl = torch.autograd.grad(rel, (metric.metric_deep, metric.metric_local))
    assert torch.isfinite(rel) and torch.isfinite(gd).all() and torch.isfinite(gl).all()

    holder = nn.Module()
    holder.m3 = nn.ModuleList([nn.ModuleList([metric, copy.deepcopy(metric)])])
    holder.head = nn.Linear(4, 2)
    groups = candidate.configure(holder, 'metric')
    names = {id(p): n for n, p in holder.named_parameters()}
    assigned = [[names[id(p)] for p in group['params']] for group in groups]
    assert not any(p.requires_grad for p in holder.head.parameters())
    assert not any(p.requires_grad for op in holder.m3[0] for p in op.reader2.parameters())
    metric_count = sum(p.numel() for n, p in holder.named_parameters() if 'metric_' in n)
    assert metric_count == 3072
    assert all('metric_' not in n for n in assigned[1])
    assert sum('metric_' in n for n in assigned[0]) == 4
    assert [g['lr'] for g in groups] == [.01, .001]
    assert len([n for group in assigned for n in group]) == len(set(n for group in assigned for n in group))
    result = dict(device='cpu', dtype=str(audit.DTYPE), seed=1030, training_run=False,
                  identity_errors=identity, metric_gradients=gradients,
                  metric_finite_difference_coordinate=coordinate,
                  metric_finite_difference_error=finite_error,
                  shared_query_reference_max_error=max(shared_errors),
                  masked_reference_value_gradient_norm=float(gref[1:3].norm()),
                  live_reference_value_gradient_norm=float(gref[[0, 3]].norm()),
                  missing_query_update_norm=float(d['update'][0].norm()),
                  masked_relation_loss=float(rel), metric_parameters=metric_count,
                  optimizer_parameter_groups=assigned, optimizer_lrs=[g['lr'] for g in groups],
                  caveat='Synthetic operator test; no full-network CPU/GPU forward or checkpoint loading.')
    (HERE/'cpu_metric_results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    prior = json.loads((HERE/'cpu_results.json').read_text(encoding='utf-8'))
    prior['historical'] = audit.historical_evidence()
    (HERE/'cpu_results.json').write_text(json.dumps(prior, indent=2, allow_nan=False), encoding='utf-8')
    print('Metric identity, gradients, finite difference, sharing, masks, optimizer groups PASS.', flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != 'optimizer_parameter_groups'}), flush=True)


if __name__ == '__main__':
    run()
