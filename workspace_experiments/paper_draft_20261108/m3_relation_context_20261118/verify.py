"""Mechanism, legacy replay and real-chip forward/backward checks, not an accuracy trial."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('verify_relation_operator', ROOT/'relation_operator.py')
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)
torch = op.torch


def unit():
    torch.manual_seed(7)
    mu = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [-1., 0.]]], requires_grad=True)
    available = torch.ones(1, 4, dtype=torch.bool)
    context, weights = op.relation_context(mu, available)
    assert weights[0, 0, 1] > weights[0, 0, 2] > weights[0, 0, 3]
    assert (weights.diagonal(dim1=-2, dim2=-1) == 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 4))
    context.square().sum().backward()
    assert torch.isfinite(mu.grad).all() and mu.grad.abs().sum() > 0
    gradient_input = torch.randn(1, 4, 5, requires_grad=True)
    _, gradient_weight = op.relation_context(gradient_input, available)
    (gradient_weight*torch.arange(4.)[None, None]).sum().backward()
    assert torch.isfinite(gradient_input.grad).all() and gradient_input.grad.abs().sum() > 0
    for mask in ([1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]):
        good = torch.tensor([mask], dtype=torch.bool)
        masked_input = mu.detach().clone().requires_grad_()
        result, w = op.relation_context(masked_input, good)
        assert torch.isfinite(result).all() and torch.isfinite(w).all()
        assert (w[:, :, ~good[0]] == 0).all() and (w[:, ~good[0]] == 0).all()
        if sum(mask) <= 1:
            assert (w == 0).all() and (result == 0).all()
        (result.square().sum()+w.square().sum()).backward()
        assert torch.isfinite(masked_input.grad).all()
    equal = torch.ones(1, 4, 3)
    _, w = op.relation_context(equal, available)
    torch.testing.assert_close(w[~torch.eye(4, dtype=torch.bool)[None]], torch.full((12,), 1/3))
    permutation = torch.tensor([2, 0, 3, 1])
    a, w = op.relation_context(mu.detach()[:, permutation], available[:, permutation])
    torch.testing.assert_close(a, context.detach()[:, permutation])
    torch.testing.assert_close(w, weights.detach()[:, permutation][:, :, permutation])

    legacy = op.MaskedCostLocalMatch(classes=4).eval()
    uniform = op.RelationContextMatch(classes=4, mode='uniform').eval()
    relation = op.RelationContextMatch(classes=4).eval()
    uniform.load_state_dict(legacy.state_dict(), strict=True)
    relation.load_state_dict(legacy.state_dict(), strict=True)
    args = (torch.randn(2, 2, 20, 768), torch.randn(4, 2, 20, 768),
            torch.ones(2, 2, 20), torch.ones(4, 2, 20), torch.rand(2, 2, 20, 10, 10),
            torch.ones(2, 4, 2, dtype=torch.bool), torch.arange(4))
    args[5][0, :, 1] = False
    args[5][1, 1:, 1] = False
    with torch.no_grad():
        baseline = legacy(*args, arm='aggregate')
        replay = uniform(*args, arm='aggregate')
        changed = relation(*args, arm='aggregate')
    for a, b in zip(baseline[:3], replay[:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.isfinite(changed[0]).all() and torch.isfinite(changed[1]).all()
    assert not torch.equal(baseline[1], changed[1])
    assert list(legacy.state_dict()) == list(relation.state_dict())
    return dict(passed=True, legacy_replay_max=0., added_parameters=0,
                local_score_max_change=float((baseline[1]-changed[1]).abs().max()))


def integration():
    import model as m
    rows = json.loads((ROOT.parent/'m3_coarse_supplement_20261118/evaluation_inputs/manifest_local.json').read_text(encoding='utf-8'))
    keys = json.loads((ROOT/'verification_inputs_local.json').read_text(encoding='utf-8'))['query_ids']
    query = [next(r for r in rows[:358] if r['id'] == key) for key in keys]
    references = rows[358:374]
    data, ref_data = m.prepare(query), m.prepare(references)
    net = m.old.Model('generic')
    net.load_state_dict(torch.load(m.START, map_location='cuda', weights_only=True), strict=True)
    net.accelerate()
    with torch.no_grad():
        ref = net.reference(ref_data)
        original = net(data, query, ref, references)[0]
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    m.op.install(net.net, mode='uniform')
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(gpu_rng, torch.cuda.get_rng_state_all()))
    net.accelerate()
    net.net.matcher.record_context = True
    with torch.no_grad():
        ref = net.reference(ref_data)
        before = net(data, query, ref, references)[0]
        torch.testing.assert_close(original, before, rtol=0, atol=0)
        net.net.matcher.mode = 'relation'
        after = net(data, query, ref, references)[0]
    change = float((after-before).abs().max())
    assert torch.isfinite(after).all() and change > 0
    traces = net.net.matcher.context_traces
    assert len(traces) == 6
    trace_output = [dict(weights=t['weights'].cpu().tolist(), available=t['available'].cpu().tolist()) for t in traces]
    net.net.matcher.record_context = False
    net.train()
    net.zero_grad(set_to_none=True)
    ref = net.reference(ref_data)
    score = net(data, query, ref, references)[0]
    score.square().mean().backward()
    groups = {'backbone': 'net.base.backbone.', 'matcher': 'net.matcher.', 'ccrel': 'ccrel.', 'pcea': 'exchange.'}
    norms = {key: sum(float(p.grad.square().sum()) for name, p in net.named_parameters()
                      if name.startswith(prefix) and p.grad is not None)**.5 for key, prefix in groups.items()}
    assert all(value > 0 and value < float('inf') for value in norms.values()), norms
    report = dict(passed=True, queries=2, references=16, context_calls=6,
                  checkpoint_install_replay_max=float((original-before).abs().max()), rng_preserved=True,
                  max_logit_change=change, gradient_norms=norms, optimizer_steps=0,
                  trained_checkpoint_saved=False, accuracy_claim=False,
                  peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
    local = dict(query_ids=[r['id'] for r in query], labels=[r['label'] for r in query],
                 uniform=before.cpu().tolist(), relation=after.cpu().tolist(), traces=trace_output)
    (ROOT/'integration_local.json').write_text(json.dumps(local, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--integration', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = {'unit': unit()}
    if args.integration:
        result['integration'] = integration()
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
