"""Test preserved correspondence, axis ambiguity and real full-model propagation."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
import axis_operator as op
torch = op.torch
ROOT = Path(__file__).resolve().parent


def maximum(a, b):
    return float((a-b).abs().max())


def mechanism():
    torch.manual_seed(41)
    net = op.AxisCoupledMatch(classes=3).double().eval()
    for layer in net.output:
        torch.nn.init.normal_(layer.weight, std=.005)
    args = [torch.randn(2, 2, 20, 768, dtype=torch.float64),
            torch.randn(3, 2, 20, 768, dtype=torch.float64),
            torch.ones(2, 2, 20, dtype=torch.float64), torch.ones(3, 2, 20, dtype=torch.float64),
            torch.rand(2, 2, 20, 10, 10, dtype=torch.float64),
            torch.ones(2, 3, 2, dtype=torch.bool), torch.arange(3)]
    # Real supported edge cases: absent polarization, partial or empty regions.
    args[2][0, 1] = 0
    args[3][1, 0, :3] = 0
    args[5][1, 2] = False
    old = op.RelationContextMatch(classes=3).double().eval()
    old.load_state_dict({k: v for k, v in net.state_dict().items() if k != 'position_strength_raw'})
    with torch.no_grad():
        net.geometry, net.coupling = 'none', 'aggregate'
        base = old(*args, arm='aggregate', details=True)
        replay = net(*args, details=True)
        for a, b in zip(base[:2], replay[:2]):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-9)
        net.geometry, net.coupling = 'axis', 'paired'
        main = net(*args, details=True)
        flipped_ref = [x.clone() for x in args]
        flipped_ref[1][1] = args[1][1, :, op.REVERSE]
        flipped_ref[3][1] = args[3][1, :, op.REVERSE]
        ref_result = net(*flipped_ref, details=True)
        flipped_query = [x.clone() for x in args]
        for index in (0, 2, 4):
            flipped_query[index][0] = args[index][0, :, op.REVERSE]
        query_result = net(*flipped_query)
        for result in (ref_result, query_result):
            for a, b in zip(main[:2], result[:2]):
                torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-7)
        # Change relative arrangement, not a valid global axis reversal; preserve content/mass sets.
        scrambled = [x.clone() for x in args]
        permutation = torch.tensor([2, 3, 0, 1, 4, 5, 6, 7]+list(range(8, 20)))
        scrambled[1] = args[1][:, :, permutation]
        scrambled[3] = args[3][:, :, permutation]
        rearranged = net(*scrambled)
        assert maximum(main[0], rearranged[0]) > 1e-5
        net.geometry = 'none'
        no_geometry = net(*args)
        scrambled_no_geometry = net(*scrambled)
        torch.testing.assert_close(no_geometry[0], scrambled_no_geometry[0], rtol=1e-8, atol=1e-7)
        assert maximum(main[0], no_geometry[0]) > 1e-5
        net.geometry, net.coupling = 'axis', 'aggregate'
        aggregate = net(*args, details=True)
        for p in range(2):
            for coupled, collapsed in zip(main[3][p]['regions'], aggregate[3][p]['regions']):
                for axis in (-1, -2):
                    torch.testing.assert_close(coupled['transport'].sum(axis), collapsed['transport'].sum(axis))
        assert maximum(main[0], aggregate[0]) > 1e-5
    net.coupling = 'paired'
    output = net(*args)
    (output[0].square().mean()+output[1][output[2]].square().mean()).backward()
    assert net.position_strength_raw.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    gradient = net.position_strength_raw.grad.tolist()
    net.zero_grad(set_to_none=True)
    missing = [x.clone() for x in args]
    missing[5].zero_()
    out = net(*missing)
    assert out[0].abs().sum() == 0 and not out[2].any()
    out[0].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)

    # Same marginals, different coupling: exact counterexample to row/column-only transfer.
    diagonal = torch.eye(4, dtype=torch.float64)/4
    reverse = diagonal.flip(-1)
    value = torch.tensor([[1., 0.], [0., 2.], [3., 0.], [0., 4.]], dtype=torch.float64)
    paired_change = maximum(diagonal@value, reverse@value)
    def collapse(t):
        return t.sum(-1)[:, None]*t.sum(-2)[None]/t.sum()
    assert paired_change > 0 and maximum(collapse(diagonal)@value, collapse(reverse)@value) == 0
    return dict(passed=True, legacy_replay_max=maximum(base[0], replay[0]),
                reference_axis_flip_max=maximum(main[0], ref_result[0]),
                query_axis_flip_max=maximum(main[0], query_result[0]),
                non_axis_rearrangement_write_change=maximum(main[0], rearranged[0]),
                no_geometry_rearrangement_change=maximum(no_geometry[0], scrambled_no_geometry[0]),
                geometry_write_change=maximum(main[0], no_geometry[0]),
                same_marginals_coupling_write_change=maximum(main[0], aggregate[0]),
                position_gradient=gradient, added_parameters=2,
                toy_same_marginals_paired_change=paired_change, toy_aggregate_change=0.)


def integration():
    spec = importlib.util.spec_from_file_location('axis_integration_model', ROOT/'model.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    rows = json.loads((ROOT.parent/'m3_coarse_supplement_20261118/evaluation_inputs/manifest_local.json').read_text(encoding='utf-8'))
    keys = json.loads((ROOT.parent/'m3_relation_context_20261118/verification_inputs_local.json').read_text(encoding='utf-8'))['query_ids']
    queries = [next(r for r in rows[:358] if r['id'] == key) for key in keys]
    references = rows[358:374]
    data, ref_data = m.prepare(queries), m.prepare(references)
    net = m.Model()
    assert isinstance(net.net.matcher, op.AxisCoupledMatch)
    net.accelerate()
    packet = {}
    def capture(module, args, kwargs):
        packet['args'] = tuple(x.detach() if torch.is_tensor(x) else x for x in args)
        packet['kwargs'] = kwargs
    handle = net.net.matcher.register_forward_pre_hook(capture, with_kwargs=True)
    scores, writes = {}, {}
    def write_capture(module, args, output):
        packet['write'] = output[0].detach()
    hook = net.net.matcher.register_forward_hook(write_capture)
    with torch.no_grad():
        ref = net.reference(ref_data)
        for name, geometry, coupling in [('legacy', 'none', 'aggregate'), ('content_pair', 'none', 'paired'),
                                         ('axis_aggregate', 'axis', 'aggregate'), ('axis_pair', 'axis', 'paired')]:
            net.net.matcher.geometry, net.net.matcher.coupling = geometry, coupling
            scores[name] = net(data, queries, ref, references)[0].detach()
            writes[name] = packet['write']
    handle.remove()
    hook.remove()
    kw = dict(packet['kwargs'], details=True)
    with torch.no_grad():
        traced = net.net.matcher(*packet['args'], **kw)
    trace_report = []
    for p, branch in enumerate(traced[3]):
        for r, item in enumerate(branch['regions']):
            t = item['transport']
            row, col = t.sum(-1), t.sum(-2)
            product = row[..., None]*col[..., None, :]/row.sum(-1)[..., None, None].clamp_min(1e-20)
            trace_report.append(dict(polarization=p, region=r, coupling_residual=float((t-product).abs().sum()),
                                     orientation=item['orientation'].cpu().tolist(),
                                     message_abs_mean=float(item['message'].abs().mean())))
    assert maximum(writes['axis_pair'], writes['axis_aggregate']) > 0
    assert maximum(scores['axis_pair'], scores['axis_aggregate']) > 0
    assert maximum(scores['axis_pair'], scores['content_pair']) > 0
    net.train()
    net.zero_grad(set_to_none=True)
    ref = net.reference(ref_data)
    z = net(data, queries, ref, references)[0]
    torch.nn.functional.cross_entropy(z, torch.tensor([r['label'] for r in queries], device=z.device)).backward()
    groups = {'backbone': 'net.base.backbone.', 'matcher': 'net.matcher.', 'ccrel': 'ccrel.', 'pcea': 'exchange.'}
    norms = {key: sum(float(p.grad.square().sum()) for name, p in net.named_parameters()
                      if name.startswith(prefix) and p.grad is not None)**.5 for key, prefix in groups.items()}
    assert all(0 < n < float('inf') for n in norms.values())
    grad = net.net.matcher.position_strength_raw.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    for p in net.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    local = dict(query_ids=keys, labels=[r['label'] for r in queries],
                 scores={k: v.cpu().tolist() for k, v in scores.items()}, traces=trace_report)
    (ROOT/'integration_local.json').write_text(json.dumps(local, indent=2), encoding='utf-8')
    return dict(passed=True, queries=len(queries), references=len(references), optimizer_steps=0,
                formal_model_changed=False, accuracy_trial=False, gradient_norms=norms,
                position_gradient=grad.detach().cpu().tolist(),
                paired_vs_aggregate_write_max=maximum(writes['axis_pair'], writes['axis_aggregate']),
                paired_vs_aggregate_logit_max=maximum(scores['axis_pair'], scores['axis_aggregate']),
                geometry_logit_max=maximum(scores['axis_pair'], scores['content_pair']),
                max_vs_legacy=maximum(scores['axis_pair'], scores['legacy']),
                prediction_changes_vs_legacy=int((scores['axis_pair'].argmax(-1) != scores['legacy'].argmax(-1)).sum()),
                peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--integration', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = {'mechanism': mechanism()}
    if args.integration:
        result['integration'] = integration()
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
