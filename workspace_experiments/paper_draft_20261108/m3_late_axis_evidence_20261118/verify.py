"""Mechanism and two-chip integration checks, with no optimizer steps."""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
import model as m
torch = m.torch
ROOT = Path(__file__).resolve().parent


def diff(a, b):
    return float((a-b).detach().abs().max())


def mechanism():
    torch.manual_seed(63)
    dt = torch.float64
    # Opposite reference values average to the query, but neither matches it.
    q0 = torch.zeros(1, 1, 1, dtype=dt)
    r0 = torch.tensor([[[-1.], [1.]]], dtype=dt)
    plans = torch.tensor([[[[[.5, 0.]], [[0., .5]]]]], dtype=dt)
    late, early, _ = m.compare_frames(q0, r0, plans, torch.full((1, 1, 2), .5, dtype=dt),
                                     torch.ones(1, 1, 1, dtype=dt))
    torch.testing.assert_close(late, torch.ones_like(late))
    torch.testing.assert_close(early, torch.zeros_like(early))
    # Hand-computed unequal retained mass: alpha=(.1,.9), values=(-1,3).
    unequal = torch.tensor([[[[[.2, 0.]], [[0., .6]]]]], dtype=dt)
    ud, ue, ut = m.compare_frames(q0, torch.tensor([[[-1.], [3.]]], dtype=dt), unequal,
        torch.tensor([[[.25, .75]]], dtype=dt), torch.ones(1, 1, 1, dtype=dt))
    torch.testing.assert_close(ut['hypothesis_weight'].flatten(), torch.tensor([.1, .9], dtype=dt))
    torch.testing.assert_close(ud, torch.full_like(ud, 8.2))
    torch.testing.assert_close(ue, torch.full_like(ue, 6.76))
    selector = m.base.selection.ContentScaleSelector().double().eval()
    net = m.LateAxisMatch(3).double().eval()
    for out in net.output:
        torch.nn.init.normal_(out.weight, std=.005)
    old = m.base.MixedMatch(3).double().eval()
    old.load_state_dict(net.state_dict(), strict=True)
    q, r = torch.randn(2, 2, 20, 768, dtype=dt), torch.randn(4, 2, 20, 768, dtype=dt)
    qm, rm = torch.rand(2, 2, 20, dtype=dt)+.1, torch.rand(4, 2, 20, dtype=dt)+.1
    qm[1, 1] = 0
    rm[2, 0, :8] = 0
    qw, rw = torch.rand(2, 2, 20, 10, 10, dtype=dt), torch.rand(4, 2, 20, 10, 10, dtype=dt)
    qw = qw/qw.sum(2, keepdim=True)*(qm > 0)[..., None, None]
    rw = rw/rw.sum(2, keepdim=True)*(rm > 0)[..., None, None]
    qe, re = torch.ones(2, 2, 8, dtype=torch.bool), torch.ones(4, 2, 8, dtype=torch.bool)
    a, b = selector(q, qm, qw, qe, 'soft'), selector(r, rm, rw, re, 'soft')
    legal, labels = torch.ones(2, 4, 2, dtype=torch.bool), torch.tensor([0, 0, 1, 2])
    def run(qa=a, rb=b, allowed=legal):
        return net(qa['desc'], rb['desc'], qa['mass'], rb['mass'], qa['write'], allowed, labels, details=True)
    with torch.no_grad():
        net.comparison = 'prototype'
        replay = run()
        before = old(a['desc'], b['desc'], a['mass'], b['mass'], a['write'], legal, labels)
        torch.testing.assert_close(replay[0], before[0], atol=0, rtol=0)
        net.comparison = 'pair_mean'
        mid = run()
        net.comparison = 'late'
        out = run()
        effect = diff(out[0], mid[0])
        assert effect > 1e-7
        for pol in range(2):
            for trace in out[3][pol]['regions']:
                assert float(trace['axis_variance'].min()) > -1e-10
        rev = torch.cdist(net.positions, -net.positions).argmin(-1)
        torch.testing.assert_close(net.positions[rev], -net.positions)
        # Reverse one reference only: global-hypothesis class prototypes fail this.
        rb = {k: v.clone() for k, v in b.items() if isinstance(v, torch.Tensor)}
        rb['desc'][0] = rb['desc'][0, :, rev]
        rb['mass'][0] = rb['mass'][0, :, rev]
        qa = {k: v.clone() for k, v in a.items() if isinstance(v, torch.Tensor)}
        for key in ('desc', 'mass', 'write'):
            qa[key][0] = qa[key][0, :, rev]
        errors = {}
        for weighting in ('content', 'equal'):
            net.weighting = weighting
            expected = run()
            rflip, qflip = run(rb=rb), run(qa=qa)
            for actual in (rflip, qflip):
                torch.testing.assert_close(actual[0], expected[0], atol=1e-9, rtol=1e-7)
                torch.testing.assert_close(actual[1], expected[1], atol=1e-9, rtol=1e-7)
            errors[weighting] = dict(reference=diff(rflip[0], expected[0]), query=diff(qflip[0], expected[0]))
        empty = run(allowed=torch.zeros_like(legal))
        assert not empty[2].any() and empty[0].abs().sum() == 0
        assert out[0][1, 1].abs().sum() == 0
    net.weighting = 'content'
    net.zero_grad(set_to_none=True)
    out = run()
    torch.nn.functional.cross_entropy(out[1], torch.tensor([0, 1])).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    grads = {name: float(p.grad.norm()) for name, p in net.named_parameters()
             if name in ('position_strength_raw', 'key.0.weight', 'value.0.weight')}
    assert all(x > 0 for x in grads.values())
    return dict(passed=True, cancellation_example=dict(early=early.item(), late=late.item()),
                unequal_mass_example=dict(late=ud.item(), early=ue.item(), expected_weights=[.1, .9]),
                legacy_replay=diff(replay[0], before[0]), late_vs_pair_mean_write=effect,
                independent_axis_flip_errors=errors, gradients=grads, added_parameters=0)


def integration():
    notes = json.loads((ROOT.parent/'m3_native_readout_20261118/case_notes_local.json').read_text(encoding='utf-8'))
    ids = [case['id'] for case in notes['cases'][:2]]
    rows = json.loads((ROOT.parent/'m3_coarse_supplement_20261118/evaluation_inputs/manifest_local.json').read_text(encoding='utf-8'))
    queries = [next(r for r in rows[:358] if r['id'] == key) for key in ids]
    refs = rows[358:374]
    data, rd = m.prepare(queries), m.prepare(refs)
    net = m.Model(selection_mode='fine')
    net.accelerate()
    scores, disabled = {}, {}
    with torch.no_grad():
        reference = net.reference(rd)
        for name, comparison, weighting in (
            ('legacy', 'prototype', 'content'), ('pair_mean', 'pair_mean', 'content'),
            ('late', 'late', 'content'), ('late_equal', 'late', 'equal')):
            net.net.matcher.comparison, net.net.matcher.weighting = comparison, weighting
            scores[name] = net(data, queries, reference, refs)[0].detach()
            disabled[name] = net(data, queries, reference, refs, m3_on=False)[0].detach()
        for z in disabled.values():
            torch.testing.assert_close(z, disabled['legacy'], atol=1e-6, rtol=0)
    effect = diff(scores['late'], scores['pair_mean'])
    assert effect > 1e-7
    net.net.matcher.comparison, net.net.matcher.weighting = 'late', 'content'
    net.selection_mode = 'auto'
    net.train()
    net.zero_grad(set_to_none=True)
    loss, _, _ = net.objective(data, queries, rd, refs, budget_weight=0.)
    loss.backward()
    gradients = {g['name']: sum(float(p.grad.square().sum()) for p in g['params'] if p.grad is not None)**.5
                 for g in net.optimizer_groups()}
    assert all(x > 0 for x in gradients.values())
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    local = dict(ids=ids, labels=[r['label'] for r in queries], reference_ids=[r['id'] for r in refs],
                 scores={k: v.cpu().tolist() for k, v in scores.items()},
                 m3_off=disabled['legacy'].cpu().tolist(), selection='fine')
    (ROOT/'integration_local.json').write_text(json.dumps(local, indent=2), encoding='utf-8')
    return dict(passed=True, queries=2, references=16, optimizer_steps=0, fixed_fine_inference=True,
                late_vs_pair_mean_logit=effect, late_vs_legacy_logit=diff(scores['late'], scores['legacy']),
                off_difference=max(diff(z, disabled['legacy']) for z in disabled.values()),
                soft_selection_training_gradients=gradients,
                peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--integration', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = (json.loads((ROOT/'verification.json').read_text(encoding='utf-8'))
              if (ROOT/'verification.json').exists() else {})
    report['mechanism'] = mechanism()
    if args.integration:
        report['integration'] = integration()
    (ROOT/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
