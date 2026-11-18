"""Bounded operator and real chip-to-class integration checks; no fitting."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mixed_scale_model', ROOT/'model.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
torch = m.torch


def maxdiff(a, b):
    return float((a-b).detach().abs().max())


def case_ids():
    notes = json.loads((ROOT.parent/'m3_native_readout_20261118/case_notes_local.json').read_text(encoding='utf-8'))
    return [case['id'] for case in notes['cases'][:2]]


def args_for(q, r, legal, labels):
    return (q['desc'], r['desc'], q['mass'], r['mass'], q['write'], legal, labels)


def mechanism():
    torch.manual_seed(2307)
    dtype = torch.float64
    selector = m.selection.ContentScaleSelector().double().eval()
    matcher = m.MixedMatch(3).double().eval()
    for module in matcher.output:
        torch.nn.init.normal_(module.weight, std=.005)
    q, r = torch.randn(2, 2, 20, 768, dtype=dtype), torch.randn(3, 2, 20, 768, dtype=dtype)
    qm, rm = torch.ones(2, 2, 20, dtype=dtype), torch.ones(3, 2, 20, dtype=dtype)
    qw, rw = torch.rand(2, 2, 20, 10, 10, dtype=dtype), torch.rand(3, 2, 20, 10, 10, dtype=dtype)
    qw = qw/qw.sum(2, keepdim=True)
    rw = rw/rw.sum(2, keepdim=True)
    qe, re = torch.ones(2, 2, 8, dtype=torch.bool), torch.ones(3, 2, 8, dtype=torch.bool)
    legal, labels = torch.ones(2, 3, 2, dtype=torch.bool), torch.arange(3)
    selected = {}
    with torch.no_grad():
        for mode in ('coarse', 'fine', 'hard', 'soft'):
            a, b = selector(q, qm, qw, qe, mode), selector(r, rm, rw, re, mode)
            torch.testing.assert_close(a['mass'].sum(-1), qm.sum(-1))
            torch.testing.assert_close(a['write'].sum(2), qw.sum(2))
            if mode in ('hard', 'coarse', 'fine'):
                for g, children in enumerate(m.selection.GROUPS):
                    child = a['mass'][:, :, [m.selection.FINE_INDEX[i] for i in children]].sum(-1)
                    parent = a['mass'][:, :, m.selection.COARSE_INDEX[g]]
                    assert not bool(((child > 0) & (parent > 0)).any())
            selected[mode] = (a, b)
        assert bool((selected['coarse'][0]['scale_trace']['active_tokens'] == 8).all())
        assert bool((selected['fine'][0]['scale_trace']['active_tokens'] == 20).all())
        original = m.axis.AxisCoupledMatch(classes=3).double().eval()
        original.mode = 'uniform'
        original.load_state_dict(matcher.state_dict(), strict=True)
        plain_fine = original(q, r, qm, rm, qw, legal, labels)
        mixed_fine = matcher(*args_for(*selected['fine'], legal, labels), details=True)
        torch.testing.assert_close(plain_fine[0], mixed_fine[0], atol=1e-8, rtol=1e-7)
        torch.testing.assert_close(plain_fine[1], mixed_fine[1], atol=1e-8, rtol=1e-7)
        # Alter child layout/content while holding every parent mean exactly fixed.
        changed = r.clone()
        delta = torch.randn_like(changed[:, :, 0])*.2
        changed[:, :, 0] += delta
        changed[:, :, 1] -= delta
        outputs = {}
        for mode in ('coarse', 'fine'):
            a, b = selected[mode]
            new_b = selector(changed, rm, rw, re, mode)
            outputs[mode] = maxdiff(matcher(*args_for(a, b, legal, labels))[0],
                                    matcher(*args_for(a, new_b, legal, labels))[0])
        assert outputs['coarse'] < 1e-9 and outputs['fine'] > 1e-6, outputs
        a, b = selected['soft']
        coupled = matcher(*args_for(a, b, legal, labels), details=True)
        reverse = torch.cdist(matcher.positions, -matcher.positions).argmin(-1)
        torch.testing.assert_close(matcher.positions[reverse], -matcher.positions)
        reversed_r = dict(b, desc=b['desc'][:, :, reverse], mass=b['mass'][:, :, reverse])
        reversed_q = dict(a, desc=a['desc'][:, :, reverse], mass=a['mass'][:, :, reverse],
                          write=a['write'][:, :, reverse])
        ref_flipped = matcher(*args_for(a, reversed_r, legal, labels))
        query_flipped = matcher(*args_for(reversed_q, b, legal, labels))
        torch.testing.assert_close(coupled[0], ref_flipped[0], atol=1e-8, rtol=1e-7)
        torch.testing.assert_close(coupled[0], query_flipped[0], atol=1e-8, rtol=1e-7)
        matcher.coupling = 'aggregate'
        collapsed = matcher(*args_for(a, b, legal, labels), details=True)
        for p in range(2):
            for pair, agg in zip(coupled[3][p]['regions'], collapsed[3][p]['regions']):
                for dim in (-1, -2):
                    torch.testing.assert_close(pair['transport'].sum(dim), agg['transport'].sum(dim))
        coupling_effect = maxdiff(coupled[0], collapsed[0])
        assert coupling_effect > 1e-6
        matcher.coupling = 'paired'
        probabilities = selected['soft'][0]['scale_trace']['probability']
        assert maxdiff(probabilities[0], probabilities[1]) > 1e-5
        # Missing polarization and unsplittable regions must remain finite/empty.
        missing = qm.clone()
        missing[:, 1] = 0
        invalid = selector(q, missing, qw*torch.tensor([1., 0.])[None, :, None, None, None], qe, 'hard')
        assert invalid['mass'][:, 1].abs().sum() == 0
        assert invalid['write'][:, 1].abs().sum() == 0
        no_split = selector(q, qm, qw, torch.zeros_like(qe), 'fine')
        torch.testing.assert_close(no_split['mass'], selected['coarse'][0]['mass'])
        none = matcher(*args_for(invalid, selected['hard'][1], torch.zeros_like(legal), labels))
        assert none[0].abs().sum() == 0 and not none[2].any()
    selector.train()
    a, b = selector(q, qm, qw, qe), selector(r, rm, rw, re)
    out = matcher(*args_for(a, b, legal, labels))
    # Classification signal only, without a budget penalty.
    loss = torch.nn.functional.cross_entropy(out[1], torch.tensor([0, 1]))
    loss.backward()
    norms = [sum(float(p.grad.square().sum()) for p in gate.parameters() if p.grad is not None)**.5
             for gate in selector.gates]
    assert all(0 < v < float('inf') for v in norms), norms
    assert all(torch.isfinite(p.grad).all() for p in selector.parameters() if p.grad is not None)
    return dict(passed=True, fine_replay_error=maxdiff(plain_fine[0], mixed_fine[0]),
                parent_preserving_child_change=outputs, coupling_write_effect=coupling_effect,
                content_probability_change=maxdiff(probabilities[0], probabilities[1]),
                reference_axis_flip_error=maxdiff(coupled[0], ref_flipped[0]),
                query_axis_flip_error=maxdiff(coupled[0], query_flipped[0]),
                classification_only_gate_gradients=norms,
                coarse_tokens=8, fine_tokens=20, padded_candidates=28)


def geometry_check():
    np = m.np
    masks = m.native.raw.tokens.slots((84, 84, 12, 3, 0), np.ones((160, 160)))
    reads = [m.native.raw.tokens.cell_mass(masks, 160//size)/m.native.geometry._cell_area(size)
             for size in (40, 20)]
    shallow_only = m.geo.can_split(reads[:1], m.selection.GROUPS)
    actual_inputs = m.geo.can_split(reads, m.selection.GROUPS)
    assert not shallow_only[6] and not shallow_only[7]
    assert actual_inputs[6] and actual_inputs[7]
    uniform = [np.ones((20, size, size)) for size in (40, 20)]
    assert not m.geo.can_split(uniform, m.selection.GROUPS).any()
    assert not m.geo.can_split([x*0 for x in uniform], m.selection.GROUPS).any()
    rows = json.loads((ROOT.parent/'m3_coarse_supplement_20261118/evaluation_inputs/manifest_local.json').read_text(encoding='utf-8'))
    row = next(r for r in rows if r['id'] == case_ids()[1])
    views = m.native.old.old.t.e.task.data.prepare(row)
    base = m.prepare([row])
    error = 0.
    for code in range(8):
        data = m.prepare([row], code)
        for key in ('images', 'valid'):
            expected = torch.rot90(base[key], code % 4, (-2, -1))
            if code >= 4:
                expected = expected.flip(-1)
            torch.testing.assert_close(data[key], expected)
        for p, view in enumerate(views):
            if view is None or not view['valid_target']:
                assert not data['eligible'][0, p].any()
                continue
            native = m.native.raw.tokens.slots(view['geometry'], view['footprint'])
            native = torch.rot90(torch.tensor(native).float()[None], code % 4, (-2, -1))
            if code >= 4:
                native = native.flip(-1)
            for key, size in (('slots', 10), ('local_slots', 20)):
                expected = m.native.raw.grid_occupancy(native, (size, size))[0].to(data[key])
                error = max(error, maxdiff(data[key][0, p], expected))
    assert error < 1e-6
    return dict(passed=True, non_nested_grid_regression=True, empty_geometry_valid=True,
                native_transforms=8, m2_coverage_max_error=error)


def integration():
    rows = json.loads((ROOT.parent/'m3_coarse_supplement_20261118/evaluation_inputs/manifest_local.json').read_text(encoding='utf-8'))
    queries = [next(r for r in rows[:358] if r['id'] == key) for key in case_ids()]
    refs = rows[358:374]
    data, rd = m.prepare(queries), m.prepare(refs)
    net = m.Model()
    net.accelerate()
    result, writes, traces, off = {}, {}, {}, {}
    packet = {}
    handle = net.net.matcher.register_forward_hook(lambda mod, args, out: packet.update(write=out[0].detach()))
    with torch.no_grad():
        for mode in ('coarse', 'fine', 'hard', 'soft'):
            net.selection_mode = mode
            reference = net.reference(rd)
            score = net(data, queries, reference, refs)[0]
            result[mode] = score.detach().cpu().tolist()
            writes[mode] = packet['write']
            traces[mode] = {k: v.cpu().tolist() for k, v in net.scale_trace.items()}
            off[mode] = net(data, queries, reference, refs, m3_on=False)[0]
        for mode in off:
            torch.testing.assert_close(off['fine'], off[mode], atol=1e-6, rtol=0)
        net.selection_mode = 'hard'
        reference = net.reference(rd)
        before = net(data, queries, reference, refs)[0]
        gate = net.scale_trace['choice']
        net.scale_override = 1-gate
        after = net(data, queries, reference, refs)[0]
        override_effect = maxdiff(before, after)
        assert override_effect > 1e-7
        net.scale_override = None
    handle.remove()
    net.selection_mode = 'auto'
    net.train()
    net.zero_grad(set_to_none=True)
    loss, score, terms = net.objective(data, queries, rd, refs, budget_weight=0.)
    loss.backward()
    groups = net.optimizer_groups()
    norms = {g['name']: sum(float(p.grad.square().sum()) for p in g['params'] if p.grad is not None)**.5
             for g in groups}
    assert all(0 < v < float('inf') for v in norms.values()), norms
    gate_norms = [sum(float(p.grad.square().sum()) for p in gate.parameters() if p.grad is not None)**.5
                  for gate in net.selector.gates]
    assert all(0 < v < float('inf') for v in gate_norms), gate_norms
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    trainable = [id(p) for g in groups for p in g['params']]
    assert len(trainable) == len(set(trainable)) == sum(p.requires_grad for p in net.parameters())
    assert net.selector.training and not net.net.base.backbone.model.bn1.training
    local = dict(ids=[r['id'] for r in queries], labels=[r['label'] for r in queries],
                 scores=result, query_gate_traces=traces, reference_ids=[r['id'] for r in refs])
    (ROOT/'integration_local.json').write_text(json.dumps(local, indent=2), encoding='utf-8')
    return dict(passed=True, queries=2, references=len(refs), optimizer_steps=0,
                untrained_selector=True, formal_model_changed=False,
                m3_off_max_difference=max(maxdiff(off['fine'], x) for x in off.values()),
                gate_override_logit_effect=override_effect,
                fine_coarse_write_difference=maxdiff(writes['fine'], writes['coarse']),
                gradient_norms=norms, classification_only_gate_gradients=gate_norms,
                selector_parameters=sum(p.numel() for p in net.selector.parameters()),
                loss=float(loss.detach()), terms={k: float(v) for k, v in terms.items()},
                peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--integration', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = (json.loads((ROOT/'verification.json').read_text(encoding='utf-8'))
              if (ROOT/'verification.json').exists() else {})
    report['mechanism'] = mechanism()
    report['geometry'] = geometry_check()
    print(json.dumps(report, indent=2), flush=True)
    if args.integration:
        report['integration'] = integration()
    (ROOT/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
