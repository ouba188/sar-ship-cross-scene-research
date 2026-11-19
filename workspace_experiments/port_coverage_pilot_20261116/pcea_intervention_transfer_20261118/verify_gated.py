"""Real-source checks of the new gate, live references, gradient and reload paths."""
import copy
import numpy as np
import torch
import gated_run as run


def difference(a, b):
    return float((a-b).abs().max())


def main():
    run.OUT.mkdir(exist_ok=True)
    inputs, fit, _ = run.base.runtime('unified')
    bank = torch.load(run.ROOT/'unified_20261119/verification_bank_local.pt', map_location='cpu', weights_only=True)
    ids = bank['ids'][[0,1,12,13]].numpy()
    assert np.isin(ids, fit).all() and not np.isin(ids, inputs.groups['references']).any()
    torch.manual_seed(17)
    net = run.make_model().eval()
    data = inputs.data(ids, 'small')
    rd = inputs.data(inputs.groups['references'], 'small')
    rows = [inputs.rows[i] for i in ids]
    state = copy.deepcopy(net.disentangler.state_dict())
    net.capture_split = True
    with torch.no_grad():
        ref = net.reference(rd)
        clean, _ = net(data, rows, ref, inputs.reference_rows)
        trace = net.split_trace
        torch.testing.assert_close(trace['content']+trace['remainder'], trace['raw'])
        assert ((trace['gate']>0) & (trace['gate']<1)).all()
        gate_sample_delta = difference(trace['gate'][0], trace['gate'][1])
        assert gate_sample_delta > 0
        donor, eligible = run.donor_permutation(data['available'], torch.Generator().manual_seed(19))
        swap, _, _ = run.forward(net, inputs, ids, 0, 'swap', donor)
        swap_effect = difference(swap, clean)
        assert swap_effect > 1e-6
        # Gate=1 must replay the ungated common-cut route including both reference caches.
        for gate in net.disentangler.gates:
            gate[-1].weight.zero_()
            gate[-1].bias.fill_(80.)
        ref = net.reference(rd)
        all_content, _ = net(data, rows, ref, inputs.reference_rows)
        raw, _ = net.forward_variant(data, rows, ref, inputs.reference_rows, 'raw')
        full_gate_replay = difference(all_content, raw)
        assert full_gate_replay < 1e-5
        # Gate=0 blocks upstream image changes on every shallow/deep readout path.
        for gate in net.disentangler.gates:
            gate[-1].bias.fill_(-80.)
        ref = net.reference(rd)
        desc, pcea = ref['desc'].clone(), net.exchange.source_raw.clone()
        zero, _ = net(data, rows, ref, inputs.reference_rows)
        handle = net.net.base.backbone.model.layer1.register_forward_hook(
            lambda module, args, output: output+1., prepend=True)
        try:
            ref2 = net.reference(rd)
            z2, _ = net(data, rows, ref2, inputs.reference_rows)
            bypass = max(difference(zero, z2), difference(desc, ref2['desc']),
                         difference(pcea, net.exchange.source_raw))
        finally:
            handle.remove()
        assert bypass < 1e-5
        net.disentangler.load_state_dict(state)
    net.capture_split = False
    groups = net.optimizer_groups()
    assigned = [id(p) for g in groups for p in g['params']]
    assert len(assigned)==len(set(assigned))
    assert set(assigned)=={id(p) for p in net.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    steps = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, stats = run.backward(net, inputs, ids, step*3, torch.Generator().manual_seed(19+step))
        norms = {g['name']: sum(float(p.grad.square().sum()) for p in g['params'] if p.grad is not None)**.5
                 for g in groups}
        assert all(np.isfinite(v) and v>0 for v in norms.values()), norms
        for part in (net.disentangler.gates, net.disentangler.fusions):
            assert sum(float(p.grad.square().sum()) for p in part.parameters() if p.grad is not None)>0
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 5., error_if_nonfinite=True)
        optimizer.step()
        steps.append(dict(**stats, gradient_norms=norms))
        print('VERIFY_STEP', step+1, stats, flush=True)
    net.eval()
    with torch.no_grad():
        z, _, _ = run.forward(net, inputs, ids, 0)
    checkpoint = run.OUT/'verification_checkpoint_local.pt'
    torch.save(net.state_dict(), checkpoint)
    net.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True), strict=True)
    with torch.no_grad():
        restored, _, _ = run.forward(net, inputs, ids, 0)
    error = difference(z, restored)
    assert error < 1e-6
    result = dict(passed=True, source_samples=4, references=16, updates=2,
        not_performance_evidence=True, gate_sample_difference=gate_sample_delta,
        swap_logit_effect=swap_effect, full_gate_replay=full_gate_replay,
        zero_gate_bypass_effect=bypass, checkpoint_error=error, steps=steps,
        peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3)
    run.write(run.OUT/'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
