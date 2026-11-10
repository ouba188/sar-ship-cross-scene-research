"""Measure and validate graph replay using trained M3 and native-chip inputs."""
from pathlib import Path
import gc
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'computational_cost_20261110'))
import benchmark as b
from graph_matcher import GraphMatcher

torch, task = b.torch, b.task


def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2), encoding='utf-8')


def subset(data, ids):
    return {k: v[ids] for k, v in data.items()}


def matcher_inputs(a, rows, reference, refs):
    legal = task.legal(rows + refs, np.arange(len(rows)), np.arange(len(rows), len(rows) + len(refs)))
    legal = legal[..., None] & a['available'][:, None].bool() & reference['available'][None].bool()
    labels = torch.tensor([r['label'] for r in refs], device=b.DEVICE)
    return a['desc'], reference['desc'], a['mass'], reference['mass'], a['write'], legal, labels


@torch.inference_mode()
def main():
    ROOT.mkdir(exist_ok=True)
    task.current.setup(b.DEVICE)
    task.tensor = b.resident_tensor
    torch.set_num_threads(2)
    rows, groups = b.e.prev.records()
    refs = [rows[i] for i in groups['references']]
    saved_inputs = json.loads((b.ROOT / 'inputs_4class.json').read_text())
    byid = {r['id']: r for r in rows}
    query_rows = [byid[i] for i in saved_inputs['query_ids']]
    qdata, rdata = b.prepare(query_rows), b.prepare(refs)
    net = b.e.Network('full', 17, b.PRETRAIN, 4)
    path = b.ChipPath(net).eval()
    state = torch.load(b.STAGED / 'full_s17_lr1/readout/final.pt', map_location='cpu', weights_only=True, mmap=True)
    net.load_state_dict(state)
    del state
    b.e.configure(net, 3)
    path.eval().requires_grad_(False)
    reference = path.reference(rdata)
    del rdata
    original = net.matcher
    results = []
    for batch in (1, 8):
        query = subset(qdata, slice(0, batch))
        qr = query_rows[:batch]
        a, c = path.extract(query, True)
        inputs = matcher_inputs(a, qr, reference, refs)
        expected_delta = original(*inputs, arm='aggregate')[0]
        expected_logits = path(query, qr, reference, refs)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        tick = time.perf_counter()
        accelerated = GraphMatcher(original)
        accelerated(*inputs, arm='aggregate')
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - tick
        graph_bytes = torch.cuda.memory_allocated() - before
        actual_delta = accelerated(*inputs, arm='aggregate')[0]
        assert torch.allclose(expected_delta, actual_delta, atol=1e-5, rtol=1e-5)
        result = dict(batch=batch, graph_setup_seconds=setup_seconds,
                      graph_allocated_mib=graph_bytes / 2**20,
                      delta_max_error=float((actual_delta - expected_delta).abs().max()))
        print('CAPTURE', batch, setup_seconds, result['delta_max_error'], flush=True)
        result['m3_original'] = b.timed(lambda: original(*inputs, arm='aggregate'), 10)
        result['m3_graph'] = b.timed(lambda: accelerated(*inputs, arm='aggregate'), 10)
        result['full_original'] = b.timed(lambda: path(query, qr, reference, refs), 10)
        net.matcher = accelerated
        actual = path(query, qr, reference, refs)
        result['full_max_error'] = float((actual - expected_logits).abs().max())
        assert result['full_max_error'] < 1e-4
        result['full_graph'] = b.timed(lambda: path(query, qr, reference, refs), 10)
        result['memory_graph'] = b.peak_memory(lambda: path(query, qr, reference, refs))
        net.matcher = original
        # New content, changing legality and empty regions must replace captured data.
        checks = []
        for variant in ('different_query', 'no_legal_refs', 'vv_missing', 'empty_region', 'changed_refs'):
            values = [v.clone() for v in inputs]
            if variant == 'different_query':
                altered = subset(qdata, slice(7, 8)) if batch == 1 else subset(qdata, torch.arange(7, -1, -1, device=b.DEVICE))
                aa, _ = path.extract(altered, True)
                values[0], values[2], values[4] = aa['desc'], aa['mass'], aa['write']
            elif variant == 'no_legal_refs':
                values[5].zero_()
            elif variant == 'vv_missing':
                values[0][:, 0].zero_(); values[2][:, 0].zero_(); values[4][:, 0].zero_(); values[5][:, :, 0] = False
            elif variant == 'empty_region':
                values[2][:, :, :8] = 0
            else:
                for i in (1, 3, 6):
                    values[i] = values[i].flip(0)
                values[5] = values[5].flip(1)
            old = original(*values, arm='aggregate')
            new = accelerated(*values, arm='aggregate')
            error = float((old[0] - new[0]).abs().max())
            local_error = float((old[1] - new[1]).abs().max())
            assert error < 1e-4 and local_error < 1e-4 and torch.equal(old[2], new[2])
            checks.append(dict(variant=variant, delta_max_error=error, local_max_error=local_error))
        result['dynamic_input_checks'] = checks
        # Keep one output across another call to check graph-storage aliasing.
        first_output = accelerated(*inputs, arm='aggregate')[0]
        preserved = first_output.clone()
        accelerated(*values, arm='aggregate')
        assert torch.equal(first_output, preserved)
        result['output_not_overwritten'] = True
        results.append(result)
        save('pilot.json', results)
        print('RESULT', batch, result['full_original']['median_ms'], result['full_graph']['median_ms'], flush=True)
        del accelerated, a, c, inputs, values, old, new, first_output, preserved
        gc.collect()
        torch.cuda.empty_cache()
    save('pilot.json', results)


if __name__ == '__main__':
    main()
