"""Correctly synchronized startup timing and direct transport-storage check."""
from pathlib import Path
import importlib.util
import json
import time
import gc

spec = importlib.util.spec_from_file_location('latency_pilot', Path(__file__).with_name('experiment.py'))
x = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x)
b, torch = x.b, x.torch


@torch.inference_mode()
def main():
    b.task.current.setup(b.DEVICE)
    b.task.tensor = b.resident_tensor
    torch.set_num_threads(2)
    rows, groups = b.e.prev.records()
    refs = [rows[i] for i in groups['references']]
    selected = json.loads((b.ROOT / 'inputs_4class.json').read_text())
    lookup = {r['id']: r for r in rows}
    queries = [lookup[i] for i in selected['query_ids']]
    qdata = b.prepare(queries)
    net = b.e.Network('full', 17, b.PRETRAIN, 4)
    path = b.ChipPath(net)
    weights = torch.load(b.STAGED / 'full_s17_lr1/readout/final.pt', map_location='cpu', weights_only=True, mmap=True)
    net.load_state_dict(weights)
    del weights
    b.e.configure(net, 3)
    path.eval().requires_grad_(False)
    reference = path.reference(b.prepare(refs))
    records = []
    for batch in (1, 8):
        a, _ = path.extract(x.subset(qdata, slice(0, batch)), True)
        inputs = x.matcher_inputs(a, queries[:batch], reference, refs)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        fast = x.GraphMatcher(net.matcher)
        fast(*inputs, arm='aggregate')
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        checks = []
        for graph in fast.graphs.values():
            values = [v.clone() for v in graph.static]
            values[1].fill_(1 / values[1].shape[-1])
            values[2].fill_(1 / values[2].shape[-1])
            values[3].fill_(True)
            first = fast.transport(*values)
            saved = first.clone()
            assert first.abs().sum().item() > 0
            values[3].zero_()
            second = fast.transport(*values)
            assert second.count_nonzero().item() == 0 and torch.equal(first, saved)
            checks.append(dict(shape=list(values[0].shape), first_nonzero=True,
                               second_all_zero=True, first_not_overwritten=True))
        records.append(dict(batch=batch, synchronized_first_setup_seconds=elapsed,
                            direct_transport_alias_checks=checks))
        del fast
        gc.collect()
        torch.cuda.empty_cache()
    x.save('runtime_checks.json', records)
    print(json.dumps(records), flush=True)


if __name__ == '__main__':
    main()
