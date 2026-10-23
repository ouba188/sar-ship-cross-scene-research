"""Bounded warm-cache algorithm cost; not end-to-end single-chip latency."""
import json
import time
import numpy as np
import torch
from run import HERE, ROOT, Matching, regional_inputs, unit


def main():
    torch.set_num_threads(2)
    cache = np.load(HERE / 'scores.npz'); ids = cache['idx'][:16]; refs = cache['refs'][:16]
    data = np.load(ROOT / 'tokens.npz'); pr = np.load(ROOT / 'projection_layer1.npz')
    x = unit((data['layer1'] - pr['mean']) @ pr['components'].T).astype(np.float32)
    x[data['mass'] <= 0] = 0
    args = regional_inputs(x[ids], data['mass'][ids], x, data['mass'], refs)
    s, a, b = [torch.tensor(v, device='cuda', dtype=torch.float32) for v in args[:3]]
    available = torch.tensor(args[3], device='cuda', dtype=torch.bool)
    model = Matching(s, a, b, available)
    timings = {}
    with torch.inference_mode():
        model.values(a[:, None])
        for arm in ('fixed', 'uncoupled', 'coupled'):
            elapsed = []
            for _ in range(3):
                torch.cuda.synchronize(); begin = time.perf_counter()
                if arm == 'fixed':
                    # Common a needs one inner solve per class, not eight copies.
                    value = model.values(a[:, None])[0][:, 0]
                    other = value[:, None].expand(-1, 8, -1).clone()
                    other.masked_fill_(torch.eye(8, device='cuda', dtype=torch.bool)[None], -1e6)
                    _ = value - other.max(-1).values
                else:
                    model.optimize(coupled=arm == 'coupled')
                torch.cuda.synchronize(); elapsed.append(time.perf_counter() - begin)
            timings[arm] = dict(seconds=elapsed, median_seconds=float(np.median(elapsed)),
                                amortized_ms_per_query=float(np.median(elapsed) * 1000 / len(ids)))
    receipt = dict(n_queries=len(ids), regions_per_query=6, candidate_classes=8,
                   cached_features_only=True, single_query_latency_measured=False,
                   hardware=torch.cuda.get_device_name(), timings=timings)
    (HERE / 'benchmark.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
