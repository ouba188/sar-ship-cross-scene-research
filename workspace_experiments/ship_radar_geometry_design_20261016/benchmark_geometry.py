"""Matched cached-feature operator timings; not end-to-end SAR inference."""

import time

import numpy as np
import pandas as pd
import torch

from run_geometry import HERE, DEVICE, GeometryReader, encode, load_data, prediction, source_pools


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    records = []
    for pool in source_pools():
        port = pool["pool"]
        data = load_data(port, target=True)
        d = data["tensors"]
        tasks = [next(t for t in data["tasks"]["target"] if t["budget"] == b and t["origin"] == "cross") for b in (16, 32, 64)]
        for mode in ("body", "A", "B", "C"):
            model = None
            if mode != "body":
                path = HERE / "trained" / port.replace(" ", "_") / f"{mode}_seed0.pt"
                model = GeometryReader(mode).to(DEVICE)
                model.load_state_dict(torch.load(path, weights_only=False)["state"])
                model.eval()
            for t in tasks:
                elapsed = []
                for repeat in range(12):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    c = None if model is None else encode(model, d, t["s"] + t["q"])
                    prediction(d, t, c)
                    torch.cuda.synchronize()
                    if repeat >= 2:
                        elapsed.append((time.perf_counter() - started) * 1000)
                records.append(dict(port=port, arm=mode, budget=t["budget"], query_count=len(t["q"]),
                    support_count=len(t["s"]), median_ms=float(np.median(elapsed)), mean_ms=float(np.mean(elapsed)),
                    std_ms=float(np.std(elapsed)), repeats=len(elapsed)))
        print("BENCHMARK", port, flush=True)
    pd.DataFrame(records).to_csv(HERE / "cached_operator_cost.csv", index=False)


if __name__ == "__main__":
    main()
