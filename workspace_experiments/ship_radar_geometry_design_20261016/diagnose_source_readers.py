"""Locate learning/geometry bottlenecks on frozen source tasks, seed 0 only."""

import numpy as np
import pandas as pd
import torch

from run_geometry import HERE, DEVICE, GeometryReader, contexts, load_data, source_pools


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    records = []
    for pool in source_pools():
        port, slug = pool["pool"], pool["pool"].replace(" ", "_")
        data = load_data(port)
        roles = {k: sorted({i for t in ts for i in t["q"] + t["s"]}) for k, ts in data["tasks"].items()}
        for mode in "ABC":
            for stage in ("initial", "selected", "last"):
                torch.manual_seed(20261016)
                model = GeometryReader(mode).to(DEVICE)
                step = 0
                if stage != "initial":
                    suffix = "_last" if stage == "last" else ""
                    saved = torch.load(HERE / "trained" / slug / f"{mode}_seed0{suffix}.pt", weights_only=False)
                    model.load_state_dict(saved["state"])
                    step = saved.get("best_step", saved.get("step", 0))
                model.eval()
                original = contexts(model, data)
                altered = {**data, "tensors": dict(data["tensors"])}
                g = data["tensors"]["geometry"].clone()
                g[:, :2] = g[:, :2] + np.pi / 4
                altered["tensors"]["geometry"] = g
                changed = contexts(model, altered)
                for split, rows in roles.items():
                    c = original[rows]
                    centered_energy = float((c - c.mean(0)).square().sum(1).mean())
                    valid = (data["tensors"]["geometry"][rows, 3:5] > 0).all(1)
                    delta = (changed[rows] - c).norm(dim=1)
                    records.append(dict(pool=port, arm=mode, seed=0, stage=stage, step=step, split=split,
                        samples=len(rows), centered_energy=centered_energy, mean_vector_norm=float(c.mean(0).norm()),
                        coordinate_change_mean=float(delta.mean()), coordinate_change_valid_mean=float(delta[valid].mean()) if valid.any() else np.nan,
                        coordinate_change_max=float(delta.max()), both_axes=int(valid.sum())))
        print("SOURCE DIAGNOSTIC", port, flush=True)
    pd.DataFrame(records).to_csv(HERE / "source_representation_diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
