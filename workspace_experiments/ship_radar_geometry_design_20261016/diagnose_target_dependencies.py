"""Post-hoc mechanism checks: constant context and unsigned-axis perturbation."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from run_geometry import HERE, DEVICE, GeometryReader, evaluate, load_data, source_pools


class ConstantContext(nn.Module):
    def forward(self, tokens, positions, areas, geometry):
        result = tokens.new_zeros((len(tokens), 64))
        result[:, 0] = 1
        return result


class RotateAxes(nn.Module):
    def __init__(self, reader):
        super().__init__()
        self.reader = reader

    def forward(self, tokens, positions, areas, geometry):
        g = geometry.clone()
        g[:, :2] += np.pi / 4
        return self.reader(tokens, positions, areas, g)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    out = HERE / "dependency_diagnostics"
    out.mkdir(exist_ok=True)
    tables, details = [], []
    for pool in source_pools():
        port, slug = pool["pool"], pool["pool"].replace(" ", "_")
        data = load_data(port, target=True)
        models = [("constant", -1, ConstantContext())]
        for seed in range(3):
            m = GeometryReader("C").to(DEVICE)
            m.load_state_dict(torch.load(HERE / "trained" / slug / f"C_seed{seed}.pt", weights_only=False)["state"])
            models.append(("C_axes45", seed, RotateAxes(m)))
        for arm, seed, model in models:
            model.eval()
            table, pred = evaluate(model, data, "target", detailed=True)
            for frame in (table, pred):
                frame["arm"], frame["seed"] = arm, seed
            tables.append(table)
            details.append(pred)
        print("DEPENDENCY", port, flush=True)
    pd.concat(tables, ignore_index=True).to_csv(out / "metrics.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    pp = pd.concat(tables).groupby(["port", "budget", "origin", "arm", "seed"])[["oa", "ba", "macro_nll", "macro_brier"]].mean().reset_index()
    pp.to_csv(out / "per_port.csv", index=False)
    means = pp.groupby(["budget", "origin", "arm", "seed"])[["oa", "ba", "macro_nll", "macro_brier"]].mean().reset_index()
    means.to_csv(out / "seed_means.csv", index=False)
    print(means.loc[means.origin.eq("cross")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
