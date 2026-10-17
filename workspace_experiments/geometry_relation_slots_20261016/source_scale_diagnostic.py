"""Source-observed amplitude diagnostic; no target fitting or alpha search."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_slots as run
from slot_reader import SlotReader

HERE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    records=[]
    for port in run.PORTS:
        data=run.load(port,True)
        slug=port.replace(" ","_")
        folder=HERE/"source_scaled_evaluation"/slug
        folder.mkdir(parents=True,exist_ok=True)
        for seed in range(3):
            run.prepare_baseline(data,seed)
            for mode in run.MODES:
                file=folder/f"{mode}_seed{seed}_predictions.csv"
                saved=torch.load(HERE/"trained"/slug/f"{mode}_seed{seed}.pt",weights_only=False)
                energy=1.; model=None
                if not saved.get("unavailable"):
                    curve=pd.read_csv(HERE/"trained"/slug/f"{mode}_seed{seed}_curve.csv")
                    energy=float(curve.loc[curve.step.eq(200),"source_centered_energy"].iloc[0])
                    model=SlotReader(mode).to(run.DEVICE)
                    model.load_state_dict(saved["state"])
                    model.scale.mul_(np.sqrt(energy))
                    model.eval()
                records.append(dict(port=port,arm=mode,seed=seed,source_energy=energy,kernel_multiplier=1/energy,trained=not saved.get("unavailable",False)))
                if file.exists():
                    continue
                metrics,predictions=run.evaluate(model,data,"target",True)
                for f in (metrics,predictions):
                    f["arm"],f["seed"]=mode,seed
                metrics.to_csv(folder/f"{mode}_seed{seed}_metrics.csv",index=False)
                predictions.to_csv(file,index=False)
                print("SOURCE SCALE",port,mode,seed,"energy",round(energy,3),flush=True)
        del data
        torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(HERE/"source_scale_factors.csv",index=False)
    metrics=pd.concat([pd.read_csv(p) for p in (HERE/"source_scaled_evaluation").glob("*/*_metrics.csv")],ignore_index=True)
    met=["oa","ba","macro_nll","macro_brier"]
    perport=metrics.groupby(["scope","origin","arm","seed","port"])[met].mean().reset_index()
    means=perport.groupby(["scope","origin","arm","seed"])[met].mean().reset_index()
    perport.to_csv(HERE/"source_scaled_per_port.csv",index=False)
    means.to_csv(HERE/"source_scaled_seed_means.csv",index=False)
    original=pd.read_csv(HERE/"seed_means.csv")
    joined=means.merge(original.loc[original.arm.eq("B")],on=["scope","origin","seed"],suffixes=("","_B"),validate="many_to_one")
    for m in met:
        joined["delta_"+m]=joined[m]-joined[m+"_B"]
    joined.to_csv(HERE/"source_scaled_contrasts.csv",index=False)
    print(means.loc[means.scope.eq("full") & means.origin.eq("cross")].groupby("arm")[met].mean().to_string())
    print(joined.loc[joined.scope.eq("full") & joined.origin.eq("cross")].groupby("arm")[["delta_oa","delta_ba"]].mean().to_string())


if __name__=="__main__":
    main()
