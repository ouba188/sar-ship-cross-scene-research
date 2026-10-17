"""Summarize matched draws, observed scale and marginal cached-operator cost."""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_slots as run
from slot_reader import SlotReader

HERE=Path(__file__).resolve().parent
KEY=["port","fold","budget","origin","draw","sample_id","seed"]
PC=[f"p{i}" for i in range(8)]


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    allpred=pd.read_csv(HERE/"predictions.csv",dtype={"mmsi":str})
    baseline=allpred.loc[allpred.arm.eq("B")]
    scaled=pd.concat([pd.read_csv(p,dtype={"mmsi":str}) for p in (HERE/"source_scaled_evaluation").glob("*/*_predictions.csv")],ignore_index=True)
    scaled.to_csv(HERE/"source_scaled_predictions.csv",index=False)
    joined=scaled.merge(baseline[KEY+["correct",*PC]],on=KEY,suffixes=("","_B"),validate="many_to_one")
    disabled=~joined.enabled
    np.testing.assert_array_equal(joined.loc[disabled,PC],joined.loc[disabled,[c+"_B" for c in PC]])
    joined["rescue"]=joined.correct & ~joined.correct_B
    joined["harm"]=~joined.correct & joined.correct_B
    joined.groupby(["arm","seed","port","origin","truth"])[["rescue","harm"]].sum().to_csv(HERE/"source_scaled_class_events.csv")
    joined.loc[joined.rescue|joined.harm].to_csv(HERE/"source_scaled_changed_cases.csv",index=False)
    data=run.load("Jebel Ali",True); run.prepare_baseline(data,0)
    item=next(t for t in data["cache"]["target"] if t["t"]["origin"]=="cross" and (data["tensors"]["valid"][t["q"]].any()) and int(data["tensors"]["valid"][t["s"]].sum())>=2)
    costs=[]
    for mode in run.MODES:
        saved=torch.load(HERE/"trained/Jebel_Ali"/f"{mode}_seed0.pt",weights_only=False)
        model=SlotReader(mode).to(run.DEVICE); model.load_state_dict(saved["state"]); model.eval()
        for _ in range(5):
            run.prediction(model,data,item)
        torch.cuda.synchronize(); start=time.perf_counter()
        for _ in range(100):
            run.prediction(model,data,item)
        torch.cuda.synchronize()
        costs.append(dict(arm=mode,additional_cached_task_ms=1000*(time.perf_counter()-start)/100,supports=len(item["s"]),queries=len(item["q"]),parameters=sum(p.numel() for p in model.parameters()),excludes_baseline_and_image_encoder=True))
    pd.DataFrame(costs).to_csv(HERE/"incremental_operator_cost.csv",index=False)
    matched=[]
    for folder in (HERE/"trained").iterdir():
        for seed in range(3):
            paths=[folder/f"{m}_seed{seed}_sampling.json" for m in run.MODES]
            if all(p.exists() for p in paths):
                trace=[json.loads(p.read_text()) for p in paths]
                assert trace[0]==trace[1]==trace[2]
                matched.append(dict(pool=folder.name,seed=seed,task_draws=len(trace[0])))
    (HERE/"SAMPLING_CHECK.json").write_text(json.dumps(matched,indent=2),encoding="utf-8")
    events=pd.read_csv(HERE/"paired_events.csv",dtype={"mmsi":str})
    e=events.loc[events.arm.eq("geometry") & events.origin.eq("cross")]
    group=e.groupby(["seed","truth"])[["rescue","harm"]].sum()
    group.to_csv(HERE/"geometry_class_events_by_seed.csv")
    (e.loc[e.rescue|e.harm].groupby(["truth"]).agg(changed_ships=("mmsi","nunique"),changed_samples=("sample_id","nunique"))).to_csv(HERE/"geometry_class_changed_ship_counts.csv")
    p=allpred.loc[allpred.arm.eq("geometry") & allpred.origin.eq("cross") & allpred.seed.eq(0) & allpred.enabled]
    coverage=dict(enabled_cross_query_samples=int(p.sample_id.nunique()),enabled_cross_ships=int(p.mmsi.nunique()),enabled_cross_scenes=int(p.acquisition.nunique()),enabled_cross_ports=sorted(p.port.unique().tolist()))
    print("COVERAGE",coverage)
    print("CLASS EVENTS",group.to_string())
    print("COST",pd.DataFrame(costs).to_string(index=False))
    (HERE/"DIAGNOSTICS_CHECK.json").write_text(json.dumps(dict(scaled_fallback_exact=True,sampling_groups_equal=len(matched),**coverage),indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
