"""Freeze a source-query coverage control while holding every support set fixed."""

import json

import numpy as np
import pandas as pd

from freeze_ring_relaxed import OUT as PARENT
from probe_ring_predictability import HERE, ROOT, audit_partitions, save_json

OUT=PARENT/"query_coverage"


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/"frozen.json").exists():
        print("Coverage control already frozen.")
        return
    frozen=json.loads((PARENT/"frozen.json").read_text(encoding="utf-8"))
    geo=pd.read_csv(HERE.parent/"context_condition_study_20261015/geometry_by_sample.csv",usecols=["sample_id","far_ok"])
    far_ok=set(geo.loc[geo.far_ok,"sample_id"])
    records,partitions=[],{}
    for pool in frozen["pools"]:
        port=pool["pool"]
        slug=port.replace(" ","_")
        folder=HERE/"body_context"/slug
        meta=pd.read_csv(folder/"metadata.csv",dtype={"mmsi":str}).set_index("position",drop=False)
        split=pd.read_csv(folder/"source_split.csv")
        for name,label in (("fit","module_fit"),("selection","selection")):
            allowed=meta.loc[split.loc[split.split.eq(label),"position"]]
            allowed=allowed.loc[allowed.sample_id.isin(far_ok)&allowed.mmsi.fillna("").str.fullmatch("[1-9][0-9]{8}")]
            for task in pool["splits"][name]:
                q=allowed.loc[allowed.port.eq(task["port"])&allowed.acquisition.eq(task["acquisition"])]
                q=q.sort_values("sample_id").drop_duplicates("mmsi")
                support=meta.loc[task["s"]]
                q=q.loc[~q.mmsi.isin(support.mmsi)]
                previous=task["q"][:]
                assert set(previous)<=set(q.index)
                task["q"]=previous+[int(i) for i in q.index if i not in previous]
                task["original_q"]=previous
                task["query_classes"]=int(q.class_id.nunique())
                task["unsupported_queries"]=int((~q.class_id.isin(support.class_id)).sum())
                assert not set(q.mmsi)&set(support.mmsi)
                # Larger budgets can remove more overlapping source identities from queries.
                # Their support prefixes are unchanged; query cohorts are therefore recorded.
            for b in pool["budgets"]:
                take=[t for t in pool["splits"][name] if t["budget"]==b]
                qidx=sorted({i for t in take for i in t["q"]})
                sidx=sorted({i for t in take for i in t["s"]})
                records.append(dict(pool=port,split=name,budget=b,tasks=len(take),
                    pairs=len({(t["port"],t["acquisition"],t["support_acquisition"]) for t in take}),
                    query_scenes=len({(t["port"],t["acquisition"]) for t in take}),
                    query_samples=len(qidx),query_ships=meta.loc[qidx].mmsi.nunique(),support_ships=meta.loc[sidx].mmsi.nunique(),
                    query_exposures=sum(len(t["q"]) for t in take),
                    original_query_exposures=sum(len(t["original_q"]) for t in take),
                    unsupported_query_exposures=sum(t["unsupported_queries"] for t in take),
                    unique_support_sets=len({tuple(sorted(t["s"])) for t in take}),
                    query_count_min=min(len(t["q"]) for t in take),query_count_max=max(len(t["q"]) for t in take)))
        target_tasks=json.loads((ROOT/slug/"tasks.json").read_text(encoding="utf-8"))
        partitions[port]=audit_partitions(folder,ROOT/slug,pool["splits"],target_tasks)
    frozen["stage"]="coverage_control_frozen_before_outcomes"
    frozen["support_unchanged"]=True
    frozen["query_count"]="all_legal_source_acquisition_ships_excluding_task_support_identities"
    save_json(OUT/"frozen.json",frozen)
    save_json(OUT/"partitions.json",partitions)
    pd.DataFrame(records).to_csv(OUT/"task_coverage.csv",index=False)
    print(pd.DataFrame(records).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
