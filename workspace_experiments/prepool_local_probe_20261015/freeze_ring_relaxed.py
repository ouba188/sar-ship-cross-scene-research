"""Freeze cross-only single-acquisition source episodes with eight fixed queries."""

import json

import numpy as np
import pandas as pd

from freeze_body_expansion import counts, tokens
from freeze_ring_matched import sample_sequence
from probe_ring_predictability import HERE, ROOT, audit_partitions, save_json

OUT=ROOT/"ring_relaxed"
SEED=20261018
B64_POOLS={"Ningbo-Zhoushan","Qingdao"}


def build(frame,budgets,seed):
    tasks=[]
    for pi,(port,pf) in enumerate(frame.groupby("port",sort=True)):
        groups={a:g.sort_values("sample_id").drop_duplicates("mmsi") for a,g in pf.groupby("acquisition",sort=True)}
        for ai,(acq,g) in enumerate(groups.items()):
            if len(g)<8 or g.class_id.nunique()<2:
                continue
            # Replay the metadata-only capacity inventory's fixed eight queries.
            rng=np.random.default_rng(SEED+ai)
            baskets=[list(rng.permutation(c.index.to_numpy())) for _,c in g.groupby("class_id",sort=True)]
            query=[]
            while len(query)<8:
                for basket in baskets:
                    if basket and len(query)<8:
                        query.append(int(basket.pop()))
            q=g.loc[query]
            assert q.mmsi.nunique()==8 and q.class_id.nunique()>=2
            for ci,(ca,c) in enumerate(groups.items()):
                if ca==acq:
                    continue
                bank=c.loc[~c.mmsi.isin(q.mmsi)]
                if bank.class_id.nunique()<2:
                    continue
                allowed=[b for b in budgets if b<=len(bank)]
                if not allowed:
                    continue
                order=tokens(counts(bank),sorted(bank.class_id.unique()),limit=max(allowed))
                for draw in range(5):
                    rng=np.random.default_rng(seed+pi*100000+ai*1000+ci*10+draw)
                    seq=sample_sequence(bank,order,rng)
                    for b in allowed:
                        support=seq[:b]
                        selected=frame.loc[support]
                        assert len(selected)==b and selected.mmsi.nunique()==b
                        assert selected.acquisition.nunique()==1 and selected.acquisition.iloc[0]!=acq
                        assert selected.port.eq(port).all() and not set(q.mmsi)&set(selected.mmsi)
                        assert selected.class_id.nunique()>=2
                        tasks.append(dict(port=port,acquisition=acq,support_acquisition=ca,
                            fold=port.replace(" ","_")+"__"+acq,budget=b,origin="cross",draw=draw,
                            q=query,s=support,support_acquisitions=1,
                            query_classes=int(q.class_id.nunique()),support_classes=int(selected.class_id.nunique()),
                            unsupported_queries=int((~q.class_id.isin(selected.class_id)).sum())))
    return tasks


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/"frozen.json").exists():
        print("Frozen episodes exist; not resampling.")
        return
    geo=pd.read_csv(HERE.parent/"context_condition_study_20261015/geometry_by_sample.csv",usecols=["sample_id","far_ok"])
    far_ok=set(geo.loc[geo.far_ok,"sample_id"])
    pools=json.loads((ROOT/"frozen.json").read_text(encoding="utf-8"))["pools"]
    records,coverage,partitions=[],[],{}
    expected32={"Jebel Ali":(30,2),"Ningbo-Zhoushan":(17,6),"Qingdao":(14,2),"Shanghai":(33,4),"Singapore":(15,6)}
    for pool_i,pool in enumerate(pools):
        if not pool["selected"]:
            continue
        port=pool["port"]
        slug=port.replace(" ","_")
        folder=HERE/"body_context"/slug
        meta=pd.read_csv(folder/"metadata.csv",dtype={"mmsi":str}).set_index("position",drop=False)
        assert np.array_equal(meta.position,np.arange(len(meta)))
        split=pd.read_csv(folder/"source_split.csv")
        budgets=[32,64] if port in B64_POOLS else [32]
        item=dict(pool=port,source_folder=str(folder),budgets=budgets,splits={})
        for si,(name,label) in enumerate((("fit","module_fit"),("selection","selection"))):
            frame=meta.loc[split.loc[split.split.eq(label),"position"]]
            frame=frame.loc[frame.sample_id.isin(far_ok)&frame.mmsi.fillna("").str.fullmatch("[1-9][0-9]{8}")]
            tasks=build(frame,budgets,SEED+pool_i*1000000+si*500000)
            item["splits"][name]=tasks
            for b in budgets:
                take=[t for t in tasks if t["budget"]==b]
                pairs={(t["port"],t["acquisition"],t["support_acquisition"]) for t in take}
                qidx=sorted({i for t in take for i in t["q"]})
                sidx=sorted({i for t in take for i in t["s"]})
                assert len(take)==len(pairs)*5
                if b==32:
                    assert len(pairs)==expected32[port][si]
                else:
                    assert len(pairs)==(3,2)[si]
                coverage.append(dict(pool=port,split=name,budget=b,pairs=len(pairs),tasks=len(take),
                    query_scenes=len({(t["port"],t["acquisition"]) for t in take}),
                    support_scenes=len({(t["port"],t["support_acquisition"]) for t in take}),
                    query_samples=len(qidx),query_ships=meta.loc[qidx].mmsi.nunique(),
                    support_ships=meta.loc[sidx].mmsi.nunique(),
                    unique_support_sets=len({tuple(sorted(t["s"])) for t in take}),
                    query_exposures=sum(len(t["q"]) for t in take),unsupported_query_exposures=sum(t["unsupported_queries"] for t in take),
                    support_classes_min=min(t["support_classes"] for t in take),support_classes_max=max(t["support_classes"] for t in take)))
            print(port,name,[(b,sum(t["budget"]==b for t in tasks)) for b in budgets],flush=True)
        target_tasks=json.loads((ROOT/slug/"tasks.json").read_text(encoding="utf-8"))
        partitions[port]=audit_partitions(folder,ROOT/slug,item["splits"],target_tasks)
        # Check query stability and support nesting across actual stored tasks.
        qmap={}
        prefixes={}
        for name,tasks in item["splits"].items():
            for t in tasks:
                key=(name,t["port"],t["acquisition"])
                qmap.setdefault(key,t["q"])
                assert qmap[key]==t["q"]
                key=(*key,t["support_acquisition"],t["draw"])
                prefixes.setdefault(key,{})[t["budget"]]=t["s"]
        for value in prefixes.values():
            if 64 in value:
                assert value[64][:32]==value[32]
        records.append(item)
    pd.DataFrame(coverage).to_csv(OUT/"task_coverage.csv",index=False)
    save_json(OUT/"partitions.json",partitions)
    save_json(OUT/"frozen.json",dict(seed=SEED,pools=records,stage="metadata_frozen_before_predictions",
        query_count=8,minimum_classes=2,single_cross_support=True,paired_same_requirement=False))
    print("FROZEN",OUT/"frozen.json",flush=True)


if __name__=="__main__":
    main()
