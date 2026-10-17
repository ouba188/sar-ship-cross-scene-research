"""Rebuild legal source episodes using target-style single-acquisition supports."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from freeze_body_expansion import counts, tokens, remove_overlap
from probe_ring_predictability import HERE, ROOT, save_json

OUT=ROOT/"ring_matched"
SEED=20261017


def sample_sequence(frame, order, rng, spread=False):
    if not spread:
        available={int(c):list(rng.permutation(g.index.to_numpy(int))) for c,g in frame.groupby("class_id")}
        return [int(available[c].pop()) for c in order]
    available={int(c):{a:list(rng.permutation(g.index.to_numpy(int))) for a,g in part.groupby("acquisition")}
               for c,part in frame.groupby("class_id")}
    acqs=list(rng.permutation(sorted(frame.acquisition.unique())))
    rank={a:i for i,a in enumerate(acqs)}
    used={a:0 for a in acqs}
    result=[]
    for c in order:
        candidates=[a for a,v in available[c].items() if v]
        chosen=min(candidates,key=lambda a:(used[a],rank[a]))
        result.append(int(available[c][chosen].pop()))
        used[chosen]+=1
    return result


def build(frame, coords, seed):
    tasks={"single":[],"mixed":[]}
    inventory=[]
    for pi,(port,portframe) in enumerate(frame.groupby("port",sort=True)):
        groups={a:g.sort_values("sample_id").drop_duplicates("mmsi") for a,g in portframe.groupby("acquisition",sort=True)}
        for ai,(acq,group) in enumerate(groups.items()):
            rec=dict(port=port,acquisition=acq,ships=len(group),classes=group.class_id.nunique(),reason="small_or_few_classes")
            inventory.append(rec)
            if len(group)<40 or group.class_id.nunique()<3:
                continue
            rng=np.random.default_rng(seed+pi*1000+ai)
            fraction=min(1/3,64/len(group))
            qi=[]
            for _,cls in group.groupby("class_id",sort=True):
                qi.extend(rng.choice(cls.index,max(1,int(np.ceil(len(cls)*fraction))),replace=False).tolist())
            query=group.loc[group.index.isin(qi)]
            same=remove_overlap(group.loc[~group.mmsi.isin(query.mmsi)],query,coords)
            rec.update(query=len(query),same=len(same),reason="insufficient_query_or_same")
            if len(query)<16 or len(same)<16:
                continue
            choices=[]
            for other_acq,other in groups.items():
                if other_acq==acq:
                    continue
                cross=other.loc[~other.mmsi.isin(query.mmsi)]
                capacity=np.minimum(counts(same),counts(cross))
                classes=[int(c) for c in sorted(query.class_id.unique()) if capacity[c]>0]
                if len(classes)<3 or capacity[classes].sum()<16:
                    continue
                choices.append((len(classes),int(capacity[classes].sum()),len(cross),other_acq,cross,tokens(capacity,classes)))
            rec["reason"]="no_single_cross_capacity"
            if not choices:
                continue
            _,_,_,cross_acq,cross,order=max(choices,key=lambda x:x[:4])
            other=portframe.loc[portframe.acquisition.ne(acq)&~portframe.mmsi.isin(query.mmsi)]
            # Preserve the single-acquisition representatives before unique-ship deduplication.
            mixed=pd.concat([cross,other]).drop_duplicates("mmsi")
            budgets=[b for b in (16,32,64) if b<=len(order)]
            rec.update(reason="eligible",cross_acquisition=cross_acq,cross=len(cross),mixed=len(mixed),
                mixed_candidate_acquisitions=mixed.acquisition.nunique(),max_budget=max(budgets))
            for draw in range(5):
                seed_draw=seed+pi*100000+ai*100+draw*3
                same_seq=sample_sequence(same,order,np.random.default_rng(seed_draw))
                single_seq=sample_sequence(cross,order,np.random.default_rng(seed_draw+1))
                mixed_seq=sample_sequence(mixed,order,np.random.default_rng(seed_draw+2),spread=True)
                for budget in budgets:
                    for org,seq in (("single",single_seq),("mixed",mixed_seq)):
                        for origin,support in (("same",same_seq[:budget]),("cross",seq[:budget])):
                            selected=frame.loc[support]
                            assert len(selected)==budget and selected.mmsi.nunique()==budget
                            assert not set(query.mmsi)&set(selected.mmsi)
                            assert selected.port.eq(port).all()
                            np.testing.assert_array_equal(counts(selected),np.bincount(order[:budget],minlength=8))
                            if origin=="cross":
                                assert selected.acquisition.ne(acq).all()
                            else:
                                assert selected.acquisition.eq(acq).all()
                            if org=="single" or origin=="same":
                                assert selected.acquisition.nunique()==1
                            tasks[org].append(dict(port=port,acquisition=acq,fold=port.replace(" ","_")+"__"+acq,
                                budget=budget,draw=draw,origin=origin,q=query.index.to_list(),s=support,
                                support_acquisitions=int(selected.acquisition.nunique())))
    return tasks,inventory


def main():
    OUT.mkdir(exist_ok=True)
    frozen_path=OUT/"frozen.json"
    if frozen_path.exists():
        print("Already frozen; preserving episodes.")
        return
    root=HERE.parent
    geometry=pd.read_csv(root/"context_condition_study_20261015/geometry_by_sample.csv",usecols=["sample_id","far_ok"])
    far_ok=set(geometry.loc[geometry.far_ok,"sample_id"])
    positions=pd.read_csv(root/"region_cue_replication_20261013/source_region_metadata.csv",usecols=[
        "sample_id","resolved_crop_left_vv","resolved_crop_top_vv","resolved_crop_left_vh","resolved_crop_top_vh"])
    shape=pd.read_csv(root/"region_cue_replication_20261013/ports/frame.csv",usecols=[
        "sample_id","scene_id","native_h_vv","native_w_vv","native_h_vh","native_w_vh"])
    coords=positions.merge(shape,on="sample_id",validate="one_to_one").set_index("sample_id")
    pools=json.loads((ROOT/"frozen.json").read_text(encoding="utf-8"))["pools"]
    records,coverage,inventory_all,partitions=[],[],[],[]
    for pool_index,pool in enumerate(pools):
        if not pool["selected"]:
            continue
        port=pool["port"]
        slug=port.replace(" ","_")
        folder=HERE/"body_context"/slug
        meta=pd.read_csv(folder/"metadata.csv",dtype={"mmsi":str}).set_index("position",drop=False)
        split=pd.read_csv(folder/"source_split.csv")
        targetmeta=pd.read_csv(ROOT/slug/"metadata.csv",dtype={"mmsi":str})
        targettasks=json.loads((ROOT/slug/"tasks.json").read_text(encoding="utf-8"))
        targetrows=sorted({i for t in targettasks for i in t["q"]+t["s"]})
        target=targetmeta.iloc[targetrows]
        splits={name:meta.loc[split.loc[split.split.eq(label),"position"]] for name,label in (("fit","module_fit"),("selection","selection"))}
        for a,b in ((splits["fit"],splits["selection"]),(splits["fit"],target),(splits["selection"],target)):
            assert not set(a.mmsi)&set(b.mmsi)
            assert not set(a.acquisition)&set(b.acquisition)
        item=dict(pool=port,source_folder=str(folder),splits={})
        for si,(name,full) in enumerate(splits.items()):
            frame=full.loc[full.sample_id.isin(far_ok)&full.mmsi.fillna("").str.fullmatch("[1-9][0-9]{8}")]
            tasks,inv=build(frame,coords,SEED+pool_index*1000000+si*100000)
            item["splits"][name]=tasks
            inventory_all.extend(dict(pool=port,split=name,**r) for r in inv)
            partitions.append(dict(pool=port,split=name,raw_rows=len(full),eligible_rows=len(frame),
                raw_ships=full.mmsi.nunique(),eligible_ships=frame.mmsi.nunique()))
            for budget in (16,32,64):
                for org,group in tasks.items():
                    take=[t for t in group if t["budget"]==budget and t["origin"]=="cross"]
                    qidx=sorted({i for t in take for i in t["q"]})
                    coverage.append(dict(pool=port,split=name,organization=org,budget=budget,tasks=len(take),
                        query_scenes=len({(t["port"],t["acquisition"]) for t in take}),
                        query_ports=len({t["port"] for t in take}),query_samples=len(qidx),
                        query_ships=meta.loc[qidx].mmsi.nunique(),
                        single_acquisition_tasks=sum(t["support_acquisitions"]==1 for t in take),
                        multi_acquisition_tasks=sum(t["support_acquisitions"]>1 for t in take)))
            print(port,name,{b:sum(t["budget"]==b and t["origin"]=="cross" for t in tasks["single"]) for b in (16,32,64)},flush=True)
        records.append(item)
    pd.DataFrame(coverage).to_csv(OUT/"task_coverage.csv",index=False)
    pd.DataFrame(inventory_all).to_csv(OUT/"candidate_inventory.csv",index=False)
    pd.DataFrame(partitions).to_csv(OUT/"partition_coverage.csv",index=False)
    save_json(frozen_path,dict(seed=SEED,pools=records,stage="metadata_only_before_predictions",
        source_split_unchanged=True,paired_class_counts=True,query_fixed_across_draws=True))
    print("FROZEN",frozen_path)


if __name__=="__main__":
    main()
