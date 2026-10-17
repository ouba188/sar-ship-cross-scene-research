"""Extend the existing source-only coordinate systems without target refitting."""

import json
import os
from pathlib import Path
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from full_readout import MANIFEST, source_gallery
from target_context_pilot import class_map, design_block, fit_head

HERE = Path(__file__).resolve().parent
OUT = HERE / "body_context_expansion"


def merge_raw(original, old_rows, extras, extra_rows, selected):
    value = np.empty((len(selected), original.shape[1]), np.float32)
    good = np.isin(selected, old_rows)
    p = np.searchsorted(old_rows, selected[good])
    np.testing.assert_array_equal(old_rows[p],selected[good])
    value[good] = original[p]
    if (~good).any():
        p = np.searchsorted(extra_rows, selected[~good])
        np.testing.assert_array_equal(extra_rows[p],selected[~good])
        value[~good] = extras[p]
    return value


def main():
    frozen=json.loads((OUT/"frozen.json").read_text(encoding="utf-8"))
    old_episodes=json.loads((HERE.parent/"context_condition_study_20261015/target_episodes.json").read_text())
    manifest=pd.read_csv(MANIFEST,usecols=["instance_id","port","product","class_id","mmsi"],dtype={"mmsi":str})
    required=pd.read_csv(OUT/"required_samples.csv")
    with threadpool_limits(limits=2):
        for pool in frozen["pools"]:
            if not pool["selected"]:
                continue
            slug=pool["port"].replace(" ","_")
            output=OUT/slug
            if (output/"prepared.json").exists():
                print("readout cache",slug,flush=True)
                continue
            assert (output/"encoded.json").exists()
            folder=Path(pool["source_folder"])
            source=folder.parent/slug
            train=pd.read_csv(folder/"source_train.csv")
            val=pd.read_csv(folder/"source_validation.csv")
            train=train.loc[~train.acquisition.isin(val.acquisition)]
            train_rows=train.row_index.to_numpy(int)
            with np.load(HERE/"body_context"/slug/"readout.npz") as cache:
                archived={k:cache[k] for k in cache.files}
            new_rows=required.loc[required.port.eq(pool["port"]),"row_index"].to_numpy(int)
            selected=np.unique(np.r_[train_rows,new_rows,archived["global_rows"]])
            local_train=np.searchsorted(selected,train_rows)
            labels=manifest.class_id.to_numpy(int)[selected]
            old_rows=np.load(source/"rows.npy")
            extra_rows=np.load(output/"rows.npy")
            features={}
            for name,file in (("base","c4"),("general","zero"),("domain","D18")):
                raw=np.array(np.load(source/(file+".npy"),mmap_mode="r"),dtype=np.float32)
                old_train=np.searchsorted(old_rows,train_rows)
                np.testing.assert_array_equal(old_rows[old_train],train_rows)
                pca=PCA(n_components=128,svd_solver="randomized",random_state=20261014)
                pca.fit(raw[old_train])
                # Preserve the archived transform batch to avoid float32 GEMM
                # shape changes perturbing frozen coordinates on known rows.
                projected=pca.transform(raw).astype(np.float32)
                extra=(pca.transform(np.load(output/(file+".npy"),mmap_mode="r")).astype(np.float32)
                       if len(extra_rows) else None)
                features[name]=merge_raw(projected,old_rows,extra,extra_rows,selected)
                del raw
            indices=np.arange(len(selected))
            base=design_block(features["base"],indices)
            expert=design_block((features["general"],features["domain"]),indices)
            prior_b=base@fit_head(features["base"],labels,local_train)
            prior_e=expert@fit_head((features["general"],features["domain"]),labels,local_train)
            relation=class_map(expert[:,:-1],labels,local_train)
            replay_positions=np.searchsorted(selected,archived["global_rows"])
            replay={}
            for name,value in (("base",base),("expert",expert),("prior_b",prior_b),("prior_e",prior_e),("relation",relation)):
                replay[name]=float(np.abs(value[replay_positions]-archived[name]).max())
                np.testing.assert_allclose(value[replay_positions],archived[name],atol=1e-6,rtol=1e-6)
            raw_local=[]
            for e in old_episodes:
                if e["port"]==pool["port"]:
                    with np.load(HERE/"localization"/f"features_{e['fold']}.npz") as cache:
                        raw_local.append({k:cache[k] for k in ("rows","ship_peak","raw_ring")})
            with np.load(output/"local.npz") as cache:
                raw_local.append({k:cache[k] for k in cache.files})
            unique,first=np.unique(np.concatenate([v["rows"] for v in raw_local]),return_index=True)
            gallery=source_gallery(folder)
            gallery=gallery[np.isin(gallery,train_rows)]
            fit_pos=np.searchsorted(unique,gallery)
            np.testing.assert_array_equal(unique[fit_pos],gallery)
            compact_rows=np.unique(np.r_[new_rows,archived["global_rows"]])
            compact_pos=np.searchsorted(selected,compact_rows)
            local_pos=np.searchsorted(unique,compact_rows)
            np.testing.assert_array_equal(unique[local_pos],compact_rows)
            vectors={}
            for name,key,width in (("ship","ship_peak",64),("ring","raw_ring",4)):
                raw=np.concatenate([v[key] for v in raw_local])[first]
                pca=PCA(n_components=width,svd_solver="randomized",random_state=20261015)
                pca.fit(raw[fit_pos])
                oldpos=np.searchsorted(compact_rows,archived["global_rows"])
                known=np.isin(compact_rows,archived["global_rows"])
                values=np.empty((len(compact_rows),width),dtype=np.float64)
                values[oldpos]=pca.transform(raw[np.searchsorted(unique,archived["global_rows"])]).astype(np.float64)
                values[~known]=pca.transform(raw[local_pos[~known]]).astype(np.float64)
                values/=np.linalg.norm(values,axis=1,keepdims=True).clip(1e-12)
                vectors[name]=values
                replay[name]=float(np.abs(values[oldpos]-archived[name]).max())
                np.testing.assert_allclose(values[oldpos],archived[name],atol=1e-6,rtol=1e-6)
            np.savez_compressed(output/"readout.npz",labels=labels[compact_pos],base=base[compact_pos],
                expert=expert[compact_pos],prior_b=prior_b[compact_pos],prior_e=prior_e[compact_pos],
                relation=relation[compact_pos],weights=archived["weights"],global_rows=compact_rows,**vectors)
            meta=manifest.iloc[compact_rows].copy().reset_index(drop=True)
            meta["sample_id"]=meta.instance_id
            meta["global_row"]=compact_rows
            meta["position"]=np.arange(len(meta))
            meta["acquisition"]=meta["product"].str.split("_").map(lambda p:"_".join((p[0],p[6],p[7])))
            meta.to_csv(output/"metadata.csv",index=False)
            lookup=pd.Series(np.arange(len(meta)),index=meta.sample_id)
            tasks=[]
            for e in frozen["episodes"]:
                if e["port"]!=pool["port"]:
                    continue
                for b in e["budgets"]:
                    for draw,pair in enumerate(e["draws"]):
                        for origin in ("same","cross"):
                            tasks.append(dict(fold=e["fold"],port=pool["port"],acquisition=e["query_acquisition"],
                                budget=b,draw=draw,origin=origin,q=lookup.loc[e["query"]].to_list(),
                                s=lookup.loc[pair[origin][:b]].to_list()))
            (output/"tasks.json").write_text(json.dumps(tasks,indent=2,ensure_ascii=False),encoding="utf-8")
            (output/"prepared.json").write_text(json.dumps(dict(source_train=len(train_rows),
                target_rows=len(new_rows),tasks=len(tasks),archived_coordinate_replay=replay),indent=2),encoding="utf-8")
            print("prepared",slug,len(tasks),"coordinate replay",max(replay.values()),flush=True)


if __name__=="__main__":
    main()
