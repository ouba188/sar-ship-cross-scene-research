"""Matched bounded training, then optional recovered-geometry relation learning."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="2"
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"geometry_relation_slots_20261016"
sys.path.insert(0,str(PREV))
import run_slots as base
from slot_reader import SlotReader
from bounded_core import OptionalGeometryReader,bounded_predict,relation_loss
from body_context_gate import _coefficients

ARMS={"bounded":("spatial","conditional","geometry"),
      "recovered":("geometry_single","geometry_pair","geometry_rank","conditional_rank")}
MET=["oa","ba","macro_nll","macro_brier"]


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def load(port,stage,target=False):
    data=base.load(port,target)
    data["stage"]=stage
    data["legacy_geometry"]=data["tensors"]["geometry"].clone()
    if stage=="recovered":
        table=pd.read_csv(HERE/"recovered_geometry.csv",low_memory=False).set_index("global_row").loc[data["meta"].global_row]
        cols=("ship_axis_rad","radar_axis_rad","incidence_rad","ship_valid","radar_valid","incidence_valid","ship_confidence")
        geo=np.nan_to_num(table[list(cols)].to_numpy(np.float32))
        data["updated_geometry"]=torch.as_tensor(geo,device=base.DEVICE)
        # Recovery changed no ship axes, so the existing rotated-slot cache is exact.
        old=data["legacy_geometry"]
        new=data["updated_geometry"]
        assert torch.equal(old[:,3],new[:,3])
        keep=old[:,3]>0
        assert torch.equal(old[keep,0],new[keep,0])
    return data


def prepare(data,seed):
    d=data["tensors"]
    d["geometry"]=data["legacy_geometry"]
    base.prepare_baseline(data,seed)
    if data["stage"]=="recovered":
        d["geometry"]=data["updated_geometry"]
        d["valid"]=d["geometry"][:,3]>0
    else:
        d["valid"]=(d["geometry"][:,3:6]>0).all(1)


def reader(stage,arm):
    cls=SlotReader if stage=="bounded" else OptionalGeometryReader
    return cls(arm.split("_")[0]).to(base.DEVICE)


def predict(model,data,item,positions=None,raw=None):
    if positions is None:
        positions=torch.arange(len(item["q"]),device=base.DEVICE)
    rows=torch.cat((item["s"],item["q"][positions]))
    values=base.descriptor(model,data["tensors"],rows) if raw is None else raw[rows]
    return bounded_predict(item,positions,values,data["tensors"]["valid"][rows],_coefficients)


@torch.no_grad()
def evaluate(model,data,split,details=False):
    d=data["tensors"]
    raw=base.raw_descriptors(model,data) if model is not None else None
    rows=[]; frames=[]
    for ti,item in enumerate(data["cache"][split]):
        prob,kernel=(item["old_prob"],item["qs"]) if model is None else predict(model,data,item,raw=raw)[:2]
        p=prob.cpu().numpy(); y=d["labels"][item["q"]].cpu().numpy()
        enabled=(d["valid"][item["q"]]&(d["valid"][item["s"]].sum()>=2)).cpu().numpy()
        common={k:item["t"][k] for k in ("port","acquisition","origin","budget","draw","fold")}
        common["task"]=ti
        for scope,keep in (("full",np.ones(len(y),bool)),("geometry",enabled)):
            if keep.any():
                rows.append(dict(**common,scope=scope,**base.old.task_metric(p[keep],y[keep]),n=int(keep.sum())))
        if not details:
            continue
        f=data["meta"].iloc[item["q"].cpu().numpy()][["sample_id","global_row","mmsi","class_id"]].reset_index(drop=True)
        for k,v in common.items():
            f[k]=v
        f["truth"],f["pred"],f["enabled"]=y,p.argmax(1),enabled
        f["correct"]=f.truth.eq(f.pred)
        f["valid_supports"]=int(d["valid"][item["s"]].sum())
        f["nll"]=-np.log(p[np.arange(len(y)),y].clip(1e-12))
        f["brier"]=((p-np.eye(8)[y])**2).sum(1)
        sy=item["sy"].cpu().numpy(); k=kernel.cpu().numpy().copy()
        k[~enabled]=item["qs"].cpu().numpy()[~enabled]
        ap=[]
        for i,cls in enumerate(y):
            hits=sy==cls; n=int(hits.sum())
            if n:
                r=hits[np.argsort(-k[i],kind="stable")[:n]]
                ap.append(float((np.cumsum(r)/np.arange(1,n+1)*r).sum()/n))
            else:
                ap.append(np.nan)
        f["ap"]=ap
        for j in range(8):
            f[f"p{j}"]=p[:,j]
        frames.append(f)
    return pd.DataFrame(rows),pd.concat(frames,ignore_index=True) if frames else None


def sample_step(data,groups,rng,paired):
    ports=list(groups); port=ports[int(rng.integers(len(ports)))]
    scenes=list(groups[port]); scene=scenes[int(rng.integers(len(scenes)))]
    ti=int(rng.choice(groups[port][scene])); item=data["cache"]["fit"][ti]
    d=data["tensors"]
    available=torch.where(d["valid"][item["q"]])[0].cpu().numpy()
    if not paired:
        pos=available if len(available)<=32 else rng.choice(available,32,replace=False)
        return [(ti,pos)]
    qids=set(item["q"][available].cpu().tolist())
    support_ids=set(item["s"].cpu().tolist())
    partners=[]
    for idx in groups[port][scene]:
        other=data["cache"]["fit"][idx]
        if idx==ti or other["t"]["support_acquisition"]!=item["t"]["support_acquisition"] or other["t"]["draw"]==item["t"]["draw"]:
            continue
        common=sorted(qids & set(other["q"][d["valid"][other["q"]]].cpu().tolist()))
        if common and support_ids!=set(other["s"].cpu().tolist()):
            partners.append((idx,common))
    if not partners:
        # Exact same task twice would create no additional support diversity.
        pos=available if len(available)<=32 else rng.choice(available,32,replace=False)
        return [(ti,pos)]
    oi,common=partners[int(rng.integers(len(partners)))]
    chosen=common if len(common)<=32 else rng.choice(common,32,replace=False).tolist()
    pairs=[]
    for idx in (ti,oi):
        q=data["cache"]["fit"][idx]["q"].cpu().tolist()
        lookup={v:j for j,v in enumerate(q)}
        pairs.append((idx,np.array([lookup[v] for v in chosen],dtype=np.int64)))
    return pairs


def train(data,stage,arm,seed):
    out=HERE/stage/"trained"/data["port"].replace(" ","_")
    out.mkdir(parents=True,exist_ok=True); path=out/f"{arm}_seed{seed}.pt"
    if path.exists():
        return
    eligible,groups,fitrows,inventory=base.fit_inventory(data)
    dump(out/"fit_inventory.json",inventory)
    if not eligible:
        torch.save(dict(unavailable=True,inventory=inventory),path); return
    torch.manual_seed(20261016+seed)
    model=reader(stage,arm); d=data["tensors"]
    with torch.no_grad():
        raw=base.descriptor(model,d,fitrows)
        energy=(raw-raw.mean(0)).square().sum(1).mean()
        assert energy>1e-10
        model.scale.copy_(energy.sqrt())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(20261016+seed)
    curve=[]; trace=[]; started=time.perf_counter(); paired=arm.endswith(("_pair","_rank")); ranking=arm.endswith("_rank")
    first_gradient=None; last_probability=None
    for step in range(201):
        if step in (0,50,100,200):
            model.eval(); entry=dict(step=step)
            for split in ("fit","selection"):
                table,_=evaluate(model,data,split)
                entry.update({split+"_"+k:v for k,v in base.summary(table).items()})
            with torch.no_grad():
                x=base.descriptor(model,d,fitrows)
                entry["raw_centered_energy"]=float((x-x.mean(0)).square().sum(1).mean())
            curve.append(entry)
            print("CURVE",stage,data["port"],arm,seed,step,round(entry["raw_centered_energy"],3),flush=True)
        if step==200:
            break
        model.train(); optimizer.zero_grad(set_to_none=True)
        sampled=sample_step(data,groups,rng,paired)
        if len(sampled)==1:
            sampled+=sample_step(data,groups,rng,False)
        logs=[]
        for ti,pos in sampled:
            item=data["cache"]["fit"][ti]
            positions=torch.as_tensor(pos,device=base.DEVICE)
            prob,_,h=predict(model,data,item,positions)
            y=d["labels"][item["q"][positions]]
            ce=base.old.loss_fn(prob,y)
            rel=h.sum()*0.; count=0
            if ranking:
                ids=torch.cat((item["s"],item["q"][positions]))
                rel,count=relation_loss(h,item["sy"],y,d["valid"][ids],item["old_prob"][positions])
            loss=ce+.1*rel
            assert torch.isfinite(loss)
            (loss/2).backward()
            logs.append((float(ce.detach()),float(rel.detach()),count,float(h.square().sum(-1).max().detach())))
            trace.append(dict(step=step+1,task=ti,query_positions=pos.tolist()))
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        if first_gradient is None:
            first_gradient=float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
            assert first_gradient>0
        optimizer.step()
        if step%25==0:
            curve.append(dict(step=step+1,train_ce=np.mean([r[0] for r in logs]),train_relation=np.mean([r[1] for r in logs]),
                              rank_queries=sum(r[2] for r in logs),max_feature_norm2=max(r[3] for r in logs)))
    torch.cuda.synchronize()
    torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},stage=stage,arm=arm,seed=seed,
                    steps=200,inventory=inventory,seconds=time.perf_counter()-started,first_gradient=first_gradient),path)
    pd.DataFrame(curve).to_csv(out/f"{arm}_seed{seed}_curve.csv",index=False)
    dump(out/f"{arm}_seed{seed}_sampling.json",trace)
    print("TRAINED",stage,data["port"],arm,seed,flush=True)


def target(data,stage,seed):
    folder=HERE/stage/"evaluation"/data["port"].replace(" ","_")
    folder.mkdir(parents=True,exist_ok=True)
    for arm in ("B",*ARMS[stage]):
        file=folder/f"{arm}_seed{seed}_predictions.csv"
        if file.exists():
            continue
        model=None
        if arm!="B":
            saved=torch.load(HERE/stage/"trained"/data["port"].replace(" ","_")/f"{arm}_seed{seed}.pt",weights_only=False)
            if not saved.get("unavailable"):
                model=reader(stage,arm); model.load_state_dict(saved["state"]); model.eval()
        metrics,predictions=evaluate(model,data,"target",True)
        for frame in (metrics,predictions):
            frame["arm"],frame["seed"]=arm,seed
        metrics.to_csv(folder/f"{arm}_seed{seed}_metrics.csv",index=False)
        predictions.to_csv(file,index=False)
        print("TARGET",stage,data["port"],arm,seed,flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=list(ARMS)); p.add_argument("phase",choices=["train","target"]); p.add_argument("--port")
    a=p.parse_args(); torch.set_num_threads(2)
    if a.phase=="target":
        expected=[HERE/a.stage/"trained"/port.replace(" ","_")/f"{arm}_seed{seed}.pt" for port in base.PORTS for arm in ARMS[a.stage] for seed in range(3)]
        assert all(f.exists() for f in expected)
        dump(HERE/a.stage/"SOURCE_FROZEN.json",dict(records=len(expected),steps=200,seeds=[0,1,2],arms=ARMS[a.stage]))
    for port in base.PORTS:
        if a.port and a.port!=port.replace(" ","_"):
            continue
        data=load(port,a.stage,a.phase=="target")
        for seed in range(3):
            prepare(data,seed)
            if a.phase=="target":
                target(data,a.stage,seed)
            else:
                for arm in ARMS[a.stage]:
                    train(data,a.stage,arm,seed)
        del data; torch.cuda.empty_cache()


if __name__=="__main__":
    main()
