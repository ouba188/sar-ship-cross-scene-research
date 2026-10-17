"""Evaluate frozen selected and 600-step models on newly frozen scene tasks."""

import json
import os
from pathlib import Path
import time
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="2"

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from body_context_gate import Gate
from run_body_context import FIELDS, evaluate, packed_task, check_real_parity, predict

HERE=Path(__file__).resolve().parent
OUT=HERE/"body_context_expansion"


def checkpoint(slug, mode, seed, version):
    root=HERE/"body_context"/slug
    if version=="selected":
        return root/f"{mode}_seed{seed}.pt"
    if seed==0:
        return OUT/"recovered_tails"/slug/f"{mode}_seed0_last.pt"
    return root/f"{mode}_seed{seed}_last.pt"


def benchmark(task, gate):
    result={}
    with torch.no_grad():
        for arm in ("original","ring","conditional_600"):
            values=[]
            for repeat in range(25):
                start=time.perf_counter()
                local=None if arm=="original" else (task["ring"] if arm=="ring" else gate(task["ship"],task["ring"]))
                predict(task,local)
                if repeat>=5:
                    values.append((time.perf_counter()-start)*1000)
            result[arm]=float(np.median(values))
    return dict(query=len(task["q"]),support=len(task["s"]),median_ms=result,
                scope="cached tensors CPU readout; excludes image encoding and ring extraction")


def main():
    torch.set_num_threads(2)
    frozen=json.loads((OUT/"frozen.json").read_text(encoding="utf-8"))
    with threadpool_limits(limits=2):
        for pool in frozen["pools"]:
            if not pool["selected"]:
                continue
            slug=pool["port"].replace(" ","_")
            folder=OUT/slug
            assert (folder/"prepared.json").exists()
            with np.load(folder/"readout.npz") as cache:
                arrays={k:cache[k] for k in (*FIELDS,"weights")}
            meta=pd.read_csv(folder/"metadata.csv",dtype={"mmsi":str})
            all_tasks=json.loads((folder/"tasks.json").read_text(encoding="utf-8"))
            for budget in sorted({t["budget"] for t in all_tasks}):
                if (folder/f"done_B{budget}.json").exists():
                    print("evaluated cache",slug,budget,flush=True)
                    continue
                tasks=[t for t in all_tasks if t["budget"]==budget]
                packed=[packed_task(arrays,t) for t in tasks]
                check_real_parity(arrays,tasks[0],packed[0])
                records,predictions=[],[]
                started=time.perf_counter()
                for arm in ("original","ring","additive"):
                    table,pred=evaluate(arrays,tasks,packed,meta,arm,predictions=True)
                    records.append(table)
                    predictions.append(pred)
                for mode in ("static","conditional"):
                    for seed in range(3):
                        for version in ("selected","600"):
                            path=checkpoint(slug,mode,seed,version)
                            saved=torch.load(path,weights_only=False)
                            gate=Gate(mode).double()
                            gate.load_state_dict(saved["state"])
                            arm=f"{mode}_{version}"
                            table,pred=evaluate(arrays,tasks,packed,meta,arm,gate,seed,predictions=True)
                            records.append(table)
                            predictions.append(pred)
                            if mode=="conditional" and version=="600" and budget==16:
                                for shuffle in range(3):
                                    table,pred=evaluate(arrays,tasks,packed,meta,arm,gate,seed,
                                                       shuffle=shuffle,predictions=True)
                                    records.append(table)
                                    predictions.append(pred)
                pd.concat(records,ignore_index=True).assign(budget=budget).to_csv(folder/f"metrics_B{budget}.csv",index=False)
                pd.concat(predictions,ignore_index=True).assign(budget=budget).to_csv(folder/f"predictions_B{budget}.csv",index=False)
                (folder/f"done_B{budget}.json").write_text(json.dumps(dict(tasks=len(tasks),
                    seconds=time.perf_counter()-started,benchmark=benchmark(packed[0],gate)),indent=2))
                print("evaluated",slug,"B",budget,"tasks",len(tasks),"seconds",round(time.perf_counter()-started,1),flush=True)


if __name__=="__main__":
    main()
