"""Selected added-head forward timing on already prepared cache inputs."""
import time
import json
import platform
import numpy as np
import pandas as pd
import torch
from run_increment import HERE,parent,ref,calibrated,inputs,restore,ARMS,SEED_PAIRS


def main():
    torch.set_num_threads(2)
    table=parent.load_table()
    cache=dict(np.load(parent.HERE/'signed_features.npz'))
    valid=cache['valid'].astype(bool)
    rows=[]
    for fold in range(3):
        d,_,_=parent.load_fold(fold,table,cache)
        outer,_=ref.load_pol(fold,table,cache,d)
        ix=d['test']
        for oldseed,seed in SEED_PAIRS:
            base,_,_=calibrated(fold,oldseed,seed,table,d)
            for arm in ARMS:
                ck,model=restore(HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt')
                x,carrier,_,_=inputs(arm,base[ix],outer[ix],valid[ix],ck['mu'],ck['scale'])
                elapsed=[]
                with torch.inference_mode():
                    for _ in range(5):model(x,carrier)
                    for _ in range(30):
                        tick=time.perf_counter();output=model(x,carrier);elapsed.append(time.perf_counter()-tick)
                assert torch.isfinite(output).all()
                rows.append(dict(fold=fold,seed=seed,arm=arm,batch_n=len(ix),repeats=30,warmup=5,
                    median_batch_ms=float(np.median(elapsed)*1000),p10_batch_ms=float(np.quantile(elapsed,.1)*1000),
                    p90_batch_ms=float(np.quantile(elapsed,.9)*1000)))
    frame=pd.DataFrame(rows)
    frame.to_csv(HERE/'inference_head_cost.csv',index=False)
    pooled=frame.groupby(['arm','seed'],as_index=False).agg(test_n=('batch_n','sum'),summed_fold_median_ms=('median_batch_ms','sum'))
    pooled['amortized_us_per_image']=pooled.summed_fold_median_ms*1000/pooled.test_n
    pooled.to_csv(HERE/'inference_head_cost_pooled.csv',index=False)
    (HERE/'inference_cost_scope.json').write_text(json.dumps(dict(device='CPU',threads=2,processor=platform.processor(),
        warmup=5,repeats=30,included='Added selected residual head only, prepared float32 inputs, batch1551-1632.',
        excluded='Old AD, frozen T, extraction of R50/C4/pol moments, I/O, descriptor/scaler, tensor conversion, score addition.',
        end_to_end_measured=False,single_image_latency_measured=False),indent=2),encoding='utf-8')
    print(pooled.to_string(index=False))


if __name__=='__main__':main()
