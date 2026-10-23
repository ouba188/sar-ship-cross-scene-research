"""Time selected small-head forwards only, never fit or select a model."""
import time
import json
import platform
import numpy as np
import pandas as pd
import torch
from run_integration import HERE, parent, ref, inputs, restore, load_old, SEED_PAIRS, ARMS


def main():
    torch.set_num_threads(2)
    table=parent.load_table()
    cache=dict(np.load(parent.HERE/'signed_features.npz'))
    valid=cache['valid'].astype(bool)
    results=[]
    for fold in range(3):
        d,_,_=parent.load_fold(fold,table,cache)
        outer,_=ref.load_pol(fold,table,cache,d)
        ix=d['test']
        for oldseed,seed in SEED_PAIRS:
            base,_=load_old(fold,oldseed,table,d)
            for arm in ARMS:
                ck,model=restore(HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt')
                x,carrier,_,_=inputs(arm,base[ix],outer[ix],valid[ix],ck['mu'],ck['scale'])
                elapsed=[]
                with torch.inference_mode():
                    for _ in range(5):model(x,carrier)
                    for _ in range(30):
                        start=time.perf_counter()
                        result=model(x,carrier)
                        elapsed.append(time.perf_counter()-start)
                assert torch.isfinite(result).all()
                results.append(dict(fold=fold,oldseed=oldseed,seed=seed,arm=arm,batch_n=len(ix),
                    repeats=30,warmup=5,median_batch_ms=float(np.median(elapsed)*1000),
                    p10_batch_ms=float(np.quantile(elapsed,.1)*1000),p90_batch_ms=float(np.quantile(elapsed,.9)*1000),
                    amortized_us_per_image=float(np.median(elapsed)*1e6/len(ix))))
    frame=pd.DataFrame(results)
    frame.to_csv(HERE/'inference_head_cost.csv',index=False)
    totals=frame.groupby(['arm','seed'],as_index=False).agg(test_n=('batch_n','sum'),
        summed_fold_median_ms=('median_batch_ms','sum'))
    totals['amortized_us_per_image']=totals.summed_fold_median_ms*1000/totals.test_n
    totals.to_csv(HERE/'inference_head_cost_pooled.csv',index=False)
    (HERE/'inference_cost_scope.json').write_text(json.dumps(dict(device='CPU',torch_threads=2,
        processor=platform.processor(),torch_version=torch.__version__,warmup=5,repeats=30,
        included='Selected head forward only on prepared float32 tensors, batch 1551-1632.',
        excluded='I/O, old A+D inference, R50/C4/footprint extraction, sigma/kappa, descriptor/scaler, tensor conversion, score addition, training.',
        end_to_end_measured=False,single_image_latency_measured=False),indent=2),encoding='utf-8')
    print(totals.to_string(index=False))


if __name__=='__main__':main()
