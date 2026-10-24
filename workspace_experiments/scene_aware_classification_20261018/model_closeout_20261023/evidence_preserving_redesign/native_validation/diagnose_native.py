"""Frozen-head component interventions; not replacement-method training."""
import json
import time
import numpy as np
import pandas as pd
import torch
import train_native as tr

@torch.inference_mode()
def inference(model,data,ids,baseline,drop_frequency=False):
    all_scores=[]
    for first in range(0,len(ids),8):
        ix=ids[first:first+8];base=torch.tensor(baseline[ix],device='cuda');scores=[]
        for k in range(4):
            b=data.batch(ix,k)
            if drop_frequency:b['x'][:,:,7]=model.mu[None,:,7,None,None]
            scores.append(model(b,base))
        all_scores.append(torch.stack(scores).mean(0).cpu().numpy())
    return np.concatenate(all_scores)

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=tr.Data();reader=dict(np.load(tr.HERE/'reader.npz'));y=data.table.iloc[data.te].class_id.to_numpy(int)
    rows=[];cost=[]
    for seed in tr.SEEDS:
        for arm in ('moment','grid_moment','axis_moment'):
            ck=torch.load(tr.HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
            model=tr.Model(arm,reader).cuda();model.load_state_dict(ck['state'])
            full=np.load(tr.HERE/'results'/f'{arm}_{seed}.npz')['score'][data.te]
            variants=['native_frequency_source_mean']+(['conv_zero'] if model.conv is not None else [])
            for variant in variants:
                if variant=='conv_zero':
                    w=model.conv.output if arm=='axis_moment' else model.conv.out
                    saved=w.detach().clone()
                    with torch.no_grad():w.zero_()
                torch.cuda.synchronize();tick=time.perf_counter()
                score=inference(model,data,data.te,data.baselines[seed],variant=='native_frequency_source_mean')
                torch.cuda.synchronize();seconds=time.perf_counter()-tick
                if variant=='conv_zero':
                    with torch.no_grad():w.copy_(saved)
                p=full.argmax(1);q=score.argmax(1)
                rows.append(dict(seed=seed,arm=arm,intervention=variant,changed=int((p!=q).sum()),
                                 full_rescues=int(((p==y)&(q!=y)).sum()),full_harms=int(((p!=y)&(q==y)).sum()),
                                 score_max_change=float(abs(full-score).max()),**tr.metrics(y,score)))
                np.savez_compressed(tr.HERE/'results'/f'diagnostic_{arm}_{variant}_{seed}.npz',score=score,indices=data.te)
                cost.append(dict(seed=seed,arm=arm,variant=variant,n=len(data.te),seconds=seconds,ms_per_query=1000*seconds/len(data.te),note='incremental branch C4, warm local cache IO; not full pipeline latency'))
            del model
    pd.DataFrame(rows).to_csv(tr.HERE/'component_interventions.csv',index=False)
    pd.DataFrame(cost).to_csv(tr.HERE/'incremental_cost.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
