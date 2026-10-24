"""Post-diagnostic source normalization control, without new spatial operators."""
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import train
from operators import Candidate, RegionHead, regional_raw

OUT = train.HERE/'conditioning_control'


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    curves,records,pending = [],[],[]
    summary_cache = None
    for seed in train.SEEDS:
        data = train.Data(seed)
        if summary_cache is None:
            m = Candidate('static_pool',data.ck,data.wproject,data.tau).cuda()
            _,summary_cache = train.predict(m,data,np.arange(len(data.table)),summaries=True)
            del m
        y = data.table.class_id.to_numpy(int)
        yt = torch.tensor(y,device='cuda')
        tr,va = data.tr,data.va
        counts = np.bincount(y[tr],minlength=8)
        weights = torch.tensor(.5+.5*len(tr)/(8*counts[y[tr]]),dtype=torch.float32,device='cuda')
        rest = torch.tensor(data.rest,device='cuda')
        rng = np.random.default_rng(seed)
        draws = rng.integers(len(tr),size=(100,64))
        for arm,desc in [('grid_refit',data.original_summary),('static_pool',summary_cache)]:
            torch.manual_seed(seed)
            raw = regional_raw(torch.tensor(data.s0),torch.tensor(desc,dtype=torch.float32),torch.tensor(data.quality)).numpy()
            fit = raw[tr].astype(np.float64)
            mu = fit.mean((0,1),keepdims=True).astype(np.float32)
            scale = np.maximum(fit.std((0,1),keepdims=True),1e-8).astype(np.float32)
            oldmu,oldscale = np.asarray(data.ck['mu'],dtype=np.float32),np.asarray(data.ck['scale'],dtype=np.float32)
            x = torch.tensor((raw-mu)/scale,device='cuda')
            head = RegionHead().cuda()
            head.load_state_dict(data.ck['state'])
            with torch.no_grad():
                expected = head(torch.tensor((raw[tr]-oldmu)/oldscale,device='cuda'))
                w = head.net[0].weight.clone()
                head.net[0].weight.mul_(torch.tensor((scale/oldscale).ravel(),device='cuda'))
                head.net[0].bias.add_(w@torch.tensor(((mu-oldmu)/oldscale).ravel(),device='cuda'))
                actual = head(x[tr])
                error = float((actual-expected).abs().max())
                assert error<5e-5,(seed,arm,error)
            best = train.rank(y[va],data.baseline[va])
            chosen,best_state = -1,{k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            optimizer = torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.01)
            for step in range(101):
                if step:
                    optimizer.zero_grad(set_to_none=True)
                    for first in range(0,64,8):
                        take = draws[step-1,first:first+8]
                        ids = tr[take]
                        s = rest[ids]+head(x[ids])
                        loss = (F.cross_entropy(s,yt[ids],reduction='none')*weights[take]).sum()/64
                        loss.backward()
                    optimizer.step()
                if step not in train.CHECKPOINTS:
                    continue
                with torch.no_grad():
                    s = (rest[va]+head(x[va])).cpu().numpy()
                curves.append(dict(seed=seed,arm=arm,step=step,**train.metrics(y[va],s)))
                value = train.rank(y[va],s)
                if value>best:
                    best,chosen = value,step
                    best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            head.load_state_dict(best_state)
            with torch.no_grad():
                score = data.baseline.copy() if chosen<0 else (rest+head(x)).cpu().numpy()
            pending.append((seed,arm,chosen,score,data.table.copy(),data.baseline.copy()))
            torch.save(dict(state=best_state,mu=mu,scale=scale,selected_step=chosen),OUT/f'{arm}_{seed}.pt')
            records.append(dict(seed=seed,arm=arm,selected_step=chosen,initial_function_error=error))
            print('CONDITIONING',records[-1],flush=True)
            del head,x,optimizer
        del data
    rows = []
    for seed,arm,chosen,score,f,old in pending:
        np.savez_compressed(OUT/f'{arm}_{seed}.npz',score=score)
        for role in ('utility_train','selection','evaluation'):
            ii = np.flatnonzero(f.role.eq(role))
            yy = f.iloc[ii].class_id.to_numpy(int)
            rows.append(dict(seed=seed,arm=arm,role=role,selected_step=chosen,**train.metrics(yy,score[ii])))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'source_curves.csv',index=False)
    pd.DataFrame(records).to_csv(OUT/'receipts.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,fits=6,scale_fit='utility_train only',initialization='first affine map transformed to preserve original function; exact replay checked',scope='post-diagnostic optimization conditioning control, not operator novelty or fresh confirmation'),indent=2))
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA','BA']].mean(),flush=True)


if __name__=='__main__':
    main()
