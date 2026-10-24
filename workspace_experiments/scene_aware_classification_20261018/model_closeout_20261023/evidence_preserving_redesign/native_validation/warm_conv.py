"""Root-cause follow-up: fixed source-selected moment head, train only convolution."""
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import train_native as tr

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=tr.Data();reader=dict(np.load(tr.HERE/'reader.npz'));y=data.table.class_id.to_numpy(int)
    receipts=[];rows=[]
    for seed in tr.SEEDS:
        parent=torch.load(tr.HERE/'models'/f'moment_{seed}.pt',map_location='cpu',weights_only=False)
        parent_score=np.load(tr.HERE/'results'/f'moment_{seed}.npz')['score']
        for arm in ('grid_moment','axis_moment'):
            path=tr.HERE/'models'/f'warm_{arm}_{seed}.pt'
            if path.exists():
                receipts.append(torch.load(path,map_location='cpu',weights_only=False)['receipt']);continue
            tick=time.perf_counter();torch.manual_seed(seed)
            model=tr.Model(arm,reader).cuda();model.load_state_dict(parent['state'],strict=False)
            model.head.requires_grad_(False)
            check=tr.predict(model,data,data.tr[:4],data.baselines[seed])
            error=float(abs(check-parent_score[data.tr[:4]]).max());assert error<3e-5
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};chosen=0
            best=tr.metrics(y[data.va],parent_score[data.va])['J']
            rng=np.random.default_rng(seed);draws=rng.integers(len(data.tr),size=(80,32));rot=rng.integers(4,size=(80,8))
            count=np.bincount(y[data.tr],minlength=8);weight=.5+.5*len(data.tr)/(8*count[y[data.tr]])
            opt=torch.optim.AdamW(model.conv.parameters(),lr=.003,weight_decay=.01)
            curve=[];gradients=[]
            for step in range(1,81):
                opt.zero_grad(set_to_none=True);loss_total=0
                for micro in range(8):
                    take=draws[step-1,micro*4:(micro+1)*4];ids=data.tr[take]
                    out=model(data.batch(ids,int(rot[step-1,micro])),torch.tensor(data.baselines[seed][ids],device='cuda'))
                    loss=(F.cross_entropy(out,torch.tensor(y[ids],device='cuda'),reduction='none')*torch.tensor(weight[take],device='cuda')).sum()/32
                    assert torch.isfinite(loss);loss.backward();loss_total+=float(loss.detach())
                if step<=3:gradients.append(dict(step=step,norms={n:float(p.grad.norm()) for n,p in model.conv.named_parameters() if p.grad is not None}))
                opt.step()
                if step in tr.CHECKPOINTS:
                    s=tr.predict(model,data,data.va,data.baselines[seed]);m=tr.metrics(y[data.va],s)
                    curve.append(dict(step=step,loss=loss_total,**m))
                    if m['J']>best+1e-12:
                        best=m['J'];chosen=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    print('WARM',arm,seed,step,'J',m['J'],'selected',chosen,flush=True)
            receipt=dict(seed=seed,arm='warm_'+arm,selected_step=chosen,seconds=time.perf_counter()-tick,parent_step=parent['receipt']['selected_step'],initial_replay_error=error,gradients=gradients)
            torch.save(dict(state=state,receipt=receipt),path);receipts.append(receipt)
            pd.DataFrame(curve).to_csv(tr.HERE/'results'/f'curve_warm_{arm}_{seed}.csv',index=False)
            del model
    (tr.HERE/'warm_receipts.json').write_text(json.dumps(receipts,indent=2))
    changes=[]
    for seed in tr.SEEDS:
        parent_score=np.load(tr.HERE/'results'/f'moment_{seed}.npz')['score']
        for arm in ('grid_moment','axis_moment'):
            ck=torch.load(tr.HERE/'models'/f'warm_{arm}_{seed}.pt',map_location='cpu',weights_only=False)
            model=tr.Model(arm,reader).cuda();model.load_state_dict(ck['state'])
            score=parent_score.copy() if ck['receipt']['selected_step']==0 else tr.predict(model,data,np.arange(data.n),data.baselines[seed])
            np.savez_compressed(tr.HERE/'results'/f'warm_{arm}_{seed}.npz',score=score,up_index=data.up)
            for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
                rows.append(dict(seed=seed,arm='warm_'+arm,role=role,**tr.metrics(y[ids],score[ids])))
            p=score[data.te].argmax(1);q=parent_score[data.te].argmax(1);yy=y[data.te]
            changes.append(dict(seed=seed,arm='warm_'+arm,selected_step=ck['receipt']['selected_step'],changed=int((p!=q).sum()),rescued=int(((p==yy)&(q!=yy)).sum()),harmed=int(((p!=yy)&(q==yy)).sum())))
            del model
    pd.DataFrame(rows).to_csv(tr.HERE/'warm_metrics.csv',index=False);pd.DataFrame(changes).to_csv(tr.HERE/'warm_changes.csv',index=False)
    print(pd.DataFrame(changes).to_string(index=False),flush=True)
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA','BA']].mean(),flush=True)

if __name__=='__main__':main()
