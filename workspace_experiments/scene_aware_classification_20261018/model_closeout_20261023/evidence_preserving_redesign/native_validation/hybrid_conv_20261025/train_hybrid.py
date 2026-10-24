"""Source-selected five-arm hybrid convolution pilot. No target tuning."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import train_native as tr
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
hybrid = tr.load('hybrid_role_operators', HERE/'operators.py')
ARMS = ('moment', 'grid_role', 'axis_difference', 'axis_early', 'axis_role')


class Model(tr.Model):
    def __init__(self, arm, reader):
        # Create the same head BEFORE arm-specific random parameters.
        super().__init__('moment', reader)
        self.arm = arm
        if arm == 'axis_difference':
            self.conv = tr.ops.AxisPairConv(8)
        elif arm != 'moment':
            self.conv = hybrid.RolePairConv(tr.ops.sample_offset, axis=arm!='grid_role', early_mean=arm=='axis_early')


def checks(data, reader):
    ids = data.tr[:2]
    b = data.batch(ids)
    base = torch.tensor(data.baselines[17][ids], device='cuda')
    rows, desc0, head0 = [], None, None
    for arm in ARMS:
        torch.manual_seed(17)
        model = Model(arm, reader).cuda()
        with torch.no_grad():
            desc = model.describe(b)
            init = float((model(b, base)-base).abs().max())
        if desc0 is None:
            desc0, head0 = desc, model.head[0].weight.detach().clone()
        replay = float((desc-desc0).abs().max())
        assert init == 0 and replay == 0 and torch.equal(head0, model.head[0].weight)
        result = dict(arm=arm, initial_score_error=init, descriptor_error=replay,
                      parameters=sum(p.numel() for p in model.parameters()),
                      conv_parameters=0 if model.conv is None else sum(p.numel() for p in model.conv.parameters()))
        if model.conv is not None:
            # Nonzero output is essential: zero initialization cannot test symmetry.
            with torch.no_grad():
                model.conv.output.copy_(torch.eye(8,device='cuda')[None].expand(6,-1,-1))
            x=torch.randn(6,8,15,15,device='cuda')
            region=torch.rand(6,3,15,15,device='cuda')*.3
            valid=torch.ones(6,1,15,15,device='cuda')
            theta=torch.linspace(-1.2,1.2,6,device='cuda')
            tid=torch.arange(6,device='cuda')
            out=model.conv(x,region,valid,theta,tid)
            reverse=model.conv(x,region,valid,theta+torch.pi,tid)
            err=float((out-reverse).abs().max().detach())
            assert err < 2e-5
            modified=x.clone(); modified[3:]*=3
            independent=model.conv(modified,region,valid,theta,tid)
            isolation=float((out[:3]-independent[:3]).abs().max().detach())
            assert isolation == 0
            absent=model.conv(x,region,torch.zeros_like(valid),theta,tid)
            assert torch.equal(absent,x)
            out.square().mean().backward()
            grad=float(model.conv.gate[0].weight.grad.norm())
            assert grad > 0 and np.isfinite(grad)
            result.update(halfturn_error=err, polarization_isolation=isolation, gate_gradient=grad)
        rows.append(result)
        del model
    return rows


def fit(data, reader, arm, seed, raw_stats):
    tick=time.perf_counter();torch.manual_seed(seed)
    baseline=data.baselines[seed]
    model=Model(arm,reader).cuda()
    model.raw_mu.copy_(torch.tensor(raw_stats[0],device='cuda'))
    model.raw_scale.copy_(torch.tensor(raw_stats[1],device='cuda'))
    y=data.table.class_id.to_numpy(int);ytr=y[data.tr]
    count=np.bincount(ytr,minlength=8)
    weights=.5+.5*len(ytr)/(8*count[ytr])
    rng=np.random.default_rng(seed)
    draws=rng.integers(len(data.tr),size=(80,32));rotations=rng.integers(4,size=(80,8))
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    best=tr.metrics(y[data.va],baseline[data.va])['J'];chosen=0
    beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    curves=[dict(seed=seed,arm=arm,step=0,loss=None,**tr.metrics(y[data.va],baseline[data.va]))]
    gradients=[]
    for step in range(1,81):
        opt.zero_grad(set_to_none=True);loss_total=0.
        for micro in range(8):
            take=draws[step-1,micro*4:(micro+1)*4];ids=data.tr[take]
            base=torch.tensor(baseline[ids],device='cuda')
            out=model(data.batch(ids,int(rotations[step-1,micro])),base)
            loss=(F.cross_entropy(out,torch.tensor(ytr[take],device='cuda'),reduction='none')
                  *torch.tensor(weights[take],device='cuda')).sum()/32
            assert torch.isfinite(loss)
            loss.backward();loss_total+=float(loss.detach())
        if step in (1,2,3,20,80):
            gradients.append(dict(step=step,norms={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}))
        opt.step()
        if step in tr.CHECKPOINTS:
            score=tr.predict(model,data,data.va,baseline)
            m=tr.metrics(y[data.va],score)
            if m['J']>best+1e-12:
                best=m['J'];chosen=step
                beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            curves.append(dict(seed=seed,arm=arm,step=step,loss=loss_total,**m))
            print('FIT',arm,seed,step,'J',m['J'],'chosen',chosen,'seconds',round(time.perf_counter()-tick,1),flush=True)
    receipt=dict(arm=arm,seed=seed,selected_step=chosen,seconds=time.perf_counter()-tick,
                 parameters=sum(p.numel() for p in model.parameters()),gradients=gradients)
    torch.save(dict(state=beststate,receipt=receipt),HERE/'models'/f'{arm}_{seed}.pt')
    pd.DataFrame(curves).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv',index=False)
    return receipt


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for name in ('models','results'):(HERE/name).mkdir(exist_ok=True)
    data=tr.Data();reader=dict(np.load(tr.HERE/'reader.npz'))
    check=checks(data,reader)
    (HERE/'checks.json').write_text(json.dumps(check,indent=2))
    print('CHECKS',check,flush=True)
    receipts=[]
    for seed in tr.SEEDS:
        torch.manual_seed(seed)
        initial=Model('moment',reader).cuda()
        stats=tr.head_stats(initial,data,data.baselines[seed]);del initial
        for arm in ARMS:
            path=HERE/'models'/f'{arm}_{seed}.pt'
            receipts.append(torch.load(path,map_location='cpu',weights_only=False)['receipt'] if path.exists() else fit(data,reader,arm,seed,stats))
            (HERE/'training_receipts.json').write_text(json.dumps(receipts,indent=2))
    # Do not inspect development outcomes until every source checkpoint is selected.
    rows=[];y=data.table.class_id.to_numpy(int)
    for seed in tr.SEEDS:
        for arm in ('old_full',*ARMS):
            if arm=='old_full':score=data.baselines[seed]
            else:
                ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
                model=Model(arm,reader).cuda();model.load_state_dict(ck['state'])
                score=data.baselines[seed].copy() if ck['receipt']['selected_step']==0 else tr.predict(model,data,np.arange(data.n),data.baselines[seed])
                del model
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',score=score,up_index=data.up)
            for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
                rows.append(dict(seed=seed,arm=arm,role=role,**tr.metrics(y[ids],score[ids])))
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv',index=False)
    (HERE/'complete.json').write_text(json.dumps(dict(fits=15,checks=check,stage='matched hybrid M2; old full/M1/M3 frozen'),indent=2))
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean(),flush=True)


if __name__=='__main__':main()
