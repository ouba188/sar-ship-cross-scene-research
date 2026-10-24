"""Matched source-only adaptation of the six native M2 class projections."""
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
from torch import nn
from torch.nn import functional as F

ARMS = ('moment', 'axis_moment')


class Model(tr.Model):
    def __init__(self, arm, reader):
        super().__init__(arm, reader)
        for name in ('W', 'bias'):
            value = getattr(self, name).clone()
            delattr(self, name)
            setattr(self, name, nn.Parameter(value))


def checks(data, reader):
    rows = []
    ids = data.tr[:2]
    baseline = torch.tensor(data.baselines[17][ids], device='cuda')
    for arm in ARMS:
        torch.manual_seed(17)
        old = tr.Model(arm, reader).cuda()
        torch.manual_seed(17)
        new = Model(arm, reader).cuda()
        b = data.batch(ids)
        with torch.no_grad():
            de = float((old.describe(b)-new.describe(b)).abs().max())
            oe = float((new(b, baseline)-baseline).abs().max())
        he = float((old.head[0].weight-new.head[0].weight).abs().max().detach())
        assert de == 0 and oe == 0 and he == 0
        assert {'W', 'bias'} <= dict(new.named_parameters()).keys()
        assert all(not getattr(new, name).requires_grad for name in ('mu', 'scale', 'tau', 'raw_mu', 'raw_scale'))
        # A nonconstant objective verifies the newly movable response projection.
        new.describe(b).square().mean().backward()
        grad = float(new.W.grad.norm())
        assert grad > 0 and np.isfinite(grad)
        rows.append(dict(arm=arm,descriptor_replay=de,initial_score_replay=oe,head_initialization=he,
                         projection_gradient=grad,parameters=sum(p.numel() for p in new.parameters())))
        del old, new
    return rows


def fit(data, reader, arm, seed):
    tick = time.perf_counter()
    torch.manual_seed(seed)
    baseline = data.baselines[seed]
    model = Model(arm, reader).cuda()
    mu, scale = tr.head_stats(model, data, baseline)
    model.raw_mu.copy_(torch.tensor(mu, device='cuda'))
    model.raw_scale.copy_(torch.tensor(scale, device='cuda'))
    y = data.table.class_id.to_numpy(int)
    ytr = y[data.tr]
    count = np.bincount(ytr, minlength=8)
    weights = .5+.5*len(ytr)/(8*count[ytr])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(data.tr), size=(80,32))
    rotations = rng.integers(4, size=(80,8))
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    best = tr.metrics(y[data.va], baseline[data.va])['J']
    chosen = 0
    beststate = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    curves = [dict(seed=seed,arm=arm,step=0,loss=None,**tr.metrics(y[data.va],baseline[data.va]))]
    gradlog = []
    for step in range(1,81):
        opt.zero_grad(set_to_none=True)
        loss_total = 0.
        for micro in range(8):
            take = draws[step-1,micro*4:(micro+1)*4]
            ids = data.tr[take]
            base = torch.tensor(baseline[ids], device='cuda')
            out = model(data.batch(ids,int(rotations[step-1,micro])),base)
            loss = (F.cross_entropy(out,torch.tensor(ytr[take],device='cuda'),reduction='none')
                    *torch.tensor(weights[take],device='cuda')).sum()/32
            assert torch.isfinite(loss)
            loss.backward()
            loss_total += float(loss.detach())
        if step <= 3:
            gradlog.append(dict(step=step,gradients={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}))
        opt.step()
        if step in tr.CHECKPOINTS:
            score = tr.predict(model,data,data.va,baseline)
            m = tr.metrics(y[data.va],score)
            if m['J'] > best+1e-12:
                best = m['J']
                chosen = step
                beststate = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            curves.append(dict(seed=seed,arm=arm,step=step,loss=loss_total,**m))
            print('FIT',arm,seed,step,'J',m['J'],'chosen',chosen,'seconds',round(time.perf_counter()-tick,1),flush=True)
    drift = {name:float((model.state_dict()[name].cpu()-torch.tensor(reader[name])).norm()) for name in ('W','bias')}
    selected_drift = {name:float((beststate[name]-torch.tensor(reader[name])).norm()) for name in ('W','bias')}
    assert drift['W'] > 0
    receipt = dict(arm=arm,seed=seed,selected_step=chosen,seconds=time.perf_counter()-tick,
                   parameters=sum(p.numel() for p in model.parameters()),gradients=gradlog,
                   terminal_projection_drift=drift,selected_projection_drift=selected_drift)
    torch.save(dict(state=beststate,receipt=receipt),HERE/'models'/f'{arm}_{seed}.pt')
    pd.DataFrame(curves).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv',index=False)
    return receipt


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for folder in ('models','results'):
        (HERE/folder).mkdir(exist_ok=True)
    data = tr.Data()
    reader = dict(np.load(tr.HERE/'reader.npz'))
    result = checks(data, reader)
    (HERE/'checks.json').write_text(json.dumps(result,indent=2))
    print('CHECKS',result,flush=True)
    receipts = []
    for seed in tr.SEEDS:
        for arm in ARMS:
            path = HERE/'models'/f'{arm}_{seed}.pt'
            receipts.append(torch.load(path,map_location='cpu',weights_only=False)['receipt'] if path.exists() else fit(data,reader,arm,seed))
            (HERE/'training_receipts.json').write_text(json.dumps(receipts,indent=2))
    # Evaluation labels enter metrics only after all six source fits are selected.
    rows = []
    y = data.table.class_id.to_numpy(int)
    for seed in tr.SEEDS:
        for arm in ARMS:
            ck = torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
            model = Model(arm,reader).cuda()
            model.load_state_dict(ck['state'])
            score = data.baselines[seed].copy() if ck['receipt']['selected_step']==0 else tr.predict(model,data,np.arange(data.n),data.baselines[seed])
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',score=score,up_index=data.up)
            for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
                rows.append(dict(seed=seed,arm=arm,role=role,**tr.metrics(y[ids],score[ids])))
            del model
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv',index=False)
    (HERE/'complete.json').write_text(json.dumps(dict(fits=6,checks=result,stage='M2 projection adaptation; old M1/M3 frozen'),indent=2))
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean(),flush=True)


if __name__ == '__main__':
    main()
