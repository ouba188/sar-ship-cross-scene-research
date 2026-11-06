"""Paired development tuning; reuse the live M1 placement experiment."""
from pathlib import Path
import sys
import json
import time
import gc
import argparse
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'m1_design_position_20261101'
sys.path.insert(0, str(OLD))
import experiment as e
task = e.task

CONFIGS = {
    'extended': dict(gamma=.1, temperature=1., inner_lr=.0001, aux=.1),
    'gain': dict(gamma=.3, temperature=1., inner_lr=.0001, aux=.1),
    'sharp': dict(gamma=.3, temperature=.5, inner_lr=.0001, aux=.1),
    'task_lr': dict(gamma=.3, temperature=.5, inner_lr=.0003, aux=.02),
}


def rank(row):
    return (row['ba'], row['oa'], -row['nll'], -row['epoch'])


def readers():
    return torch.load(OLD/'source_reader_calibrated/readers.pt', map_location='cpu', weights_only=True)


def network(position, design, config):
    net = e.Network(position+'_'+design, readers())
    for op in net.m1:
        op.gamma.data.fill_(config['gamma'])
        op.temperature = config['temperature']
    return net


def weights(rows, train):
    hard = set(json.loads((task.ROOT/'source_difficulty.json').read_text())['hard'])
    frame = pd.DataFrame([rows[i] for i in train])
    value = np.zeros(len(rows), np.float32)
    value[train] = 1/(frame.groupby('label').mmsi.transform('nunique').to_numpy()*frame.groupby('mmsi').mmsi.transform('size').to_numpy())
    value[train] *= [2. if i in hard else 1. for i in train]
    value[train] /= value[train].mean()
    return torch.tensor(value, device=task.DEVICE)


def fit(position, design, name, seed, epochs, rows, train, hold, a, c, fixed_epoch=None):
    cfg = CONFIGS[name]
    dest = ROOT/f'{position}_{design}_{name}_{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'receipt.json').exists():
        receipt = json.loads((dest/'receipt.json').read_text())
        if receipt['parameters'] != cfg or receipt['epochs'] != epochs or receipt['fixed_epoch'] != fixed_epoch:
            raise ValueError(f'Existing run has different settings; preserve it and use a new run name: {dest}')
        return receipt
    torch.manual_seed(seed)
    net = network(position, design, cfg)
    params = {n:p for n,p in net.named_parameters() if p.requires_grad}
    groups = {}
    for n,p in params.items():
        lr = 1e-5 if n.startswith('base.') else (.001 if '.output.' in n or '.contrast_output.' in n else cfg['inner_lr'])
        groups.setdefault(lr, []).append(p)
    opt = torch.optim.AdamW([dict(params=p,lr=lr) for lr,p in groups.items()],weight_decay=.01)
    y = np.array([r['label'] for r in rows])
    weight = weights(rows,train)
    initial = e.predict(net,a,c,hold)
    np.savez_compressed(dest/'epoch0.npz',**initial)
    curve = [dict(epoch=0,**task.metrics(y[hold],initial['score']))]
    best = curve[0]
    torch.save(net.live_state(),dest/'best.pt')
    rng = np.random.default_rng(seed)
    orders, norms = [], []
    tick = time.perf_counter()
    for epoch in range(1,epochs+1):
        order = rng.permutation(train)
        orders.append(order.tolist())
        losses,auxs = [],[]
        for start in range(0,len(order),24):
            ids = order[start:start+24]
            labels = torch.tensor(y[ids],device=task.DEVICE)
            opt.zero_grad(set_to_none=True)
            score,aux,_ = net(a,c,ids)
            ce = (F.cross_entropy(score,labels,reduction='none')*weight[ids]).mean()
            av = task.tensor(a['available'],ids)
            aux_ce = F.cross_entropy(aux.flatten(0,1),labels[:,None].expand(-1,2).flatten(),reduction='none').reshape(len(ids),2)
            aux_loss = ((aux_ce*av).sum(1)/av.sum(1).clamp_min(1)*weight[ids]).mean()
            loss = ce+cfg['aux']*aux_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss: {dest.name}, epoch {epoch}')
            loss.backward()
            norms.append(float(torch.nn.utils.clip_grad_norm_(list(params.values()),5.)))
            opt.step()
            losses.append(float(ce.detach()))
            auxs.append(float(aux_loss.detach()))
        pred = e.predict(net,a,c,hold)
        np.savez_compressed(dest/f'epoch{epoch}.npz',**pred)
        row = dict(epoch=epoch,ce=float(np.mean(losses)),auxiliary_ce=float(np.mean(auxs)),
                   vv_update=float(pred['update_ratio'][:,0].mean()),vh_update=float(pred['update_ratio'][:,1].mean()),
                   **task.metrics(y[hold],pred['score']))
        curve.append(row)
        if (fixed_epoch is None and rank(row)>rank(best)) or epoch==fixed_epoch:
            best = row
            torch.save(net.live_state(),dest/'best.pt')
        pd.DataFrame(curve).to_csv(dest/'curve.csv',index=False)
        print('EPOCH',dest.name,epoch,row['oa'],row['ba'],row['nll'],flush=True)
    torch.save(net.live_state(),dest/'last.pt')
    net.restore(torch.load(dest/'best.pt',map_location='cpu',weights_only=True))
    for mode in (None,'off','uniform_context','spatial_mean','vv_only','vh_only'):
        if design=='mean_gate' and mode in ('uniform_context','spatial_mean'):
            continue
        np.savez_compressed(dest/('selected.npz' if mode is None else mode+'.npz'),**e.predict(net,a,c,hold,mode))
    source=e.predict(net,a,c,train)
    np.savez_compressed(dest/'source.npz',**source)
    selected=np.load(dest/'selected.npz')['score']
    expected=np.load(dest/f'epoch{best["epoch"]}.npz')['score']
    reload_error=float(abs(selected-expected).max())
    assert reload_error==0, reload_error
    receipt=dict(position=position,design=design,config=name,parameters=cfg,seed=seed,epochs=epochs,
                 selected=best,last=curve[-1],selection='development BA/OA/NLL' if fixed_epoch is None else 'frozen epoch from seed17',
                 fixed_epoch=fixed_epoch,source=task.metrics(y[train],source['score']),
                 trainable=sum(p.numel() for p in params.values()),steps=len(orders)*30,seconds=time.perf_counter()-tick,
                 reload_error=reload_error,clipped_steps=sum(v>5 for v in norms),max_gradient_norm=max(norms,default=0.))
    task.write(dest/'orders.json',orders)
    task.write(dest/'receipt.json',receipt)
    print('DONE',dest.name,best,flush=True)
    del net,opt
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def summarize():
    rows=[]
    for path in ROOT.glob('*/receipt.json'):
        r=json.loads(path.read_text())
        rows.append(dict(run=path.parent.name,position=r['position'],design=r['design'],config=r['config'],seed=r['seed'],
                         **r['selected'],last_oa=r['last']['oa'],last_ba=r['last']['ba'],seconds=r['seconds']))
    df=pd.DataFrame(rows)
    df.to_csv(ROOT/'summary.csv',index=False)
    if len(df):
        print(df[['run','epoch','oa','ba','nll','last_oa','last_ba']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--configs',nargs='+',default=list(CONFIGS))
    parser.add_argument('--positions',nargs='+',default=['pre','post'])
    parser.add_argument('--designs',nargs='+',default=['class_gate','mean_gate'])
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--fixed-epoch',type=int)
    args=parser.parse_args()
    task.current.setup(task.DEVICE)
    rows,train,hold,rid=task.records()
    a,c=task.arrays(),e.caches()
    if not (ROOT/'checks.json').exists():
        ids=np.r_[train[:2],hold[:2]]
        baseline=e.predict(e.Network('none'),a,c,ids)['score']
        errors={}
        for position in args.positions:
            net=network(position,'class_gate',CONFIGS['task_lr'])
            errors[position]=float(abs(e.predict(net,a,c,ids)['score']-baseline).max())
            assert errors[position]<1e-5
            net.design='mean_gate'
            for op in net.m1:
                torch.nn.init.normal_(op.output.weight,std=.01)
            first=e.predict(net,a,c,ids)['score']
            for op in net.m1:op.temperature=1.
            second=e.predict(net,a,c,ids)['score']
            errors[position+'_mean_temperature']=float(abs(first-second).max())
            assert errors[position+'_mean_temperature']==0
            del net
        task.write(ROOT/'checks.json',dict(errors=errors,source=len(train),development=len(hold),references=len(rid)))
    for position in args.positions:
        for name in args.configs:
            for design in args.designs:
                actual_name = 'gain' if name=='sharp' and design=='mean_gate' else name
                fit(position,design,actual_name,args.seed,args.epochs,rows,train,hold,a,c,args.fixed_epoch)
                summarize()


if __name__=='__main__':main()
