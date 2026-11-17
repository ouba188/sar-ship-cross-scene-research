"""Matched lightweight residual readouts; hyperparameters selected inside source."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
ARMS = ('summary','detail_mean','detail_grid')
SEEDS = (17,29,43)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def metrics(y,z):
    pred = z.argmax(1)
    recall = [float((pred[y==c]==c).mean()) for c in np.unique(y)]
    f1 = []
    for c in range(z.shape[1]):
        tp = ((pred==c)&(y==c)).sum()
        f1.append(float(2*tp/max(1,(pred==c).sum()+(y==c).sum())))
    t = torch.from_numpy(z).double()
    return dict(oa=100*float((pred==y).mean()),ba=100*float(np.mean(recall)),
        macro_f1=100*float(np.mean(f1)),nll=float(F.cross_entropy(t,torch.from_numpy(y).long())))


def key(m):
    return m['ba'],m['oa'],-m['nll']


def split_source(data, meta):
    eligible = np.flatnonzero(data['active'])
    refs = meta['references']
    ref_pass = {r['passage'] for r in refs}
    ref_vessel = {r['vessel'] for r in refs}
    rows = meta['train']
    candidate = sorted({rows[i]['passage'] for i in eligible}-ref_pass)
    rng = np.random.default_rng(20261117)
    rng.shuffle(candidate)
    held = set()
    for p in candidate:
        held.add(p)
        n = sum(rows[i]['passage'] in held and rows[i]['vessel'] not in ref_vessel for i in eligible)
        if n >= .2*len(eligible): break
    valid = np.array([i for i in eligible if rows[i]['passage'] in held and rows[i]['vessel'] not in ref_vessel])
    vessels = {rows[i]['vessel'] for i in valid}
    fit = np.array([i for i in eligible if rows[i]['passage'] not in held and rows[i]['vessel'] not in vessels])
    assert not {rows[i]['passage'] for i in fit}&{rows[i]['passage'] for i in valid}
    assert not {rows[i]['vessel'] for i in fit}&vessels
    assert set(data['label'][fit]) == set(range(8)) and set(data['label'][valid]) == set(range(8))
    def counts(ix):
        return dict(n=len(ix),classes=np.bincount(data['label'][ix],minlength=8).tolist(),
            ports=len({rows[i]['port'] for i in ix}),passages=len({rows[i]['passage'] for i in ix}),
            vessels=len({rows[i]['vessel'] for i in ix}))
    write(ROOT/'source_split_local.json',dict(fit=fit.tolist(),inner=valid.tolist()))
    write(ROOT/'source_scope.json',dict(fit=counts(fit),inner=counts(valid),
        excluded_from_dual=len(eligible)-len(fit)-len(valid),source_dual=len(eligible),
        note='encoder has previously seen source data; split is only for readout selection'))
    return fit, valid


class Head(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d,32),nn.GELU(),nn.Linear(32,8))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self,x):
        return self.body(x)


def feature(data,arm):
    return np.concatenate((data['base'],data[arm]),1).astype(np.float32)


def normalize(x,mean,scale):
    return np.clip((x-mean)/scale,-5,5).astype(np.float32)


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    data = dict(np.load(ROOT/'train_local.npz'))
    meta = read(ROOT/'metadata_local.json')
    fit,inner = split_source(data,meta)
    y = data['label']
    baseline = data['logits']
    weights = data['weight'][fit].copy()
    weights /= weights.mean()
    outcomes = []
    tick = time.perf_counter()
    for arm in ARMS:
        x = feature(data,arm)
        mean = x[fit].mean(0)
        std = x[fit].std(0)
        scale = np.maximum(std,max(float(np.median(std[std>1e-6]))*.05,1e-4))
        tx = torch.from_numpy(normalize(x[fit],mean,scale)).cuda()
        vx = torch.from_numpy(normalize(x[inner],mean,scale)).cuda()
        ty = torch.from_numpy(y[fit]).long().cuda()
        tz = torch.from_numpy(baseline[fit]).cuda()
        vz = torch.from_numpy(baseline[inner]).cuda()
        tw = torch.from_numpy(weights).float().cuda()
        for seed in SEEDS:
            for wd in (.01,.1):
                name = f'{arm}_s{seed}_wd{wd:g}'
                folder = ROOT/'fits'/name
                if (folder/'receipt.json').exists():
                    outcomes.append(read(folder/'receipt.json')); continue
                folder.mkdir(parents=True,exist_ok=True)
                torch.manual_seed(seed)
                head = Head(x.shape[1]).cuda()
                opt = torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=wd)
                initial = metrics(y[inner],baseline[inner])
                best = dict(epoch=0,inner=initial)
                state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                rng = np.random.default_rng(seed)
                curve = []
                for epoch in range(1,31):
                    head.train()
                    loss_sum = 0.
                    for first in range(0,len(fit),128):
                        if first == 0: order = rng.permutation(len(fit))
                        ix = torch.as_tensor(order[first:first+128],device='cuda')
                        z = tz[ix]+head(tx[ix])
                        loss = (F.cross_entropy(z,ty[ix],reduction='none')*tw[ix]).mean()
                        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                        loss_sum += float(loss.detach())*len(ix)
                    head.eval()
                    with torch.no_grad():
                        score = (vz+head(vx)).cpu().numpy()
                        tr = (tz+head(tx)).cpu().numpy()
                    metric = metrics(y[inner],score)
                    curve.append(dict(epoch=epoch,loss=loss_sum/len(fit),inner=metric,
                                      fit=metrics(y[fit],tr)))
                    if key(metric)>key(best['inner']):
                        best=dict(epoch=epoch,inner=metric)
                        state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                torch.save(state,folder/'best.pt')
                np.savez(folder/'normalization.npz',mean=mean,scale=scale)
                receipt = dict(name=name,arm=arm,seed=seed,weight_decay=wd,best=best,
                    input_dimensions=x.shape[1],parameters=sum(p.numel() for p in head.parameters()),
                    epochs_trained=30,source_fit=len(fit),source_inner=len(inner),
                    baseline_inner=initial,validation_used_for_selection=False)
                write(folder/'receipt.json',receipt);write(folder/'curve.json',curve)
                outcomes.append(receipt)
                write(ROOT/'progress.json',dict(phase='fit',completed=len(outcomes),total=18))
                print('FIT',name,'best',best,'seconds',round(time.perf_counter()-tick,1),flush=True)
        del tx,vx
    # External validation is first evaluated only after all source-only choices are fixed.
    chosen = [max((r for r in outcomes if r['arm']==a and r['seed']==s),key=lambda r:key(r['best']['inner']))
              for a in ARMS for s in SEEDS]
    write(ROOT/'selected.json',chosen)
    validation = dict(np.load(ROOT/'validation_local.npz'))
    active = validation['active']
    results=[]
    for row in chosen:
        folder=ROOT/'fits'/row['name']
        norm=dict(np.load(folder/'normalization.npz'))
        x=normalize(feature(validation,row['arm']),**norm)
        head=Head(x.shape[1]).cuda()
        head.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True),strict=True)
        head.eval()
        with torch.no_grad():
            score=validation['logits']+head(torch.from_numpy(x).cuda()).cpu().numpy()*active[:,None]
        np.savez(ROOT/(row['arm']+f'_s{row["seed"]}_predictions_local.npz'),
                 ids=validation['ids'],label=validation['label'],score=score)
        results.append(dict(**row,validation=metrics(validation['label'],score),
            dual=metrics(validation['label'][active],score[active])))
    write(ROOT/'results.json',dict(baseline=metrics(validation['label'],validation['logits']),
        baseline_dual=metrics(validation['label'][active],validation['logits'][active]),
        results=results,elapsed_seconds=time.perf_counter()-tick,test_access=False))
    write(ROOT/'progress.json',dict(phase='complete',completed=18,total=18))
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
