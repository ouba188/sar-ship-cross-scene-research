"""Three predeclared jointCE controls; primary J selection, secondary NLL."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ALIGNMENT = HERE.parent
sys.path.insert(0, str(ALIGNMENT))
from run_alignment import parent, inputs, assessment, exact_j, frozen_state

ARMS = ['score_only', 'conditional_std', 'linear_distribution']
SEEDS = [17, 29, 43]


class ScoreOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3,16), nn.GELU(), nn.Linear(16,3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, moments):
        return self.net(x)


def make_model(arm):
    return ScoreOnly() if arm == 'score_only' else parent.Correction(arm)


def raw_input(arm, base, moments):
    if arm == 'score_only':
        return base-base.mean(1, keepdims=True)
    return parent.descriptors(base, moments)


def main():
    assert (HERE/'PROTOCOL_ZH.md').exists()
    start = time.perf_counter()
    torch.set_num_threads(2)
    (HERE/'models').mkdir(exist_ok=True)
    table = pd.read_csv(parent.EXP/'readout/targets.csv',dtype={'mmsi':str})
    y = table.class_id.to_numpy()
    yy = torch.tensor(y,dtype=torch.long)
    cache = np.load(parent.CACHE/'signed_features.npz')
    names = list(cache['reader_names'])
    curves, selections, costs, checks = [], [], [], []
    for fold in range(6):
        d, outer, oof, _, _, _ = inputs(fold,cache,names)
        tr, va = d['train'], d['validation']
        b = torch.tensor(d['oof'],dtype=torch.float32)
        bv = torch.tensor(d['base'],dtype=torch.float32)
        m, mv = torch.tensor(oof,dtype=torch.float32), torch.tensor(outer,dtype=torch.float32)
        for arm in ARMS:
            raw = raw_input(arm,d['oof'],oof)
            rawv = raw_input(arm,d['base'],outer)
            axes = 0 if arm=='score_only' else (0,1)
            mu = raw[tr].mean(axes)
            scale = raw[tr].std(axes).clip(1e-8)
            x = torch.tensor((raw-mu)/scale,dtype=torch.float32)
            xv = torch.tensor((rawv-mu)/scale,dtype=torch.float32)
            for seed in [-1] if arm=='linear_distribution' else SEEDS:
                tick = time.perf_counter()
                torch.manual_seed(0 if seed<0 else seed)
                model = make_model(arm)
                parameters = sum(p.numel() for p in model.parameters())
                assert parameters == {'score_only':115,'conditional_std':81,'linear_distribution':3,'signed_only':113}[arm]
                with torch.no_grad():
                    zero = float(model(xv[va],mv[va]).abs().max())
                    assert zero == 0
                risk0 = float(parent.balanced_loss(bv[va],yy[va]))
                j0 = exact_j(y[va],bv[va].numpy().argmax(1))
                initial = frozen_state(model)
                best = {'J':dict(epoch=0,state=initial,value=j0),
                        'balancedNLL':dict(epoch=0,state=initial,value=risk0)}
                meta = dict(fold=fold,arm=arm,seed=seed)
                trace = [dict(**meta,epoch=0,train_loss=np.nan,
                    **assessment(y[va],bv[va].numpy().astype(float)),selection_balancedNLL=risk0)]
                opt = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                for epoch in range(1,201):
                    model.train()
                    opt.zero_grad()
                    score = b[tr]+model(x[tr],m[tr])
                    loss = .5*parent.balanced_loss(score,yy[tr])+.5*nn.functional.cross_entropy(score,yy[tr])
                    loss.backward()
                    opt.step()
                    if epoch%5:
                        continue
                    model.eval()
                    with torch.no_grad():
                        val_score = bv[va]+model(xv[va],mv[va])
                        risk = float(parent.balanced_loss(val_score,yy[va]))
                    arr = val_score.numpy().astype(float)
                    jval = exact_j(y[va],arr.argmax(1))
                    trace.append(dict(**meta,epoch=epoch,train_loss=float(loss.detach()),
                        **assessment(y[va],arr),selection_balancedNLL=risk))
                    if jval > best['J']['value']:
                        best['J'] = dict(epoch=epoch,state=frozen_state(model),value=jval)
                    if risk < best['balancedNLL']['value']-1e-8:
                        best['balancedNLL'] = dict(epoch=epoch,state=frozen_state(model),value=risk)
                costs.append(dict(**meta,parameters=parameters,steps=200,seconds=time.perf_counter()-tick))
                curves.extend(trace)
                for selector, chosen in best.items():
                    path = HERE/'models'/f'fold{fold}_{arm}_{selector}_seed{seed}.pt'
                    ck = dict(state=chosen['state'],arm=arm,seed=seed,fold=fold,selector=selector,
                        mu=mu,scale=scale,epoch=chosen['epoch'])
                    torch.save(ck,path)
                    row = next(row for row in trace if row['epoch']==chosen['epoch'])
                    selections.append(dict(**meta,selector=selector,epoch=chosen['epoch'],parameters=parameters,
                        checkpoint=str(path),validation_OA=row['OA'],validation_BA=row['BA'],validation_J=row['J'],
                        validation_NLL=row['NLL'],validation_balancedNLL=row['selection_balancedNLL'],
                        J_gain=row['J']-float(j0),balancedNLL_gain=row['selection_balancedNLL']-risk0))
                    clone = make_model(arm)
                    clone.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state'])
                    clone.eval()
                    with torch.no_grad():
                        check_score = bv[va]+clone(xv[va],mv[va])
                        check_risk = float(parent.balanced_loss(check_score,yy[va]))
                    error = abs(check_risk-row['selection_balancedNLL'])
                    assert error<2e-6
                    assert abs(float(exact_j(y[va],check_score.numpy().argmax(1)))-row['J'])<1e-12
                    checks.append(dict(**meta,selector=selector,zero_head_error=zero,validation_replay_error=error))
                print(f'fold={fold} arm={arm} seed={seed} J={best["J"]["epoch"]} NLL={best["balancedNLL"]["epoch"]}',flush=True)
    pd.DataFrame(curves).to_csv(HERE/'selection_curves.csv',index=False)
    pd.DataFrame(selections).to_csv(HERE/'selections.csv',index=False)
    pd.DataFrame(costs).to_csv(HERE/'trajectory_costs.csv',index=False)
    (HERE/'selection_frozen.json').write_text(json.dumps(dict(trajectories=len(costs),selected_heads=len(selections),
        test_evaluated=False,seconds=time.perf_counter()-start),indent=2),encoding='utf8')

    eval_start = time.perf_counter()
    predictions, validation = [], []
    for fold in range(6):
        d = dict(np.load(parent.NEW/f'fold{fold}_inputs.npz'))
        outer = cache['moments'][names.index(f'fold{fold}_outer')]
        for row in (r for r in selections if r['fold']==fold):
            ck = torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
            model = make_model(ck['arm'])
            model.load_state_dict(ck['state'])
            model.eval()
            raw = raw_input(ck['arm'],d['base'],outer)
            for role in ['validation','test']:
                ix = d[role]
                with torch.inference_mode():
                    delta = model(torch.tensor((raw[ix]-ck['mu'])/ck['scale'],dtype=torch.float32),
                        torch.tensor(outer[ix],dtype=torch.float32)).numpy().astype(float)
                base = d['base'][ix].astype(float)
                score = base+delta
                g = table.iloc[ix][['instance_id','class_id','mmsi','pass_group','port']].copy()
                for key in ['fold','arm','seed','selector','epoch']:
                    g[key] = row[key]
                g['prediction'],g['base_prediction'] = score.argmax(1),base.argmax(1)
                g['net'] = (g.prediction==g.class_id).astype(int)-(g.base_prediction==g.class_id).astype(int)
                g['outcome'] = np.select([g.net.eq(1),g.net.eq(-1),g.prediction.eq(g.class_id)],['rescued','harmed','CC'],default='WW')
                for c in range(3):
                    g[f's{c}'],g[f'b{c}'],g[f'delta{c}'] = score[:,c],base[:,c],delta[:,c]
                (predictions if role=='test' else validation).append(g)
    pred = pd.concat(predictions,ignore_index=True)
    pred.to_csv(HERE/'predictions.csv',index=False)
    pd.concat(validation,ignore_index=True).to_csv(HERE/'validation_predictions.csv',index=False)
    pooled,classes,acquisition = [],[],[]
    for (arm,selector,seed),g in pred.groupby(['arm','selector','seed']):
        assert len(g)==len(table) and g.instance_id.is_unique
        meta = dict(arm=arm,selector=selector,seed=seed)
        pooled.append(dict(**meta,rescued=int(g.net.eq(1).sum()),harmed=int(g.net.eq(-1).sum()),net=int(g.net.sum()),
            **assessment(g.class_id.to_numpy(),g[['s0','s1','s2']].to_numpy())))
        for c,h in g.groupby('class_id'):
            classes.append(dict(**meta,class_id=c,n=len(h),net=int(h.net.sum()),
                rescued=int(h.net.eq(1).sum()),harmed=int(h.net.eq(-1).sum()),
                base_recall=100*h.base_prediction.eq(c).mean(),recall=100*h.prediction.eq(c).mean()))
        for unit in ['fold','pass_group','port']:
            for name,h in g.groupby(unit):
                acquisition.append(dict(**meta,unit=unit,name=name,n=len(h),net=int(h.net.sum()),
                    rescued=int(h.net.eq(1).sum()),harmed=int(h.net.eq(-1).sum())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    pd.DataFrame(classes).to_csv(HERE/'class_effects.csv',index=False)
    pd.DataFrame(acquisition).to_csv(HERE/'acquisition_effects.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'replay_checks.csv',index=False)
    receipt = dict(trajectories=len(costs),selected_outputs=len(selections),training_seconds=sum(r['seconds'] for r in costs),
        evaluation_seconds=time.perf_counter()-eval_start,total_seconds=time.perf_counter()-start,
        cache=str(parent.CACHE/'signed_features.npz'),test_used_for_selection=False,candidate_retrained='signed_only' in ARMS)
    (HERE/'summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
