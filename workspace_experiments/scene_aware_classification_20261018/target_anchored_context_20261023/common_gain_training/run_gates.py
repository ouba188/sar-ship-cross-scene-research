"""Matched training of zero-common and range-matched gates on frozen inputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
NEW = PARENT/'partition_replication'
sys.path.insert(0,str(PARENT))
from run_anchored import EXP, Anchored, KINDS, SEEDS, weights, risk, metrics

ARMS = ['contrast_only_gate','range_matched_unrestricted']


def active_mask(kind):
    mask = torch.zeros(32)
    for name in KINDS[kind]:
        mask[:16] = 1 if name=='stem' else mask[:16]
        if name != 'stem':
            mask[16:] = 1
    return mask


class GroupGate(Anchored):
    def __init__(self,kind,arm):
        super().__init__()
        self.arm = arm
        self.register_buffer('active',active_mask(kind))

    def coefficients(self,env):
        a = torch.tanh(self.net(env))
        if self.arm == 'contrast_only_gate':
            a = .5*a
            mean = (a*self.active).sum(1,keepdim=True)/self.active.sum()
            return (a-mean)*self.active
        return a*self.active

    def forward(self,env,q):
        return (self.coefficients(env)[:,:,None]*q).sum(1)


def load_inputs(cohort,fold):
    folder = PARENT if cohort=='old' else NEW
    with np.load(folder/f'fold{fold}_inputs.npz') as cache:
        d = {k:cache[k] for k in ['base','oof','q','oq','train','validation','test']}
        kind = str(cache['selected_kind']) if cohort=='old' else 'layer3'
        if cohort=='new':
            d['env'] = cache['env']
    env_path = folder/f'fold{fold}_inputs.npz'
    if cohort=='old':
        env_path = PARENT/f'fold{fold}_own_input.npz'
        with np.load(env_path) as cache:
            d['env'] = cache['env']
    inactive = ~active_mask(kind).numpy().astype(bool)
    assert not d['q'][:,inactive].any() and not d['oq'][:,inactive].any()
    assert d['env'].shape == (5695,79)
    assert all(np.isfinite(d[k]).all() for k in ['env','base','oof','q','oq'])
    return d,kind,str(folder/f'fold{fold}_inputs.npz'),str(env_path)


def make_predictions(t,test,score,base,cohort,fold,arm,seed):
    truth = t.class_id.to_numpy()[test]
    pred,bp = score.argmax(1),base.argmax(1)
    frame = t.iloc[test][['instance_id','class_id','mmsi','port','pass_group','neighbors']].copy()
    frame['cohort'],frame['fold'],frame['arm'],frame['seed'] = cohort,fold,arm,seed
    frame['prediction'],frame['base_prediction'] = pred,bp
    frame['net'] = (pred==truth).astype(int)-(bp==truth).astype(int)
    frame['outcome'] = np.select([frame.net.eq(1),frame.net.eq(-1),pred==truth],['rescued','harmed','CC'],default='WW')
    for c in range(3):
        frame[f's{c}'],frame[f'b{c}'] = score[:,c],base[:,c]
    return frame


def test_group_rule():
    # Catch averaging the 16 inactive slots, even when real q there is zero.
    for kind in ['layer3','stem_layer3']:
        gate = GroupGate(kind,'contrast_only_gate')
        with torch.no_grad():
            gate.net[-1].bias.copy_(torch.linspace(-2,2,32))
            env = torch.zeros(3,79)
            a = .5*torch.tanh(gate.net(env))
            active = gate.active.bool()
            expected = a[:,active]-a[:,active].mean(1,keepdim=True)
            actual = gate.coefficients(env)
        torch.testing.assert_close(actual[:,active],expected,atol=2e-7,rtol=0)
        assert not actual[:,~active].any()
        assert actual.sum(1).abs().max() < 3e-6


def main():
    torch.set_num_threads(2)
    test_group_rule()
    started = time.perf_counter()
    (HERE/'models').mkdir(exist_ok=True)
    t = pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    assert len(t)==5695 and set(t.class_id)=={0,1,2}
    y,ids = t.class_id.to_numpy(),t.mmsi.to_numpy()
    rows,baseline,records,curves,checks,sources,times = [],[],[],[],[],[],[]
    for cohort in ['old','new']:
        for fold in range(6):
            d,kind,input_path,env_path = load_inputs(cohort,fold)
            train,val,test = [d[k] for k in ['train','validation','test']]
            for aa,bb in [(train,val),(train,test),(val,test)]:
                assert not set(ids[aa])&set(ids[bb])
                assert not set(t.iloc[aa].pass_group)&set(t.iloc[bb].pass_group)
            baseline.append(make_predictions(t,test,d['base'][test],d['base'][test],cohort,fold,'A',-1))
            sources.append(dict(cohort=cohort,fold=fold,selected_kind=kind,groups=int(active_mask(kind).sum()),
                train_n=len(train),validation_n=len(val),test_n=len(test),inputs=input_path,env=env_path,
                q_and_oq='unchanged cached original-neural-channel contributions; selected kind fixed within fold',
                scope='A and contribution OOF; env outer-source projection; not whole-pipeline OOF'))
            env = torch.tensor(d['env'],dtype=torch.float32,device='cuda')
            q = torch.tensor(d['q'],dtype=torch.float32,device='cuda')
            oq = torch.tensor(d['oq'],dtype=torch.float32,device='cuda')
            oo = torch.tensor(d['oof'],dtype=torch.float32,device='cuda')
            yy = torch.tensor(y,dtype=torch.long,device='cuda')
            w = torch.tensor(weights(y[train],ids[train]),dtype=torch.float32,device='cuda')
            for seed in SEEDS:
                for arm in ARMS:
                    start_head = time.perf_counter()
                    torch.manual_seed(seed)
                    model = GroupGate(kind,arm).cuda()
                    opt = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                    best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    best_risk,best_epoch = risk(y[val],d['base'][val]),0
                    with torch.no_grad():
                        assert not model(env,q).any()
                    torch.cuda.synchronize()
                    start_loop = time.perf_counter()
                    for epoch in range(1,201):
                        opt.zero_grad()
                        score = oo[train]+model(env[train],oq[train])
                        loss = (torch.nn.functional.cross_entropy(score,yy[train],reduction='none')*w).mean()
                        loss.backward()
                        opt.step()
                        if epoch % 5 == 0:
                            with torch.no_grad():
                                ss = d['base'][val]+model(env[val],q[val]).cpu().numpy()
                            vr = risk(y[val],ss)
                            curves.append(dict(cohort=cohort,fold=fold,arm=arm,seed=seed,epoch=epoch,loss=float(loss.detach()),val_nll=vr))
                            if vr < best_risk-1e-6:
                                best_risk,best_epoch = vr,epoch
                                best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    torch.cuda.synchronize()
                    loop_seconds = time.perf_counter()-start_loop
                    model.load_state_dict(best)
                    model.eval()
                    with torch.inference_mode():
                        coeff = model.coefficients(env).cpu().numpy()
                        delta = model(env,q).cpu().numpy()
                        null = model(env[test],torch.zeros_like(q[test])).cpu().numpy()
                    active = active_mask(kind).numpy().astype(bool)
                    zero_sum_error = float(abs(coeff[:,active].sum(1)).max())
                    if arm == 'contrast_only_gate':
                        assert zero_sum_error < 4e-6
                    assert not coeff[:,~active].any() and not null.any()
                    assert abs(coeff).max() <= 1.000001 and np.isfinite(delta).all()
                    if best_epoch==0:
                        assert not delta.any()
                    score = d['base']+delta
                    path = HERE/'models'/f'{cohort}_fold{fold}_{arm}_{seed}.pt'
                    torch.save(best,path)
                    replay = GroupGate(kind,arm).cuda().eval()
                    replay.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
                    with torch.inference_mode():
                        back = d['base'][test]+replay(env[test],q[test]).cpu().numpy()
                    error = float(abs(back-score[test]).max())
                    assert error < 1e-5
                    np.testing.assert_array_equal(back.argmax(1),score[test].argmax(1))
                    frame = make_predictions(t,test,score[test],d['base'][test],cohort,fold,arm,seed)
                    rows.append(frame)
                    records.append(dict(cohort=cohort,fold=fold,arm=arm,seed=seed,selected_kind=kind,
                        epoch=best_epoch,val_nll=best_risk,parameters=sum(p.numel() for p in model.parameters()),
                        rescued=int(frame.net.eq(1).sum()),harmed=int(frame.net.eq(-1).sum()),net=int(frame.net.sum()),**metrics(y[test],score[test])))
                    checks.append(dict(cohort=cohort,fold=fold,arm=arm,seed=seed,groups=int(active.sum()),
                        active_sum_error=zero_sum_error,inactive_exact=True,zero_q_exact=True,replay_error=error,
                        gain_min=float((1+coeff[:,active]).min()),gain_max=float((1+coeff[:,active]).max()),zero_step_fallback=True))
                    times.append(dict(cohort=cohort,fold=fold,arm=arm,seed=seed,train_and_val_seconds=loop_seconds,total_head_seconds=time.perf_counter()-start_head))
                    print(cohort,fold,arm,seed,'epoch',best_epoch,'net',int(frame.net.sum()),flush=True)
            for name,values in [('metrics',records),('curves',curves),('verification',checks),('sources',sources),('timing',times)]:
                pd.DataFrame(values).to_csv(HERE/(name+'.csv'),index=False)
            pd.concat(rows,ignore_index=True).to_csv(HERE/'predictions.csv',index=False)
            pd.concat(baseline,ignore_index=True).to_csv(HERE/'baseline_predictions.csv',index=False)
    p = pd.concat(rows,ignore_index=True)
    assert len(records)==72 and not p.duplicated(['cohort','arm','seed','instance_id']).any()
    assert p.groupby(['cohort','arm','seed']).size().eq(len(t)).all()
    pooled = []
    for (cohort,arm,seed),part in p.groupby(['cohort','arm','seed']):
        pooled.append(dict(cohort=cohort,arm=arm,seed=seed,n=len(part),net=int(part.net.sum()),
            rescued=int(part.net.eq(1).sum()),harmed=int(part.net.eq(-1).sum()),**metrics(part.class_id.to_numpy(),part[['s0','s1','s2']].to_numpy())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    b = pd.concat(baseline,ignore_index=True)
    pd.DataFrame([dict(cohort=c,**metrics(part.class_id.to_numpy(),part[['s0','s1','s2']].to_numpy())) for c,part in b.groupby('cohort')]).to_csv(HERE/'baseline_pooled.csv',index=False)
    summary = dict(status='complete',seconds=time.perf_counter()-started,heads=72,reader_refits=0,targets=len(t),
        seeds=SEEDS,steps=200,arms=ARMS,cohorts=['old','new'],
        train_and_val_seconds=sum(x['train_and_val_seconds'] for x in times),
        max_contrast_zero_sum_error=max(x['active_sum_error'] for x in checks if x['arm']=='contrast_only_gate'),
        max_gpu_replay_error=max(x['replay_error'] for x in checks),
        boundary='Frozen three-class development data and two existing splits; no weighted-neutral training; no new readout arm.')
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
