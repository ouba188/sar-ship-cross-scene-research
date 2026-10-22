"""One score-scale decoupling experiment; frozen readers and inputs."""
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
sys.path.insert(0,str(PARENT/'common_gain_training'))
from run_gates import EXP, Anchored, SEEDS, active_mask, load_inputs, weights, risk, metrics, make_predictions

ARM = 'score_decoupled'


class ScoreDecoupled(Anchored):
    def __init__(self,kind):
        super().__init__()
        self.register_buffer('active',active_mask(kind))

    def parts(self,env):
        a = .5*torch.tanh(self.net(env))
        m = (a*self.active).sum(1,keepdim=True)/self.active.sum()
        d = (a-m)*self.active
        return a,m,d

    def forward(self,env,q,base):
        _,m,d = self.parts(env)
        unscaled = base+(d[:,:,None]*q).sum(1)
        return (1+m)*unscaled


def probabilistic_metrics(y,score):
    z = np.asarray(score,dtype=np.float64)
    z = z-z.max(1,keepdims=True)
    logp = z-np.log(np.exp(z).sum(1,keepdims=True))
    p = np.exp(logp)
    nll = -logp[np.arange(len(y)),y]
    brier = ((p-np.eye(3)[y])**2).sum(1)
    return dict(NLL=float(nll.mean()),balanced_NLL=float(np.mean([nll[y==c].mean() for c in range(3)])),
                Brier=float(brier.mean()),balanced_Brier=float(np.mean([brier[y==c].mean() for c in range(3)])))


def gate_summary(a,m,d,active):
    mean = m.reshape(-1)
    sat = (abs(a[:,active])>.49).mean(1)
    spread = d[:,active].std(1)
    return dict(m_mean=float(mean.mean()),m_sd=float(mean.std()),
        **{f'm_p{p:02d}':float(np.quantile(mean,p/100)) for p in [0,5,25,50,75,95,100]},
        d_sd_mean=float(spread.mean()),d_sd_p50=float(np.median(spread)),d_sd_p95=float(np.quantile(spread,.95)),
        saturation_fraction=float(sat.mean()),all_active_saturated_fraction=float((sat==1).mean()),
        m_gt_049_fraction=float((mean>.49).mean()))


def main():
    torch.set_num_threads(2)
    start = time.perf_counter()
    (HERE/'models').mkdir(exist_ok=True)
    t = pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    y,ids = t.class_id.to_numpy(),t.mmsi.to_numpy()
    rows,records,curves,checks,stats,samples,sources,times = [],[],[],[],[],[],[],[]
    for cohort in ['old','new']:
        for fold in range(6):
            d,kind,input_path,env_path = load_inputs(cohort,fold)
            train,val,test = [d[k] for k in ['train','validation','test']]
            env = torch.tensor(d['env'],dtype=torch.float32,device='cuda')
            q = torch.tensor(d['q'],dtype=torch.float32,device='cuda')
            oq = torch.tensor(d['oq'],dtype=torch.float32,device='cuda')
            # Training precision follows the parent CE; evaluation preserves A
            # exactly, including its stored float64 direct-reader scores.
            oo = torch.tensor(d['oof'],dtype=torch.float32,device='cuda')
            outer = torch.tensor(d['base'],dtype=torch.float64,device='cuda')
            yy = torch.tensor(y,dtype=torch.long,device='cuda')
            w = torch.tensor(weights(y[train],ids[train]),dtype=torch.float32,device='cuda')
            active = active_mask(kind).numpy().astype(bool)
            sources.append(dict(cohort=cohort,fold=fold,selected_kind=kind,groups=int(active.sum()),inputs=input_path,env=env_path,
                training_base='cached oof[train], paired with oq[train]',validation_and_test_base='cached base, paired with outer q',
                scope='original three-class frozen data; base/contribution OOF, not full-pipeline OOF'))
            for seed in SEEDS:
                tick = time.perf_counter()
                torch.manual_seed(seed)
                model = ScoreDecoupled(kind).cuda()
                opt = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                br,be = risk(y[val],d['base'][val]),0
                with torch.no_grad():
                    torch.testing.assert_close(model(env,q,outer),outer,atol=0,rtol=0)
                    torch.testing.assert_close(model(env[train],oq[train],oo[train]),oo[train],atol=0,rtol=0)
                for epoch in range(1,201):
                    opt.zero_grad()
                    score = model(env[train],oq[train],oo[train])
                    loss = (torch.nn.functional.cross_entropy(score,yy[train],reduction='none')*w).mean()
                    loss.backward()
                    opt.step()
                    if epoch % 5 == 0:
                        with torch.no_grad():
                            ss = model(env[val],q[val],outer[val]).cpu().numpy()
                            aa,mm,dd = [x.cpu().numpy() for x in model.parts(env[train])]
                        vr = risk(y[val],ss)
                        curves.append(dict(cohort=cohort,fold=fold,seed=seed,epoch=epoch,loss=float(loss.detach()),val_nll=vr,
                                           **gate_summary(aa,mm,dd,active)))
                        if vr < br-1e-6:
                            br,be = vr,epoch
                            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(best)
                model.eval()
                with torch.inference_mode():
                    aa,mm,dd = model.parts(env)
                    core = outer+(dd[:,:,None]*q).sum(1)
                    score = model(env,q,outer)
                    zero = model(env,torch.zeros_like(q),torch.zeros_like(outer))
                    qzero = model(env,torch.zeros_like(q),outer)
                assert not zero.any()
                torch.testing.assert_close(qzero,(1+mm)*outer,atol=0,rtol=0)
                assert (1+mm>0).all()
                assert torch.equal(score.argmax(1),core.argmax(1))
                assert torch.equal(qzero.argmax(1),outer.argmax(1))
                sum_error = float(dd[:,active].sum(1).abs().max())
                assert sum_error < 4e-6 and not dd[:,~active].any()
                if be==0:
                    torch.testing.assert_close(score,outer,atol=0,rtol=0)
                aa,mm,dd,score,core = [x.cpu().numpy() for x in [aa,mm,dd,score,core]]
                path = HERE/'models'/f'{cohort}_fold{fold}_{seed}.pt'
                torch.save(best,path)
                replay = ScoreDecoupled(kind).cuda().eval()
                replay.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
                with torch.inference_mode():
                    back = replay(env[test],q[test],outer[test]).cpu().numpy()
                error = float(abs(back-score[test]).max())
                assert error < 1e-5 and np.array_equal(back.argmax(1),score[test].argmax(1))
                frame = make_predictions(t,test,score[test],d['base'][test],cohort,fold,ARM,seed)
                for c in range(3):
                    frame[f'u{c}'] = core[test,c]
                frame['m'] = mm[test,0]
                frame['d_sd'] = dd[test][:,active].std(1)
                frame['saturation_fraction'] = (abs(aa[test][:,active])>.49).mean(1)
                rows.append(frame)
                records.append(dict(cohort=cohort,fold=fold,seed=seed,epoch=be,val_nll=br,parameters=sum(p.numel() for p in model.parameters()),
                    rescued=int(frame.net.eq(1).sum()),harmed=int(frame.net.eq(-1).sum()),net=int(frame.net.sum()),
                    **metrics(y[test],score[test]),**probabilistic_metrics(y[test],score[test])))
                checks.append(dict(cohort=cohort,fold=fold,seed=seed,active_sum_error=sum_error,zero_step_exact=True,
                    all_sample_argmax_changes=0,zero_signal_exact=True,qzero_is_only_temperature=True,groups=int(active.sum()),
                    scale_min=float((1+mm).min()),scale_max=float((1+mm).max()),replay_error=error))
                for role,ix in [('train',train),('validation',val),('test',test)]:
                    stats.append(dict(cohort=cohort,fold=fold,seed=seed,role=role,n=len(ix),**gate_summary(aa[ix],mm[ix],dd[ix],active)))
                samples.append(pd.DataFrame(dict(cohort=cohort,fold=fold,seed=seed,instance_id=t.iloc[test].instance_id.to_numpy(),
                    m=mm[test,0],d_sd=dd[test][:,active].std(1),saturation_fraction=(abs(aa[test][:,active])>.49).mean(1))))
                times.append(dict(cohort=cohort,fold=fold,seed=seed,seconds=time.perf_counter()-tick))
                print(cohort,fold,seed,'epoch',be,'net',int(frame.net.sum()),'test_m',stats[-1]['m_mean'],'test_d_sd',stats[-1]['d_sd_mean'],flush=True)
            for name,values in [('metrics',records),('curves',curves),('verification',checks),('gate_stats',stats),('sources',sources),('timing',times)]:
                pd.DataFrame(values).to_csv(HERE/(name+'.csv'),index=False)
            pd.concat(rows,ignore_index=True).to_csv(HERE/'predictions.csv',index=False)
            pd.concat(samples,ignore_index=True).to_csv(HERE/'sample_gates.csv',index=False)
    p = pd.concat(rows,ignore_index=True)
    assert len(records)==36 and p.groupby(['cohort','seed']).size().eq(len(t)).all()
    pooled=[]
    for (cohort,seed),part in p.groupby(['cohort','seed']):
        pooled.append(dict(cohort=cohort,seed=seed,net=int(part.net.sum()),rescued=int(part.net.eq(1).sum()),harmed=int(part.net.eq(-1).sum()),
            **metrics(part.class_id.to_numpy(),part[['s0','s1','s2']].to_numpy()),
            **probabilistic_metrics(part.class_id.to_numpy(),part[['s0','s1','s2']].to_numpy())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    summary = dict(status='complete',seconds=time.perf_counter()-start,heads=36,reader_refits=0,seeds=SEEDS,steps=200,
        max_active_sum_error=max(x['active_sum_error'] for x in checks),all_sample_argmax_changes=0,
        max_gpu_replay_error=max(x['replay_error'] for x in checks),
        definition='score=(1+m)*(base+sum(d*q)); m and d share bounded a=.5tanh(net)',
        no_extra_arm=True,no_temperature_selection=True,no_calibration_head=True)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
