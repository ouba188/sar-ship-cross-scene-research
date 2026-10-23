"""Learn shared relation maps across legal reference organizations."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bank_intervention',HERE.parent/'run.py')
diag=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
exp=diag.exp
ARMS=(*diag.ARMS,'excess_shuffle101')


def reset_bank(reader,bank):
    reader.bank=bank
    reader.sb=reader.b[:,bank.ravel()]
    reader.sn=reader.n[:,bank.ravel()]
    reader.ms=reader.counts[:,bank.ravel()]
    reader.rv=reader.ms>=2
    reader.ready[:]=False
    reader.reference.fill_(float('nan'))


def schedule(table,mass,old,legal):
    fit=np.load(exp.edge.support.UP/'indices.npz')['fit_train']
    pool=table.iloc[fit[mass[fit,:,:8].sum((1,2))>0]].drop_duplicates('mmsi').index.to_numpy()
    patterns=legal.sum(-1)>=2
    groups=[]
    for c in range(8):
        for signature in np.unique(patterns[old[c]],axis=0):
            slots=np.flatnonzero((patterns[old[c]]==signature).all(-1))
            options=pool[(table.iloc[pool].class_id.to_numpy()==c)&(patterns[pool]==signature).all(-1)]
            assert len(options)>=len(slots)
            groups.append((c,slots,options))
    banks=[]
    for step in range(1,201):
        rng=np.random.default_rng(10000+step)
        bank=np.empty_like(old)
        for c,slots,options in groups:
            bank[c,slots]=rng.choice(options,len(slots),replace=False)
        assert table.iloc[bank.ravel()].mmsi.nunique()==64
        np.testing.assert_array_equal(patterns[bank],patterns[old])
        banks.append(bank)
    return banks


def select_score(readers,model,ids,full,y):
    predictions={}
    for name,reader in readers.items():
        predictions[name]=exp.infer(reader,model,ids,full)[0]
    score=sum(exp.dep.exact_j(y[ids],v.argmax(-1)) for v in predictions.values())/len(predictions)
    return score,predictions


def restore_checkpoint(model,cp,step):
    model.restore({k:cp[f'{k}_{step}'] for k in exp.STATE_KEYS})


def main():
    started=time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    table,x,mass,old,idx,tr,va,te=exp.source_data()
    y=table.class_id.to_numpy(int)
    _,_,legal,_=exp.edge.edges(x,mass)
    banks,_=diag.banks_from_source(table,mass,legal,old)
    training_banks=schedule(table,mass,old,legal)
    np.savez_compressed(HERE/'schedule.npz',banks=np.array(training_banks),**banks)
    scales=pd.read_csv(exp.HERE/'scales.csv')
    metrics,pairs,curves,receipts,classes=[],[],[],[],[]
    weights={}
    checks={}
    for arm in ARMS:
        divisor=torch.tensor(scales[scales.arm.eq(arm)].sort_values('feature').divisor.to_numpy(),device='cuda',dtype=torch.float32)
        readers={}
        for name,bank in banks.items():
            r=exp.Reader(x,mass,bank,arm)
            r.divisor=divisor
            r.cache_reference(idx[va])
            readers[name]=r
        trainreader=exp.Reader(x,mass,old,arm)
        trainreader.divisor=divisor
        # Rebinding only references must agree with constructing a fresh reader.
        reset_bank(trainreader,banks['101'])
        probe=idx[tr[:3]]
        model=exp.Metric()
        with torch.no_grad():
            a=trainreader.scores(probe,*model.matrices())[0]
            b=readers['101'].scores(probe,*model.matrices())[0]
            error=float((a-b).abs().max().cpu())
            assert error<1e-6,error
            trainreader.cache_reference(probe)
            assert float((a-trainreader.reference[probe]).abs().max().cpu())==0
        checks[arm]=dict(rebound_reader_error=error,initial_delta_zero=True)
        for seed in exp.SEEDS:
            tick=time.perf_counter()
            full=np.load(exp.HERE/f'predictions_{seed}.npz')['full']
            model=exp.Metric()
            best,initial_pred=select_score(readers,model,idx[va],full,y)
            for v in initial_pred.values():
                np.testing.assert_array_equal(v,full[idx[va]])
            initial=best
            # Match selector for saved fixed-reference training before comparing methods.
            cp=np.load(exp.HERE/f'checkpoints_{arm}_{seed}.npz')
            fixedbest,fixedstep,fixedstate=initial,0,model.state()
            for step in exp.CHECKPOINTS:
                restore_checkpoint(model,cp,step)
                j,pred=select_score(readers,model,idx[va],full,y)
                curves.append(dict(arm=arm,seed=seed,method='fixed_reselected',step=step,selection_J=float(j)))
                if j>fixedbest:
                    fixedbest,fixedstep,fixedstate=j,step,model.state()
            model=exp.Metric()
            opt=torch.optim.Adam(model.parameters(),lr=.01)
            beststep,beststate=0,model.state()
            ids=idx[tr]
            target=torch.tensor(y[ids],device='cuda')
            counts=np.bincount(y[ids],minlength=8)
            w=torch.tensor(.5/len(ids)+.5/(8*counts[y[ids]]),device='cuda',dtype=torch.float32)
            baseline=torch.tensor(full[ids],device='cuda')
            states={}
            for step in range(201):
                if step:
                    reset_bank(trainreader,training_banks[step-1])
                    trainreader.cache_reference(ids)
                    opt.zero_grad(set_to_none=True)
                    for first in range(0,len(ids),128):
                        part=ids[first:first+128]
                        z,_,_=trainreader.scores(part,*model.matrices())
                        score=baseline[first:first+128]+exp.delta(z-trainreader.reference[part],model.a)
                        loss=(F.cross_entropy(score,target[first:first+128],reduction='none')*w[first:first+128]).sum()
                        loss.backward()
                    model.regularizer().backward()
                    opt.step()
                if step not in exp.CHECKPOINTS:
                    continue
                j,pred=select_score(readers,model,idx[va],full,y)
                state=model.state()
                states.update({f'{k}_{step}':v for k,v in state.items()})
                row=dict(arm=arm,seed=seed,method='episodic',step=step,selection_J=float(j),
                    Mb_change=float(np.linalg.norm(state['Mb']-np.eye(32))),Mn_change=float(np.linalg.norm(state['Mn']-np.eye(32))))
                for name,v in pred.items():
                    row['selection_J_'+name]=exp.dep.metrics(y[idx[va]],v)['J']
                curves.append(row)
                assert np.isfinite(float(j))
                if j>best:
                    best,beststep,beststate=j,step,state
                print('CHECK',arm,seed,step,'J',float(j),'seconds',round(time.perf_counter()-tick,1),flush=True)
            np.savez_compressed(HERE/f'checkpoints_{arm}_{seed}.npz',**states)
            saved={'idx':idx,'y':y[idx],'full':full[idx]}
            for method,state,step,chosen_j in [('fixed_reselected',fixedstate,fixedstep,fixedbest),('episodic',beststate,beststep,best)]:
                model.restore(state)
                for k,v in state.items():
                    weights[f'{method}_{arm}_{seed}_{k}']=v
                receipts.append(dict(arm=arm,seed=seed,method=method,selected_step=step,selection_J=float(chosen_j),
                    baseline_selection_J=float(initial),parameters=516,seconds=time.perf_counter()-tick))
                for name,r in readers.items():
                    r.cache_reference(idx)
                    pred,z,_=exp.infer(r,model,idx,full)
                    saved[f'{method}_{name}']=pred
                    if name=='original':
                        saved[f'{method}_{name}_parts']=z*state['a']
                    for role,loc in [('utility_train',tr),('selection',va),('evaluation',te)]:
                        meta=dict(method=method,arm=arm,seed=seed,bank=name,role=role,step=step)
                        metrics.append(dict(**meta,**exp.dep.metrics(y[idx[loc]],pred[loc])))
                        oldcorrect=full[idx[loc]].argmax(-1)==y[idx[loc]]
                        correct=pred[loc].argmax(-1)==y[idx[loc]]
                        pairs.append(dict(**meta,rescued=int((~oldcorrect&correct).sum()),harmed=int((oldcorrect&~correct).sum()),net=int(correct.sum()-oldcorrect.sum())))
                        for c in range(8):
                            mask=y[idx[loc]]==c
                            classes.append(dict(**meta,class_id=c,n=int(mask.sum()),correct=int(correct[mask].sum()),full_correct=int(oldcorrect[mask].sum())))
            np.savez_compressed(HERE/f'predictions_{arm}_{seed}.npz',**saved)
            np.savez_compressed(HERE/'weights.npz',**weights)
            for name,rows in [('metrics',metrics),('paired',pairs),('curves',curves),('receipts',receipts),('classes',classes)]:
                pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
            print('SELECTED',arm,seed,fixedstep,beststep,flush=True)
    frame=pd.DataFrame(metrics)
    summary=frame[frame.role.eq('evaluation')].groupby(['method','arm','bank'])[['OA','BA','J','NLL','Brier']].mean()
    summary.to_csv(HERE/'summary.csv')
    receipt=dict(fits=12,steps=200,schedule_seeds='10001..10200',shared_schedule_all_arms=True,
        parameters=516,configs=list(exp.SEEDS),arms=list(ARMS),transform_init_seed=17,
        main_deployed_bank='original only; others are stress tests, no inference ensemble',
        source_selection='mean exact J over 6 frozen organizations',evaluation_after_selection=True,
        old_training_reselected_same_rule=True,checks=checks,seconds=time.perf_counter()-started)
    (HERE/'execution.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(summary.to_string(),flush=True)
    print(receipt,flush=True)


if __name__=='__main__':
    main()
