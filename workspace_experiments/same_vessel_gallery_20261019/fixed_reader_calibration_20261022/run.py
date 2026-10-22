"""Calibrate a residual gate without changing its upstream reader at deployment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import gc
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'held_vessel_gate_20261022'
spec=importlib.util.spec_from_file_location('held_gate',PREV/'run.py')
held=importlib.util.module_from_spec(spec);spec.loader.exec_module(held)
old=held.old
ARMS=('seen_constant','calibration_constant','seen_score','calibration_score')


def source_pools(f,tr,assignment,fold,inner):
    fit=tr[assignment!=inner];cal=tr[assignment==inner]
    rng=np.random.default_rng(20261022+31*fold+113*inner)
    seen=[]
    for c in range(8):
        n=int((f.iloc[cal].class_id==c).sum())
        choices=fit[f.iloc[fit].class_id.to_numpy()==c]
        seen.extend(rng.choice(choices,size=n,replace=False).tolist())
    def ordered(ii):
        return f.loc[ii].sort_values(['class_id','obs_id']).index.to_numpy()
    cal,seen=ordered(cal),ordered(seen)
    np.testing.assert_array_equal(f.iloc[cal].class_id,f.iloc[seen].class_id)
    return fit,cal,seen


def main():
    torch.set_num_threads(2);start=time.perf_counter()
    f,splits=old.relation.load_data();y=f.class_id.to_numpy()
    raw=np.load(old.five.AXIS/'descriptors.npz');raw_base=np.concatenate((raw['base'],raw['axis']),1)
    raw_five=np.load(old.strip.PREV/'keep_refs.npy',mmap_mode='r').reshape(len(f),60,64)
    cache=np.load(old.ROOT/'features.npz')
    logs=[];metrics=[];predictions=[];checks=[];coverage=[];weights=[];pools=[];signals=[]
    for fold,(tr,val,test) in enumerate(splits):
        oof=np.load(PREV/f'oof_f{fold}.npz');np.testing.assert_array_equal(oof['rows'],tr)
        assignment=oof['inner']
        for inner in range(3):
            fit,cal,seen=source_pools(f,tr,assignment,fold,inner)
            assert not set(f.iloc[cal].mmsi)&set(f.iloc[fit].mmsi)
            assert not set(f.iloc[cal].mmsi)&set(f.iloc[val].mmsi)
            assert not set(f.iloc[cal].mmsi)&set(f.iloc[test].mmsi)
            for name,ii in (('upstream_fit',fit),('calibration',cal),('seen_matched',seen),('val',val),('test',test)):
                coverage.append(dict(fold=fold,inner=inner,partition=name,n=len(ii),ships=f.iloc[ii].mmsi.nunique(),
                    classes=f.iloc[ii].class_id.nunique(),products=f.iloc[ii]['product'].nunique()))
                pools.extend(dict(fold=fold,inner=inner,partition=name,row=int(i),obs_id=int(f.iloc[i].obs_id)) for i in ii)
            norm=np.load(PREV/f'inner_f{fold}_k{inner}_normalization.npz')
            x=(raw_base-norm['ridge_mean'])/norm['ridge_std']/np.sqrt(raw_base.shape[1])
            ridge=(x@norm['ridge_coef'].T+norm['ridge_intercept']).astype(np.float32)
            xf=torch.empty(raw_five.shape,device='cuda',dtype=torch.float32)
            for first in range(0,len(f),512):
                xf[first:first+512]=torch.tensor((raw_five[first:first+512]-norm['five_mean'])/norm['five_std'],device='cuda')
            matrices,stats=old.standardize(cache,fit)
            for name,array in stats.items():np.testing.assert_array_equal(array,norm[name])
            xd=torch.tensor(matrices['separated'],device='cuda');del matrices,x
            for seed in range(3):
                head=old.five.Residual().cuda().eval()
                hh=torch.load(PREV/f'inner_f{fold}_k{inner}_keep_refs_s{seed}.pt',map_location='cpu',weights_only=False)
                head.load_state_dict(hh['state']);a=ridge+held.predict(head,xf)
                residual=old.Residual().cuda().eval()
                rr=torch.load(PREV/f'inner_f{fold}_k{inner}_neighborhood_s{seed}.pt',map_location='cpu',weights_only=False)
                residual.load_state_dict(rr['state']);d=held.predict(residual,xd)
                ids=tr[assignment==inner];mask=assignment==inner
                ea=float(np.abs(a[ids]-oof['base'][seed,mask]).max())
                ed=float(np.abs(d[ids]-oof['residual'][seed,mask]).max())
                assert max(ea,ed)<2e-5
                checks.append(dict(fold=fold,inner=inner,seed=seed,check='frozen_reader_OOF_replay',error=max(ea,ed)))
                reference=a+d;active=bool(np.abs(d).max()>1e-8)
                np.savez_compressed(ROOT/f'inputs_f{fold}_k{inner}_s{seed}.npz',base=a,residual=d)
                for name,ii in (('seen',seen),('calibration',cal),('val',val),('test',test)):
                    bc=a[ii].argmax(1)==y[ii];rc=reference[ii].argmax(1)==y[ii]
                    signals.append(dict(fold=fold,inner=inner,seed=seed,partition=name,n=len(ii),active=active,
                        base_correct=int(bc.sum()),reference_correct=int(rc.sum()),rescue=int((rc&~bc).sum()),
                        harm=int((~rc&bc).sum()),residual_abs=float(np.abs(d[ii]).mean())))
                ta=torch.tensor(a,device='cuda');td=torch.tensor(d,device='cuda');yy=torch.tensor(y,device='cuda')
                for arm in ARMS:
                    train=cal if arm.startswith('calibration') else seen
                    sw=torch.tensor(old.prior.weights(f,train),device='cuda',dtype=torch.float32)
                    torch.manual_seed(20261022+seed)
                    gate=held.Gate(arm.endswith('constant')).cuda()
                    optimizer=torch.optim.AdamW(gate.parameters(),lr=.001,weight_decay=.001)
                    rng=np.random.default_rng(20261022+seed+31*fold+113*inner)
                    best=None;state=None;chosen=0
                    for step in range(151):
                        if step in (0,50,100,150):
                            with torch.no_grad():ss=(ta[val]+gate(ta[val],td[val])[:,None]*td[val]).cpu().numpy()
                            met=old.control.metrics(y[val],ss)
                            logs.append(dict(fold=fold,inner=inner,seed=seed,arm=arm,step=step,**met))
                            rank=(met['ba'],-met['brier'])
                            if best is None or rank>best:
                                best=rank;chosen=step;state={k:v.detach().cpu().clone() for k,v in gate.state_dict().items()}
                            if step==0:np.testing.assert_array_equal(ss,reference[val])
                            if step==150:break
                        pick=rng.integers(len(train),size=128);ii=train[pick]
                        score=ta[ii]+gate(ta[ii],td[ii])[:,None]*td[ii]
                        target=2*F.one_hot(yy[ii],8).float()-1
                        loss=((score-target).square().mean(1)*sw[pick]).mean();assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                    gate.load_state_dict(state)
                    torch.save(dict(state=state,step=chosen,arm=arm),ROOT/f'head_f{fold}_k{inner}_{arm}_s{seed}.pt')
                    with torch.no_grad():w=gate(ta,td).cpu().numpy()
                    score=a+w[:,None]*d
                    for name,ii in (('seen',seen),('calibration',cal),('val',val),('test',test)):
                        metrics.append(dict(fold=fold,inner=inner,seed=seed,arm=arm,partition=name,step=chosen,
                            active=active,**old.control.metrics(y[ii],score[ii])))
                        weights.append(dict(fold=fold,inner=inner,seed=seed,arm=arm,partition=name,step=chosen,
                            mean=float(w[ii].mean()),std=float(w[ii].std()),minimum=float(w[ii].min()),maximum=float(w[ii].max()),active=active))
                    variants={arm:(score[test],w[test])}
                    if arm.endswith('score'):
                        wm=np.full(len(test),w[cal].mean(),np.float32)
                        variants[arm+'_cal_mean']=(a[test]+wm[:,None]*d[test],wm)
                    if arm==ARMS[0]:
                        variants.update(reference=(reference[test],np.ones(len(test))),baseline=(a[test],np.zeros(len(test))))
                        full=np.load(held.PREVIOUS/f'inputs_f{fold}_s{seed}.npz')
                        variants['old_full_reference']=(full['base'][test]+full['residual'][test],np.ones(len(test)))
                    for name,(ss,ww) in variants.items():
                        records=[];held.record_predictions(records,f,test,fold,seed,name,ss,ww,chosen if name in ARMS else -1)
                        predictions.extend(dict(**row,inner=inner,active=active) for row in records)
                    np.savez_compressed(ROOT/f'scores_f{fold}_k{inner}_{arm}_s{seed}.npz',rows=test,
                        scores=score[test],weight=w[test],reference=reference[test])
                    del gate,optimizer
                print('done',fold,inner,seed,'D_step',rr['step'],'cal',len(cal),flush=True)
                del head,residual,ta,td
            del xf,xd;gc.collect()
        for name,records in (('validation_selection',logs),('metrics',metrics),('predictions',predictions),
            ('checks',checks),('coverage',coverage),('weights',weights),('partitions',pools),('signals',signals)):
            pd.DataFrame(records).to_csv(ROOT/f'{name}.csv',index=False,lineterminator='\n')
    (ROOT/'TRAINING.json').write_text(json.dumps(dict(status='complete',seconds=time.perf_counter()-start,
        upstream_fits=0,reused_frozen_stacks=27,gate_fits=108,updates=150,
        seeds=3,calibration_splits=3,inference_ensemble=False,test_observations=4744),indent=2),encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
