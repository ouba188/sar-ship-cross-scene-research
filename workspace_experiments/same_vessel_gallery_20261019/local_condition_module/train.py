"""Matched local residual training with a frozen, replayed conditional baseline."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
import copy
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits
from model import LocalReader

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'conditional_evidence_pilot'))
import run_pilot as prior


def conditions(f,train):
    data=np.load(ROOT/'conditions.npz'); raw=data['values']; valid=data['valid']
    result=np.zeros_like(raw)
    for pol in range(2):
        for j in range(3):
            used=train[valid[train,pol,j]>0]
            mean=raw[used,pol,j].mean() if len(used) else 0
            std=max(raw[used,pol,j].std(),1e-6) if len(used) else 1
            result[:,pol,j]=np.where(valid[:,pol,j]>0,(raw[:,pol,j]-mean)/std,0)
    return np.concatenate((result,valid),axis=-1).astype(np.float32)


def frozen_scores(f,train,val,test,fold):
    x=prior.image_features(f,train,fold)
    c=prior.standardize(f[['cos2','sin2','incidence']].to_numpy(float),train)
    features=prior.product_features(x,c); features=prior.standardize(features,train)/np.sqrt(features.shape[1])
    previous=pd.read_csv(prior.ROOT/'metrics.csv')
    alpha=previous.query('fold==@fold and arm=="interaction" and partition=="test"').iloc[0].alpha
    model=RidgeClassifier(alpha=alpha,solver='cholesky').fit(features[train],f.class_id.to_numpy()[train],sample_weight=prior.weights(f,train))
    scores=model.decision_function(features).astype(np.float32)
    old=pd.read_csv(prior.ROOT/'predictions.csv'); ref=old.query('fold==@fold and arm=="interaction"').set_index('obs_id').loc[f.iloc[test].obs_id]
    assert np.array_equal(scores[test].argmax(1),ref.prediction.to_numpy())
    other=scores[test].copy(); other[np.arange(len(test)),f.iloc[test].class_id]=-np.inf
    margins=scores[test,np.array(f.iloc[test].class_id)]-other.max(1)
    assert np.allclose(margins,ref.margin,atol=1e-6)
    np.save(ROOT/f'baseline_fold{fold}.npy',scores)
    del x,features; gc.collect()
    return scores


def main():
    started=time.perf_counter(); torch.set_num_threads(2)
    f=pd.read_csv(prior.ROOT/'cohort.csv',dtype={'mmsi':str}); y=f.class_id.to_numpy()
    maps=np.load(ROOT/'maps.npy',mmap_mode='r'); masks=np.load(ROOT/'masks.npy',mmap_mode='r')
    alignment=pd.read_csv(ROOT/'crossview_alignment.csv').set_index('obs_id').loc[f.obs_id]
    conflict=alignment['disable_new_vh'].to_numpy(bool)
    results=[]; predictions=[]; history=[]; interventions=[]; used_states=[]
    device='cuda'
    def batch(ii,cond):
        x=torch.from_numpy(np.array(maps[ii],dtype=np.float32)).to(device)
        m=torch.from_numpy(np.array(masks[ii],dtype=np.float32)).to(device)
        m[torch.as_tensor(conflict[ii],device=device),1]=0
        c=torch.from_numpy(cond[ii]).to(device)
        return x,m,c
    def evaluate(net,ii,cond,base,drop=None):
        output=[]; net.eval()
        with torch.no_grad():
            for first in range(0,len(ii),128):
                jj=ii[first:first+128]; x,m,c=batch(jj,cond)
                output.append(base[jj]+net(x,m,c,drop_role=drop).cpu().numpy())
        return np.concatenate(output)
    for fold in range(3):
        train,val,test=prior.indices(f,fold)
        # Preserve the original split assignments; quarantine only confirmed bad pairs.
        excluded=pd.read_csv(ROOT/'EXCLUSIONS.csv').obs_id.to_numpy()
        train,val,test=[ii[~f.iloc[ii].obs_id.isin(excluded).to_numpy()] for ii in (train,val,test)]
        base=frozen_scores(f,train,val,test,fold); correct=conditions(f,train)
        wrong=correct.copy(); rng=np.random.default_rng(20261019+fold)
        for ii in (train,val,test):
            for _,g in f.iloc[ii].groupby(['port','platform','quarter']):
                jj=g.index.to_numpy(); wrong[jj]=correct[rng.permutation(jj)]
        np.savez(ROOT/f'condition_fold{fold}.npz',correct=correct,wrong=wrong)
        w=prior.weights(f,train); probabilities=w/w.sum()
        for seed in (20261019,20261020,20261021):
            for arm in ('plain','post','pre','wrong'):
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                net=LocalReader(arm).to(device)
                cond=wrong if arm=='wrong' else correct
                optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
                rng=np.random.default_rng(seed+fold*100)
                best=prior.metrics(y[val],base[val]); best_step=0; state=copy.deepcopy(net.state_dict())
                history.append(dict(fold=fold,seed=seed,arm=arm,step=0,train_loss=np.nan,**best))
                began=time.perf_counter()
                for step in range(1,151):
                    net.train(); jj=rng.choice(train,size=48,replace=True,p=probabilities)
                    x,m,c=batch(jj,cond)
                    target=2*F.one_hot(torch.as_tensor(y[jj],device=device),8).float()-1
                    scores=torch.as_tensor(base[jj],device=device)+net(x,m,c)
                    loss=F.mse_loss(scores,target)
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                    if step%50==0:
                        val_scores=evaluate(net,val,cond,base); met=prior.metrics(y[val],val_scores)
                        history.append(dict(fold=fold,seed=seed,arm=arm,step=step,train_loss=float(loss.detach().cpu()),**met))
                        if (met['ba'],-met['brier'])>(best['ba'],-best['brier']):
                            best=met; best_step=step; state=copy.deepcopy(net.state_dict())
                endpoint=evaluate(net,test,cond,base)
                results.append(dict(fold=fold,seed=seed,arm=arm,selection='endpoint150',step=150,seconds=time.perf_counter()-began,**prior.metrics(y[test],endpoint)))
                net.load_state_dict(state)
                scores=evaluate(net,test,cond,base)
                results.append(dict(fold=fold,seed=seed,arm=arm,selection='source_selected',step=best_step,seconds=time.perf_counter()-began,**prior.metrics(y[test],scores)))
                torch.save({k:v.cpu() for k,v in state.items()},ROOT/f'model_f{fold}_s{seed}_{arm}.pt')
                used_states.append(dict(fold=fold,seed=seed,arm=arm,selected_step=best_step,parameter_count=sum(p.numel() for p in net.parameters())))
                for name,values in [('base',base[test]),('source_selected',scores),('endpoint150',endpoint)]:
                    pred=values.argmax(1); other=values.copy(); other[np.arange(len(test)),y[test]]=-np.inf
                    margin=values[np.arange(len(test)),y[test]]-other.max(1)
                    for k,i in enumerate(test):
                        predictions.append(dict(fold=fold,seed=seed,arm=arm,selection=name,obs_id=int(f.iloc[i].obs_id),mmsi=f.iloc[i].mmsi,port=f.iloc[i].port,class_id=int(y[i]),prediction=int(pred[k]),correct=bool(pred[k]==y[i]),margin=float(margin[k]),crossview_conflict=bool(conflict[i])))
                if arm=='pre':
                    for label,cc,drop in [('drop_target',correct,0),('drop_other',correct,1),('drop_rest',correct,2),('wrong_condition',wrong,None)]:
                        alt=evaluate(net,test,cc,base,drop)
                        interventions.append(dict(fold=fold,seed=seed,intervention=label,**prior.metrics(y[test],alt)))
                        np.savez_compressed(ROOT/f'intervention_f{fold}_s{seed}_{label}.npz',obs_id=f.iloc[test].obs_id.to_numpy(),scores=alt)
                pd.DataFrame(results).to_csv(ROOT/'metrics.csv',index=False)
                pd.DataFrame(history).to_csv(ROOT/'history.csv',index=False)
                pd.DataFrame(predictions).to_csv(ROOT/'predictions.csv',index=False)
                pd.DataFrame(interventions).to_csv(ROOT/'interventions.csv',index=False)
                pd.DataFrame(used_states).to_csv(ROOT/'selected_states.csv',index=False)
                print('DONE',fold,seed,arm,'step',best_step,prior.metrics(y[test],scores),'seconds',round(time.perf_counter()-began,1),flush=True)
                del net,state,optimizer; gc.collect(); torch.cuda.empty_cache()
    (ROOT/'RECEIPT.json').write_text(json.dumps(dict(status='complete',fits=36,seconds=time.perf_counter()-started,neural_steps_per_fit=150,scope='local residual on frozen global probe, not support-adapted full system',conflict_vh_disabled=int(conflict.sum())),indent=2),encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
