"""Source-only bounded gates over an unchanged neighborhood residual."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('level_gate',ROOT.parent/'neighborhood_level_split_20261021/run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
ARMS=('constant','content','geometry','joint','sham')


class Gate(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        if arm=='constant':self.bias=nn.Parameter(torch.zeros(()))
        else:
            self.first=nn.Linear(40,8);self.last=nn.Linear(8,1)
            nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)

    def forward(self,c,g):
        if self.arm=='constant':raw=self.bias.expand(len(c))
        else:
            if self.arm=='content':g=torch.cat((torch.zeros_like(g[:,:6]),g[:,6:]),1)
            if self.arm=='geometry':c=torch.zeros_like(c)
            raw=self.last(F.gelu(self.first(torch.cat((c,g),1)))).squeeze(1)
            if self.arm=='geometry':raw=raw*(g[:,6:].sum(1)>0)
        return 2*torch.sigmoid(raw)


def geometry(f,usable):
    raw,valid=old.relation.conditions(f);valid &= usable.astype(bool)
    raw[:,:,2]=(raw[:,:,2]-35)/10
    raw*=valid[:,:,None]
    assert np.isfinite(raw).all()
    return np.concatenate((raw.reshape(len(f),6),valid.astype(np.float32)),1),valid


def mismatch(g,f,splits,fold):
    wrong=g.copy();records=[];donors=np.arange(len(f))
    rng=np.random.default_rng(20261021+fold)
    groups=f[['port','platform','quarter']].copy()
    groups['pattern']=g[:,6]+2*g[:,7]
    for part,ii in zip(('train','val','test'),splits):
        for _,block in groups.iloc[ii].groupby(['port','platform','quarter','pattern']):
            ids=block.index.to_numpy();donors[ids]=rng.permutation(ids)
        assert np.isin(donors[ii],ii).all()
        wrong[ii]=g[donors[ii]]
        np.testing.assert_array_equal(wrong[ii,6:],g[ii,6:])
        records.append(dict(fold=fold,partition=part,n=len(ii),donor_changed=int((donors[ii]!=ii).sum()),
                            geometry_changed=int((np.abs(wrong[ii,:6]-g[ii,:6]).max(1)>1e-6).sum())))
    return wrong,donors,records


def fit():
    torch.set_num_threads(2);start=time.perf_counter()
    f,splits=old.relation.load_data();y=f.class_id.to_numpy();yy=torch.tensor(y,device='cuda')
    cache=np.load(old.ROOT/'features.npz');g,valid=geometry(f,cache['usable'])
    raw=np.load(old.five.AXIS/'descriptors.npz');raw_base=np.concatenate((raw['base'],raw['axis']),1)
    raw_five=np.load(old.strip.PREV/'keep_refs.npy',mmap_mode='r').reshape(len(f),60,64)
    metrics=[];selection=[];predictions=[];checks=[];donor_stats=[];coverage=[];weight_stats=[]
    for fold,(tr,val,test) in enumerate(splits):
        material,_=old.standardize(cache,tr);x=torch.tensor(material['separated'],device='cuda')
        wrong,donors,info=mismatch(g,f,(tr,val,test),fold);donor_stats+=info
        np.savez_compressed(ROOT/f'geometry_f{fold}.npz',geometry=g,sham=wrong,donors=donors)
        for part,ii in zip(('train','val','test'),(tr,val,test)):
            coverage.append(dict(fold=fold,partition=part,n=len(ii),vv=int(valid[ii,0].sum()),
                                 vh=int(valid[ii,1].sum()),both=int(valid[ii].all(1).sum())))
        geo=torch.tensor(g,device='cuda');sham=torch.tensor(wrong,device='cuda')
        weights=torch.tensor(old.prior.weights(f,tr),device='cuda',dtype=torch.float32)
        for seed in range(3):
            a=old.strip.frozen_scores(fold,seed,raw_base,raw_five).astype(np.float32)
            saved=torch.load(old.ROOT/f'head_f{fold}_separated_s{seed}.pt',map_location='cpu',weights_only=False)
            d=old.Residual().cuda().eval();d.load_state_dict(saved['state'])
            with torch.no_grad():
                r=d(x);h=F.gelu(d.content(x));mean=h[tr].mean(0);std=h[tr].std(0,unbiased=False).clamp_min(.001)
                c=torch.tanh((h-mean)/std)
            rn=r.cpu().numpy();reference=a+rn;base=torch.tensor(a,device='cuda')
            prior=np.load(old.ROOT/f'scores_f{fold}_separated_s{seed}.npz')
            np.testing.assert_array_equal(test,prior['rows'])
            err=float(np.abs(reference[test]-prior['scores']).max());assert err<2e-5
            checks.append(dict(fold=fold,seed=seed,test='frozen_reference',error=err))
            np.savez_compressed(ROOT/f'inputs_f{fold}_s{seed}.npz',content=c.cpu().numpy(),
                hidden_mean=mean.cpu().numpy(),hidden_std=std.cpu().numpy(),base=a,residual=rn)
            for arm in ARMS:
                torch.manual_seed(20261021+seed);gate=Gate(arm).cuda();gg=sham if arm=='sham' else geo
                optimizer=torch.optim.AdamW(gate.parameters(),lr=.001,weight_decay=.001)
                rng=np.random.default_rng(20261021+seed+31*fold)
                best=None;chosen=0;state=None
                for step in range(151):
                    if step in (0,50,100,150):
                        with torch.no_grad():score=(base[val]+gate(c[val],gg[val])[:,None]*r[val]).cpu().numpy()
                        met=old.control.metrics(y[val],score);selection.append(dict(fold=fold,seed=seed,arm=arm,step=step,**met))
                        rank=(met['ba'],-met['brier'])
                        if best is None or rank>best:
                            best=rank;chosen=step;state={k:v.detach().cpu().clone() for k,v in gate.state_dict().items()}
                        if step==0:np.testing.assert_array_equal(score,reference[val])
                    if step==150:break
                    pick=rng.integers(len(tr),size=128);ii=tr[pick]
                    score=base[ii]+gate(c[ii],gg[ii])[:,None]*r[ii]
                    target=2*F.one_hot(yy[ii],8).float()-1
                    loss=((score-target).square().mean(1)*weights[pick]).mean()
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                for part,ii in (('terminal_train',tr),('terminal_val',val)):
                    with torch.no_grad():score=(base[ii]+gate(c[ii],gg[ii])[:,None]*r[ii]).cpu().numpy()
                    metrics.append(dict(fold=fold,seed=seed,arm=arm,partition=part,step=150,**old.control.metrics(y[ii],score)))
                gate.load_state_dict(state)
                torch.save(dict(state=state,step=chosen,arm=arm),ROOT/f'head_f{fold}_{arm}_s{seed}.pt')
                with torch.no_grad():w=gate(c,gg).cpu().numpy()
                score=a+w[:,None]*rn
                replay=Gate(arm).cuda();replay.load_state_dict(state)
                with torch.no_grad():wr=replay(c,gg).cpu().numpy()
                np.testing.assert_array_equal(w,wr)
                checks.append(dict(fold=fold,seed=seed,arm=arm,test='selected_gate_replay',error=0.))
                for part,ii in (('train',tr),('val',val),('test',test)):
                    metrics.append(dict(fold=fold,seed=seed,arm=arm,partition=part,step=chosen,**old.control.metrics(y[ii],score[ii])))
                    weight_stats.append(dict(fold=fold,seed=seed,arm=arm,partition=part,step=chosen,
                        mean=float(w[ii].mean()),std=float(w[ii].std()),minimum=float(w[ii].min()),maximum=float(w[ii].max()),
                        residual_nonzero=int((np.abs(rn[ii]).max(1)>1e-8).sum())))
                np.savez_compressed(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz',rows=test,scores=score[test],
                                    weight=w[test],reference=reference[test],base=a[test])
                variants={arm:(score[test],w[test])}
                if arm=='joint':
                    for name,geom in (('joint_zero_geometry',torch.cat((torch.zeros_like(geo[:,:6]),geo[:,6:]),1)),
                                      ('joint_shuffled_geometry',sham)):
                        with torch.no_grad():ww=gate(c,geom).cpu().numpy()
                        variants[name]=(a[test]+ww[test,None]*rn[test],ww[test])
                    wm=np.full(len(test),w[tr].mean(),np.float32)
                    variants['joint_source_mean_weight']=(a[test]+wm[:,None]*rn[test],wm)
                if arm=='constant':
                    variants.update(reference=(reference[test],np.ones(len(test))),baseline=(a[test],np.zeros(len(test))))
                for name,(ss,ww) in variants.items():
                    wrongs=ss.copy();wrongs[np.arange(len(test)),y[test]]=-np.inf
                    for j,i in enumerate(test):
                        predictions.append(dict(fold=fold,seed=seed,arm=name,obs_id=int(f.iloc[i].obs_id),mmsi=f.iloc[i].mmsi,
                            class_id=int(y[i]),port=f.iloc[i].port,prediction=int(ss[j].argmax()),
                            correct=bool(ss[j].argmax()==y[i]),margin=float(ss[j,y[i]]-wrongs[j].max()),weight=float(ww[j]),
                            geometry_both=bool(valid[i].all()),r_nonzero=bool(np.abs(rn[i]).max()>1e-8),step=chosen if name in ARMS else -1))
                print('fit',fold,seed,arm,chosen,metrics[-1]['oa'],metrics[-1]['ba'],flush=True)
            for name,data in (('metrics',metrics),('validation_selection',selection),('predictions',predictions),
                              ('checks',checks),('weights',weight_stats),('geometry_coverage',coverage),('mismatch_coverage',donor_stats)):
                pd.DataFrame(data).to_csv(ROOT/f'{name}.csv',index=False,lineterminator='\n')
    (ROOT/'TRAINING.json').write_text(json.dumps(dict(status='complete',fits=45,seconds=time.perf_counter()-start,
        selected_head_replays=45,reference_replays=9,source_only=True,gate_parameters=337,constant_parameters=1),indent=2),encoding='utf-8')


if __name__=='__main__':fit()
