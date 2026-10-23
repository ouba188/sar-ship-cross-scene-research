"""Equal-vessel conditional operators estimated before class score aggregation."""
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
spec=importlib.util.spec_from_file_location('pooled_existing_episode',HERE.parent/'reference_resampling/episodic/run.py')
epi=importlib.util.module_from_spec(spec)
spec.loader.exec_module(epi)
exp=epi.exp
ARMS=('independent','covariance','transfer','excess','excess_shuffle101')


class Reader(exp.Reader):
    def scores(self,ids,mb,mn,trace=False):
        qb,qn=self.b[:,ids]@mb,self.n[:,ids]@mn
        sb=(self.sb@mb).reshape(4,8,8,8,32)
        sn=(self.sn@mn).reshape(4,8,8,8,32)
        ms=self.ms.reshape(4,8,8)
        rv=ms>=2
        k=rv.sum(-1)
        m=ms.clamp_min(1)
        weight=rv.to(sb.dtype)/(k.clamp_min(1)[...,None]*m)
        ub=sb.transpose(-1,-2)@sb
        un=sn.transpose(-1,-2)@sn
        cbn=sb.transpose(-1,-2)@sn
        G=(ub*weight[...,None,None]).sum(2)
        C=(cbn*weight[...,None,None]).sum(2)
        Gn=(un*weight[...,None,None]).sum(2)
        lam=(ub.diagonal(dim1=-2,dim2=-1).sum(-1)*weight/m).sum(-1).clamp_min(1e-8)
        I=torch.eye(32,device=G.device,dtype=G.dtype)
        R=torch.linalg.solve_ex(G+lam[...,None,None]*I,I.expand_as(G),check_errors=False)[0]
        operator=R@C
        mq=self.counts[:,ids].clamp_min(1)
        Gq=(qb.transpose(-1,-2)@qb)/mq[...,None,None]
        Cq=(qb.transpose(-1,-2)@qn)/mq[...,None,None]
        Nq=(qn.transpose(-1,-2)@qn)/mq[...,None,None]
        linear=2*torch.einsum('gbij,gcij->gbc',Cq,operator)
        energy=torch.einsum('gbij,gcij->gbc',Gq,operator@operator.transpose(-1,-2))
        T=linear-energy
        bsum=sb.sum(-2)
        zero_center=ub-bsum[...,None]*bsum[...,None,:]/m[...,None,None]
        gamma=sn.square().sum((-2,-1))/(ms-1).clamp_min(1)
        D=(zero_center*(gamma*weight.square())[...,None,None]).sum(2)
        V=torch.einsum('gbij,gcij->gbc',Gq,R@D@R.transpose(-1,-2))
        if self.arm=='independent':
            lamn=(un.diagonal(dim1=-2,dim2=-1).sum(-1)*weight/m).sum(-1).clamp_min(1e-8)
            Pn=torch.linalg.solve_ex(Gn+lamn[...,None,None]*I,Gn,check_errors=False)[0]
            Pb=R@G
            body=torch.einsum('gbij,gcij->gbc',Gq,2*Pb-Pb@Pb.transpose(-1,-2))
            near=torch.einsum('gbij,gcij->gbc',Nq,2*Pn-Pn@Pn.transpose(-1,-2))
            value=.5*(body+near)
        elif self.arm=='covariance':
            value=2*torch.einsum('gbij,gcij->gbc',Cq,C)
        else:
            value=T if self.arm=='transfer' else T+V
        valid=((self.counts[:,ids]>=2)[:,:,None]&(k>0)[:,None]).permute(1,2,0)
        raw=value.permute(1,2,0)
        raw=torch.where(valid,raw,0.)
        center=(raw*valid).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)
        z=torch.where(valid,(raw-center)/self.divisor,0.)
        if trace:
            return z,raw,valid,dict(qb=qb,qn=qn,sb=sb,sn=sn,ms=ms,mq=mq,rv=rv,weight=weight,
                G=G,C=C,R=R,operator=operator,T=T,V=V,lambda_body=lam)
        return z,raw,valid


def smoke(data):
    table,x,mass,bank,idx,tr,va,te=data
    utility=idx[tr]
    ids=utility[:3]
    y=table.class_id.to_numpy(int)
    base=np.load(exp.HERE/'predictions_17.npz')['full'][ids]
    rows={}
    for arm in ARMS:
        reader=Reader(x,mass,bank,arm,device='cpu',dtype=torch.float64)
        reader.initialize(utility)
        reader.cache_reference(ids)
        model=exp.Metric(device='cpu',dtype=torch.float64)
        z,raw,valid,detail=reader.scores(ids,*model.matrices(),trace=True)
        assert torch.count_nonzero(z-reader.reference[ids])==0
        # An explicit sum of transferred reference responses checks omitted cross terms.
        g,c,q=0,0,0
        qb,qn=detail['qb'][g,q],detail['qn'][g,q]
        H=torch.zeros_like(qn)
        null=torch.tensor(0.,dtype=torch.float64)
        for r in range(8):
            if not detail['rv'][g,c,r]:
                continue
            sb,sn=detail['sb'][g,c,r],detail['sn'][g,c,r]
            m=detail['ms'][g,c,r]
            A=(qb@detail['R'][g,c]@sb.T)*detail['weight'][g,c,r]
            H+=A@sn
            null+=sn.square().sum()/(m-1)*(A.square().sum()-A.sum(-1).square().sum()/m)/detail['mq'][g,q]
        T=(2*(qn*H).sum()-H.square().sum())/detail['mq'][g,q]
        err=max(abs(float(T-detail['T'][g,q,c])),abs(float(null-detail['V'][g,q,c])))
        assert err<1e-10,err
        def loss():
            zz=reader.scores(ids,*model.matrices())[0]
            return F.cross_entropy(torch.tensor(base,dtype=torch.float64)+exp.delta(zz-reader.reference[ids],model.a),torch.tensor(y[ids]))
        loss().backward()
        grads={}
        for key in ('Vb','Vn'):
            param=getattr(model,key)
            pos=np.unravel_index(int(param.grad.abs().argmax()),param.shape)
            exact=float(param.grad[pos]); eps=1e-5
            assert abs(exact)>1e-12,(arm,key)
            with torch.no_grad():
                param[pos]=eps; plus=float(loss())
                param[pos]=-eps; minus=float(loss())
                param[pos]=0
            numerical=(plus-minus)/(2*eps)
            assert abs(exact-numerical)<1e-8,(arm,key,exact,numerical)
            grads[key]=dict(analytic=exact,numeric=numerical,error=abs(exact-numerical))
        rows[arm]=dict(explicit_prediction_error=err,initial_zero=True,gradients=grads)
    (HERE/'smoke.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(rows,flush=True)


def main(data):
    started=time.perf_counter()
    table,x,mass,old,idx,tr,va,te=data
    y=table.class_id.to_numpy(int)
    _,_,legal,_=exp.edge.edges(x,mass)
    banks,_=epi.diag.banks_from_source(table,mass,legal,old)
    training_banks=epi.schedule(table,mass,old,legal)
    metrics,pairs,curves,receipts,classes,scales=[],[],[],[],[],[]
    weights={}
    for arm in ARMS:
        trainreader=Reader(x,mass,old,arm)
        rms,count=trainreader.initialize(idx[tr])
        for g in range(4):
            scales.append(dict(arm=arm,feature=g,group=exp.edge.GROUPS[g],rms=float(rms[g]),divisor=float(trainreader.divisor[g].cpu()),n=int(count[g])))
        pd.DataFrame(scales).to_csv(HERE/'scales.csv',index=False)
        readers={}
        for name,bank in banks.items():
            reader=Reader(x,mass,bank,arm)
            reader.divisor=trainreader.divisor
            reader.cache_reference(idx[va])
            readers[name]=reader
        for seed in exp.SEEDS:
            tick=time.perf_counter()
            full=np.load(exp.HERE/f'predictions_{seed}.npz')['full']
            model=exp.Metric()
            best,pred=epi.select_score(readers,model,idx[va],full,y)
            for v in pred.values():
                np.testing.assert_array_equal(v,full[idx[va]])
            beststep,beststate=0,model.state()
            opt=torch.optim.Adam(model.parameters(),lr=.01)
            ids=idx[tr]
            target=torch.tensor(y[ids],device='cuda')
            count=np.bincount(y[ids],minlength=8)
            w=torch.tensor(.5/len(ids)+.5/(8*count[y[ids]]),device='cuda',dtype=torch.float32)
            baseline=torch.tensor(full[ids],device='cuda')
            states={}
            for step in range(201):
                if step:
                    epi.reset_bank(trainreader,training_banks[step-1])
                    trainreader.cache_reference(ids)
                    opt.zero_grad(set_to_none=True)
                    for first in range(0,len(ids),128):
                        part=ids[first:first+128]
                        z,_,_=trainreader.scores(part,*model.matrices())
                        logits=baseline[first:first+128]+exp.delta(z-trainreader.reference[part],model.a)
                        loss=(F.cross_entropy(logits,target[first:first+128],reduction='none')*w[first:first+128]).sum()
                        loss.backward()
                    model.regularizer().backward()
                    opt.step()
                if step not in exp.CHECKPOINTS:
                    continue
                j,pred=epi.select_score(readers,model,idx[va],full,y)
                state=model.state()
                states.update({f'{k}_{step}':v for k,v in state.items()})
                row=dict(arm=arm,seed=seed,step=step,selection_J=float(j),Mb_change=float(np.linalg.norm(state['Mb']-np.eye(32))),Mn_change=float(np.linalg.norm(state['Mn']-np.eye(32))))
                row.update({'selection_J_'+name:exp.dep.metrics(y[idx[va]],v)['J'] for name,v in pred.items()})
                curves.append(row)
                assert all(np.isfinite(v).all() for v in pred.values())
                if j>best:
                    best,beststep,beststate=j,step,state
                print('CHECK',arm,seed,step,'J',float(j),'seconds',round(time.perf_counter()-tick,1),flush=True)
            np.savez_compressed(HERE/f'checkpoints_{arm}_{seed}.npz',**states)
            model.restore(beststate)
            weights.update({f'{arm}_{seed}_{k}':v for k,v in beststate.items()})
            saved=dict(idx=idx,y=y[idx],full=full[idx])
            for name,reader in readers.items():
                reader.cache_reference(idx)
                score,z,_=exp.infer(reader,model,idx,full)
                saved[name]=score
                if name=='original':
                    saved['parts']=z*beststate['a']
                for role,loc in [('utility_train',tr),('selection',va),('evaluation',te)]:
                    meta=dict(arm=arm,seed=seed,bank=name,role=role,step=beststep)
                    metrics.append(dict(**meta,**exp.dep.metrics(y[idx[loc]],score[loc])))
                    before=full[idx[loc]].argmax(-1)==y[idx[loc]]
                    after=score[loc].argmax(-1)==y[idx[loc]]
                    pairs.append(dict(**meta,rescued=int((~before&after).sum()),harmed=int((before&~after).sum()),net=int(after.sum()-before.sum())))
                    for c in range(8):
                        mask=y[idx[loc]]==c
                        classes.append(dict(**meta,class_id=c,n=int(mask.sum()),correct=int(after[mask].sum()),full_correct=int(before[mask].sum())))
            np.savez_compressed(HERE/f'predictions_{arm}_{seed}.npz',**saved)
            np.savez_compressed(HERE/'weights.npz',**weights)
            receipts.append(dict(arm=arm,seed=seed,selected_step=beststep,selection_J=float(best),parameters=516,seconds=time.perf_counter()-tick))
            for name,rows in [('metrics',metrics),('paired',pairs),('curves',curves),('receipts',receipts),('classes',classes)]:
                pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
            print('SELECTED',arm,seed,beststep,flush=True)
    frame=pd.DataFrame(metrics)
    summary=frame[frame.role.eq('evaluation')].groupby(['arm','bank'])[['OA','BA','J','NLL','Brier']].mean()
    summary.to_csv(HERE/'summary.csv')
    receipt=dict(fits=15,steps=200,configs=list(exp.SEEDS),arms=list(ARMS),parameters=516,
        init_seed=17,schedule='same10001..10200',source_selection='mean six-bank exact J',
        main_bank='original only',per_arm_original_utility_identity_RMS=True,seconds=time.perf_counter()-started)
    (HERE/'execution.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(summary.to_string(),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    data=exp.source_data()
    if '--smoke' in sys.argv:
        smoke(data)
    else:
        assert (HERE/'smoke.json').exists()
        main(data)
