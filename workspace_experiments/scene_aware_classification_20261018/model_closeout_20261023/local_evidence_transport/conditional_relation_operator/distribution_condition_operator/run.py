"""Conditional relational risk with nonlinear neighborhood feature outputs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('input_kernel_parent',HERE.parent/'kernel_condition_operator/run.py')
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
exp=prior.exp
ARMS=prior.ARMS


class Reader(prior.Reader):
    def __init__(self,*args,output_kernel='rbf',**kwargs):
        super().__init__(*args,**kwargs)
        self.output_kernel=output_kernel

    def output_grams(self,q,s,ql,sl,kind):
        ms=sl.sum(-1)
        sl=sl*(ms>=2)[...,None]
        hs0=prior.projector(sl)
        hs=torch.einsum('gcrij,rs->gcrisj',hs0,torch.eye(8,device=q.device,dtype=q.dtype)).reshape(4,8,64,64)
        hq=prior.projector(ql)
        lss=s@s.transpose(-1,-2)
        lqs=torch.einsum('gbid,gcjd->gbcij',q,s)
        lqq=q@q.transpose(-1,-2)
        if self.output_kernel=='rbf':
            ss=s.square().sum(-1)
            qq=q.square().sum(-1)
            lss=torch.exp(-(ss[...,None]+ss[...,None,:]-2*lss).clamp_min(0)/self.width[:,kind,None,None,None])
            lqs=torch.exp(-(qq[:,:,None,:,None]+ss[:,None,:,None,:]-2*lqs).clamp_min(0)/self.width[:,kind,None,None,None,None])
            lqq=torch.exp(-(qq[...,None]+qq[...,None,:]-2*lqq).clamp_min(0)/self.width[:,kind,None,None,None])
        return hq[:,:,None]@lqs@hs[:,None],hs@lss@hs,hq@lqq@hq

    def scores(self,ids,mb,mn,trace=False):
        qb,qn=self.b[:,ids]@mb,self.n[:,ids]@mn
        sb=(self.sb@mb).reshape(4,8,64,32)
        sn=(self.sn@mn).reshape(4,8,64,32)
        ql=self.legal[:,ids]
        sl=self.legal[:,self.bank].reshape(4,8,8,8)
        ms=sl.sum(-1)
        rv=ms>=2
        mq=self.counts[:,ids].clamp_min(1)
        A,_=self.coefficients(qb,sb,ql,sl,0)
        Lqs,Lss,Lqq=self.output_grams(qn,sn,ql,sl,1)
        linear=2*(A*Lqs).sum((-2,-1))/mq[:,:,None]
        energy=((A@Lss[:,None])*A).sum((-2,-1))/mq[:,:,None]
        T=linear-energy
        ar=A.reshape(4,len(ids),8,8,8,8)
        gamma=Lss.diagonal(dim1=-2,dim2=-1).reshape(4,8,8,8).sum(-1)/(ms-1).clamp_min(1)
        null=ar.square().sum((-3,-1))-ar.sum(-1).square().sum(-2)/ms[:,None].clamp_min(1)
        V=(null*gamma[:,None]).sum(-1)/mq[:,:,None]
        if self.arm=='independent':
            An,_=self.coefficients(qn,sn,ql,sl,1)
            Bqs,Bss,_=self.output_grams(qb,sb,ql,sl,0)
            body=(2*(A*Bqs).sum((-2,-1))-((A@Bss[:,None])*A).sum((-2,-1)))/mq[:,:,None]
            near=(2*(An*Lqs).sum((-2,-1))-((An@Lss[:,None])*An).sum((-2,-1)))/mq[:,:,None]
            value=.5*(body+near)
        elif self.arm=='covariance':
            value=linear
        else:
            value=T if self.arm=='transfer' else T+V
        valid=((self.counts[:,ids]>=2)[:,:,None]&rv.any(-1)[:,None]).permute(1,2,0)
        raw=torch.where(valid,value.permute(1,2,0),0.)
        mean=(raw*valid).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)
        z=torch.where(valid,(raw-mean)/self.divisor,0.)
        if trace:
            detail=dict(A=A,Lqs=Lqs,Lss=Lss,Lqq=Lqq,T=T,V=V,qb=qb,qn=qn,
                sb=sb.reshape(4,8,8,8,32),sn=sn.reshape(4,8,8,8,32),mq=mq,ms=ms,rv=rv,gamma=gamma)
            if self.arm=='independent':
                detail.update(An=An,Bqs=Bqs,Bss=Bss)
            return z,raw,valid,detail
        return z,raw,valid


def smoke(data):
    table,x,mass,bank,idx,tr,va,te=data
    ids=idx[tr[:3]]
    _,_,legal,_=exp.edge.edges(x,mass)
    I=torch.eye(32,dtype=torch.float64)
    errors=[]
    for arm in ARMS:
        for partial in (False,True):
            override=legal.copy()
            if partial:
                override[ids[0],0,0]=False
                override[ids[1],2:]=False
                override[bank[0,0],0,1:4]=False
                override[bank[7],3]=False
            new=Reader(x,mass,bank,arm,device='cpu',dtype=torch.float64,output_kernel='linear',legal_override=override)
            old=prior.Reader(x,mass,bank,arm,device='cpu',dtype=torch.float64,legal_override=override)
            a=new.scores(ids,I,I)
            b=old.scores(ids,I,I)
            err=float((a[1]-b[1]).abs().max())
            assert err<1e-10,(arm,partial,err)
            assert torch.equal(a[2],b[2])
            errors.append(err)
    reader=Reader(x,mass,bank,'excess',device='cpu',dtype=torch.float64)
    reader.initialize(idx[tr])
    reader.cache_reference(ids)
    model=exp.Metric(device='cpu',dtype=torch.float64)
    z=reader.scores(ids,*model.matrices())[0]
    assert torch.count_nonzero(z-reader.reference[ids])==0
    y=torch.tensor(table.iloc[ids].class_id.to_numpy())
    def loss():
        return torch.nn.functional.cross_entropy(exp.delta(reader.scores(ids,*model.matrices())[0],model.a),y)
    loss().backward()
    gradients={}
    for name in ('Vb','Vn'):
        p=getattr(model,name)
        pos=np.unravel_index(int(p.grad.abs().argmax()),p.shape)
        actual=float(p.grad[pos]); eps=1e-5
        with torch.no_grad():
            p[pos]=eps; plus=float(loss()); p[pos]=-eps; minus=float(loss()); p[pos]=0
        err=abs(actual-(plus-minus)/(2*eps))
        assert err<1e-8 and abs(actual)>1e-12
        gradients[name]=dict(analytic=actual,error=err)
    result=dict(linear_output_replay_max_error=max(errors),partial_mask=True,initial_zero=True,gradients=gradients)
    (HERE/'smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result,flush=True)


def probe(data):
    table,x,mass,bank,idx,tr,va,te=data
    labels=table.class_id.to_numpy(int)
    rows=[]
    for output in ('linear','rbf'):
        for arm in ARMS:
            reader=Reader(x,mass,bank,arm,output_kernel=output)
            reader.initialize(idx[tr])
            I=torch.eye(32,device='cuda')
            for role,locs in [('utility_train',tr),('selection',va)]:
                values=[]; masks=[]
                with torch.no_grad():
                    for start in range(0,len(locs),128):
                        z,_,v=reader.scores(idx[locs[start:start+128]],I,I)
                        values.append(z.cpu().numpy()); masks.append(v.cpu().numpy())
                z=np.concatenate(values); valid=np.concatenate(masks)
                y=labels[idx[locs]]; pos=np.arange(len(y))
                for g,group in enumerate(exp.edge.GROUPS):
                    true=z[pos,y,g]
                    rival=np.where(np.arange(8)[None,:]==y[:,None],-np.inf,z[:,:,g]).max(-1)
                    mean=(z[:,:,g].sum(-1)-true)/7
                    for c in range(8):
                        keep=(y==c)&valid[pos,y,g]
                        if not keep.any():
                            continue
                        rows.append(dict(output=output,arm=arm,role=role,group=group,class_id=c,n=int(keep.sum()),
                            mean_gap=float((true-mean)[keep].mean()),best_gap=float((true-rival)[keep].mean()),
                            top1=float((z[:,:,g].argmax(-1)[keep]==c).mean())))
    frame=pd.DataFrame(rows)
    frame.to_csv(HERE/'probe_classes.csv',index=False)
    summary=frame.groupby(['output','arm','role','group'])[['mean_gap','best_gap','top1']].mean()
    summary.to_csv(HERE/'probe_summary.csv')
    passed=[]
    for g in exp.edge.GROUPS:
        signal=[summary.loc[('rbf','excess',role,g),'mean_gap'] for role in ('utility_train','selection')]
        shuffled=[summary.loc[('rbf','excess_shuffle101',role,g),'mean_gap'] for role in ('utility_train','selection')]
        if all(a>0 and a>b for a,b in zip(signal,shuffled)):
            passed.append(g)
    result=dict(non_degenerate_groups=passed,eligible_for_planned_fit=bool(passed),no_evaluation=True,
        criterion='same group positive macro true-minus-mean-wrong in utility and selection, exceeding fixed shuffle')
    (HERE/'probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(summary.xs('excess',level='arm').to_string(),flush=True)
    print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    data=exp.source_data()
    if '--smoke' in sys.argv:
        smoke(data)
    elif '--probe' in sys.argv:
        probe(data)
    else:
        assert (HERE/'smoke.json').exists()
        assert json.loads((HERE/'probe.json').read_text())['eligible_for_planned_fit']
        prior.pool.HERE=HERE
        prior.pool.Reader=Reader
        prior.pool.main(data)
