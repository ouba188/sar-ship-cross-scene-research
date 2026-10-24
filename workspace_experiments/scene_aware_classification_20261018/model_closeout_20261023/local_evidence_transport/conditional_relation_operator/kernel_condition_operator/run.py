"""Within-vessel centered nonlinear conditional transfer before aggregation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import itertools
import json
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('linear_class_pool',HERE.parent/'class_pooled_operator/run.py')
pool=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)
exp=pool.exp
ARMS=pool.ARMS


def projector(mask):
    v=mask.to(torch.float64) if not mask.is_floating_point() else mask
    m=v.sum(-1).clamp_min(1)
    return torch.diag_embed(v)-v[...,None]*v[...,None,:]/m[...,None,None]


class Reader(exp.Reader):
    def __init__(self,*args,kernel='rbf',**kwargs):
        super().__init__(*args,**kwargs)
        self.kernel=kernel
        self.legal=torch.as_tensor(self.numpy_legal.transpose(1,0,2).copy(),device=self.b.device,dtype=self.b.dtype)
        self.width=torch.as_tensor(json.loads((HERE/'bandwidth.json').read_text())['widths'],device=self.b.device,dtype=self.b.dtype)

    def coefficients(self,q,s,ql,sl,kind):
        # sl groups,class,reference,edge; each reference has its own centering.
        ms=sl.sum(-1)
        rv=ms>=2
        sl=sl*rv[...,None]
        m=ms.clamp_min(1)
        count=rv.sum(-1).clamp_min(1)
        w=(sl/(count[...,None,None]*m[...,None])).flatten(-2)
        hs0=projector(sl)
        eye=torch.eye(8,device=q.device,dtype=q.dtype)
        hs=torch.einsum('gcrij,rs->gcrisj',hs0,eye).reshape(4,8,64,64)
        hq=projector(ql)
        kss=s@s.transpose(-1,-2)
        kqs=torch.einsum('gbid,gcjd->gbcij',q,s)
        if self.kernel=='rbf':
            ss=s.square().sum(-1)
            qq=q.square().sum(-1)
            dss=(ss[...,None]+ss[...,None,:]-2*kss).clamp_min(0)
            dqs=(qq[:,:,None,:,None]+ss[:,None,:,None,:]-2*kqs).clamp_min(0)
            kss=torch.exp(-dss/self.width[:,kind,None,None,None])
            kqs=torch.exp(-dqs/self.width[:,kind,None,None,None,None])
        kss=hs@kss@hs
        kqs=hq[:,:,None]@kqs@hs[:,None]
        if self.arm=='covariance':
            return kqs*w[:,None,:,None,:],kss
        mi=m[...,None].expand(-1,-1,-1,8).reshape(4,8,64)
        lam=(kss.diagonal(dim1=-2,dim2=-1)*w/mi).sum(-1).clamp_min(1e-8)
        sw=w.sqrt()
        system=sw[...,None]*kss*sw[...,None,:]+lam[...,None,None]*torch.eye(64,device=q.device,dtype=q.dtype)
        solve=torch.linalg.solve_ex(system,torch.diag_embed(sw),check_errors=False)[0]
        a=(kqs*sw[:,None,:,None,:])@solve[:,None]
        return a,kss

    def scores(self,ids,mb,mn,trace=False):
        qb,qn=self.b[:,ids]@mb,self.n[:,ids]@mn
        sb=(self.sb@mb).reshape(4,8,64,32)
        sn=(self.sn@mn).reshape(4,8,64,32)
        ql=self.legal[:,ids]
        sl=self.legal[:,self.bank].reshape(4,8,8,8)
        ms=sl.sum(-1)
        rv=ms>=2
        mq=self.counts[:,ids].clamp_min(1)
        A,kss=self.coefficients(qb,sb,ql,sl,0)
        H=A@sn[:,None]
        T=(2*(qn[:,:,None]*H).sum((-2,-1))-H.square().sum((-2,-1)))/mq[:,:,None]
        ar=A.reshape(4,len(ids),8,8,8,8)
        gamma=sn.reshape(4,8,8,8,32).square().sum((-2,-1))/(ms-1).clamp_min(1)
        null=ar.square().sum((-3,-1))-ar.sum(-1).square().sum(-2)/ms[:,None].clamp_min(1)
        V=(null*gamma[:,None]).sum(-1)/mq[:,:,None]
        if self.arm=='independent':
            An,_=self.coefficients(qn,sn,ql,sl,1)
            hb=A@sb[:,None]
            hn=An@sn[:,None]
            body=(2*(qb[:,:,None]*hb).sum((-2,-1))-hb.square().sum((-2,-1)))/mq[:,:,None]
            near=(2*(qn[:,:,None]*hn).sum((-2,-1))-hn.square().sum((-2,-1)))/mq[:,:,None]
            value=.5*(body+near)
        elif self.arm=='covariance':
            value=2*(qn[:,:,None]*H).sum((-2,-1))/mq[:,:,None]
        else:
            value=T if self.arm=='transfer' else T+V
        valid=((self.counts[:,ids]>=2)[:,:,None]&rv.any(-1)[:,None]).permute(1,2,0)
        raw=torch.where(valid,value.permute(1,2,0),0.)
        mean=(raw*valid).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)
        z=torch.where(valid,(raw-mean)/self.divisor,0.)
        if trace:
            return z,raw,valid,dict(A=A,H=H,T=T,V=V,qb=qb,qn=qn,sb=sb.reshape(4,8,8,8,32),
                sn=sn.reshape(4,8,8,8,32),mq=mq,ms=ms,rv=rv,kss=kss)
        return z,raw,valid


def bandwidth(data):
    table,x,mass,bank,idx,tr,va,te=data
    b,n,legal,_=exp.edge.edges(x,mass)
    b,n,counts=exp.fixed.center(b,n,legal)
    fit=np.load(exp.edge.support.UP/'indices.npz')['fit_train']
    ids=table.iloc[fit[mass[fit,:,:8].sum((1,2))>0]].drop_duplicates('mmsi').index.to_numpy()
    rng=np.random.default_rng(613)
    widths=[]
    for g in range(4):
        row=[]
        for v in (b,n):
            z=v[ids,g][legal[ids,g]]
            pairs=rng.integers(len(z),size=(4096,2))
            d=((z[pairs[:,0]]-z[pairs[:,1]])**2).sum(-1)
            row.append(float(np.median(d[d>0])))
        widths.append(row)
    (HERE/'bandwidth.json').write_text(json.dumps(dict(seed=613,source='fit_train canonical MMSI',
        n_vessels=len(ids),widths=widths),indent=2),encoding='utf-8')


def checks(data):
    table,x,mass,bank,idx,tr,va,te=data
    utility=idx[tr]
    _,_,legal,_=exp.edge.edges(x,mass)
    ids=np.unique(np.r_[utility[:3],utility[~legal[utility,2].any(-1)][:1]])
    I=torch.eye(32,dtype=torch.float64)
    errors=[]
    for arm in ARMS:
        for partial in (False,True):
            override=legal.copy()
            if partial:
                override[ids[0],0,0]=False
                override[bank[0,0],0,1:4]=False
                override[bank[7],3]=False
            new=Reader(x,mass,bank,arm,device='cpu',dtype=torch.float64,kernel='linear',legal_override=override)
            old=pool.Reader(x,mass,bank,arm,device='cpu',dtype=torch.float64,legal_override=override)
            a=new.scores(ids,I,I)
            b=old.scores(ids,I,I)
            error=float((a[1]-b[1]).abs().max())
            assert error<1e-10,(arm,partial,error)
            assert torch.equal(a[2],b[2])
            errors.append(error)
    reader=Reader(x,mass,bank,'excess',device='cpu',dtype=torch.float64)
    reader.initialize(utility)
    reader.cache_reference(ids)
    model=exp.Metric(device='cpu',dtype=torch.float64)
    z,_,_=reader.scores(ids,*model.matrices())
    assert torch.count_nonzero(z-reader.reference[ids])==0
    y=torch.tensor(table.iloc[ids].class_id.to_numpy())
    def loss():
        value=reader.scores(ids,*model.matrices())[0]
        return torch.nn.functional.cross_entropy(exp.delta(value,model.a),y)
    loss().backward()
    gradients={}
    for name in ('Vb','Vn'):
        par=getattr(model,name)
        pos=np.unravel_index(int(par.grad.abs().argmax()),par.shape)
        actual=float(par.grad[pos]); eps=1e-5
        with torch.no_grad():
            par[pos]=eps; plus=float(loss()); par[pos]=-eps; minus=float(loss()); par[pos]=0
        error=abs(actual-(plus-minus)/(2*eps))
        assert error<1e-8 and abs(actual)>1e-12
        gradients[name]=dict(analytic=actual,error=error)
    # Exhaustive independent within-reference permutation checks the null, not autograd.
    rng=np.random.default_rng(71)
    q=rng.normal(size=(3,32)); target=rng.normal(size=(3,32)); target-=target.mean(0)
    source=rng.normal(size=(6,32)); neighbors=rng.normal(size=(2,3,32)); neighbors-=neighbors.mean(1,keepdims=True)
    def k(a,b): return np.exp(-((a[:,None]-b[None])**2).sum(-1)/32)
    h=np.eye(3)-np.ones((3,3))/3
    hs=np.kron(np.eye(2),h)
    ks=hs@k(source,source)@hs
    lam=np.trace(ks)/18
    A=h@k(q,source)@hs@np.linalg.inv(ks+6*lam*np.eye(6))
    def score(nn):
        H=A@nn.reshape(6,32)
        T=(2*np.sum(target*H)-np.sum(H*H))/3
        V=sum(np.sum(nn[r]**2)/2*(np.sum(A[:,r*3:(r+1)*3]**2)-np.sum(A[:,r*3:(r+1)*3].sum(-1)**2)/3)/3 for r in range(2))
        return T+V,np.mean(np.sum((target-H)**2,-1))
    real,realrisk=score(neighbors)
    vals=[]
    for p1,p2 in itertools.product(itertools.permutations(range(3)),repeat=2):
        vals.append(score(np.stack([neighbors[0,list(p1)],neighbors[1,list(p2)]])))
    vals=np.array(vals)
    nullerr=max(abs(vals[:,0].mean()),abs(vals[:,1].mean()-realrisk-real))
    assert nullerr<1e-11
    out=dict(linear_primal_dual_max_error=max(errors),partial_masks=True,initial_zero=True,
        gradients=gradients,independent_36_permutation_risk_error=float(nullerr))
    (HERE/'smoke.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(out,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    data=exp.source_data()
    if '--smoke' in sys.argv:
        bandwidth(data)
        checks(data)
    else:
        assert (HERE/'smoke.json').exists()
        pool.HERE=HERE
        pool.Reader=Reader
        pool.main(data)
