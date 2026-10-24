"""Source-only reference-deletion covariance and actual bank sensitivity."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('uncertainty_previous',HERE.parent/'competitive_evidence_operator/run.py')
prev=importlib.util.module_from_spec(spec); spec.loader.exec_module(prev)


@torch.no_grad()
def profiles(q,mask,bank,ids):
    s=q[bank]; sm=mask[bank]; count=sm.sum(-1); rv=count>0
    valid=rv.any(1).T
    n=rv.sum(1).T
    means=[]; covs=[]; refs=[]
    for first in range(0,len(ids),64):
        query=q[ids[first:first+64]]
        sim=torch.einsum('bgtd,crgud->bgtcru',query,s)*np.sqrt(32)
        sim=sim.masked_fill(~sm.permute(2,0,1,3)[None,:,None],-1e4)
        per=torch.logsumexp(sim,-1)-count.clamp_min(1).log().permute(2,0,1)[None,:,None]
        per=per.masked_fill(~rv.permute(2,0,1)[None,:,None],-1e4)
        # Relative positive values avoid subtracting nearly equal exponentials.
        h=(per-per.max(-1,keepdim=True).values).exp()*rv.permute(2,0,1)[None,:,None]
        total=h.sum(-1,keepdim=True)
        leave=(total-h).clamp_min(1e-20).log()-(n-1).clamp_min(1).log()[None,:,None,:,None]
        rmask=rv.permute(2,0,1)[None,:,None]
        mean=(leave*rmask).sum(-1,keepdim=True)/n[None,:,None,:,None].clamp_min(1)
        centered=(leave-mean)*rmask
        cov=torch.einsum('bgtcr,bgucr->bgctu',centered,centered)
        cov*=((n-1).clamp_min(0)/n.clamp_min(1))[None,:,:,None,None]
        cov*= (n>=2)[None,:,:,None,None]
        value=torch.logsumexp(per,-1)-n.clamp_min(1).log()[None,:,None]
        value=torch.where(valid[None,:,None],value,0.)
        means.append(value.cpu()); covs.append(cov.cpu()); refs.append(per.cpu())
    return torch.cat(means),torch.cat(covs),valid.cpu(),torch.cat(refs),n.cpu()


def main():
    torch.set_num_threads(2)
    table,x,mass,bank,idx,tr,va,te=prev.exp.source_data()
    ids=idx[np.r_[tr,va]]
    q,m=prev.organize(x,mass); q=torch.tensor(q,device='cuda'); m=torch.tensor(m,device='cuda')
    _,_,legal,_=prev.exp.edge.edges(x,mass)
    banks,_=prev.epi.diag.banks_from_source(table,mass,legal,bank)
    values=[]
    for name,b in banks.items():
        v,c,cv,r,n=profiles(q,m,b,ids)
        mm=m[ids].cpu().float(); w=mm/mm.sum(-1,keepdim=True).clamp_min(1)
        pool=(v*w[...,None]).sum(2)
        values.append(pool.numpy())
        if name=='original': original=(v,c,cv,r,n)
        print('SOURCE BANK',name,flush=True)
    v,c,cv,r,n=original
    previous,_=prev.evidence(q,m,bank,ids)
    cent=(v*cv[None,:,None]).sum(-1,keepdim=True)/cv.sum(-1)[None,:,None,None].clamp_min(1)
    replay=float(abs(torch.where(cv[None,:,None],v-cent,0.)-previous).max())
    assert replay<3e-6,replay
    # One literal deletion recomputation detects subtraction and reference-axis errors.
    rr=0; gg=0; cc=0; tt=2; loc=7
    per=r[loc,gg,tt,cc].numpy(); good=np.flatnonzero(per>-9999)
    leave=np.array([np.log(np.exp(per[np.setdiff1d(good,[j])]-per[good].max()).mean()) for j in good])
    scalar=((len(good)-1)/len(good))*np.sum((leave-leave.mean())**2)
    ce=abs(scalar-float(c[loc,gg,cc,tt,tt])); assert ce<1e-6,ce
    yy=table.iloc[ids].class_id.to_numpy(int); meta=table.iloc[ids].reset_index(drop=True)
    base=np.load(prev.exp.HERE/'predictions_17.npz')['full'][ids]
    top=np.argsort(-base,axis=-1)[:,:2]
    means=np.stack(values)
    rows=[]
    for g,name in enumerate(prev.GROUPS):
        weights=w[:,g].numpy(); covariance=c[:,g].numpy()
        for i in range(len(ids)):
            if not mm[i,g].any() or not cv[g].all() or (n[g]<2).any(): continue
            a,b=top[i]; cov=covariance[i,a]+covariance[i,b]
            predvar=max(float(weights[i]@cov@weights[i]),0)
            changes=means[:,i,g,a]-means[:,i,g,b]
            local=means[0,i,g]; order=np.argsort(-local); winner=order[0]; runner=order[1]
            lc=covariance[i,winner]+covariance[i,runner]
            risk=max(float(weights[i]@lc@weights[i]),0)
            rows.append(dict(role=meta.role[i],obs_id=int(meta.obs_id[i]),mmsi=str(meta.mmsi[i]),class_id=int(yy[i]),group=name,
                predicted_sd=np.sqrt(predvar),actual_sd=float(np.std(changes,ddof=1)),local_margin=float(local[winner]-local[runner]),
                local_correct=int(winner==yy[i]),local_sd=np.sqrt(risk),offdiag_fraction=float((np.abs(cov).sum()-np.abs(np.diag(cov)).sum())/max(np.abs(cov).sum(),1e-12))))
    frame=pd.DataFrame(rows); frame.to_csv(HERE/'source_probe.csv',index=False)
    summaries=[]; strata=[]
    for (role,group),f in frame.groupby(['role','group']):
        corr=spearmanr(f.predicted_sd,f.actual_sd).statistic
        f=f.copy(); f['margin_bin']=pd.qcut(f.local_margin,4,labels=False,duplicates='drop')
        gaps=[]
        for b,part in f.groupby('margin_bin'):
            threshold=part.local_sd.median(); low=part[part.local_sd<=threshold]; high=part[part.local_sd>threshold]
            gap=high.local_correct.mean()-low.local_correct.mean(); gaps.append(gap)
            strata.append(dict(role=role,group=group,margin_bin=int(b),low_n=len(low),high_n=len(high),low_correct=low.local_correct.mean(),high_correct=high.local_correct.mean()))
        classgaps=[]
        for cl,part in f.groupby('class_id'):
            part=part.copy(); part['class_margin_bin']=pd.qcut(part.local_margin,2,labels=False,duplicates='drop')
            cg=[]
            for b,cell in part.groupby('class_margin_bin'):
                low=cell[cell.local_sd<=cell.local_sd.median()]; high=cell[cell.local_sd>cell.local_sd.median()]
                if min(len(low),len(high))<3: continue
                cg.append(high.local_correct.mean()-low.local_correct.mean())
                strata.append(dict(role=role,group=group,class_id=int(cl),class_margin_bin=int(b),low_n=len(low),high_n=len(high),low_correct=low.local_correct.mean(),high_correct=high.local_correct.mean()))
            if cg: classgaps.append(np.mean(cg))
        summaries.append(dict(role=role,group=group,n=len(f),bank_sd_spearman=float(corr),offdiag_median=f.offdiag_fraction.median(),high_uncertainty_minus_low_accuracy=float(np.mean(gaps)),class_adjusted_gap=float(np.mean(classgaps)),classes=len(classgaps)))
    summary=pd.DataFrame(summaries); summary.to_csv(HERE/'probe_summary.csv',index=False); pd.DataFrame(strata).to_csv(HERE/'margin_strata.csv',index=False)
    np.savez_compressed(HERE/'source_profiles.npz',ids=ids,values=v.numpy(),cov=c.numpy(),reference=r.numpy(),valid=cv.numpy(),count=n.numpy(),banks=means)
    receipt=dict(n=len(ids),evaluation_used=False,previous_evidence_replay=replay,scalar_deletion_covariance_error=float(ce),
        interpretation='reference deletion sensitivity; not calibrated uncertainty or physical noise',
        scope='within this source bank and six alternatives, records are repeated query-group not independent samples')
    (HERE/'probe_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__': main()
