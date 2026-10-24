"""Frozen broader evaluation and native-chip inspection, without new selection."""
import os
import sys
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
def loadmodule(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
run=loadmodule('competition_run',HERE/'run.py')
ev=loadmodule('competition_external',run.ROOT/'conditional_relation_operator/distribution_condition_operator/frozen_external/run.py')


@torch.no_grad()
def main():
    source,target,x,mass,bank,idx,tr,va,te,ids=ev.load()
    table=pd.concat([source,target],ignore_index=True)
    out=HERE/'analysis'; out.mkdir(exist_ok=True)
    q,mask=run.organize(x,mass)
    q=torch.tensor(q,device='cuda'); mask=torch.tensor(mask,device='cuda')
    original=np.load(HERE/'inputs.npz')
    scale=torch.tensor(original['scale'])
    allids=np.r_[idx,ids]
    e,cv=run.evidence(q,mask,bank,allids)
    err=float(abs(e[:len(idx)].numpy()-original['e']).max())
    assert err<1e-5,err
    e=e/scale[None,:,None,None]; cv=cv.cuda()
    y=target.class_id.to_numpy(int)
    rows=[]; pairs=[]; classes=[]; diagnostics=[]; replays=[]
    modelstore={}; predictions={}
    for seed in (17,29,43):
        strategy=np.load(ev.EXT/f'strategies_seed{seed}.npz')
        np.testing.assert_array_equal(strategy['instance_ids'].astype(str),target.instance_id)
        full=strategy['logits'][:,2].astype(np.float32)
        scores=dict(full=full,old_independent=np.load(ev.EXT/'results'/f'independent_{seed}.npz')['score'])
        for arm in run.ARMS:
            model=run.Pool(arm,seed).cuda()
            model.load_state_dict(torch.load(HERE/f'model_{arm}_{seed}.pt',weights_only=True))
            modelstore[arm,seed]=model
            values=[]; simple=[]
            uniform=run.Pool('uniform').cuda(); uniform.a.copy_(model.a)
            for first in range(0,len(allids),64):
                pos=np.arange(first,min(first+64,len(allids))); take=allids[pos]
                args=(e[pos].cuda(),q[take],mask[take],cv)
                values.append(model(*args).cpu().numpy()); simple.append(uniform(*args).cpu().numpy())
            delta=np.concatenate(values); sim=np.concatenate(simple)
            saved=np.load(HERE/f'predictions_{arm}_{seed}.npz')
            replay=saved['full']+delta[:len(idx)]
            error=float(abs(replay-saved['original']).max()); assert error<1e-5
            replays.append(dict(arm=arm,seed=seed,error=error))
            scores[arm]=full+delta[len(idx):]
            scores[arm+'_uniform_gate']=full+sim[len(idx):]
            for role,loc in [('utility',tr),('selection',va),('development',te),('broader',np.arange(len(idx),len(allids)))]:
                yy=table.iloc[allids[loc]].class_id.to_numpy(int)
                d=delta[loc]; s=sim[loc]; rr=np.arange(len(loc))
                best=np.where(np.arange(8)[None]==yy[:,None],-np.inf,d).max(-1)
                sbest=np.where(np.arange(8)[None]==yy[:,None],-np.inf,s).max(-1)
                diagnostics.append(dict(arm=arm,seed=seed,role=role,n=len(loc),
                    changed_local_gap_mean=float((d[rr,yy]-best-s[rr,yy]+sbest).mean()),
                    changed_logit_rms=float(np.sqrt(np.mean((d-s)**2)))))
        predictions[seed]=scores
        np.savez_compressed(HERE/f'external_{seed}.npz',**scores)
        for arm,score in scores.items():
            rows.append(dict(arm=arm,seed=seed,**run.exp.dep.metrics(y,score)))
            for baseline in ('full','uniform','own','old_independent'):
                before=scores[baseline].argmax(-1)==y; after=score.argmax(-1)==y
                pairs.append(dict(arm=arm,seed=seed,reference=baseline,rescued=int((~before&after).sum()),harmed=int((before&~after).sum()),net=int(after.sum()-before.sum())))
            for c in range(8):
                keep=y==c; correct=score.argmax(-1)==y; old=full.argmax(-1)==y
                classes.append(dict(arm=arm,seed=seed,class_id=c,n=int(keep.sum()),net=int(correct[keep].sum()-old[keep].sum())))
    for name,data in [('external_metrics',rows),('external_paired',pairs),('external_classes',classes),('gate_diagnostics',diagnostics)]: pd.DataFrame(data).to_csv(HERE/f'{name}.csv',index=False)
    summary=pd.DataFrame(rows).groupby('arm')[['OA','BA','J','NLL','Brier']].mean()
    summary.to_csv(HERE/'external_summary.csv'); print(summary.to_string(),flush=True)
    # Deterministic representative cases, not prevalence estimates.
    seed=17; scores=predictions[seed]; ok=scores['full'].argmax(-1)==y; nk=scores['pair'].argmax(-1)==y
    own=scores['own'].argmax(-1)==y
    cases=[]; contributions=[]; refs=[]; weights=[]; margins=[]
    for name,keep in [('rescued',~ok&nk),('harmed',ok&~nk),('CC',ok&nk),('WW',~ok&~nk),('better_than_own',~own&nk),('worse_than_own',own&~nk)]:
        options=np.flatnonzero(keep)
        if not len(options): continue
        loc=int(options[np.argmin(target.iloc[options].obs_id.to_numpy())]); i=int(ids[loc]); pos=len(idx)+loc
        c=int(y[loc]); baseline='pair' if name in ('harmed','worse_than_own') else 'full'
        d=int(np.argmax(np.where(np.arange(8)==c,-np.inf,scores[baseline][loc])))
        model=modelstore['pair',seed]
        _,parts,w,edges=model(e[pos:pos+1].cuda(),q[i:i+1],mask[i:i+1],cv,True)
        ee=e[pos].cuda(); ii,jj=model.pairs.T
        tokenedge=w[0]*(ee[...,ii]-ee[...,jj])*(cv[:,ii]&cv[:,jj])[:,None]
        allocation=torch.zeros((6,8,8),device='cuda')
        allocation.index_add_(2,ii,tokenedge); allocation.index_add_(2,jj,-tokenedge)
        allocation=allocation/cv.sum(-1)[:,None,None].clamp_min(1)*model.a[:,None,None]
        token=(allocation[:,:,c]-allocation[:,:,d]).cpu().numpy()
        observed=scores['pair'][loc,c]-scores['pair'][loc,d]-scores['full'][loc,c]+scores['full'][loc,d]
        replay_error=abs(float(token.sum()-observed)); assert replay_error<1e-5
        g=int(abs(token.sum(-1)).argmax()); picked=[]
        for cl in (c,d):
            s=q[bank[cl],g]; sm=mask[bank[cl],g]; count=sm.sum(-1)
            sim=torch.einsum('td,rud->tru',q[i,g],s)*np.sqrt(32)
            logref=torch.logsumexp(sim.masked_fill(~sm[None],-1e4),-1)-count.clamp_min(1).log()[None]
            responsibility=logref.masked_fill(count[None]==0,-1e4).softmax(-1)
            importance=torch.tensor(abs(token[g]),device='cuda')*mask[i,g]
            rank=(responsibility*importance[:,None]).sum(0)
            chosen=int(rank.argmax()); picked.append(int(bank[cl,chosen]))
            for r in range(8): refs.append(dict(case=name,query_obs=int(table.iloc[i].obs_id),class_id=cl,ref_obs=int(table.iloc[bank[cl,r]].obs_id),group=run.GROUPS[g],responsibility=float(rank[r]),valid=bool(count[r]>0)))
        for group in range(6):
            for t in range(8): contributions.append(dict(case=name,obs_id=int(table.iloc[i].obs_id),group=run.GROUPS[group],token=t,contribution=float(token[group,t]),valid=bool(mask[i,group,t])))
        for arm in ('full','pair','own','uniform','pair_uniform_gate','old_independent'):
            v=scores[arm][loc]; margins.append(dict(case=name,obs_id=int(table.iloc[i].obs_id),arm=arm,pred=int(v.argmax()),margin=float(v[c]-v[d])))
        cases.append(dict(case=name,obs_id=int(table.iloc[i].obs_id),i=i,loc=loc,seed=seed,true=c,rival=d,refs=picked,group=g,n=int(keep.sum()),replay_error=replay_error))
    from extract_tokens import geo,slots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wanted={int(table.iloc[i].obs_id) for ca in cases for i in [ca['i'],*ca['refs']]}
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    valid=np.concatenate([np.load(run.ROOT/'tokens.npz')['valid'],np.load(ev.EXT/'tokens.npz')['valid']])
    provenance=[]
    for ca in cases:
        fig,axes=plt.subplots(2,3,figsize=(12,8),squeeze=False)
        for col,(i,label) in enumerate(zip([ca['i'],*ca['refs']],('QUERY','TRUE REF','RIVAL REF'))):
            for p,pol in enumerate(('vv','vh')):
                pixels,fp,geometry,place=geo.native(table,snapshots,i,p); ax=axes[p,col]
                ax.imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if valid[i,p]:
                    masks=slots(geometry,fp)
                    for (lo,hi),color in zip(((0,8),(8,16),(16,20)),('#24a878','#2c84c6','#d39222')):
                        region=masks[lo:hi].sum(0)
                        if region.min()<.5<region.max(): ax.contour(region,levels=[.5],colors=[color],linewidths=.7)
                row=table.iloc[i]
                ax.set(title=f"{pol.upper()} {label} {int(row.obs_id)}\n{row.class_name}; {row.port}\nvalid={int(valid[i,p])}",xticks=[0,80,159],yticks=[0,80,159])
                provenance.append(dict(case=ca['case'],obs_id=int(row.obs_id),pol=pol,path=row['image_'+pol],placement=place,valid=bool(valid[i,p])))
        fig.suptitle(f"Competitive pooling | {ca['case']} {ca['obs_id']} | config{seed}; {run.GROUPS[ca['group']]}\nReferences ranked by compatibility responsibility, NOT additive causal attribution")
        text='; '.join(f"{r['arm']}: {r['pred']} ({r['margin']:+.3f})" for r in margins if r['case']==ca['case'])
        fig.text(.02,.02,text+'\nNative160 crop/pad, no resize. Green body / blue side / amber end.',fontsize=7)
        fig.subplots_adjust(top=.84,bottom=.12,wspace=.3,hspace=.5)
        fig.savefig(out/f"case_{ca['case']}_{ca['obs_id']}.png",dpi=130); plt.close(fig)
    for name,data in [('tokens',contributions),('margins',margins),('references',refs)]: pd.DataFrame(data).to_csv(out/f'{name}.csv',index=False)
    (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (out/'image_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    receipt=dict(n=len(target),ships=int(target.mmsi.nunique()),passes=int(target.pass_group.nunique()),historical=True,new_selection=False,evidence_replay=err,prediction_replays=replays,raw_panels=len(cases),raw_review='pending')
    (HERE/'external_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(pd.DataFrame(margins).to_string(index=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; main()
