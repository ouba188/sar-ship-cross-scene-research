"""Inspect the learned competition, without refitting or choosing a variant."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('competitive_diagnostic',HERE/'evaluate.py')
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
run=ev.run


@torch.no_grad()
def main():
    source,target,x,mass,bank,idx,tr,va,te,ids=ev.ev.load()
    qn,mn=run.organize(x,mass)
    q=torch.tensor(qn,device='cuda'); mask=torch.tensor(mn,device='cuda')
    take=np.r_[idx,ids]
    e,cv=run.evidence(q,mask,bank,take)
    # Independent scalar reference mixture checks weights, axes and masked classes.
    error=0.
    for loc in (0,700,len(idx),len(take)-1):
        i=take[loc]
        for g in range(6):
            raw=[]; avail=[]
            for c in range(8):
                per=[]
                for r in bank[c]:
                    if not mn[r,g].any(): continue
                    values=qn[r,g,mn[r,g]]@qn[i,g,0]*np.sqrt(32)
                    per.append(logsumexp(values)-np.log(len(values)))
                avail.append(bool(per)); raw.append(logsumexp(per)-np.log(len(per)) if per else 0.)
            raw=np.array(raw); avail=np.array(avail)
            if avail.any(): raw=np.where(avail,raw-raw[avail].mean(),0.)
            error=max(error,float(abs(raw-e[loc,g,0].numpy()).max()))
    assert error<3e-6,error
    scale=torch.tensor(np.load(HERE/'inputs.npz')['scale'])
    e=e/scale[None,:,None,None]; cv=cv.cuda()
    roles=dict(utility=tr,selection=va,development=te,broader=np.arange(len(idx),len(take)))
    table=pd.concat([source,target],ignore_index=True)
    y=table.iloc[take].class_id.to_numpy(int)
    rows=[]; stats=[]; effects=[]; coefficients=[]; case_rows=[]
    caseids={c['i']:c for c in json.loads((HERE/'analysis/cases.json').read_text())}
    for seed in (17,29,43):
        model=run.Pool('pair',seed).cuda()
        model.load_state_dict(torch.load(HERE/f'model_pair_{seed}.pt',weights_only=True))
        full=np.concatenate([np.load(HERE/f'predictions_pair_{seed}.npz')['full'],np.load(HERE/f'external_{seed}.npz')['full']])
        result={k:[] for k in ('pair','shared_weight','content_only')}
        for g,name in enumerate(run.GROUPS): coefficients.append(dict(seed=seed,group=name,coefficient=float(model.a[g])))
        for first in range(0,len(take),64):
            pos=np.arange(first,min(first+64,len(take))); ti=take[pos]
            ee=e[pos].cuda(); qq=q[ti]; mm=mask[ti]
            pred,parts,w,edges=model(ee,qq,mm,cv,True)
            result['pair'].append(pred.cpu().numpy())
            ii,jj=model.pairs.T; pv=cv[:,ii]&cv[:,jj]
            shared=(w*pv[None,:,None]).sum(-1)/pv.sum(-1)[None,:,None].clamp_min(1)
            pooled=(ee*shared[...,None]).sum(2)*cv[None]
            pooled=(pooled-pooled.sum(-1,keepdim=True)/cv.sum(-1)[None,:,None].clamp_min(1))*cv[None]
            result['shared_weight'].append((pooled*model.a[None,:,None]).sum(1).cpu().numpy())
            features=torch.cat((torch.zeros((*qq.shape[:-1],4),device='cuda'),qq),-1)
            gate=model.net(features).squeeze(-1)
            cw=gate.masked_fill(~mm,-1e4).softmax(2)*mm
            cw=cw/cw.sum(2,keepdim=True).clamp_min(1e-8)
            pp=(ee*cw[...,None]).sum(2)*cv[None]
            pp=(pp-pp.sum(-1,keepdim=True)/cv.sum(-1)[None,:,None].clamp_min(1))*cv[None]
            result['content_only'].append((pp*model.a[None,:,None]).sum(1).cpu().numpy())
            for g,name in enumerate(run.GROUPS):
                good=mm[:,g].any(-1)
                if not good.any(): continue
                wg=w[good,g][:,:,pv[g]]
                neff=1/wg.square().sum(1).clamp_min(1e-8)
                for b in range(int(good.sum())):
                    stats.append(dict(seed=seed,group=name,loc=int(pos[good.cpu().numpy()][b]),
                        effective_tokens=float(neff[b].mean()),weight_range_across_pairs=float((wg[b].max(-1).values-wg[b].min(-1).values).mean())))
            if seed==17:
                for b,i in enumerate(ti):
                    if int(i) not in caseids: continue
                    ca=caseids[int(i)]; c,d=ca['true'],ca['rival']; pair=(min(c,d),max(c,d))
                    pi=[tuple(v) for v in model.pairs.tolist()].index(pair)
                    for g,name in enumerate(run.GROUPS):
                        for t in range(8):
                            if not mm[b,g,t]: continue
                            case_rows.append(dict(case=ca['case'],obs_id=ca['obs_id'],group=name,token=t,true_e=float(ee[b,g,t,c]),rival_e=float(ee[b,g,t,d]),pair_weight=float(w[b,g,t,pi]),shared_weight=float(shared[b,g,t]),content_weight=float(cw[b,g,t])))
        result={k:full+np.concatenate(v) for k,v in result.items()}
        for role,loc in roles.items():
            for arm,score in result.items():
                rows.append(dict(seed=seed,role=role,arm=arm,**run.exp.dep.metrics(y[loc],score[loc])))
            actual=result['pair'][loc].argmax(-1)==y[loc]
            for arm in ('shared_weight','content_only'):
                before=result[arm][loc].argmax(-1)==y[loc]
                effects.append(dict(seed=seed,role=role,reference=arm,rescued=int((~before&actual).sum()),harmed=int((before&~actual).sum()),net=int(actual.sum()-before.sum())))
    for name,data in [('intervention_metrics',rows),('intervention_pairs',effects),('coefficients',coefficients),('attention_samples',stats),('case_evidence',case_rows)]: pd.DataFrame(data).to_csv(HERE/f'{name}.csv',index=False)
    pd.DataFrame(rows).groupby(['role','arm'])[['OA','BA','J']].mean().to_csv(HERE/'intervention_summary.csv')
    (HERE/'diagnostic_checks.json').write_text(json.dumps(dict(scalar_reference_error=error,interventions='frozen weights only, not retrained controls'),indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).groupby(['role','arm'])[['OA','BA','J']].mean().to_string())
    print(pd.DataFrame(stats).groupby(['seed','group'])[['effective_tokens','weight_range_across_pairs']].median().to_string())


if __name__=='__main__':
    torch.set_num_threads(2); main()
