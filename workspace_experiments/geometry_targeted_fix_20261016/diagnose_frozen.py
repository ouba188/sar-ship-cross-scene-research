"""Explain frozen kernel effects on the two target ports with usable ship axes."""
import json
import numpy as np
import pandas as pd
import torch
from run_fix import HERE,base,load,prepare,reader,predict,relation_loss


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    rows=[]
    for port in ('Jebel Ali','Singapore'):
        data=load(port,'recovered',True)
        for seed in range(3):
            prepare(data,seed)
            d=data['tensors']
            for arm in ('geometry_pair','geometry_rank','conditional_rank'):
                saved=torch.load(HERE/'recovered'/'trained'/port.replace(' ','_')/f'{arm}_seed{seed}.pt',weights_only=False)
                for state in ('initial','final'):
                    torch.manual_seed(20261016+seed)
                    model=reader('recovered',arm)
                    if state=='final':
                        model.load_state_dict(saved['state'])
                    else:
                        model.scale.copy_(saved['state']['scale'].to(base.DEVICE))
                    model.eval(); raw=base.raw_descriptors(model,data)
                    for item in data['cache']['target']:
                        if item['t']['origin']!='cross':
                            continue
                        s,q=item['s'],item['q']; ns=len(s)
                        valid=d['valid'][torch.cat((s,q))]
                        active=valid[ns:]&(valid[:ns].sum()>=2)
                        if not active.any():
                            continue
                        prob,qs,h=predict(model,data,item,raw=raw)
                        loss,n=relation_loss(h,item['sy'],d['labels'][q],valid,item['old_prob'])
                        sy=item['sy'].cpu().numpy(); y=d['labels'][q].cpu().numpy()
                        sim=(h[ns:]@h[:ns].T).cpu().numpy()
                        sv=valid[:ns].cpu().numpy(); qi=active.cpu().numpy()
                        ap=[]; labels=[]
                        for i in np.flatnonzero(qi):
                            hits=sy[sv]==y[i]; r=int(hits.sum())
                            if r:
                                ranked=hits[np.argsort(-sim[i,sv],kind='stable')[:r]]
                                ap.append(float((np.cumsum(ranked)/np.arange(1,r+1)*ranked).sum()/r)); labels.append(y[i])
                        macro_ap=pd.DataFrame(dict(ap=ap,cls=labels)).groupby('cls').ap.mean().mean() if ap else np.nan
                        delta=(qs-item['qs'])[active]
                        newscore=item['prior_q']+qs@base._coefficients(item['ss']+.75*h[:ns]@h[:ns].T,item['prior_s'],item['sy'])
                        oldscore=item['prior_q']+item['qs']@base._coefficients(item['ss'],item['prior_s'],item['sy'])
                        rows.append(dict(port=port,seed=seed,arm=arm,state=state,fold=item['t']['fold'],draw=item['t']['draw'],
                                         relation_loss=float(loss),relation_queries=n,new_descriptor_mapr=macro_ap,
                                         active_queries=int(active.sum()),valid_supports=int(valid[:ns].sum()),
                                         kernel_delta_rms=float(delta.square().mean().sqrt()),kernel_delta_max=float(delta.abs().max()),
                                         old_kernel_rms=float(item['qs'][active].square().mean().sqrt()),
                                         expert_logit_delta_rms=float((newscore-oldscore)[active].square().mean().sqrt()),
                                         centered_norm2=float(h[ns:][active].square().sum(-1).mean()),
                                         near_saturation=float((h[ns:][active].square().sum(-1)>.9).double().mean())))
        del data; torch.cuda.empty_cache()
    frame=pd.DataFrame(rows)
    frame.to_csv(HERE/'recovered'/'frozen_kernel_diagnostics.csv',index=False)
    summary=frame.groupby(['port','arm','state'])[[c for c in frame.columns if c not in ('port','seed','arm','state','fold','draw')]].mean()
    summary.to_csv(HERE/'recovered'/'frozen_kernel_summary.csv')
    assert frame.kernel_delta_max.max()<=.750001
    print(summary[['relation_loss','new_descriptor_mapr','kernel_delta_rms','old_kernel_rms','expert_logit_delta_rms']].to_string())


if __name__=='__main__':
    main()
