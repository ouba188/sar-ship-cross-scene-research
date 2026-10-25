"""Trace full source interventions through kernels, local evidence and class margins."""
import train as run
import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    data=run.Data();rows=[]
    for seed in run.SEEDS:
        reader=run.Reader(data,seed)
        z=np.load(run.HERE/'results'/f'source_interventions_{seed}.npz')
        lookup={int(j):k for k,j in enumerate(z['ids'])}
        for t in range(18):
            p,r,b=t//9,(t%9)//3,t%3
            # Diagnose the strongest source-selection effect of each actual action.
            effects=z['effect'][[lookup[int(j)] for j in data.va],t]
            j=int(data.va[np.argmax(abs(effects))]);ii=torch.tensor([j],device='cuda')
            k=reader.kernels[r]
            k0=k['k0'][ii,:,:,p];dk=k['dk'][ii,:,:,p,b]
            qa,sa=k['qa'][ii,:,:,p],k['sa'][ii,:,:,p]
            mask=(qa>0)[..., :,None]&(sa>0)[...,None,:]
            tr,_,_,_=run.prior.old.sinkhorn(k0+dk,qa,sa)
            evidence=run.prior.old.pair_score(tr,k0+dk,qa,sa).mean(2)
            oldlocal=reader.local0[ii,:,:,r][:,:,p]
            change=evidence-oldlocal
            rawidx=reader.ix[ii];ref=reader.old.old.refs[rawidx]
            region=run.prior.old.REGIONS[r]
            dq=reader.old.old.d[rawidx][:,p,region]
            ds=reader.old.old.d[ref][...,p,region,:]
            eq,es=dq.clone(),ds.clone();sl=slice(32*b,32*(b+1))
            mu=torch.tensor(reader.replacement_means[r,p,sl],device='cuda')
            eq[...,sl]=mu;es[...,sl]=mu
            metric=reader.old.frozen
            w0=metric.weights(metric.project(dq)[:,None,None,:,None],metric.project(ds)[...,None,:,:])
            w1=metric.weights(metric.project(eq)[:,None,None,:,None],metric.project(es)[...,None,:,:])
            relative=w1/w0
            normalized=relative/relative.mean(-1,keepdim=True)
            rows.append(dict(seed=seed,type=t,polarization=('VV','VH')[p],region=('body','sides','ends')[r],
                wavelength_px=(4,8,16)[b],obs_id=int(data.table.iloc[j].obs_id),
                source_selection_fixed_margin_effect=float(z['effect'][lookup[j],t]),
                query_descriptor_change_rms=float((eq-dq).square().mean().sqrt()),
                relative_weight_contrast_rms=float((normalized-1).square().mean().sqrt()),
                mean_abs_K_change=float(dk[mask].abs().mean()),max_abs_K_change=float(dk[mask].abs().max()),
                local_E_max_change=float(change[...,0].abs().max()),
                local_mass_max_change=float(change[...,1].abs().max())))
        del reader;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(run.HERE/'operator_scale.csv',index=False)
    print(pd.DataFrame(rows).groupby('seed')[['query_descriptor_change_rms','relative_weight_contrast_rms','mean_abs_K_change','local_E_max_change']].median().to_string())


if __name__=='__main__':main()
