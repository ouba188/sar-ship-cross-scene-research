"""Review actual source chips behind sensitivity and reference-scope diagnostics."""
import os
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from scipy.special import softmax, logsumexp

HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
spec=importlib.util.spec_from_file_location('case_probe',HERE/'probe.py')
probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)


def main():
    table,x,mass,bank,idx,tr,va,te=probe.prev.exp.source_data()
    z=np.load(HERE/'source_profiles.npz'); ids=z['ids']; per=z['reference']; cov=z['cov']
    scores=np.load(HERE/'reference_scope_scores.npz'); yy=table.iloc[ids].class_id.to_numpy(int)
    meta=table.iloc[ids].reset_index(drop=True); source=meta.role.eq('selection').to_numpy()
    mask=probe.prev.organize(x,mass)[1][ids]
    body=(x[:,:,:8]*mass[:,:,:8,None]).sum(2)/np.maximum(mass[:,:,:8].sum(2)[...,None],1e-8)
    body=body/np.maximum(np.linalg.norm(body,axis=-1,keepdims=True),1e-8)
    affinity=np.einsum('nd,crd->ncr',body.reshape(len(x),-1)[ids],body.reshape(len(x),-1)[bank])
    # VV body is fixed for all case definitions; local classifications, not full model.
    g=0; a=scores['all8'][:,g]; b=scores['nearest2'][:,g]; ok=a.argmax(-1)==yy; nk=b.argmax(-1)==yy
    frame=pd.read_csv(HERE/'source_probe.csv').query("group=='VV_body'")
    threshold=frame.query("role=='utility_train'").local_sd.median()
    sd=frame.set_index('obs_id').local_sd.reindex(meta.obs_id).to_numpy()
    definitions=[('nearest_rescue',source&~ok&nk),('nearest_harm',source&ok&~nk),('CC',source&ok&nk),('WW',source&~ok&~nk),
        ('sensitive_correct',source&ok&(sd>threshold)),('stable_wrong',source&~ok&(sd<=threshold))]
    cases=[]; refs=[]; tokens=[]
    for name,eligible in definitions:
        choices=np.flatnonzero(eligible)
        if not len(choices): continue
        loc=int(choices[np.argmin(meta.iloc[choices].obs_id.to_numpy())]); i=int(ids[loc]); c=int(yy[loc])
        rival=int(np.where(np.arange(8)==c,-np.inf,b[loc] if name=='nearest_harm' else a[loc]).argmax())
        w=mask[loc,g]/max(mask[loc,g].sum(),1)
        chosen=[]
        for cl in (c,rival):
            responsibility=softmax(per[loc,g,:,cl],axis=-1)
            importance=(responsibility*w[:,None]).sum(0)
            order=np.argsort(-importance); chosen.append(int(bank[cl,order[0]]))
            valid=per[loc,g,0,cl]>-9999
            selected=np.argsort(-np.where(valid,affinity[loc,cl],-np.inf))[:2]
            full_score=logsumexp(per[loc,g,:,cl],axis=-1)-np.log(valid.sum())
            for j in range(8):
                keep=valid.copy(); keep[j]=False
                deleted=logsumexp(np.where(keep[None],per[loc,g,:,cl],-np.inf),axis=-1)-np.log(max(keep.sum(),1))
                refs.append(dict(case=name,query_obs=int(meta.iloc[loc].obs_id),class_id=cl,ref_obs=int(table.iloc[bank[cl,j]].obs_id),responsibility=float(importance[j]),nearest_selected=bool(j in selected and valid[j]),body_affinity=float(affinity[loc,cl,j]),deletion_score_change=float((deleted-full_score)@w)))
        for t in np.flatnonzero(mask[loc,g]): tokens.append(dict(case=name,obs_id=int(meta.iloc[loc].obs_id),token=int(t),true_mean=float(z['values'][loc,g,t,c]),rival_mean=float(z['values'][loc,g,t,rival]),true_var=float(cov[loc,g,c,t,t]),rival_var=float(cov[loc,g,rival,t,t])))
        cases.append(dict(case=name,loc=loc,i=i,obs_id=int(meta.iloc[loc].obs_id),true=c,rival=rival,refs=chosen,
            all_pred=int(a[loc].argmax()),near_pred=int(b[loc].argmax()),all_margin=float(a[loc,c]-a[loc,rival]),near_margin=float(b[loc,c]-b[loc,rival]),sensitivity=float(sd[loc]),n=int(eligible.sum())))
    from extract_tokens import geo,slots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=HERE/'analysis'; out.mkdir(exist_ok=True)
    wanted={int(table.iloc[i].obs_id) for ca in cases for i in [ca['i'],*ca['refs']]}
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    valid=np.load(probe.prev.ROOT/'tokens.npz')['valid']; provenance=[]
    for ca in cases:
        fig,axes=plt.subplots(2,3,figsize=(12,8),squeeze=False)
        for col,(i,label) in enumerate(zip([ca['i'],*ca['refs']],('QUERY','TRUE REF','RIVAL REF'))):
            for p,pol in enumerate(('vv','vh')):
                pixels,fp,geometry,placement=geo.native(table,snapshots,i,p); ax=axes[p,col]
                ax.imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if valid[i,p]:
                    region=slots(geometry,fp)[:8].sum(0)
                    if region.min()<.5<region.max(): ax.contour(region,levels=[.5],colors=['#24a878'],linewidths=.8)
                row=table.iloc[i]; ax.set(title=f"{pol.upper()} {label} {int(row.obs_id)}\n{row.class_name}; {row.port}\nvalid={int(valid[i,p])}",xticks=[0,80,159],yticks=[0,80,159])
                provenance.append(dict(case=ca['case'],obs_id=int(row.obs_id),pol=pol,path=row['image_'+pol],valid=bool(valid[i,p]),placement=placement))
        fig.suptitle(f"SOURCE selection | {ca['case']} {ca['obs_id']} | VV body local classification ONLY\nAll8: {ca['all_pred']}, margin {ca['all_margin']:+.3f}; nearest2: {ca['near_pred']}, margin {ca['near_margin']:+.3f}; sensitivity {ca['sensitivity']:.3f}")
        fig.text(.02,.02,'Native160 crop/pad, no resize. References: highest all8 compatibility responsibility. NOT causal attribution.',fontsize=8)
        fig.subplots_adjust(top=.84,bottom=.1,wspace=.3,hspace=.5); fig.savefig(out/f"{ca['case']}_{ca['obs_id']}.png",dpi=130); plt.close(fig)
    pd.DataFrame(refs).to_csv(out/'references.csv',index=False); pd.DataFrame(tokens).to_csv(out/'tokens.csv',index=False)
    (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8'); (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(pd.DataFrame(cases).to_string(index=False))


if __name__=='__main__': main()
