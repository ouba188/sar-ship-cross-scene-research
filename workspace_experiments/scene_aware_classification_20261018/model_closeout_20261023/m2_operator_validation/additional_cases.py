"""Raw-chip checks for direct convolution effects and normalization control."""
import json
import numpy as np
import pandas as pd
import torch
import analyze as a
import train
from operators import Candidate


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    f=pd.read_csv(train.HERE/'table.csv')
    te=np.flatnonzero(f.role.eq('evaluation'))
    y=f.class_id.to_numpy(int)
    chosen=[]
    full=np.load(train.HERE/'results/conv_learned_pool_43.npz')['score']
    z=np.load(train.HERE/'results/intervention_conv_learned_pool_43_conv_zero.npz')
    off=np.zeros_like(full);off[z['index']]=z['score']
    for kind,mask in [('rescue',(full[te].argmax(1)==y[te])&(off[te].argmax(1)!=y[te])),('harm',(full[te].argmax(1)!=y[te])&(off[te].argmax(1)==y[te]))]:
        ii=te[mask]
        j=int(ii[np.argmax(abs(full[ii,y[ii]]-off[ii,y[ii]]))])
        chosen.append(dict(family='direct_conv',kind=kind,seed=43,j=j,before=off[j],after=full[j]))
    before=np.load(train.HERE/'results/static_pool_17.npz')['score']
    after=np.load(train.HERE/'conditioning_control/static_pool_17.npz')['score']
    bc,ac=before[te].argmax(1)==y[te],after[te].argmax(1)==y[te]
    for kind,mask in [('rescue',ac&~bc),('harm',bc&~ac),('CC',ac&bc),('WW',~ac&~bc)]:
        ii=te[mask]
        if len(ii):
            change=a.margin(after[ii],y[ii])-a.margin(before[ii],y[ii])
            j=int(ii[np.argmax(abs(change))])
            chosen.append(dict(family='normalization',kind=kind,seed=17,j=j,before=before[j],after=after[j]))
    j=int(np.flatnonzero(f.obs_id.eq(25398))[0])
    chosen.append(dict(family='normalization_rare',kind='harm',seed=43,j=j,before=np.load(train.HERE/'results/static_pool_43.npz')['score'][j],after=np.load(train.HERE/'conditioning_control/static_pool_43.npz')['score'][j]))
    geo=a.load('additional_geo',train.HERE.parent.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    snaps={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    table=pd.read_csv(train.UP/'table.csv')
    refs=np.load(train.LOCAL/'scores_layer1.npz')['refs']
    valid=np.load(train.UP/'summaries.npz')['valid']
    out=train.HERE/'analysis'
    records=[]
    for case in chosen:
        j,seed=case['j'],case['seed'];c=int(y[j]);i=int(f.iloc[j].up_index);obs=int(f.iloc[j].obs_id)
        before,after=case.pop('before'),case.pop('after')
        rival=int(after.argmax()) if case['kind']=='harm' else int(np.where(np.arange(8)==c,-np.inf,before).argmax())
        rr=refs[i,[c,rival]].ravel()
        fig,ax=a.plt.subplots(2,5,figsize=(16,7));paths=[]
        for p in range(2):
            for k,ri in enumerate([i,*rr]):
                pix,fp,g,_=geo.native(table,snaps,int(ri),p)
                ax[p,k].imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if g is not None:
                    for r,col in enumerate(('#20a780','#d77329','#3287d0')):
                        m=geo.soft_regions(g)[r]*fp
                        if m.max()>.5:ax[p,k].contour(m,levels=[.5],colors=[col],linewidths=.55)
                label='QUERY' if k==0 else ('TRUE' if k<3 else 'RIVAL')
                ax[p,k].set_title(f'{("VV","VH")[p]} {label} {int(table.iloc[ri].obs_id)} valid={int(valid[ri,p])}\n{table.iloc[ri].class_name}',fontsize=8)
                ax[p,k].set(xticks=[],yticks=[])
                paths.append(dict(obs_id=int(table.iloc[ri].obs_id),polarization=p,path=table.iloc[ri]['image_'+('vv','vh')[p]]))
        fig.suptitle(f"{case['family']} {case['kind']} seed{seed}: {before.argmax()} -> {after.argmax()}, true {c}\nFixed rival {rival}: {before[c]-before[rival]:+.4f} -> {after[c]-after[rival]:+.4f}")
        fig.text(.02,.02,'Native160 crop/pad; no resize. Green=body, orange=end, blue=side. M3 references unchanged.',fontsize=9)
        fig.subplots_adjust(top=.83,bottom=.09,wspace=.12,hspace=.15)
        path=out/f"{case['family']}_{case['kind']}_{obs}_{seed}.png";fig.savefig(path,dpi=120);a.plt.close(fig)
        att=None
        if case['family']=='direct_conv':
            data=train.Data(seed);m=Candidate('conv_learned_pool',data.ck,data.wproject,data.tau).cuda()
            m.load_state_dict(torch.load(train.HERE/'models/conv_learned_pool_43.pt',map_location='cpu',weights_only=False)['state'])
            att=a.attribution(data,m,j,c,rival,reference='conv_zero')
            np.testing.assert_allclose(att['new_fixed_margin'],after[c]-after[rival],atol=2e-5)
            np.testing.assert_allclose(att['old_fixed_margin'],before[c]-before[rival],atol=2e-5)
            del data,m
        records.append(dict(**case,obs_id=obs,true_class=c,rival=rival,before_pred=int(before.argmax()),after_pred=int(after.argmax()),before_margin=float(before[c]-before[rival]),after_margin=float(after[c]-after[rival]),attribution=att,reference_obs=table.iloc[rr].obs_id.astype(int).tolist(),provenance=paths,path=str(path)))
    (out/'additional_cases.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(pd.DataFrame(records)[['family','kind','obs_id','before_margin','after_margin','attribution']].to_string(index=False),flush=True)


if __name__=='__main__':main()
