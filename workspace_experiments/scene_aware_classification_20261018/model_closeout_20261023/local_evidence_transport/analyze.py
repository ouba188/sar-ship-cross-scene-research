"""Paired class/acquisition effects and actual query-reference chip panels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from extract_tokens import geo,slots
from match_readout import unit,sinkhorn
from train_readout import Head,inputs

HERE=Path(__file__).resolve().parent
UP=HERE.parent/'regional_utility_pilot/upstream'
OUT=HERE/'analysis';OUT.mkdir(exist_ok=True)


def score(layer,arm,seed):
    return np.load(HERE/'results'/f'{layer}_{arm}_{seed}.npz')['score']


def main():
    table=pd.read_csv(UP/'table.csv',dtype={'mmsi':str})
    te=np.flatnonzero(table.role.eq('evaluation')); y=table.iloc[te].class_id.to_numpy()
    met=pd.concat([pd.read_csv(HERE/'metrics.csv'),pd.read_csv(HERE/'controls_metrics.csv')])
    summary=met[met.role.eq('evaluation')].groupby(['layer','arm'])[['OA','BA','NLL','Brier']].agg(['mean','std'])
    summary.to_csv(OUT/'summary.csv')
    rows=[]; classes=[]; predictions=[]; acquisition=[]
    for layer in ['stem','layer1']:
        for seed in [17,29,43]:
            new=score(layer,'anchored',seed)
            old=np.load(UP/f'strategies_seed{seed}.npz')['logits']
            methods={a:score(layer,a,seed) for a in ['pooled','fixed','free','independent','joint','mispaired','body_match']}
            methods.update(full=old[:,2],body=old[:,1],score_only=score('none','score_only',seed))
            for arm,s in methods.items():
                p=new[te].argmax(1); q=s[te].argmax(1)
                a=p==y; b=q==y; d=a.astype(float)-b
                rows.append(dict(layer=layer,seed=seed,comparison=arm,n=len(te),rescue=int((a&~b).sum()),harm=int((~a&b).sum()),delta_OA=100*d.mean(),delta_BA=100*np.mean([d[y==c].mean() for c in range(8)])))
                for c in range(8):
                    take=y==c
                    classes.append(dict(layer=layer,seed=seed,comparison=arm,class_id=c,n=int(take.sum()),rescue=int((a[take]&~b[take]).sum()),harm=int((~a[take]&b[take]).sum()),delta_recall=100*d[take].mean()))
                for passage,idx in table.iloc[te].groupby('pass_group').groups.items():
                    loc=np.flatnonzero(np.isin(te,idx))
                    acquisition.append(dict(layer=layer,seed=seed,comparison=arm,pass_group=passage,n=len(loc),net=int(d[loc].sum())))
            for j,i in enumerate(te):
                predictions.append(dict(layer=layer,seed=seed,local_index=i,obs_id=table.iloc[i].obs_id,class_id=int(y[j]),prediction=int(new[i].argmax()),**{f's{c}':float(new[i,c]) for c in range(8)}))
    pd.DataFrame(rows).to_csv(OUT/'paired.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'classes.csv',index=False)
    pd.DataFrame(acquisition).to_csv(OUT/'acquisitions.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'predictions.csv',index=False)
    print(summary.to_string(),flush=True)
    print(pd.DataFrame(rows).groupby(['layer','comparison'])[['delta_OA','delta_BA']].mean().to_string(),flush=True)
    # Preset seed17, minimum obs_id within each outcome; not best seed/case.
    layer='layer1'; seed=17
    new=score(layer,'anchored',seed)
    references=np.load(HERE/f'scores_{layer}.npz')['refs']
    whole=np.load(UP/f'strategies_seed{seed}.npz')['logits'][:,2]
    independent=score(layer,'independent',seed)
    cases=[]
    for arm,other in [('full',whole),('independent',independent)]:
        a=new[te].argmax(1)==y; b=other[te].argmax(1)==y
        outcomes={'rescued':a&~b,'harmed':~a&b}
        if arm=='full':
            outcomes.update(CC=a&b,WW=~a&~b)
        for outcome,mask in outcomes.items():
            ids=te[mask]
            if len(ids):
                i=int(table.iloc[ids].sort_values('obs_id').index[0])
                cases.append(dict(comparison=arm,outcome=outcome,group_n=len(ids),local_index=i,obs_id=int(table.iloc[i].obs_id)))
    pd.DataFrame(cases).to_csv(OUT/'case_selection.csv',index=False)
    wanted=set(table.obs_id)
    snapshots={s['obs_id']:s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if s['obs_id'] in wanted}
    tokens=np.load(HERE/'tokens.npz')
    projection=np.load(HERE/f'projection_{layer}.npz')
    f=tokens[layer]; x=unit((f-projection['mean'])@projection['components'].T).astype(np.float32)
    x[tokens['mass']<=0]=0
    module=torch.load(HERE/'models'/f'{layer}_anchored_{seed}.pt',map_location='cpu',weights_only=False)
    head=Head();head.load_state_dict(module['state']);head.eval()
    tr=np.flatnonzero(table.role.eq('utility_train'))
    local=np.load(HERE/f'scores_{layer}.npz')['anchored']
    xx,_,_,_=inputs(local,whole,tr)
    details=[]
    for case in cases:
        i=case['local_index']; row=table.iloc[i]; true=int(row.class_id)
        rival=int(np.argmax(np.where(np.arange(8)==true,-np.inf,new[i])))
        ids=[i,*references[i,true],*references[i,rival]]
        title=f"{case['outcome']} vs {case['comparison']} | query {row.obs_id} {row.class_name} | seed17 layer1"
        fig,axes=plt.subplots(2,5,figsize=(17,8),layout='constrained')
        for p,pol in enumerate(['vv','vh']):
            for j,idx in enumerate(ids):
                rr=table.iloc[idx]
                pix,fp,g,place=geo.native(table,snapshots,int(idx),p)
                ax=axes[p,j]
                ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                ax.set_xlim(24,136);ax.set_ylim(136,24);ax.set_aspect('equal')
                if g is not None and tokens['valid'][idx,p]:
                    m=slots(g,fp)
                    for k,color in [(slice(0,8),'#24a878'),(slice(8,16),'#2c84c6'),(slice(16,20),'#d39222')]:
                        area=m[k].sum(0)
                        if area.max()>0:
                            ax.contour(area,levels=[.5],colors=[color],linewidths=.7)
                    for t in range(8):
                        area=m[t].sum()
                        if area:
                            cx=(m[t]*geo.XX).sum()/area;cy=(m[t]*geo.YY).sum()/area
                            ax.text(cx,cy,str(t),fontsize=7,color='#ff514f',ha='center')
                label='QUERY' if j==0 else ('TRUE reference' if j<3 else 'RIVAL reference')
                ax.set_title(f'{pol.upper()} {label}\nobs{rr.obs_id}, {rr.class_name}\n{rr.port}',fontsize=9)
                if not tokens['valid'][idx,p]:
                    ax.text(80,80,'INVALID POL',color='red',ha='center')
            for c in [true,rival]:
                for k,idx in enumerate(references[i,c]):
                    sim=torch.tensor(x[i,p,:8]@x[idx,p,:8].T)
                    t,_,_,_=sinkhorn(sim,torch.tensor(tokens['mass'][i,p,:8]),torch.tensor(tokens['mass'][idx,p,:8]))
                    links=np.dstack(np.unravel_index(np.argsort(-t.numpy().ravel())[:4],(8,8)))[0]
                    for qa,sb in links:
                        details.append(dict(**case,pol=pol,query_class=true,reference_class=int(c),reference_obs=int(table.iloc[idx].obs_id),reference_slot=k,query_token=int(qa),reference_token=int(sb),mass=float(t[qa,sb]),cosine=float(sim[qa,sb])))
        margin=lambda s:float(s[i,true]-s[i,rival])
        text=f"true={true}, rival={rival}; full margin={margin(whole):+.3f}; independent={margin(independent):+.3f}; anchored={margin(new):+.3f}\nNative pixels on a common 112-pixel display window. Green=body; blue=side; amber=end. No resize of model input."
        fig.suptitle(title+'\n'+text,fontsize=11)
        fig.savefig(OUT/f"case_{row.obs_id}_{case['comparison']}_{case['outcome']}.png",dpi=140)
        plt.close(fig)
    pd.DataFrame(details).to_csv(OUT/'actual_correspondences.csv',index=False)


if __name__=='__main__':
    main()
