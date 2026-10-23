"""Four preset seed17 role-contrast versus ordered cases; frozen replay only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
import importlib.util
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXT = ROOT/'external_confirmation'
UP = ROOT.parent/'regional_utility_pilot/upstream'
OUT = HERE/'analysis'
OUT.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT/'.mplcache')
sys.path.insert(0,str(ROOT))
spec = importlib.util.spec_from_file_location('relational_operator',HERE/'run.py')
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)
from extract_tokens import geo,slots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 17
REGIONS = ('body','side','end')


def select(table, scores):
    y = table.class_id.to_numpy()
    a = scores['role_contrast'].argmax(1)==y
    b = scores['ordered'].argmax(1)==y
    rows = []
    for outcome,mask in [('rescued',a&~b),('harmed',~a&b),('CC',a&b),('WW',~a&~b)]:
        ids = np.flatnonzero(mask)
        if not len(ids):
            rows.append(dict(outcome=outcome,group_n=0,selected=False))
            continue
        i = int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy())])
        true = int(y[i])
        role_pred = int(scores['role_contrast'][i].argmax())
        ordered_pred = int(scores['ordered'][i].argmax())
        role_rival = int(np.argmax(np.where(np.arange(8)==true,-np.inf,scores['role_contrast'][i])))
        rival = ordered_pred if outcome=='rescued' else role_rival
        rows.append(dict(outcome=outcome,group_n=len(ids),selected=True,external_index=i,
            obs_id=int(table.iloc[i].obs_id),true_class=true,role_prediction=role_pred,
            ordered_prediction=ordered_pred,fixed_comparison_rival=rival,
            rival_reason='ordered_actual_error' if outcome=='rescued' else 'role_highest_nontrue',
            role_final_rival=role_rival))
    return rows


def exact_terms(q,s,qm,sm):
    q=q[:,None,None];qm=qm[:,None,None]
    b,bb=q[...,:8,:],s[...,:8,:]
    a,aa=qm[...,:8],sm[...,:8]
    body=operator.dot(b,bb)
    t,_,_,_=operator.sinkhorn(body,a,aa)
    own = operator.pair_score(t,body,a,aa)
    ordered,cross,wedge,masks = [own[...,0]], [torch.zeros_like(own[...,0])], [own[...,0]], [own[...,1]]
    availability=[(a.sum(-1)>0)&(aa.sum(-1)>0)]
    for region in ('side','end'):
        ix=np.arange(8,16) if region=='side' else operator.END
        n,nn=q[...,ix,:],s[...,ix,:]
        am,bm=qm[...,ix],sm[...,ix]
        if region=='end':
            f=torch.as_tensor(operator.IS_END,device=q.device)
            am=am*f;bm=bm*f
        o=body*operator.dot(n,nn)
        c=operator.dot(b,nn)*operator.dot(n,bb)
        ov=operator.pair_score(t,o,am,bm)
        cv=operator.pair_score(t,c,am,bm)
        wv=operator.pair_score(t,o-c,am,bm)
        ordered.append(ov[...,0]);cross.append(cv[...,0]);wedge.append(wv[...,0]);masks.append(ov[...,1])
        availability.append(availability[0]&(am.sum(-1)>0)&(bm.sum(-1)>0))
    per={k:torch.stack(v,-1).cpu().numpy() for k,v in
         [('ordered',ordered),('cross',cross),('wedge',wedge),('matched_mass',masks)]}
    ok=torch.stack(availability,-1).cpu().numpy()
    count=ok.sum(2)
    mean={k:(v*ok).sum(2)/np.maximum(count,1) for k,v in per.items()}
    return per,mean,ok,count


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    table=pd.read_csv(EXT/'table.csv',dtype={'mmsi':str,'instance_id':str})
    source=pd.read_csv(UP/'table.csv',dtype={'mmsi':str,'instance_id':str})
    scores={arm:np.load(HERE/'results'/f'external_{arm}_{SEED}.npz')['score'] for arm in operator.ARMS}
    base=np.load(EXT/f'strategies_seed{SEED}.npz')['logits'][:,2].astype(np.float32)
    scores['full']=base
    selections=select(table,scores)
    pd.DataFrame(selections).to_csv(OUT/'case_selection.csv',index=False)
    cases=[r for r in selections if r['selected']]
    ids=np.array([r['external_index'] for r in cases])
    assert len(ids)<=4
    cached=np.load(HERE/'external_local.npz')
    refs=cached['refs'][ids]
    srcids,inverse=np.unique(refs,return_inverse=True)
    mapped=inverse.reshape(refs.shape)
    sc=np.load(ROOT/'tokens.npz');qc=np.load(EXT/'tokens.npz')
    np.testing.assert_array_equal(sc['instance_ids'].astype(str),source.instance_id.to_numpy())
    np.testing.assert_array_equal(qc['instance_ids'].astype(str),table.instance_id.to_numpy())
    pr=np.load(ROOT/'projection_layer1.npz')
    qm,sm=qc['mass'][ids],sc['mass'][srcids]
    q=operator.project(dict(layer1=qc['layer1'][ids],mass=qm),pr)
    s=operator.project(dict(layer1=sc['layer1'][srcids],mass=sm),pr)
    rng=np.random.default_rng(17)
    ps=np.stack([rng.permutation(8) for _ in range(len(source)*2)]).reshape(len(source),2,8)
    pq=np.stack([rng.permutation(8) for _ in range(len(table)*2)]).reshape(len(table),2,8)
    misq,misqm=operator.shuffled(q,qm,pq[ids])
    miss,missm=operator.shuffled(s,sm,ps[srcids])
    args=[torch.tensor(v,device='cuda') for v in (q,s[mapped],qm,sm[mapped],misq,miss[mapped],misqm,missm[mapped])]
    with torch.inference_mode():
        replay,count=operator.compute(*args)
        per,means,ok,n=exact_terms(*args[:4])
    np.testing.assert_array_equal(count.cpu().numpy(),cached['count'][ids])
    np.testing.assert_array_equal(n,cached['count'][ids])
    checks=[];epochs={};outputs={};head_rows=[]
    for arm in operator.ARMS:
        loc=replay[arm].cpu().numpy()
        err=float(abs(loc-cached[arm][ids]).max())
        assert err<5e-5,(arm,err)
        ck=torch.load(HERE/'models'/f'{arm}_{SEED}.pt',map_location='cpu',weights_only=False)
        raw=operator.raw_inputs(loc,base[ids])
        x=np.where(ck['active'],(raw-ck['mu'])/ck['scale'],0).astype(np.float32)
        head=operator.Head().eval().requires_grad_(False)
        head.load_state_dict(ck['state'])
        with torch.inference_mode():
            pred=base[ids]+head(torch.tensor(x)).numpy()
        e=float(abs(pred-scores[arm][ids]).max())
        assert e<5e-5,(arm,e)
        np.testing.assert_array_equal(pred.argmax(1),scores[arm][ids].argmax(1))
        checks.append(dict(arm=arm,local_max_error=err,head_max_error=e,top1_exact=True,selected_step=int(ck['selected_step'])))
        epochs[arm]=int(ck['selected_step']);outputs[arm]=pred
    closed=float(abs(means['wedge']-(means['ordered']-means['cross'])).max())
    assert closed<2e-6
    assert np.max(abs(means['ordered']-cached['ordered'][ids,...,0]))<5e-5
    assert np.max(abs(means['wedge']-cached['role_contrast'][ids,...,0]))<5e-5
    localrows=[];perrows=[];scorerows=[]
    for j,case in enumerate(cases):
        i=case['external_index'];true=case['true_class'];rival=case['fixed_comparison_rival']
        for arm,v in scores.items():
            final_rival=int(np.argmax(np.where(np.arange(8)==true,-np.inf,v[i])))
            margin=float(v[i,true]-v[i,rival])
            head_rows.append(dict(obs_id=case['obs_id'],outcome=case['outcome'],arm=arm,
                selected_step=None if arm=='full' else epochs[arm],prediction=int(v[i].argmax()),
                true_class=true,fixed_rival=rival,fixed_margin=margin,final_rival=final_rival,
                own_top1_margin=float(v[i,true]-v[i,final_rival]),
                residual_fixed_margin=margin-float(base[i,true]-base[i,rival])))
            for c in range(8):
                scorerows.append(dict(obs_id=case['obs_id'],arm=arm,class_id=c,score=float(v[i,c])))
        for c in range(8):
            for p,pol in enumerate(('VV','VH')):
                for r,region in enumerate(REGIONS):
                    localrows.append(dict(obs_id=case['obs_id'],candidate_class=c,pol=pol,region=region,
                        available_refs=int(n[j,c,p,r]),**{key:float(v[j,c,p,r]) for key,v in means.items()}))
                    for k,ref in enumerate(refs[j,c]):
                        perrows.append(dict(obs_id=case['obs_id'],candidate_class=c,reference_slot=k,
                            reference_source_index=int(ref),reference_obs=int(source.iloc[int(ref)].obs_id),
                            pol=pol,region=region,available=bool(ok[j,c,k,p,r]),
                            **{key:float(v[j,c,k,p,r]) for key,v in per.items()}))
    pd.DataFrame(head_rows).to_csv(OUT/'case_margins.csv',index=False)
    pd.DataFrame(scorerows).to_csv(OUT/'case_scores.csv',index=False)
    pd.DataFrame(localrows).to_csv(OUT/'local_terms.csv',index=False)
    pd.DataFrame(perrows).to_csv(OUT/'per_reference_terms.csv',index=False)
    metadata=[]
    wanted=set(table.iloc[ids].obs_id)|set(source.iloc[srcids].obs_id)
    snapshots={s['obs_id']:s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if s['obs_id'] in wanted}
    for j,case in enumerate(cases):
        i=case['external_index'];true=case['true_class'];rival=case['fixed_comparison_rival']
        views=[(table,i,qc,'QUERY')]
        views.extend((source,int(ix),sc,f'TRUE REF {k+1}') for k,ix in enumerate(refs[j,true]))
        views.extend((source,int(ix),sc,f'FIXED RIVAL REF {k+1}') for k,ix in enumerate(refs[j,rival]))
        fig,axes=plt.subplots(2,5,figsize=(22,13))
        fig.subplots_adjust(left=.03,right=.985,top=.85,bottom=.28,wspace=.16,hspace=.34)
        # Native-image/region conventions reused from external_cases.py; no transport lines.
        for p,pol in enumerate(('vv','vh')):
            for k,(tab,idx,cache,label) in enumerate(views):
                row=tab.iloc[idx]
                pix,fp,g,place=geo.native(tab,snapshots,idx,p)
                ax=axes[p,k]
                ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                ax.set(xlim=(-.5,159.5),ylim=(159.5,-.5),xticks=[0,80,159],yticks=[0,80,159],aspect='equal')
                valid=bool(cache['valid'][idx,p])
                if g is not None and valid:
                    masks=slots(g,fp)
                    for sl,color in [(slice(0,8),'#24a878'),(slice(8,16),'#2c84c6'),(slice(16,20),'#d39222')]:
                        field=masks[sl].sum(0)
                        if field.min()<.5<field.max():
                            ax.contour(field,levels=[.5],colors=[color],linewidths=.75)
                if not valid:
                    ax.text(80,80,'INVALID LOCAL INPUT',ha='center',color='red',fontsize=9,bbox=dict(facecolor='white',alpha=.85))
                ax.set_title(f'{pol.upper()} {label} | obs{row.obs_id}\n'+textwrap.fill(str(row.class_name),25)+
                    f'\n{row.port} | valid={int(valid)}',fontsize=9)
                metadata.append(dict(query_obs=case['obs_id'],panel_column=k,pol=pol,
                    scope='external' if k==0 else 'source',local_index=idx,obs_id=int(row.obs_id),
                    instance_id=str(row.instance_id),mmsi=str(row.mmsi),class_id=int(row.class_id),
                    pass_group=str(row.pass_group),path=str(row['image_'+pol]),valid_local_tokens=valid,
                    native_w=snapshots[int(row.obs_id)]['images'][pol]['w'],
                    native_h=snapshots[int(row.obs_id)]['images'][pol]['h'],placement=place))
        fig.suptitle(f"EXTERNAL development | {case['outcome']} role_contrast vs ordered | obs{case['obs_id']} | seed17\n"
            f"true={true}; fixed rival={rival} ({case['rival_reason']}); role final rival={case['role_final_rival']} | group {case['group_n']}/{len(table)}",
            fontsize=13)
        lines=[]
        for arm in ('full','ordered','role_contrast','mispaired'):
            row=next(v for v in head_rows if v['obs_id']==case['obs_id'] and v['arm']==arm)
            step='' if arm=='full' else f", step={epochs[arm]}"
            lines.append(f"{arm}: pred={row['prediction']}, fixed margin={row['fixed_margin']:+.5f}, own top1 margin={row['own_top1_margin']:+.5f}{step}")
        for p,pol in enumerate(('VV','VH')):
            for r,region in [(1,'side'),(2,'end')]:
                delta={k:float(v[j,true,p,r]-v[j,rival,p,r]) for k,v in means.items()}
                lines.append(f"{pol} {region} input-margin: ordered={delta['ordered']:+.5f} - cross={delta['cross']:+.5f} = wedge={delta['wedge']:+.5f}; refs true/rival={n[j,true,p,r]}/{n[j,rival,p,r]}")
        fig.text(.035,.105,'\n'.join(lines),fontsize=10)
        fig.text(.035,.025,'Actual native160 crop/pad, no model-input resize. Green body / blue side / amber end.\n'
            'Local kernel terms are input descriptors, not additive logit attribution through the nonlinear head.\n'
            'Layer1 RF35 mixes nearby support. Fixed-rival margin and own top1 margin are distinct; no pixel-causal claim.',fontsize=10)
        fig.savefig(OUT/f"case_{case['obs_id']}.png",dpi=140)
        plt.close(fig)
    (OUT/'image_records.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (OUT/'replay_checks.json').write_text(json.dumps(dict(seed=SEED,external_n=len(table),cases=selections,
        checks=checks,ordered_minus_cross_closure=closed,availability_count_exact=True,
        no_training=True,no_backbone_extraction=True),indent=2),encoding='utf-8')
    print(pd.DataFrame(head_rows).to_string(index=False),flush=True)
    print('CHECKS',json.dumps(checks),'closure',closed,flush=True)


if __name__=='__main__':
    main()
