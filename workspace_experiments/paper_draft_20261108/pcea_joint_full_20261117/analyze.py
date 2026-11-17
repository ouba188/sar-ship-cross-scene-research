"""Paired classification accounting and selected-model pathway interventions."""
from pathlib import Path
import argparse
import csv
import html
import importlib.util
import sys
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


run=load('joint_full_analyze_run',ROOT/'run.py')
h=load('joint_full_analyze_helper',ROOT.parent/'pcea_readout_repair_20261116/analysis.py')
read,write=h.read,h.write


def entries(seed):
    selection=read(ROOT/'selection.json')
    if seed==17:return selection['selected']
    return {a:read(ROOT/'fits'/f"{a}_s{seed}_{r['config']}"/'receipt.json')
            for a,r in selection['selected'].items()}


def scores_for(seed):
    chosen=entries(seed)
    with np.load(chosen['full']['validation_predictions']) as f:
        ids,y=f['ids'],f['label']
    scores={a:h.load_score(r['validation_predictions'],ids,y) for a,r in chosen.items()}
    return ids,y,scores,chosen


def summarize():
    selection=read(ROOT/'selection.json')
    seeds=[17]
    if (ROOT/'confirmation.json').exists() and read(ROOT/'confirmation.json')['complete']:
        seeds += [29,43]
    manifest=read(ROOT.parent.parent/'m1_coverage_20261101/manifest.json')
    rows,strata,paired_configs,cases=[],[],[],{}
    for seed in seeds:
        ids,y,scores,chosen=scores_for(seed)
        cases[str(seed)]={}
        for arm,z in scores.items():
            m=h.metrics(y,z)
            rows.append(dict(seed=seed,arm=arm,config=chosen[arm]['config'],epoch=chosen[arm]['best']['epoch'],
                epochs_run=chosen[arm]['epochs'],oa=m['oa'],ba=m['ba'],nll=chosen[arm]['best']['metrics']['nll'],
                **h.paired(y,z,scores['full'])))
            for name,mask in [(h.NAMES[k],y==k) for k in range(4)]+[
                ('dual',np.array([manifest[i]['vv'] is not None and manifest[i]['vh'] is not None for i in ids])),
                ('single',np.array([manifest[i]['vv'] is None or manifest[i]['vh'] is None for i in ids]))]+[
                ('pass:'+p,np.array([manifest[i]['pass_group']==p for i in ids]))
                for p in sorted({manifest[i]['pass_group'] for i in ids})]:
                if mask.any():
                    strata.append(dict(seed=seed,arm=arm,group=name,n=int(mask.sum()),
                        correct=int((z[mask].argmax(1)==y[mask]).sum()),
                        full_correct=int((scores['full'][mask].argmax(1)==y[mask]).sum())))
            old=scores['full'].argmax(1)==y;new=z.argmax(1)==y
            cases[str(seed)][arm]=[dict(index=int(ids[j]),id=manifest[ids[j]]['id'],label=int(y[j]),
                prediction=int(z[j].argmax()),state='rescue' if new[j] else 'harm',
                margin=float(h.margin(z,y)[j]),full_margin=float(h.margin(scores['full'],y)[j]))
                for j in np.flatnonzero(old!=new)]
        paired_configs.append(dict(seed=seed,configuration='independently validation-selected',
            local_vs_mean=h.paired(y,scores['cross_local'],scores['cross_mean'])))
        if seed==17:
            for config in run.CONFIGS:
                z={a:h.load_score(ROOT/'fits'/f'{a}_s17_{config}'/'validation.npz',ids,y) for a in run.ARMS}
                paired_configs.append(dict(seed=seed,configuration=config,
                    mean_vs_full=h.paired(y,z['cross_mean'],z['full']),
                    local_vs_full=h.paired(y,z['cross_local'],z['full']),
                    local_vs_mean=h.paired(y,z['cross_local'],z['cross_mean'])))
    with (ROOT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(ROOT/'summary.json',dict(rows=rows,strata=strata,matched=paired_configs,cases=cases,
        note='Repeatedly used development validation; not independent test; no 24-port claim.'))
    print('SUMMARY',rows,flush=True)


def diagnose(seed, arm='cross_local'):
    model,inputs=run.runtime()
    torch,full=model.torch,model.full
    ids,y,scores,chosen=scores_for(seed)
    net=model.Model(seed,arm)
    net.load_state_dict(torch.load(chosen[arm]['checkpoint'],map_location='cpu',weights_only=True))
    net.eval().requires_grad_(False);net.accelerate()
    actual=full.predict(net,inputs,ids,'validation')
    err=float(abs(actual-scores[arm]).max())
    assert err<1e-3 and np.array_equal(actual.argmax(1),scores[arm].argmax(1)),err
    variants={}
    for flag in ('uniform_attention','no_detail','no_product','disabled'):
        setattr(net.exchange,flag,True)
        variants[flag]=full.predict(net,inputs,ids,'validation')
        setattr(net.exchange,flag,False)
    prior_mode=net.exchange.mode
    net.exchange.mode='independent'
    variants['own_context']=full.predict(net,inputs,ids,'validation')
    net.exchange.mode=prior_mode
    with torch.no_grad():
        ref=net.reference(inputs.data(inputs.groups['references'],'small'))
        part=ids[:1]
        single=net(inputs.data(part,'validation'),[inputs.rows[int(part[0])]],ref,inputs.reference_rows)[0]
    batch_error=float(abs(single.cpu().numpy()[0]-actual[0]).max())
    assert batch_error<1e-3
    write(ROOT/f'diagnostics_{arm}_s{seed}.json',dict(arm=arm,replay_error=err,single_batch_error=batch_error,
        variants=[dict(flag=k,metrics=h.metrics(y,z),paired=h.paired(y,z,actual),
                       max_logit_change=float(abs(z-actual).max())) for k,z in variants.items()]))
    np.savez_compressed(ROOT/f'diagnostics_{arm}_s{seed}.npz',ids=ids,label=y,actual=actual,**variants)
    gallery(model,inputs,net,ids,y,scores,variants,seed,arm)


def gallery(model,inputs,net,ids,y,scores,variants,seed,arm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype']='none'
    folder=ROOT/f'gallery_{arm}_s{seed}';folder.mkdir(exist_ok=True)
    chosen,counts=h.choose_cases(y,dict(scores,initial_full=scores['full']),arm,ids)
    comparator='cross_mean' if arm=='cross_local' else 'cross_local'
    good=scores[arm].argmax(1)==y;other_good=scores[comparator].argmax(1)==y
    for state,mask in [(arm+'_vs_'+comparator+'_rescue',good & ~other_good),(arm+'_vs_'+comparator+'_harm',~good & other_good)]:
        ix=np.flatnonzero(mask)
        if len(ix):
            j=int(ix[np.argmin(abs(h.margin(scores[arm],y)[ix]))])
            if j not in [n for _,n in chosen]:chosen.append((state,j))
    torch=model.torch
    with torch.no_grad():ref=net.reference(inputs.data(inputs.groups['references'],'small'))
    records=[]
    for state,j in chosen:
        idx=int(ids[j]);row=inputs.rows[idx]
        with torch.no_grad():
            z,_=net(inputs.data([idx],'validation'),[row],ref,inputs.reference_rows)
        assert float(abs(z.cpu().numpy()[0]-scores[arm][j]).max())<1e-3
        delta=net.exchange.trace['delta'][0].square().mean(1).sqrt().cpu().numpy()
        legal=model.full.staged.task.legal([row]+inputs.reference_rows,np.array([0]),np.arange(1,33))[0]
        if torch.is_tensor(legal):legal=legal.cpu().numpy()
        rival=int(h.fixed_rival(scores['full'],y)[j])
        fig,axes=plt.subplots(2,4,figsize=(13,7));refs=[]
        for p,pol in enumerate(('VV','VH')):
            for ax in axes[p]:ax.axis('off')
            axes[p,0].imshow(inputs.pixels[idx,p],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[p,0].set_title(pol+' native160 crop/pad',fontsize=9)
            axes[p,1].imshow(delta[p],cmap='magma',vmin=0,vmax=max(float(delta.max()),1e-8),interpolation='nearest')
            axes[p,1].set_title('PCEA update RMS 10x10',fontsize=9)
            for col,k in ((2,int(y[j])),(3,rival)):
                usable=[int(i) for i,ok in zip(inputs.groups['references'],legal) if ok and inputs.y[i]==k and inputs.available[i,p]]
                if usable:
                    ri=usable[0]
                    axes[p,col].imshow(inputs.pixels[ri,p],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    axes[p,col].set_title(('True ' if col==2 else 'Rival ')+inputs.rows[ri]['id'],fontsize=7)
                    refs.append(dict(index=ri,id=inputs.rows[ri]['id'],polarization=pol,role='true' if col==2 else 'rival'))
        details={k:dict(prediction=int(z[j].argmax()),margin=float(h.margin(z,y)[j]),logits=z[j].tolist()) for k,z in dict(scores,**variants).items()}
        fig.suptitle(f'{row["id"]} | {h.NAMES[int(y[j])]} | {arm} seed{seed} | {state}\n'+
                     ' / '.join(f'{k}: {details[k]["margin"]:+.4f}' for k in scores),fontsize=10)
        fig.text(.5,.02,'Legal reference examples, not PCEA attention keys. Update RMS is not causal attribution. Same native preprocessing.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.02,right=.98,top=.83,bottom=.07,wspace=.15,hspace=.25)
        stem=f'{state}_{idx}';fig.savefig(folder/(stem+'.png'),dpi=130);fig.savefig(folder/(stem+'.svg'));plt.close(fig)
        records.append(dict(index=idx,id=row['id'],label=int(y[j]),state=state,image=stem+'.png',details=details,references=refs,
                            vv=row['vv'],vh=row['vh']))
    write(folder/'cases_local.json',dict(seed=seed,counts=counts,records=records))
    body=''.join(f'<section><h2>{html.escape(r["state"]+" "+r["id"])}</h2><img src="{r["image"]}" style="max-width:100%"><pre>{html.escape(str(r["details"]))}</pre></section>' for r in records)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>PCEA Joint Training Cases</title><h1>Selected joint-training model</h1><p>Development cases; VV/VH native inputs. Update magnitude is not attribution.</p>'+body,encoding='utf-8')
    print('GALLERY',folder,len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--diagnose',type=int)
    p.add_argument('--arm',choices=('cross_mean','cross_local'),default='cross_local');args=p.parse_args()
    if args.diagnose is None:summarize()
    else:diagnose(args.diagnose,args.arm)
