"""Matched classification summaries and native-chip diagnostics after unfreezing."""
from pathlib import Path
import argparse
import csv
import gc
import html
import importlib.util
import sys
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


h = load('unfrozen_analysis_helper', ROOT.parent/'pcea_readout_repair_20261116/analysis.py')
read, write = h.read, h.write


def cached(seed, last=False, config=None):
    data = np.load(ROOT.parent/f'full_finetune_20261111/selected_s{seed}/validation.npz')
    ids, y = data['ids'], data['label']
    scores = {'original': data['score']}
    entries = read(ROOT/'selected_checkpoints.json')[str(seed)]
    for arm, item in entries.items():
        if config:
            assert seed == 17
            item['checkpoint']=str(ROOT/f'tuning/{config}/{arm}_s17/best.pt')
            item['validation_predictions']=str(ROOT/f'tuning/{config}/{arm}_s17/validation.npz')
        if last:
            dest=Path(item['checkpoint']).parent
            item['checkpoint']=str(dest/'last.pt')
            item['validation_predictions']=str(dest/'lastvalidation.npz')
        scores[arm] = h.load_score(item['validation_predictions'], ids, y)
    return ids, y, scores, entries


def summarize():
    all_rows, matched, strata, endpoints = [], [], [], []
    manifest = read(ROOT.parent.parent/'m1_coverage_20261101/manifest.json')
    cases = {}
    for seed in map(int, read(ROOT/'selected_checkpoints.json')):
        ids, y, scores, entries = cached(seed)
        cases[str(seed)] = {}
        dual=np.array([manifest[i]['vv'] is not None and manifest[i]['vh'] is not None for i in ids])
        for arm, z in scores.items():
            metric = h.metrics(y, z)
            for control in ('original', 'full', 'cross_mean'):
                if control in scores:
                    row = dict(seed=seed, arm=arm, control=control,
                        oa=metric['oa'], ba=metric['ba'], **h.paired(y,z,scores[control]),
                        epoch=entries.get(arm,{}).get('epoch'), config=entries.get(arm,{}).get('config'))
                    all_rows.append(row)
            for group,mask in [('dual',dual),('single',~dual)]+[(h.NAMES[k],y==k) for k in range(4)]:
                if mask.any():
                    strata.append(dict(seed=seed,arm=arm,group=group,metrics=h.metrics(y[mask],z[mask]),
                        paired_to_full=h.paired(y[mask],z[mask],scores['full'][mask])))
            good = z.argmax(1) == y
            prior = scores['full'].argmax(1) == y
            cases[str(seed)][arm] = [dict(index=int(ids[j]),id=manifest[ids[j]]['id'],
                label=int(y[j]), prediction=int(z[j].argmax()), state='rescued' if good[j] else 'harmed',
                margin=float(h.margin(z,y)[j]),full_margin=float(h.margin(scores['full'],y)[j]))
                for j in np.flatnonzero(good!=prior)]
        if seed == 17:
            for config in ('base','slow'):
                zs = {a:h.load_score(ROOT/f'tuning/{config}/{a}_s17/validation.npz',ids,y)
                    for a in ('full','cross_mean','cross_local')}
                last_scores={a:h.load_score(ROOT/f'tuning/{config}/{a}_s17/lastvalidation.npz',ids,y) for a in zs}
                for a,z in zs.items():
                    for c in ('full','cross_mean'):
                        matched.append(dict(config=config,arm=a,control=c,metrics=h.metrics(y,z),
                            paired=h.paired(y,z,zs[c])))
                for arm in zs:
                    dest=ROOT/f'tuning/{config}/{arm}_s17'
                    receipt=read(dest/'receipt.json')
                    curve=read(dest/'curve.json')
                    last=h.load_score(dest/'lastvalidation.npz',ids,y)
                    endpoints.append(dict(config=config,arm=arm,best_epoch=receipt['best']['epoch'],
                        best=receipt['best']['metrics'],last=receipt['last'],
                        train_oa_start=curve[0]['train_oa'],train_oa_end=curve[-1]['train_oa'],
                        train_loss_start=curve[0]['train_loss'],train_loss_end=curve[-1]['train_loss'],
                        last_vs_original=h.paired(y,last,scores['original']),seconds=receipt['seconds'],
                        last_vs_full=h.paired(y,last,last_scores['full']),
                        last_vs_mean=h.paired(y,last,last_scores['cross_mean']),
                        last_confusion=h.metrics(y,last)['confusion'],
                        trainable=receipt['trainable'],peak_gib=receipt['peak_gib']))
    with (ROOT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader();writer.writerows(all_rows)
    write(ROOT/'summary.json',dict(rows=all_rows,matched_config=matched,cases=cases,strata=strata,endpoints=endpoints))
    print('SUMMARY', [r for r in all_rows if r['control']=='full'], flush=True)


def diagnose(seed, last=False, config=None):
    run=load('unfrozen_diagnose_runner',ROOT/'run.py')
    model,inputs=run.runner.runtime()
    torch,full=model.torch,model.full
    ids,y,scores,entries=cached(seed,last,config)
    suffix=('_'+config if config else '')+('_last' if last else '')
    arm='cross_local'
    net=model.Model(seed,arm)
    net.load_state_dict(torch.load(entries[arm]['checkpoint'],map_location='cpu',weights_only=True))
    net.eval().requires_grad_(False);net.accelerate()
    actual=full.predict(net,inputs,ids,'validation')
    replay=run.runner.replay(model,ids,actual,entries[arm]['validation_predictions'])
    variants={}
    for flag in ('uniform_attention','no_detail','no_product','disabled'):
        setattr(net.exchange,flag,True)
        variants[flag]=full.predict(net,inputs,ids,'validation')
        setattr(net.exchange,flag,False)
    write(ROOT/f'diagnostics_s{seed}{suffix}.json',dict(replay=replay,config=config or 'selected_configuration',checkpoint_kind='last_not_selected' if last else 'selected',variants=[dict(flag=k,
        metrics=h.metrics(y,z),paired=h.paired(y,z,actual),max_logit_change=float(abs(z-actual).max()))
        for k,z in variants.items()]))
    np.savez_compressed(ROOT/f'diagnostics_s{seed}{suffix}.npz',ids=ids,label=y,actual=actual,**variants)
    make_gallery(model,inputs,net,ids,y,scores,variants,seed,suffix)
    del net
    gc.collect();torch.cuda.empty_cache()


def make_gallery(model,inputs,net,ids,y,scores,variants,seed,suffix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype']='none'
    folder=ROOT/f'gallery_s{seed}{suffix}';folder.mkdir(exist_ok=True)
    chosen,counts=h.choose_cases(y,dict(scores,initial_full=scores['full']),'cross_local',ids)
    original_cases,original_counts=h.choose_cases(y,dict(scores,initial_full=scores['original']),'cross_local',ids)
    for state,j in original_cases:
        if j not in [n for _,n in chosen]:chosen.append(('vs_original_'+state,j))
    localgood=scores['cross_local'].argmax(1)==y
    meangood=scores['cross_mean'].argmax(1)==y
    for name,mask in (('local_only_rescue',localgood & ~meangood),('local_only_harm',~localgood & meangood)):
        ix=np.flatnonzero(mask)
        if len(ix):
            j=int(ix[np.argmin(abs(h.margin(scores['cross_local'],y)[ix]))])
            if j not in [n for _,n in chosen]:chosen.append((name,j))
    for sample in ('000955','003656','000414'):
        for j,i in enumerate(ids):
            if inputs.rows[i]['id'].endswith('_'+sample) and j not in [n for _,n in chosen]:
                chosen.append(('preset_'+sample,j))
    with model.torch.no_grad():ref=net.reference(inputs.data(inputs.groups['references'],'small'))
    records=[]
    for state,j in chosen:
        idx=int(ids[j]);row=inputs.rows[idx]
        with model.torch.no_grad():
            z,_=net(inputs.data([idx],'validation'),[row],ref,inputs.reference_rows)
        assert np.max(abs(z.cpu().numpy()[0]-scores['cross_local'][j]))<1e-3
        update=net.exchange.trace['delta'][0].square().mean(1).sqrt().cpu().numpy()
        legal=model.full.staged.task.legal([row]+inputs.reference_rows,np.array([0]),np.arange(1,33))[0]
        if model.torch.is_tensor(legal):legal=legal.cpu().numpy()
        rival=int(h.fixed_rival(scores['full'],y)[j])
        fig,axes=plt.subplots(2,4,figsize=(13,7))
        references=[]
        for p,pol in enumerate(('VV','VH')):
            for ax in axes[p]:ax.axis('off')
            axes[p,0].imshow(inputs.pixels[idx,p],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[p,0].set_title(pol+' native160 (crop/pad)',fontsize=9)
            axes[p,1].imshow(update[p],cmap='magma',vmin=0,vmax=max(float(update.max()),1e-8),interpolation='nearest')
            axes[p,1].set_title('PCEA update RMS / 10x10',fontsize=9)
            for col,klass in ((2,int(y[j])),(3,rival)):
                usable=[int(i) for i,ok in zip(inputs.groups['references'],legal) if ok and inputs.y[i]==klass and inputs.available[i,p]]
                if usable:
                    ri=usable[0]
                    axes[p,col].imshow(inputs.pixels[ri,p],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    axes[p,col].set_title(('True' if col==2 else 'Rival')+' '+inputs.rows[ri]['id'],fontsize=7)
                    references.append(dict(index=ri,id=inputs.rows[ri]['id'],polarization=pol,role='true' if col==2 else 'rival'))
        details={k:dict(prediction=int(z[j].argmax()),margin=float(h.margin(z,y)[j]),logits=z[j].tolist()) for k,z in dict(scores,**variants).items()}
        fig.suptitle(f'{row["id"]} | {h.NAMES[int(y[j])]} | seed{seed}{suffix} | {state}\n'+
            ' / '.join(f'{k}: {details[k]["margin"]:+.4f}' for k in scores),fontsize=10)
        fig.text(.5,.02,'References are legal context examples, NOT PCEA keys. Update magnitude is not causal attribution. Original preprocessing retained.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.02,right=.98,top=.83,bottom=.07,wspace=.15,hspace=.25)
        name=f'{state}_{idx}'
        fig.savefig(folder/(name+'.png'),dpi=130);fig.savefig(folder/(name+'.svg'));plt.close(fig)
        records.append(dict(index=idx,id=row['id'],label=int(y[j]),state=state,image=name+'.png',
            scores=details,references=references,source_paths=row.get('source_paths'),
            active=net.exchange.trace['active'][0].cpu().tolist()))
    write(folder/'cases_local.json',dict(counts=counts,counts_vs_original=original_counts,cases=records))
    page='<meta charset="utf-8"><title>PCEA late unfreezing</title><style>body{font:15px sans-serif;margin:24px}img{max-width:100%}</style>'
    for record in records:
        page+=f'<h2>{html.escape(record["id"])} / {record["state"]}</h2><img src="{record["image"]}">'
    (folder/'index.html').write_text(page,encoding='utf-8')
    print('GALLERY',folder,len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--diagnose',action='store_true');p.add_argument('--seed',type=int,default=17);p.add_argument('--last',action='store_true');p.add_argument('--config',choices=('base','slow'))
    args=p.parse_args();summarize()
    if args.diagnose:diagnose(args.seed,args.last,args.config)
