"""Paired pilot summary and fixed-weight local-information diagnostics."""
import argparse
import csv
import gc
import html
import importlib.util
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'pcea_conditional_modulation_20261117'
FULL = ROOT.parent/'full_finetune_20261111'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helper = load('local_cross_analysis_helper', ROOT.parent/'pcea_readout_repair_20261116/analysis.py')
read, write = helper.read, helper.write


def cached(seed=17):
    base = np.load(FULL/f'selected_s{seed}/validation.npz')
    ids, y = base['ids'], base['label']
    scores = {'full': base['score']}
    old = read(OLD/'selected_checkpoints.json')[str(seed)]['affine_frozen']
    scores['pooled'] = helper.load_score(old['validation_predictions'], ids, y)
    entries = read(ROOT/'selected_checkpoints.json')[str(seed)]
    for arm, entry in entries.items():
        scores[arm] = helper.load_score(entry['validation_predictions'], ids, y)
    return ids, y, scores, entries


def summarize():
    manifest = read(ROOT.parent.parent/'m1_coverage_20261101/manifest.json')
    rows, changed = [], {}
    for key in read(ROOT/'selected_checkpoints.json'):
        seed = int(key)
        ids, y, scores, entries = cached(seed)
        dual = np.array([manifest[i]['vv'] is not None and manifest[i]['vh'] is not None for i in ids])
        changed[key] = {}
        for arm, z in scores.items():
            metric = helper.metrics(y, z)
            logsum = np.log(np.exp(z-z.max(1, keepdims=True)).sum(1))+z.max(1)
            metric['nll'] = float((logsum-z[np.arange(len(y)), y]).mean())
            for control in ('full', 'pooled', 'independent', 'cross_mean'):
                if control not in scores:
                    continue
                rows.append(dict(seed=seed, arm=arm, control=control,
                    **{k:v for k,v in metric.items() if k!='confusion'},
                    **helper.paired(y,z,scores[control]),
                    single_polar_changes=int(((z.argmax(1)!=scores['full'].argmax(1)) & ~dual).sum()),
                    epoch=entries.get(arm,{}).get('epoch'), config=entries.get(arm,{}).get('config')))
            diff = np.flatnonzero(z.argmax(1)!=scores['pooled'].argmax(1))
            changed[key][arm] = [dict(index=int(ids[j]), id=manifest[ids[j]]['id'], label=int(y[j]),
                pred=int(z[j].argmax()), old_pred=int(scores['pooled'][j].argmax()),
                margin=float(helper.margin(z,y)[j]), dual=bool(dual[j])) for j in diff]
    write(ROOT/'summary.json',dict(rows=rows,changed=changed))
    with (ROOT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print({'rows':[r for r in rows if r['control']=='cross_mean'],'changed':changed},flush=True)


def gallery(model, inputs, net, ids, y, scores, arm, variants, seed, suffix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype']='none'
    folder=ROOT/f'gallery_{arm}_s{seed}{suffix}'
    folder.mkdir(exist_ok=True)
    chosen, counts=helper.choose_cases(y,dict(scores,initial_full=scores['pooled']),arm,ids)
    dual=np.array([inputs.available[i].all() for i in ids],dtype=bool)
    delta_margin=helper.margin(scores[arm],y)-helper.margin(scores['cross_mean'],y)
    candidates=np.flatnonzero(dual)
    extras=[('max_vs_mean',int(candidates[np.argmax(delta_margin[candidates])])),
            ('min_vs_mean',int(candidates[np.argmin(delta_margin[candidates])]))]
    for ending in ('000955','003656','000414'):
        extras += [('preset_'+ending,j) for j,i in enumerate(ids) if inputs.rows[i]['id'].endswith('_'+ending)]
    for state,j in extras:
        if j not in [k for _,k in chosen]:
            chosen.append((state,j))
    with model.torch.inference_mode():
        ref=net.reference(inputs.data(inputs.groups['references'],'small'))
    def retain_masks(module,args):
        net.last_reader_masks=args[4].detach()
    mask_hook=net.exchange.register_forward_pre_hook(retain_masks)
    records=[]
    for state,j in chosen:
        idx,row=int(ids[j]),inputs.rows[ids[j]]
        net.exchange.capture_attention=True
        with model.torch.inference_mode():
            z,_=net(inputs.data([idx],'validation'),[row],ref,inputs.reference_rows)
        net.exchange.capture_attention=False
        assert np.max(abs(z.cpu().numpy()[0]-scores[arm][j]))<1e-3
        trace=net.exchange.trace
        update=trace['delta'][0].square().mean(1).sqrt().cpu().numpy()
        attention=trace['attention'][0].cpu().numpy()
        # Display exactly the masks consumed by this prediction.
        reader_mask=np.asarray(net.last_reader_masks[0].cpu())
        legal=model.full.staged.task.legal([row]+inputs.reference_rows,np.array([0]),np.arange(1,33))[0]
        legal=legal.cpu().numpy() if model.torch.is_tensor(legal) else np.asarray(legal)
        rival=int(helper.fixed_rival(scores['pooled'],y)[j])
        fig,axes=plt.subplots(2,5,figsize=(15,7))
        refs, locations=[],[]
        for p,pol in enumerate(('VV','VH')):
            for ax in axes[p]: ax.axis('off')
            axes[p,0].imshow(inputs.pixels[idx,p],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[p,0].set_title(pol+' native160',fontsize=9)
            axes[p,1].imshow(update[p],cmap='magma',vmin=0,vmax=max(float(update.max()),1e-8),interpolation='nearest')
            axes[p,1].set_title('Writeback RMS 10x10',fontsize=9)
            # Choose highest body-mask occupancy, not a truth-dependent "important" point.
            query=int(reader_mask[p,0].argmax())
            heat=attention[p,0,0,query].reshape(20,20)
            axes[p,2].imshow(heat,cmap='viridis',interpolation='nearest')
            context_label='Own' if arm=='independent' else 'Other'
            axes[p,2].set_title(f'{context_label} polar body / mean-slot\nquery20=({query//20},{query%20})',fontsize=8)
            locations.append(dict(polarization=pol,query_cell=[query//20,query%20],
                selection='max body-mask occupancy; diagnostic, not attribution ranking',
                maximum_key=[int(heat.argmax()//20),int(heat.argmax()%20)]))
            for col,klass in ((3,int(y[j])),(4,rival)):
                usable=[int(i) for i,ok in zip(inputs.groups['references'],legal) if ok and inputs.y[i]==klass and inputs.available[i,p]]
                if usable:
                    ri=usable[0]
                    axes[p,col].imshow(inputs.pixels[ri,p],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    axes[p,col].set_title(('True' if col==3 else 'Rival')+' '+inputs.rows[ri]['id'],fontsize=7)
                    refs.append(dict(index=ri,id=inputs.rows[ri]['id'],polarization=pol,role='true' if col==3 else 'rival'))
        details={name:dict(prediction=int(z[j].argmax()),margin=float(helper.margin(z,y)[j]),logits=z[j].tolist()) for name,z in dict(scores,**variants).items()}
        attention_stats=[]
        for p in range(2):
            for r in range(3):
                target=reader_mask[p,r].reshape(-1)
                source=reader_mask[1-p if arm!='independent' else p,r].reshape(-1)
                target=target/max(float(target.sum()),1e-8)
                source=source/max(float(source.sum()),1e-8)
                tv=.5*np.abs(attention[p,r]-source[None,None,:]).sum(-1)
                attention_stats.append(dict(polarization=('VV','VH')[p],region=('body','ends','side')[r],
                    area_weighted_total_variation=float((tv*target[None,:]).sum(-1).mean())))
        fig.suptitle(f'{row["id"]} | {helper.NAMES[int(y[j])]} | {arm} seed{seed} | {state}\n'+
            ' / '.join(f'{k}: {details[k]["margin"]:+.4f}' for k in ('full','pooled','independent','cross_mean','cross_local')),fontsize=10)
        fig.text(.5,.02,'Native crop/pad. Reference chips are legal examples, not influence rankings. Attention is a diagnostic, not physical correspondence.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.02,right=.98,top=.82,bottom=.07,wspace=.15,hspace=.3)
        name=f'{state}_{idx}'
        fig.savefig(folder/(name+'.png'),dpi=120)
        fig.savefig(folder/(name+'.svg'))
        plt.close(fig)
        records.append(dict(id=row['id'],index=idx,label=int(y[j]),state=state,references=refs,
            active=trace['active'][0].cpu().tolist(),
            source_paths=row.get('source_paths'),scores=details,attention_locations=locations,
            attention_vs_area=attention_stats,image=name+'.png'))
    mask_hook.remove()
    write(folder/'cases_local.json',dict(counts=counts,cases=records))
    page='<meta charset="utf-8"><title>PCEA local interaction diagnostics</title><style>body{font:15px sans-serif;margin:24px}img{max-width:100%}</style>'
    for case in records:
        page+=f'<h2>{html.escape(case["id"])} / {case["state"]}</h2><img src="{case["image"]}">'
    (folder/'index.html').write_text(page,encoding='utf-8')


def diagnose(arms,seed=17,last=False):
    run=load('local_cross_diagnose_run',ROOT/'run.py')
    model,inputs=run.runner.runtime()
    torch,full=model.torch,model.full
    ids,y,scores,entries=cached(seed)
    result={}
    for arm in arms:
        entry=dict(entries[arm])
        if last:
            dest=Path(entry['checkpoint']).parent
            entry['checkpoint']=str(dest/'last.pt')
            entry['validation_predictions']=str(dest/'lastvalidation.npz')
            scores[arm]=helper.load_score(entry['validation_predictions'],ids,y)
        net=model.Model(seed,arm)
        net.load_state_dict(torch.load(entry['checkpoint'],map_location='cpu',weights_only=True))
        net.eval().requires_grad_(False)
        net.accelerate()
        traces=[]
        def capture(module,args,output):
            traces.append({k:module.trace[k].cpu().numpy() for k in ('active','effective_cells','detail_rms','input_rms','update_rms','context_spatial_rms','token_spatial_rms')})
        handle=net.exchange.register_forward_hook(capture)
        actual=full.predict(net,inputs,ids,'validation')
        handle.remove()
        arrays={k:np.concatenate([a[k] for a in traces]) for k in traces[0]}
        replay=run.runner.replay(model,ids,actual,entry['validation_predictions'])
        variants={'actual':actual}
        for flag in ('uniform_attention','other_mean','no_detail','no_product','disabled'):
            setattr(net.exchange,flag,True)
            variants[flag]=full.predict(net,inputs,ids,'validation')
            setattr(net.exchange,flag,False)
        suffix='_last' if last else ''
        np.savez_compressed(ROOT/f'diagnostics_{arm}_s{seed}{suffix}.npz',ids=ids,label=y,**variants)
        np.savez_compressed(ROOT/f'trace_{arm}_s{seed}{suffix}.npz',ids=ids,**arrays)
        eligible=arrays['active'].any(1)
        rows=[dict(variant=name,metrics=helper.metrics(y,z),paired_to_actual=helper.paired(y,z,actual),
            max_logit_change=float(abs(z-actual).max())) for name,z in variants.items()]
        result[arm]=dict(replay=replay,variants=rows,trace_summary=dict(
            eligible=int(eligible.sum()),detail_rms=helper.stats(arrays['detail_rms'][eligible]),
            token_spatial_rms=helper.stats(arrays['token_spatial_rms'][eligible]),
            context_spatial_rms=helper.stats(arrays['context_spatial_rms'][eligible]),
            update_to_input=helper.stats((arrays['update_rms']/np.maximum(arrays['input_rms'],1e-8))[eligible]),
            common_shift=helper.constant_shift(actual,scores['full'],y,eligible)))
        gallery(model,inputs,net,ids,y,scores,arm,variants,seed,suffix)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    write(ROOT/f'diagnostics_s{seed}{"_last" if last else ""}.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--diagnose',action='store_true')
    p.add_argument('--arms',nargs='+',default=['cross_local'])
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--last',action='store_true')
    a=p.parse_args()
    summarize()
    if a.diagnose: diagnose(a.arms,a.seed,a.last)
