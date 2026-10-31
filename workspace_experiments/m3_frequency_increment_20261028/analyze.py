"""Paired evidence, native-chip cases and fixed-path checks for this pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run as exp

HERE=exp.HERE


def relation_metrics(y, ry, score, valid):
    ap, top, mass=[] ,[],[]
    for c,s,v in zip(y,score,valid):
        ids=np.flatnonzero(v)
        positives=ry[ids]==c
        count=int(positives.sum())
        if not count:
            continue
        ordered=positives[np.argsort(-s[ids],kind='stable')]
        ap.append(float((np.cumsum(ordered[:count])/np.arange(1,count+1)*ordered[:count]).sum()/count))
        top.append(float(ordered[0]))
        p=np.exp(s[ids]-s[ids].max());p/=p.sum()
        mass.append(float(p[positives].sum()))
    return dict(map_at_r=float(np.mean(ap)),reference_top1=float(np.mean(top)),same_class_mass=float(np.mean(mass)))


def tables():
    rows,nf,refs,rid,arrays=exp.load()
    y=np.array([r['label'] for r in rows[nf:]])
    ry=np.array([r['label'] for r in refs])
    summary,classes,pairs=[],[],[]
    for seed in exp.SEEDS:
        outcomes={}
        for arm in exp.ARMS:
            z=dict(np.load(HERE/'training'/f'{arm}_{seed}'/'final.npz'))
            outcomes[arm]=z
            summary.append(dict(arm=arm,seed=seed,**exp.metrics(y,z['score']),
                **relation_metrics(y,ry,z['relation'],z['valid'])))
            for c in np.unique(y):
                classes.append(dict(arm=arm,seed=seed,class_id=int(c),n=int((y==c).sum()),
                    recall=float((z['score'].argmax(1)[y==c]==c).mean()*100)))
        for new,base in (('B_current','A_no_frequency'),('C_band_layout','A_no_frequency'),
                         ('D_spatial_layout','A_no_frequency'),('C_band_layout','D_spatial_layout')):
            a,b=outcomes[new]['score'],outcomes[base]['score']
            ca,cb=a.argmax(1)==y,b.argmax(1)==y
            ma,mb=exp.metrics(y,a),exp.metrics(y,b)
            pairs.append(dict(new=new,base=base,seed=seed,rescue=int((ca&~cb).sum()),harm=int((~ca&cb).sum()),
                **{k:ma[k]-mb[k] for k in ma}))
    pd.DataFrame(summary).to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(HERE/'class_recall.csv',index=False)
    pd.DataFrame(pairs).to_csv(HERE/'paired.csv',index=False)
    aggregate=pd.DataFrame(summary).groupby('arm').agg({k:['mean','std'] for k in summary[0] if k not in ('arm','seed')})
    aggregate.to_csv(HERE/'means.csv')
    print(aggregate.to_string(),flush=True)
    print(pd.DataFrame(pairs).to_string(index=False),flush=True)
    exp.write(HERE/'coverage.json',dict(fit=nf,development=len(y),
        development_vessels=len({r['mmsi'] for r in rows[nf:]}),
        development_passes=len({r['pass_group'] for r in rows[nf:]}),
        development_ports=sorted({r['port'] for r in rows[nf:]}),
        development_classes={str(int(c)):int((y==c).sum()) for c in np.unique(y)},
        valid_slots=int((arrays['mass']>0).sum()),
        native_single_pixel_fraction=float((arrays['native_count'][arrays['mass']>0]<=1).mean()),
        native_under16_pixel_fraction=float((arrays['native_count'][arrays['mass']>0]<16).mean())))


def restore(arm,seed,arrays,nf):
    model,norm=exp.make(arm,seed,arrays,nf)
    blob=torch.load(HERE/'training'/f'{arm}_{seed}'/'selector.pt',map_location='cpu',weights_only=True)
    status=model.load_state_dict(blob,strict=False)
    assert not status.unexpected_keys
    return model,norm


@torch.no_grad()
def diagnostics():
    rows,nf,refs,rid,arrays=exp.load()
    ids=np.arange(nf,len(rows))
    y=np.array([r['label'] for r in rows[nf:]])
    summary=[]
    for arm in exp.ARMS:
        for seed in exp.SEEDS:
            model,norm=restore(arm,seed,arrays,nf)
            replay=exp.predict(model,arrays,ids,rid,rows,norm)
            saved=np.load(HERE/'training'/f'{arm}_{seed}'/'final.npz')['score']
            err=float(abs(saved-replay['score']).max())
            assert err<1e-5,err
            if arm!='A_no_frequency':
                for op in model.m3[0]: op.frequency=False
                off=exp.predict(model,arrays,ids,rid,rows,norm)
                np.savez_compressed(HERE/'training'/f'{arm}_{seed}'/'frequency_off.npz',**off)
                ca,cb=saved.argmax(1)==y,off['score'].argmax(1)==y
                summary.append(dict(arm=arm,seed=seed,replay_error=err,
                    rescue=int((ca&~cb).sum()),harm=int((~ca&cb).sum()),
                    score_delta=float(abs(saved-off['score']).max()),
                    **{k:exp.metrics(y,saved)[k]-exp.metrics(y,off['score'])[k] for k in ('oa','ba','nll','brier')}))
            del model
            torch.cuda.empty_cache()
    pd.DataFrame(summary).to_csv(HERE/'path_dependency.csv',index=False)
    model,norm=exp.make('B_current',None,arrays,nf)
    original=exp.predict(model,arrays,ids,rid,rows,norm)
    np.savez_compressed(HERE/'original_frozen.npz',**original)
    exp.write(HERE/'original_frozen_metrics.json',exp.metrics(y,original['score']))
    del model
    evidence_rows=[]
    path_cache={}
    for arm in ('B_current','C_band_layout','D_spatial_layout'):
        model,norm=restore(arm,17,arrays,nf)
        buffers={(p,r):[] for p in range(2) for r in range(3)}
        weights={(p,r):[] for p in range(2) for r in range(3)}
        updates=[]
        reference_weights=[]
        for first in range(0,len(ids),24):
            batch=ids[first:first+24]
            _,info=exp.forward(model,arrays,batch,rid,rows,norm,True)
            updates.append(torch.stack([d['update'] for d in info],1).cpu().numpy())
            reference_weights.append(torch.stack([d['reference_weights'] for d in info],1).cpu().numpy())
            legal=exp.data.pair_mask([rows[i] for i in batch],[rows[i] for i in rid],exp.DEVICE)
            for p,d in enumerate(info):
                for r,ix in enumerate(exp.tokens.REGIONS):
                    mass=exp.tensor(arrays['mass'],batch)[:,p,ix]
                    rm=exp.tensor(arrays['mass'],rid)[:,p,ix]
                    live=(mass[:,None,:,None]>0)&(rm[None,:,None,:]>0)&legal[:,:,None,None]
                    buffers[p,r].append(d['branch_similarity'][r][live].cpu().numpy())
                    weights[p,r].append(d['branch_weights'][r][live].cpu().numpy())
        for key in buffers:
            a=np.concatenate(buffers[key]);w=np.concatenate(weights[key]);corr=np.corrcoef(a.T)
            for branch in range(5):
                evidence_rows.append(dict(arm=arm,pol=('VV','VH')[key[0]],region=('body','side','end')[key[1]],
                    branch=branch,comparisons=len(a),similarity_std=float(a[:,branch].std()),
                    correlation_with_fine_mean=float(corr[1,branch]),
                    mean_weight=float(w[:,branch].mean()),weight_std=float(w[:,branch].std())))
        path_cache[arm]=(np.concatenate(updates),np.concatenate(reference_weights))
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(evidence_rows).to_csv(HERE/'evidence_diagnostics.csv',index=False)
    model,norm=restore('A_no_frequency',17,arrays,nf)
    updates=[];reference_weights=[]
    for first in range(0,len(ids),24):
        _,ds=exp.forward(model,arrays,ids[first:first+24],rid,rows,norm,True)
        updates.append(torch.stack([d['update'] for d in ds],1).cpu().numpy())
        reference_weights.append(torch.stack([d['reference_weights'] for d in ds],1).cpu().numpy())
    ua,wa=np.concatenate(updates),np.concatenate(reference_weights)
    qnorm=np.linalg.norm(np.asarray(arrays['x'][ids]).reshape(len(ids),-1),axis=1)
    unorm=np.linalg.norm(ua.reshape(len(ids),-1),axis=1)
    path_rows=[]
    for arm,(update,w) in path_cache.items():
        diff=np.linalg.norm((update-ua).reshape(len(ids),-1),axis=1)
        for j,idx in enumerate(ids):
            path_rows.append(dict(arm=arm,id=rows[idx]['id'],
                reference_weight_l1=float(abs(w[j]-wa[j]).sum(axis=1).mean()),
                update_relative_l2=float(diff[j]/max(unorm[j],1e-20)),
                update_to_feature_l2=float(diff[j]/max(qnorm[j],1e-20))))
    pd.DataFrame(path_rows).to_csv(HERE/'path_change.csv',index=False)
    for op in model.m3[0]:op.gain.zero_()
    no_m3=exp.predict(model,arrays,ids,rid,rows,norm)
    np.savez_compressed(HERE/'fixed_no_m3.npz',**no_m3)
    exp.write(HERE/'fixed_no_m3_metrics.json',exp.metrics(y,no_m3['score']))
    del model
    source_rows=[]
    for arm in ('A_no_frequency','C_band_layout'):
        model,norm=restore(arm,17,arrays,nf)
        train=exp.predict(model,arrays,np.arange(nf),rid,rows,norm)
        ty=np.array([r['label'] for r in rows[:nf]])
        ry=np.array([r['label'] for r in refs])
        source_rows.append(dict(arm=arm,**exp.metrics(ty,train['score']),
            **relation_metrics(ty,ry,train['relation'],train['valid'])))
        del model
    pd.DataFrame(source_rows).to_csv(HERE/'training_domain_diagnostic.csv',index=False)
    print('DIAGNOSTICS complete',flush=True)


def margin(score,y):
    competitor=score.copy();competitor[np.arange(len(y)),y]=-np.inf
    return score[np.arange(len(y)),y]-competitor.max(1)


@torch.no_grad()
def gallery():
    rows,nf,refs,rid,arrays=exp.load()
    y=np.array([r['label'] for r in rows[nf:]])
    directory=HERE/'gallery';directory.mkdir(exist_ok=True)
    selections=[]
    for new,base,groups in (('C_band_layout','A_no_frequency',('rescue','harm','CC','WW')),
                           ('C_band_layout','D_spatial_layout',('rescue','harm','CC','WW')),
                           ('B_current','A_no_frequency',('rescue','harm','CC','WW'))):
        a=np.load(HERE/'training'/f'{new}_17'/'final.npz')['score']
        b=np.load(HERE/'training'/f'{base}_17'/'final.npz')['score']
        ca,cb=a.argmax(1)==y,b.argmax(1)==y
        conditions=dict(rescue=ca&~cb,harm=~ca&cb,CC=ca&cb,WW=~ca&~cb)
        for group in groups:
            ix=np.flatnonzero(conditions[group])
            if len(ix)==0:continue
            pos=int(ix[np.argmax(abs(margin(a,y)-margin(b,y))[ix])])
            selections.append(dict(new=new,base=base,group=group,position=pos,index=nf+pos,
                id=rows[nf+pos]['id'],truth=int(y[pos]),new_pred=int(a[pos].argmax()),base_pred=int(b[pos].argmax()),
                new_margin=float(margin(a,y)[pos]),base_margin=float(margin(b,y)[pos])))
    metadata=[]
    reference_views=[exp.data.prepare(r) for r in refs]
    for case in selections:
        idx=case['index'];record=rows[idx];traces={};scores={}
        for arm in (case['base'],case['new']):
            model,norm=restore(arm,17,arrays,nf)
            score,ds=exp.forward(model,arrays,np.array([idx]),rid,rows,norm,True)
            traces[arm]=[d['reference_weights'][0].mean(-1).cpu().numpy() for d in ds]
            scores[arm]=score[0].cpu().numpy()
            del model
        qviews=exp.data.prepare(record)
        fig,axes=plt.subplots(2,6,figsize=(15,6.8))
        competitor=scores[case['new']].copy();competitor[case['truth']]=-np.inf
        conf=int(competitor.argmax());details=[]
        for p in range(2):
            pol=('VV','VH')[p]
            for ax in axes[p]: ax.axis('off')
            if qviews[p] is None:continue
            view=qviews[p]
            axes[p,0].imshow(view['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[p,0].set_title(f'{pol} query {case["id"]}',fontsize=9)
            axes[p,1].imshow(view['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            masks=exp.tokens.slots(view['geometry'],view['footprint'])
            for region,color in zip(exp.tokens.REGIONS,('red','cyan','lime')):
                if masks[region].sum()>0:axes[p,1].contour(masks[region].sum(0),levels=[.5],colors=[color],linewidths=.6)
            axes[p,1].set_title('body / side / ends',fontsize=9)
            for arm,start in ((case['base'],2),(case['new'],4)):
                w=traces[arm][p]
                for offset,c in enumerate((case['truth'],conf)):
                    candidates=[i for i,r in enumerate(refs) if r['label']==c and reference_views[i][p] is not None]
                    chosen=max(candidates,key=lambda i:w[i]);rv=reference_views[chosen][p]
                    axes[p,start+offset].imshow(rv['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    axes[p,start+offset].set_title(f'{arm[:1]} {"same" if offset==0 else "conf"} {refs[chosen]["id"]}\nw={w[chosen]:.4f}',fontsize=8)
                    details.append(dict(arm=arm,pol=pol,class_id=c,reference_id=refs[chosen]['id'],weight=float(w[chosen])))
        fig.suptitle(f'{case["group"]}: {case["base"]} -> {case["new"]}; query {case["id"]}, truth {case["truth"]}\n'
            f'prediction {case["base_pred"]} -> {case["new_pred"]}; margin {case["base_margin"]:+.3f} -> {case["new_margin"]:+.3f}',fontsize=10)
        fig.subplots_adjust(left=.01,right=.99,bottom=.03,top=.81,hspace=.42,wspace=.06)
        name=f'{case["new"]}_vs_{case["base"]}_{case["group"]}_{case["id"]}.png'
        fig.savefig(directory/name,dpi=130);plt.close(fig)
        metadata.append(dict(**case,record=record,references=details,panel=name))
    exp.write(directory/'cases.json',metadata)
    print('GALLERY',len(metadata),flush=True)


if __name__=='__main__':
    exp.current.setup(exp.DEVICE)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('tables','diagnostics','gallery','all'))
    args=parser.parse_args()
    if args.action in ('tables','all'):tables()
    if args.action in ('diagnostics','all'):diagnostics()
    if args.action in ('gallery','all'):gallery()
