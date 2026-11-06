"""Paired outcomes, actual chip/reference panels and regional contribution traces."""
import html
import importlib.util
import study as s
from study import ROOT,prev,task,torch,np,pd


def compare(z,b,y):
    good,old=z.argmax(1)==y,b.argmax(1)==y
    a,c=task.metrics(y,z),task.metrics(y,b)
    return dict(rescues=int((good&~old).sum()),harms=int((~good&old).sum()),
                changes=int((z.argmax(1)!=b.argmax(1)).sum()),
                delta_oa=a['oa']-c['oa'],delta_ba=a['ba']-c['ba'],delta_nll=a['nll']-c['nll'])


def main():
    rows,groups=prev.records();y=np.array([r['label'] for r in rows])
    cache={p.stem:np.load(p,mmap_mode='r') for p in s.CACHE_ROOT.glob('*.npy')}
    entries={};summary=[];pairs=[];classes=[]
    for path in ROOT.glob('*/receipt.json'):
        r=json.loads(path.read_text());arm,seed=r['arm'],r['seed']
        for split in ('validation','evaluation'):
            for mode in ('','_off','_uniform','_shuffle') if arm=='class' else ('','_off'):
                f=np.load(path.parent/(split+mode+'.npz'));assert np.array_equal(f['ids'],groups[split])
                z=f['score'];entries[(arm+mode,seed,split)]=z
                summary.append(dict(arm=arm+mode,seed=seed,split=split,**task.metrics(y[groups[split]],z)))
                for cls in range(4):
                    mask=y[groups[split]]==cls
                    classes.append(dict(arm=arm+mode,seed=seed,split=split,label=cls,n=int(mask.sum()),correct=int((z[mask].argmax(1)==cls).sum())))
    assert len(list(ROOT.glob('*/receipt.json')))==9
    for seed in (17,29,43):
        orders=[json.loads((ROOT/f'{arm}_s{seed}/orders.json').read_text()) for arm in ('mean','content','class')]
        assert orders[0]==orders[1]==orders[2]
        for split in ('validation','evaluation'):
            yy=y[groups[split]]
            controls=[('class','mean'),('class','content'),('class','class_uniform'),('class','class_shuffle')]
            controls += [(arm,arm+'_off') for arm in ('mean','content','class')]
            for candidate,control in controls:
                pairs.append(dict(seed=seed,split=split,comparison=candidate+'_vs_'+control,**compare(entries[(candidate,seed,split)],entries[(control,seed,split)],yy)))
            for arm in ('mean','content','class'):
                pairs.append(dict(seed=seed,split=split,comparison=arm+'_vs_frozen',**compare(entries[(arm,seed,split)],cache['score'][groups[split]],yy)))
    frame=pd.DataFrame(summary);frame.to_csv(ROOT/'summary.csv',index=False)
    pair=pd.DataFrame(pairs);pair.to_csv(ROOT/'pairs.csv',index=False)
    pd.DataFrame(classes).to_csv(ROOT/'classes.csv',index=False)
    avg=frame.groupby(['split','arm'])[['oa','ba','nll']].agg(['mean','std']);avg.to_csv(ROOT/'averages.csv')
    pavg=pair.groupby(['split','comparison'])[['delta_oa','delta_ba','delta_nll','rescues','harms']].mean();pavg.to_csv(ROOT/'pair_averages.csv')
    spec=importlib.util.spec_from_file_location('guidance_gallery',task.ROOT/'review_task.py')
    gallery=importlib.util.module_from_spec(spec);spec.loader.exec_module(gallery);gallery.ROOT=ROOT
    a,_=prev.arrays();cases=[];used=set();statuses=[]
    split='evaluation';ids=groups[split];yy=y[ids]
    definitions=[('content_rescue','content',True),('content_harm','content',False),
                 ('mean_rescue','mean',True),('mean_harm','mean',False),
                 ('direct_rescue','class_off',True),('direct_harm','class_off',False),
                 ('stable_correct','content',None),('stable_wrong','content',None)]
    for name,control,rescue in definitions:
        choices=[]
        for seed in (17,29,43):
            z,b=entries[('class',seed,split)],entries[(control,seed,split)]
            good,old=z.argmax(1)==yy,b.argmax(1)==yy
            mask=(good&old if name=='stable_correct' else ~good&~old) if rescue is None else (good&~old if rescue else ~good&old)
            delta=abs(gallery.margins(z,yy)-gallery.margins(b,yy))
            choices += [(float(delta[i]),seed,int(i)) for i in np.flatnonzero(mask)]
        choices.sort(reverse=True);pick=next((v for v in choices if int(ids[v[2]]) not in used),None)
        statuses.append(dict(group=name,events=len(choices),selected=pick))
        if pick is None:continue
        _,seed,pos=pick;idx=int(ids[pos]);used.add(idx)
        scores={arm:entries[(arm,seed,split)] for arm in ('mean','content','class','class_off','class_uniform','class_shuffle')}
        scores['baseline']=scores['mean']
        case=gallery.draw_case(rows,a,groups['references'],idx,f'{name}_s{seed}',scores,yy,pos,focus='class')
        case.update(seed=seed,split=split,control=control)
        cases.append(case)
    # Exact linear head decomposition of the added regional path on selected chips.
    plt=gallery.plt;traces=[]
    for case in cases:
        idx=case['index'];seed=case['seed'];true=rows[idx]['label']
        net=s.RegionalGuidance('class').to(task.DEVICE)
        net.load_state_dict(torch.load(ROOT/f'class_s{seed}/final.pt',map_location=task.DEVICE,weights_only=True));net.eval()
        b={k:torch.as_tensor(np.array(v[idx:idx+1],copy=True),device=task.DEVICE) for k,v in cache.items() if k!='score'}
        with torch.no_grad():
            z,detail=net(b,details=True);off=net(b,'off')
        wrong=int(z[0].clone().index_fill(0,torch.tensor([true],device=task.DEVICE),-torch.inf).argmax())
        views=task.data.prepare(rows[idx]);fig,axes=plt.subplots(2,4,figsize=(12,6))
        exact=[];contributions=[];weights_info=[]
        for p,view in enumerate(views):
            W=net.head.weight[:,p*3584+2048:(p+1)*3584].reshape(4,3,512)
            contribution=torch.einsum('krc,rc->rk',W,detail[p]['delta'][0]).detach().cpu().numpy()
            contributions.append(contribution);exact.append(contribution.sum(0))
            weight=detail[p]['weights'][0].detach().cpu().numpy()
            uniform=detail[p]['uniform'][0].detach().cpu().numpy()
            info=[]
            for r in range(3):
                w=weight[r];u=uniform[r]
                info.append(dict(region=r,mass=float(b['mask'][0,p,r].sum()),
                    l1_to_uniform=np.abs(w-u[None]).sum(-1).tolist(),
                    effective_cells=(1/np.maximum((w*w).sum(-1),1e-12)).tolist(),
                    true_vs_competitor_increment=float(contribution[r,true]-contribution[r,wrong])))
            weights_info.append(info)
            for col in range(4):
                ax=axes[p,col];ax.axis('off')
                if view is None:ax.text(.5,.5,'Unavailable',ha='center');continue
                ax.imshow(view['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if col==0:ax.set_title(('VV' if p==0 else 'VH')+' native input')
                else:
                    r=col-1;u=uniform[r];w=weight[r,true];ix=np.where(u>0)[0]
                    ratio=w[ix]/np.maximum(u[ix],1e-8)
                    points=ax.scatter((ix%20)*8,(ix//20)*8,c=ratio,cmap='coolwarm',vmin=0,vmax=2,s=16,alpha=.65)
                    ax.set_title(('body','axis ends','sides')[r]+f' / true-class weight\ncontrast delta {info[r]["true_vs_competitor_increment"]:+.4f}',fontsize=9)
            net_score=(z-off)[0].detach().cpu().numpy()
        err=float(abs(np.sum(exact,0)-net_score).max());assert err<1e-4,err
        fig.suptitle(case['id']+f' seed{seed}: attention / area-uniform weight; all classes jointly produce updates',fontsize=10)
        fig.tight_layout();name=case['group']+'_attention.png';fig.savefig(ROOT/name,dpi=140);plt.close(fig)
        case['attention_image']=name
        traces.append(dict(id=case['id'],seed=seed,true=true,competitor=wrong,regions=weights_info,
            class_score_increment=np.sum(exact,0).tolist(),linear_reconstruction_error=err,
            note='Attention shown for true class is posthoc illustration; forward uses all four candidates. Region score increments sum exactly; not physical causality.'))
    s.write('cases.json',dict(cases=cases,groups=statuses,selection='Largest absolute paired margin change per outcome group, unique queries where possible; not prevalence.'))
    s.write('region_traces.json',traces)
    body=['<html><meta charset="utf-8"><title>M1 guided M2</title><style>body{font-family:Arial;margin:24px}img{max-width:100%}td,th{padding:5px;border:1px solid #ccc}table{border-collapse:collapse}</style>',
          '<h1>M1 responses guiding M2 regional pooling</h1><p>Frozen upstream seed17; 3 readout-training seeds; source719/dev228/evaluation378. Historical evaluation, not blind. No formal replacement.</p>',
          '<p><a href="FINDINGS_ZH.md">Findings</a> | <a href="PROTOCOL.md">Protocol</a></p>',avg.to_html(),pavg.to_html()]
    for c in cases:
        body += ['<h2>'+html.escape(c['group']+' '+c['id'])+'</h2>',
            '<img src="'+c['image']+'"><img src="'+c['attention_image']+'">',
            '<pre>'+html.escape(json.dumps(c['evidence'],indent=2))+'</pre>']
    (ROOT/'index.html').write_text('\n'.join(body+['</html>']),encoding='utf-8')
    s.write('analysis_status.json',dict(fits=9,paired_orders=True,native_panels=len(cases),formal_changed=False,git_synced=False))
    print(avg.to_string(),flush=True);print(pavg.to_string(),flush=True)


import json
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-head',action='store_true');args=parser.parse_args()
    if args.freeze_head:ROOT=s.ROOT=s.ROOT/'frozen_head'
    task.current.setup(task.DEVICE)
    main()
