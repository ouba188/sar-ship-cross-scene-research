"""Paired coverage results and native query/reference galleries; no fitting."""
import html
import importlib.util
import sys
from pathlib import Path

_spec=importlib.util.spec_from_file_location('coverage_experiment',Path(__file__).with_name('experiment.py'))
_experiment=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=_experiment
_spec.loader.exec_module(_experiment)
globals().update({k:v for k,v in vars(_experiment).items() if not k.startswith('_')})


def compare(z, b, y):
    good, old_good = z.argmax(1)==y, b.argmax(1)==y
    zz, bb = task.metrics(y,z), task.metrics(y,b)
    return dict(rescues=int((good & ~old_good).sum()),harms=int((~good & old_good).sum()),
                delta_oa=zz['oa']-bb['oa'],delta_ba=zz['ba']-bb['ba'],delta_nll=zz['nll']-bb['nll'])


def main():
    rows,groups=records(); a,c=arrays()
    yall=np.array([r['label'] for r in rows])
    initial=np.load(ROOT/'initial_source.npz')['score']
    source_summary=[]
    for cohort in ('small','selection','expanded'):
        for label in (None,0,1,2,3):
            ids=np.array([i for i in groups[cohort] if label is None or rows[i]['label']==label])
            yy=yall[ids]; zz=initial[ids]
            margin=zz[np.arange(len(ids)),yy]-np.where(np.arange(4)[None]==yy[:,None],-np.inf,zz).max(1)
            source_summary.append(dict(cohort=cohort,label='all' if label is None else label,n=len(ids),
                hard=int((margin<1).sum()),**task.metrics(yy,zz)))
    pd.DataFrame(source_summary).to_csv(ROOT/'source_initial_metrics.csv',index=False)
    entries={}; summary=[]; events=[]; classes=[]; passes=[]
    for path in ROOT.glob('*/receipt.json'):
        receipt=json.loads(path.read_text())
        cohort,arm,seed=receipt['cohort'],receipt['arm'],receipt['seed']
        for split in ('validation','evaluation'):
            with np.load(path.parent/(split+'.npz')) as file:
                assert np.array_equal(file['ids'],groups[split])
                score=file['score'].copy()
            entries[(cohort,arm,seed,split)]=score
            yy=yall[groups[split]]
            summary.append(dict(cohort=cohort,arm=arm,seed=seed,split=split,**task.metrics(yy,score),
                                seconds=receipt['seconds'],n_train=receipt['n_train'],steps=receipt['steps']))
            margin=score[np.arange(len(yy)),yy]-np.where(np.arange(4)[None]==yy[:,None],-np.inf,score).max(1)
            for i,idx in enumerate(groups[split]):
                events.append(dict(cohort=cohort,arm=arm,seed=seed,split=split,index=int(idx),id=rows[idx]['id'],
                    mmsi=rows[idx]['mmsi'],orbit=task.orbit(rows[idx]),label=int(yy[i]),prediction=int(score[i].argmax()),margin=float(margin[i])))
            for label in range(4):
                mask=yy==label
                classes.append(dict(cohort=cohort,arm=arm,seed=seed,split=split,label=label,n=int(mask.sum()),correct=int((score[mask].argmax(1)==yy[mask]).sum())))
            for orbit in sorted({task.orbit(rows[i]) for i in groups[split]}):
                mask=np.array([task.orbit(rows[i])==orbit for i in groups[split]])
                passes.append(dict(cohort=cohort,arm=arm,seed=seed,split=split,orbit=orbit,n=int(mask.sum()),**task.metrics(yy[mask],score[mask])))
    if not summary:
        return
    pd.DataFrame(summary).to_csv(ROOT/'summary.csv',index=False)
    pd.DataFrame(events).to_csv(ROOT/'events.csv',index=False)
    pd.DataFrame(classes).to_csv(ROOT/'classes.csv',index=False)
    pd.DataFrame(passes).to_csv(ROOT/'passes.csv',index=False)
    average=pd.DataFrame(summary).groupby(['split','cohort','arm'])[['oa','ba','nll']].agg(['mean','std'])
    average.to_csv(ROOT/'averages.csv')
    pairs=[]
    for (cohort,arm,seed,split),score in entries.items():
        yy=yall[groups[split]]
        if arm=='class':
            for control in ('mean','no_cov'):
                key=(cohort,control,seed,split)
                if key in entries:
                    pairs.append(dict(cohort=cohort,seed=seed,split=split,comparison='class_vs_'+control,**compare(score,entries[key],yy)))
        if cohort=='expanded' and ('small',arm,seed,split) in entries:
            pairs.append(dict(cohort='expanded_minus_small',seed=seed,split=split,comparison=arm,**compare(score,entries[('small',arm,seed,split)],yy)))
    pair_table=pd.DataFrame(pairs)
    pair_table.to_csv(ROOT/'pairs.csv',index=False)
    if not pairs:
        return
    pair_average=pair_table.groupby(['split','cohort','comparison'])[['delta_oa','delta_ba','delta_nll']].mean()
    pair_average.to_csv(ROOT/'pair_averages.csv')
    # Equal sample order is required within each data size/seed, not across sizes.
    for cohort in ('small','expanded'):
        for seed in (17,29,43):
            files=[ROOT/f'{cohort}_{arm}_s{seed}'/'orders.json' for arm in ('mean','class','no_cov')]
            if all(f.exists() for f in files):
                orders=[json.loads(f.read_text()) for f in files]
                assert orders[0]==orders[1]==orders[2]
    spec=importlib.util.spec_from_file_location('coverage_native_gallery',task.ROOT/'review_task.py')
    gallery=importlib.util.module_from_spec(spec);spec.loader.exec_module(gallery)
    gallery.ROOT=ROOT
    figure,axes=gallery.plt.subplots(2,2,figsize=(9,6),squeeze=False)
    frame=pd.DataFrame(summary)
    for col,split in enumerate(('validation','evaluation')):
        for row,metric in enumerate(('oa','ba')):
            ax=axes[row,col]
            for arm,color,marker in [('mean','#555555','s'),('class','#b12635','o'),('no_cov','#177a66','^')]:
                part=frame[(frame.split==split)&(frame.arm==arm)]
                stats=part.groupby('n_train')[metric].agg(['mean','std']).sort_index()
                ax.errorbar(stats.index,stats['mean'],yerr=stats['std'].fillna(0),
                            color=color,marker=marker,capsize=3,label=arm)
            ax.set_xticks([719,913]);ax.set_xlabel('Source observations')
            ax.set_ylabel(metric.upper()+' (%)');ax.grid(alpha=.2)
            if row==0:ax.set_title('Historical development228' if split=='validation' else 'Frozen evaluation378')
            if row==1 and col==1:ax.legend(fontsize=9)
    figure.suptitle('Fixed180 steps: mean +/- training-seed SD')
    figure.tight_layout();figure.savefig(ROOT/'coverage_results.png',dpi=160);gallery.plt.close(figure)
    cases=[];statuses=[]; seen=set()
    for split in ('evaluation','validation'):
        ids=groups[split];yy=yall[ids]
        for group,control,rescue in [('cov_rescue','no_cov',True),('cov_harm','no_cov',False),
                                     ('mean_rescue','mean',True),('mean_harm','mean',False),
                                     ('stable_correct','mean',None),('stable_wrong','mean',None)]:
            eligible=[]
            for seed in (17,29,43):
                key=('expanded','class',seed,split);other=('expanded',control,seed,split)
                if key not in entries or other not in entries:
                    continue
                z,b=entries[key],entries[other]
                good,old_good=z.argmax(1)==yy,b.argmax(1)==yy
                if rescue is None:
                    mask=(good & old_good) if group=='stable_correct' else (~good & ~old_good)
                else:
                    mask=(good & ~old_good) if rescue else (~good & old_good)
                gap=abs(gallery.margins(z,yy)-gallery.margins(b,yy))
                eligible += [(float(gap[i]),seed,int(i)) for i in np.flatnonzero(mask)]
            eligible.sort(reverse=True)
            picked=next((item for item in eligible if int(ids[item[2]]) not in seen),None)
            statuses.append(dict(split=split,group=group,events=len(eligible),selected=picked))
            if picked:
                _,seed,pos=picked;seen.add(int(ids[pos]))
                scores={arm:entries[('expanded',arm,seed,split)] for arm in ('class','mean','no_cov')}
                scores['baseline']=scores['mean']
                if ('small','class',seed,split) in entries:
                    scores['small_class']=entries[('small','class',seed,split)]
                label=f'{split}_{group}_s{seed}'
                case=gallery.draw_case(rows,a,groups['references'],int(ids[pos]),label,scores,yy,pos,focus='class')
                case.update(seed=seed,split=split,control=control)
                cases.append(case)
    # Include the smaller-source covariance-specific events, not just expanded runs.
    split='evaluation';ids=groups[split];yy=yall[ids]
    for rescue in (True,False):
        eligible=[]
        for seed in (17,29,43):
            z=entries[('small','class',seed,split)]
            b=entries[('small','no_cov',seed,split)]
            good,old_good=z.argmax(1)==yy,b.argmax(1)==yy
            mask=(good & ~old_good) if rescue else (~good & old_good)
            gap=abs(gallery.margins(z,yy)-gallery.margins(b,yy))
            eligible += [(float(gap[i]),seed,int(i)) for i in np.flatnonzero(mask)]
        if eligible:
            _,seed,pos=max(eligible)
            scores={arm:entries[('small',arm,seed,split)] for arm in ('class','mean','no_cov')}
            scores['baseline']=scores['mean']
            label=f'small_cov_{"rescue" if rescue else "harm"}_s{seed}'
            case=gallery.draw_case(rows,a,groups['references'],int(ids[pos]),label,scores,yy,pos,focus='class')
            case.update(seed=seed,split=split,cohort='small',control='no_cov')
            cases.append(case)
    write('cases.json',dict(cases=cases,groups=statuses,selection='Largest absolute paired margin difference within outcome group; not prevalence estimates.'))
    body=['<html><meta charset="utf-8"><title>M1 coverage and covariance replacement</title>',
          '<style>body{font-family:Arial;margin:24px}img{max-width:100%;height:auto}td,th{padding:5px;border:1px solid #bbb}table{border-collapse:collapse}</style>',
          '<h1>M1: 719 vs 913 source observations</h1>',
          '<p>Three seeds, fixed180 steps, native160 crop/pad, fixed32 source references. Historical selection194 reassigned to training. Evaluation378 is historically exposed, not a new blind test. No formal-model replacement.</p>',
          '<p><a href="FINDINGS_ZH.md">Findings</a> | <a href="PROTOCOL.md">Protocol</a></p>',
          '<img src="coverage_results.png">',
          average.to_html(),'<h2>Paired differences</h2>',pair_average.to_html(),pair_table.to_html(index=False)]
    for case in cases:
        body += ['<h2>'+html.escape(case['group']+' '+case['id'])+'</h2>',
                 '<img src="'+case['image']+'">','<pre>'+html.escape(json.dumps(case['evidence'],indent=2))+'</pre>']
    (ROOT/'index.html').write_text('\n'.join(body+['</html>']),encoding='utf-8')
    write('analysis_status.json',dict(fits=len(summary)//2,complete=len(summary)==36,
        native_panels=len(cases),orders_matched=True,formal_changed=False,git_synced=False))
    print(average.to_string(),flush=True)
    print(pair_average.to_string(),flush=True)


if __name__=='__main__':
    main()
