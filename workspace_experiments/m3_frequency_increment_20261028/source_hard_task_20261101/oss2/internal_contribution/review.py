"""Fixed-epoch internal ablations, same-weight mechanism tests and native cases."""
import html
import pandas as pd
from ablate import HERE, AblationModel
from model_task import *
import review_task as gallery
from review_repeats import load_scores, paired

SEEDS = (17,29,43)
VARIANTS = ('shared_metric','no_competition')


@torch.no_grad()
def trace_effect(model,a,rows,ids,rid,seed,cohort):
    inputs=[tensor(a[k],ix) for k,ix in [('desc',ids),('desc',rid),('mass',ids),('mass',rid),('write',ids)]]
    pair=legal(rows,ids,rid)[...,None]&tensor(a['available'],ids)[:,None].bool()&tensor(a['available'],rid)[None].bool()
    labels=torch.tensor([rows[i]['label'] for i in rid],device=DEVICE)
    t={}
    for v in ('full',*VARIANTS):
        model.matcher.variant=v
        delta,_,_,traces=model.matcher(*inputs,pair,labels,arm='aggregate',details=True)
        t[v]=(delta,traces)
    result=[]
    for v in VARIANTS:
        record=dict(seed=seed,cohort=cohort,variant=v,ids=ids.tolist(),
                    delta_relative=float((t[v][0]-t['full'][0]).norm()/t['full'][0].norm().clamp_min(1e-20)))
        for field in ('metric','original_transport','prototype','message'):
            f=torch.cat([r[field].flatten() for p in t['full'][1] for r in p['regions']])
            z=torch.cat([r[field].flatten() for p in t[v][1] for r in p['regions']])
            record[field+'_relative']=float((z-f).norm()/f.norm().clamp_min(1e-20))
        spans=[]
        for p in t['full'][1]:
            for r in p['regions']:
                mask=r['available'].sum(-1)>1
                if mask.any():spans.append(r['metric'][mask].std(1).mean())
        record['full_class_metric_std']=float(torch.stack(spans).mean())
        result.append(record)
    return result


def main():
    current.setup(DEVICE)
    rows,train,hold,rid=records();a=arrays();y=np.array([rows[i]['label'] for i in hold])
    summary=[];pairs=[];queries=[];all_scores={};classes=[];mechanisms=[]
    for seed in SEEDS:
        old=ROOT/('aggregate_'+str(seed))
        episodes=json.loads((old/'episodes.json').read_text())
        s={'baseline':np.load(ROOT/'baseline.npz')['score'][hold],
           'full':load_scores(seed,'aggregate','last_validation',hold),
           'full_off':load_scores(seed,'aggregate','last_off',hold),
           'none':load_scores(seed,'none','last_validation',hold)}
        for variant in VARIANTS:
            d=HERE/variant/('aggregate_'+str(seed))
            receipt=json.loads((d/'receipt.json').read_text())
            assert receipt['steps']==120 and receipt['reload_error']==0
            assert json.loads((d/'episodes.json').read_text())==episodes
            for suffix,filename in (('','last_validation'),('_off','last_off')):
                z=np.load(d/(filename+'.npz'));assert np.array_equal(z['ids'],hold)
                s[variant+suffix]=z['score']
            z=np.load(d/'validation.npz')
            summary.append(dict(seed=seed,scope='selected',variant=variant,epoch=receipt['selected_epoch'],**metrics(y,z['score'])))
        rec=json.loads((old/'receipt.json').read_text())
        summary.append(dict(seed=seed,scope='selected',variant='full',epoch=rec['selected_epoch'],
                            **metrics(y,load_scores(seed,'aggregate','validation',hold))))
        model=AblationModel('aggregate',seed,'full')
        model.restore(torch.load(old/'last.pt',map_location='cpu',weights_only=True)['trainable_state'])
        for variant in VARIANTS:
            model.matcher.variant=variant
            p=predict(model,a,rows,hold,rid)
            np.savez_compressed(HERE/f'same_weights_{variant}_{seed}.npz',**p)
            s['switch_'+variant]=p['score']
        for cohort,pool in [('source',train),('development',hold)]:
            ids=np.concatenate([np.array([i for i in pool if rows[i]['label']==c][:3]) for c in range(4)])
            mechanisms.extend(trace_effect(model,a,rows,ids,rid,seed,cohort))
        del model
        for name,z in s.items():
            summary.append(dict(seed=seed,scope='last',variant=name,epoch=4,**metrics(y,z)))
            for cls in range(4):
                mask=y==cls
                classes.append(dict(seed=seed,variant=name,label=cls,n=int(mask.sum()),correct=int((z.argmax(1)[mask]==y[mask]).sum())))
        comparisons=[('full','none'),('full','full_off')]
        for v in VARIANTS:comparisons += [('full',v),('full','switch_'+v),(v,v+'_off'),(v,'none'),('full_off',v+'_off')]
        for new,old_name in comparisons:pairs.append(dict(seed=seed,candidate=new,control=old_name,**paired(s[new],s[old_name],y)))
        for j,i in enumerate(hold):
            row=dict(seed=seed,id=rows[i]['id'],label=int(y[j]),pass_group=rows[i]['pass_group'])
            for k,z in s.items():
                row[k+'_correct']=int(z[j].argmax()==y[j])
                row[k+'_margin']=float(gallery.margins(z[j:j+1],y[j:j+1])[0])
            queries.append(row)
        all_scores[seed]=s
    df=pd.DataFrame(summary);pf=pd.DataFrame(pairs);qf=pd.DataFrame(queries)
    df.to_csv(HERE/'metrics.csv',index=False);pf.to_csv(HERE/'pairs.csv',index=False)
    write(HERE/'mechanism_traces.json',mechanisms)
    pd.DataFrame(classes).to_csv(HERE/'class_counts.csv',index=False);qf.to_csv(HERE/'queries.csv',index=False)
    means=df.groupby(['scope','variant'])[['oa','ba','nll']].agg(['mean','std'])
    means.to_csv(HERE/'mean_std.csv')
    pf.groupby(['candidate','control'])[['delta_oa','delta_ba','delta_nll']].agg(['mean','std']).to_csv(HERE/'paired_mean_std.csv')
    qf.groupby(['seed','pass_group'])[[k for k in qf if k.endswith('_correct')]].sum().to_csv(HERE/'pass_counts.csv')
    gallery.ROOT=HERE;cases=[];used=set();statuses=[]
    good={k:np.stack([all_scores[s][k].argmax(1)==y for s in SEEDS]) for k in ('full',*VARIANTS,'full_off','none')}
    for v in (*VARIANTS,'full_off'):
        for role,events in [('rescue',good['full']&~good[v]),('harm',~good['full']&good[v])]:
            pool=sorted(np.flatnonzero(events.any(0)),key=lambda j:-events[:,j].sum())
            j=next((int(j) for j in pool if j not in used),None)
            statuses.append(dict(group=v+'_'+role,eligible=len(pool),selected=j))
            if j is None:continue
            used.add(j);seed=SEEDS[int(np.flatnonzero(events[:,j])[0])]
            case=gallery.draw_case(rows,a,rid,int(hold[j]),v+'_'+role+'_s'+str(seed),all_scores[seed],y,j,focus='full')
            case['seed']=seed;case['across_seeds']=[r for r in queries if r['id']==rows[hold[j]]['id']];cases.append(case)
    for role,mask in [('stable_correct',np.stack(list(good.values())).all((0,1))),('stable_wrong',~np.stack(list(good.values())).any((0,1)))]:
        pool=np.flatnonzero(mask);j=next((int(j) for j in pool if j not in used),None)
        statuses.append(dict(group=role,eligible=len(pool),selected=j))
        if j is not None:
            used.add(j);case=gallery.draw_case(rows,a,rid,int(hold[j]),role,all_scores[17],y,j,focus='full')
            case['seed']=17;case['across_seeds']=[r for r in queries if r['id']==rows[hold[j]]['id']];cases.append(case)
    write(HERE/'cases.json',dict(cases=cases,groups=statuses))
    body=['<html><meta charset="utf-8"><title>M3 internal contribution</title><style>body{font-family:Arial;margin:24px}img{max-width:100%}td,th{padding:5px;border:1px solid #aaa}table{border-collapse:collapse}</style>',
          '<h1>M3 internal contribution</h1><p>OSS2 four-class, same historical development228. Fixed epoch4. Paired training and same-weight switches are distinct tests.</p>',means.to_html(),pf.to_html(index=False),pd.DataFrame(statuses).to_html(index=False)]
    for c in cases:body += ['<h2>'+html.escape(c['group']+' '+c['id'])+'</h2>',pd.DataFrame(c['across_seeds']).to_html(index=False),'<img src="'+c['image']+'">']
    (HERE/'index.html').write_text('\n'.join(body)+'</html>',encoding='utf-8')
    print(means.to_string());print(pf.to_string(index=False))


if __name__=='__main__':main()
