"""Cohort-level paired estimates and raw query/reference review."""
import html
import pandas as pd
from run import *
import review_task as gallery
from review_repeats import paired

SEEDS=(17,29,43)
NAMES=('evidence','reference','none','evidence_off','reference_off')


def main():
    rows,train,rid,groups=expanded_records()
    ids=np.concatenate(list(groups.values()));lookup={int(i):j for j,i in enumerate(ids)}
    cohorts=json.loads((HERE/'cohorts.json').read_text())
    for name,ix in cohorts['strict_groups'].items():
        if len(ix)!=len(groups[name]):groups[name+'_strict']=np.array(ix)
    groups['added_combined']=np.concatenate([groups['selection'],groups['evaluation']])
    groups['added_strict']=np.array(cohorts['strict_groups']['selection']+cohorts['strict_groups']['evaluation'])
    y=np.array([rows[i]['label'] for i in ids]);scores={}
    for seed in SEEDS:
        scores[seed]={}
        for name in NAMES:
            z=np.load(HERE/f'{name}_{seed}.npz');assert np.array_equal(ids,z['ids'])
            scores[seed][name]=z['score']
        scores[seed]['baseline']=np.load(HERE/'baseline.npz')['score'][ids]
    summary=[];pairs=[];classes=[];queries=[];subgroups=[]
    a={p.stem:np.load(p,mmap_mode='r') for p in (HERE/'cache').glob('*.npy')}
    for cohort,ix in groups.items():
        j=np.array([lookup[int(i)] for i in ix]);yy=y[j]
        for seed in SEEDS:
            s={k:z[j] for k,z in scores[seed].items()}
            for name,z in s.items():
                summary.append(dict(cohort=cohort,n=len(j),seed=seed,variant=name,**metrics(yy,z)))
                for c in range(4):
                    mask=yy==c
                    classes.append(dict(cohort=cohort,seed=seed,variant=name,label=c,n=int(mask.sum()),correct=int((z[mask].argmax(1)==c).sum())))
            for new,old in [('evidence','reference'),('evidence','none'),('reference','none'),
                            ('evidence','evidence_off'),('reference','reference_off'),('evidence_off','reference_off')]:
                pairs.append(dict(cohort=cohort,seed=seed,candidate=new,control=old,**paired(s[new],s[old],yy)))
            if cohort in ('selection','evaluation'):
                for pos,i in enumerate(ix):
                    r=rows[i];item=dict(cohort=cohort,seed=seed,id=r['id'],index=int(i),label=r['label'],mmsi=r['mmsi'],
                        orbit=orbit(r),vv=int(a['available'][i,0]),vh=int(a['available'][i,1]))
                    for name,z in s.items():
                        item[name+'_correct']=int(z[pos].argmax()==yy[pos])
                        item[name+'_margin']=float(gallery.margins(z[pos:pos+1],yy[pos:pos+1])[0])
                    queries.append(item)
                for key,values in [('orbit',np.array([orbit(rows[i]) for i in ix])),
                                  ('polarization',np.array([''.join(str(int(v)) for v in a['available'][i]) for i in ix]))]:
                    for value in np.unique(values):
                        mask=values==value
                        for name,z in s.items():subgroups.append(dict(cohort=cohort,seed=seed,group=key,value=value,n=int(mask.sum()),variant=name,**metrics(yy[mask],z[mask])))
    df=pd.DataFrame(summary);pf=pd.DataFrame(pairs);qf=pd.DataFrame(queries)
    for name,frame in [('metrics',df),('pairs',pf),('classes',pd.DataFrame(classes)),('queries',qf),('subgroups',pd.DataFrame(subgroups))]:frame.to_csv(HERE/f'{name}.csv',index=False)
    means=df.groupby(['cohort','variant'])[['oa','ba','nll']].agg(['mean','std']);means.to_csv(HERE/'mean_std.csv')
    pf.groupby(['cohort','candidate','control'])[['delta_oa','delta_ba','delta_nll']].agg(['mean','std']).to_csv(HERE/'paired_mean_std.csv')
    gallery.ROOT=HERE;cases=[]
    for cohort in ('selection','evaluation'):
        ix=groups[cohort];j=np.array([lookup[int(i)] for i in ix]);yy=y[j]
        good={name:np.stack([scores[seed][name][j].argmax(1)==yy for seed in SEEDS]) for name in NAMES}
        stack=np.stack(list(good.values()))
        masks=[('rescue',good['evidence']&~good['reference']),('harm',~good['evidence']&good['reference']),
               ('direct_rescue',good['evidence']&~good['evidence_off']),
               ('stable_correct',np.broadcast_to(stack.all((0,1)),(3,len(j)))),
               ('stable_wrong',np.broadcast_to(~stack.any((0,1)),(3,len(j))))]
        used=set()
        for name,events in masks:
            ranked=sorted(np.flatnonzero(events.any(0)),key=lambda k:(-events[:,k].sum(),int(ix[k])))
            k=next((int(k) for k in ranked if k not in used),None)
            if k is None:continue
            used.add(k);seed=SEEDS[int(np.flatnonzero(events[:,k])[0])]
            s={name:z[j] for name,z in scores[seed].items()}
            case=gallery.draw_case(rows,a,rid,int(ix[k]),cohort+'_'+name,s,yy,k,focus='evidence')
            case.update(cohort=cohort,seed=seed,eligible=len(ranked),across_seeds=qf[qf.id==rows[ix[k]]['id']].to_dict('records'))
            cases.append(case)
    write(HERE/'cases.json',cases)
    body=['<html><meta charset="utf-8"><title>Frozen M3 extension</title><style>body{font-family:Arial;margin:24px}img{max-width:100%}td,th{padding:4px;border:1px solid #aaa}table{border-collapse:collapse}</style>',
          '<h1>Frozen M3 evidence vs feature writeback</h1><p>OSS2, epoch4, seeds17/29/43, fixed32 source references. Historical expansion, NOT an untouched blind test. No training or checkpoint reselection.</p>',means.to_html(),pf.to_html(index=False)]
    for case in cases:
        body += ['<h2>'+html.escape(case['group']+' '+case['id'])+'</h2>',pd.DataFrame(case['across_seeds']).to_html(index=False),'<img src="'+case['image']+'">']
    (HERE/'index.html').write_text('\n'.join(body)+'</html>',encoding='utf-8')
    print(means.to_string());print(pf[pf.cohort.isin(['selection','evaluation','selection_strict'])].to_string(index=False))


if __name__=='__main__':main()
