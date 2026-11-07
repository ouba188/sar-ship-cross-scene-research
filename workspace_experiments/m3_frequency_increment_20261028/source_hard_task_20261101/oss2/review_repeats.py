"""Paired seeds, fixed-epoch attribution, and native query/reference examples."""
import pandas as pd
import review_task as gallery
from model_task import *

SEEDS=(17,29,43)
OUT=ROOT/'paired_repeats'


def load_scores(seed,arm,mode,ids):
    if seed==17 and mode.startswith('last'):
        suffix={'last_validation':'','last_off':'_off','last_aggregate':'_pool'}[mode]
        path=ROOT/'last_epoch'/(arm+suffix+'.npz')
    else:path=ROOT/(arm+'_'+str(seed))/(mode+'.npz')
    with np.load(path) as z:
        assert np.array_equal(z['ids'],ids)
        return z['score']


def paired(new,old,y):
    a,b=new.argmax(1)==y,old.argmax(1)==y
    ma,mb=metrics(y,new),metrics(y,old)
    d=gallery.margins(new,y)-gallery.margins(old,y)
    return dict(delta_oa=ma['oa']-mb['oa'],delta_ba=ma['ba']-mb['ba'],delta_nll=ma['nll']-mb['nll'],
        balanced_margin_delta=float(np.mean([d[y==c].mean() for c in range(4)])),
        rescues=int((a&~b).sum()),harms=int((~a&b).sum()),changes=int((new.argmax(1)!=old.argmax(1)).sum()))


def main():
    rows,train,hold,rid=records();a=arrays();y=np.array([rows[i]['label'] for i in hold])
    OUT.mkdir(exist_ok=True);gallery.ROOT=OUT
    baseline=np.load(ROOT/'baseline.npz')['score'][hold]
    scores={};summary=[];pairs=[];classes=[];per_query=[];episodes=[]
    for seed in SEEDS:
        es=[json.loads((ROOT/(arm+'_'+str(seed))/'episodes.json').read_text()) for arm in ('none','aggregate','competitive')]
        assert es[0]==es[1]==es[2] and len(es[0])==120
        episodes.append(es[0])
        for scope in ('last','selected'):
            s={'baseline':baseline}
            for arm in ('none','aggregate','competitive'):
                receipt=json.loads((ROOT/(arm+'_'+str(seed))/'receipt.json').read_text())
                assert receipt['seed']==seed and receipt['reload_error']==0
                s[arm]=load_scores(seed,arm,'last_validation' if scope=='last' else 'validation',hold)
                summary.append(dict(seed=seed,scope=scope,arm=arm,epoch=4 if scope=='last' else receipt['selected_epoch'],**metrics(y,s[arm])))
                if arm!='none':s[arm+'_off']=load_scores(seed,arm,'last_off' if scope=='last' else 'same_weights_off',hold)
                if arm=='competitive':s[arm+'_pool']=load_scores(seed,arm,'last_aggregate' if scope=='last' else 'same_weights_aggregate',hold)
            scores[seed,scope]=s
            comparisons=[('competitive','aggregate'),('competitive','none'),('aggregate','none'),
                ('competitive','competitive_pool'),('competitive_pool','competitive_off'),('competitive','competitive_off'),
                ('competitive_off','none'),('aggregate_off','none'),('aggregate','aggregate_off'),('competitive_off','aggregate_off')]
            for new,old in comparisons:
                pairs.append(dict(seed=seed,scope=scope,candidate=new,control=old,**paired(s[new],s[old],y)))
            for name,z in s.items():
                for cls in range(4):
                    mask=y==cls
                    classes.append(dict(seed=seed,scope=scope,arm=name,label=cls,n=int(mask.sum()),correct=int((z.argmax(1)[mask]==y[mask]).sum())))
            if scope=='last':
                m={k:gallery.margins(v.astype(np.float64),y) for k,v in s.items()}
                total=m['competitive']-m['none']
                pieces=[m['competitive_off']-m['none'],m['competitive_pool']-m['competitive_off'],m['competitive']-m['competitive_pool']]
                assert np.allclose(total,sum(pieces),atol=2e-6)
                for j,i in enumerate(hold):
                    row=dict(seed=seed,id=rows[i]['id'],mmsi=rows[i]['mmsi'],label=int(y[j]),pass_group=rows[i]['pass_group'],
                        head_delta=float(pieces[0][j]),pooled_write_delta=float(pieces[1][j]),fine_coupling_delta=float(pieces[2][j]))
                    for k,z in s.items():row[k+'_correct']=int(z[j].argmax()==y[j]);row[k+'_margin']=float(m[k][j])
                    per_query.append(row)
    assert episodes[0]!=episodes[1] and episodes[1]!=episodes[2]
    df=pd.DataFrame(summary);pf=pd.DataFrame(pairs)
    df.to_csv(OUT/'metrics.csv',index=False);pf.to_csv(OUT/'pairs.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_counts.csv',index=False)
    qf=pd.DataFrame(per_query);qf.to_csv(OUT/'query_attribution.csv',index=False)
    means=df.groupby(['scope','arm'])[['oa','ba','nll']].agg(['mean','std'])
    deltas=pf.groupby(['scope','candidate','control'])[['delta_oa','delta_ba','balanced_margin_delta']].agg(['mean','std'])
    means.to_csv(OUT/'mean_std.csv');deltas.to_csv(OUT/'paired_mean_std.csv')
    strat=[]
    for (seed,group),g in qf.groupby(['seed','pass_group']):
        strat.append(dict(seed=seed,pass_group=group,n=len(g),
            soft_correct=int(g.competitive_correct.sum()),aggregate_correct=int(g.aggregate_correct.sum()),
            none_correct=int(g.none_correct.sum()),off_correct=int(g.competitive_off_correct.sum())))
    pd.DataFrame(strat).to_csv(OUT/'pass_counts.csv',index=False)
    good={k:np.stack([scores[seed,'last'][k].argmax(1)==y for seed in SEEDS]) for k in ('competitive','aggregate','none','competitive_off','competitive_pool')}
    rescue=good['competitive']&~good['aggregate'];harm=~good['competitive']&good['aggregate']
    direct=good['competitive']&~good['competitive_off'];direct_harm=~good['competitive']&good['competitive_off']
    fine=good['competitive']&~good['competitive_pool']
    groups=[('repeat_rescue',rescue.sum(0)>=2,rescue),('once_rescue',rescue.sum(0)==1,rescue),
        ('repeat_harm',harm.sum(0)>=2,harm),('once_harm',harm.sum(0)==1,harm),
        ('direct_m3_rescue',direct.sum(0)>=2,direct),('direct_m3_harm',direct_harm.sum(0)>=2,direct_harm),
        ('fine_only_rescue',fine.any(0),fine),
        ('stable_correct',(good['competitive']&good['aggregate']&good['none']).all(0),good['competitive']),
        ('stable_wrong',(~good['competitive']&~good['aggregate']&~good['none']).all(0),~good['competitive'])]
    rank=np.mean([abs(gallery.margins(scores[seed,'last']['competitive'],y)-gallery.margins(scores[seed,'last']['aggregate'],y)) for seed in SEEDS],axis=0)
    used=set();cases=[];statuses=[]
    for name,mask,events in groups:
        pool=sorted(np.flatnonzero(mask),key=lambda i:-rank[i]);j=next((int(j) for j in pool if j not in used),None)
        statuses.append(dict(group=name,eligible=len(pool),selected=j))
        if j is None:continue
        used.add(j);si=int(np.flatnonzero(events[:,j])[0]);seed=SEEDS[si]
        case=gallery.draw_case(rows,a,rid,int(hold[j]),name+'_s'+str(seed),scores[seed,'last'],y,j)
        case.update(seed=seed,rescue_seeds=[SEEDS[s] for s in np.flatnonzero(rescue[:,j])],
            harm_seeds=[SEEDS[s] for s in np.flatnonzero(harm[:,j])],
            across_seeds=[r for r in per_query if r['id']==rows[hold[j]]['id']])
        cases.append(case)
    write(OUT/'cases.json',dict(cases=cases,groups=statuses,seeds=list(SEEDS),
        scope='Fixed epoch4, same historical validation228. seed17 pilot plus29/43 repeats. No independent test.'))
    body=['<html><meta charset="utf-8"><title>OSS2 M3 paired seeds</title><style>body{font-family:Arial;margin:24px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ccc}</style>',
        '<h1>OSS2 paired seeds 17 / 29 / 43</h1><p>Same historical development set. Fixed epoch4 is primary; selected checkpoints also reported. Reference pictures are raw-cosine examples, not highest attention claims.</p>',means.to_html(),
        pf[pf.scope=='last'].to_html(index=False),'<h2>Case coverage</h2>',pd.DataFrame(statuses).to_html(index=False)]
    for c in cases:body+=['<h2>'+c['group']+' '+c['id']+'</h2>',pd.DataFrame(c['across_seeds']).to_html(index=False),'<img src="'+c['image']+'">']
    (OUT/'index.html').write_text('\n'.join(body)+'</html>',encoding='utf-8')
    print(df.to_string(index=False));print(pf[pf.scope=='last'].to_string(index=False));print(means.to_string())


if __name__=='__main__':main()
