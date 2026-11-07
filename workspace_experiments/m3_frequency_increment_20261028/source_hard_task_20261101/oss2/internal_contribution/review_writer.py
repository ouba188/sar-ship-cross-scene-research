"""Matched feature-writeback comparison; keep all seeds and checkpoint rules."""
import html
import pandas as pd
from ablate import HERE
from model_task import *
from review_repeats import load_scores,paired
import review_task as gallery

OUT=HERE/'writer_review'
SEEDS=(17,29,43)


def main():
    OUT.mkdir(exist_ok=True);gallery.ROOT=OUT
    rows,train,hold,rid=records();a=arrays();y=np.array([rows[i]['label'] for i in hold])
    scores={};summary=[];pairs=[];queries=[];class_rows=[];checks=[]
    for seed in SEEDS:
        full_dir=ROOT/f'aggregate_{seed}';ref_dir=HERE/'reference_message'/f'aggregate_{seed}'
        fr=json.loads((full_dir/'receipt.json').read_text());rr=json.loads((ref_dir/'receipt.json').read_text())
        assert json.loads((full_dir/'episodes.json').read_text())==json.loads((ref_dir/'episodes.json').read_text())
        assert rr['steps']==120 and rr['reload_error']==0
        initial_error=float(np.abs(np.load(full_dir/'initial_validation.npz')['score']-np.load(ref_dir/'initial_validation.npz')['score']).max())
        assert initial_error==0
        checks.append(dict(seed=seed,initial_error=initial_error,common_episodes=120,
                           full_params=sum(fr['trainable'].values()),reference_params=sum(rr['trainable'].values())))
        for scope in ('last','selected'):
            mode='last_validation' if scope=='last' else 'validation'
            off='last_off' if scope=='last' else 'same_weights_off'
            s={'full':load_scores(seed,'aggregate',mode,hold),'full_off':load_scores(seed,'aggregate',off,hold),
               'none':load_scores(seed,'none',mode,hold),'baseline':np.load(ROOT/'baseline.npz')['score'][hold]}
            for key,name in [('reference',mode),('reference_off',off)]:
                z=np.load(ref_dir/(name+'.npz'));assert np.array_equal(z['ids'],hold);s[key]=z['score']
            for key,z in s.items():summary.append(dict(seed=seed,scope=scope,variant=key,**metrics(y,z)))
            for new,old in [('full','reference'),('reference','none'),('full','none'),('reference','reference_off'),('full','full_off'),('full_off','reference_off')]:
                pairs.append(dict(seed=seed,scope=scope,candidate=new,control=old,**paired(s[new],s[old],y)))
            if scope=='last':
                scores[seed]=s
                for j,i in enumerate(hold):
                    row=dict(seed=seed,id=rows[i]['id'],label=int(y[j]),pass_group=rows[i]['pass_group'])
                    for name,z in s.items():
                        row[name+'_correct']=int(z[j].argmax()==y[j])
                        row[name+'_margin']=float(gallery.margins(z[j:j+1],y[j:j+1])[0])
                    queries.append(row)
                for name,z in s.items():
                    for c in range(4):
                        mask=y==c
                        class_rows.append(dict(seed=seed,variant=name,label=c,n=int(mask.sum()),correct=int((z.argmax(1)[mask]==y[mask]).sum())))
    df=pd.DataFrame(summary);pf=pd.DataFrame(pairs)
    df.to_csv(OUT/'metrics.csv',index=False);pf.to_csv(OUT/'pairs.csv',index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'class_counts.csv',index=False)
    pd.DataFrame(queries).to_csv(OUT/'queries.csv',index=False);write(OUT/'checks.json',checks)
    means=df.groupby(['scope','variant'])[['oa','ba','nll']].agg(['mean','std']);means.to_csv(OUT/'mean_std.csv')
    good={k:np.stack([scores[s][k].argmax(1)==y for s in SEEDS]) for k in ('full','reference','none','full_off','reference_off')}
    groups=[]
    for name,new,old in [('evidence_vs_feature','full','reference'),('feature_direct','reference','reference_off')]:
        groups += [(name+'_rescue',good[new]&~good[old]),(name+'_harm',~good[new]&good[old])]
    stack=np.stack(list(good.values()))
    groups += [('stable_correct',np.broadcast_to(stack.all((0,1)),(len(SEEDS),len(y)))),
               ('stable_wrong',np.broadcast_to(~stack.any((0,1)),(len(SEEDS),len(y))))]
    used=set();cases=[];statuses=[]
    for name,events in groups:
        pool=sorted(np.flatnonzero(events.any(0)),key=lambda j:-events[:,j].sum())
        j=next((int(j) for j in pool if j not in used),None)
        statuses.append(dict(group=name,eligible=len(pool),selected=j))
        if j is None:continue
        used.add(j);seed=SEEDS[int(np.flatnonzero(events[:,j])[0])]
        c=gallery.draw_case(rows,a,rid,int(hold[j]),name+'_s'+str(seed),scores[seed],y,j,focus='full')
        c['seed']=seed;c['across_seeds']=[r for r in queries if r['id']==c['id']];cases.append(c)
    write(OUT/'cases.json',dict(cases=cases,groups=statuses))
    body=['<html><meta charset="utf-8"><title>Evidence vs feature writeback</title><style>body{font-family:Arial;margin:24px}img{max-width:100%}td,th{padding:5px;border:1px solid #aaa}table{border-collapse:collapse}</style>',
          '<h1>Evidence vs reference-feature writeback</h1><p>OSS2 four-class historical development228; three paired seeds. Source-only initial message energy matching. Feature control has 1024 extra trainable parameters.</p>',means.to_html(),pf.to_html(index=False)]
    for c in cases:body += ['<h2>'+html.escape(c['group']+' '+c['id'])+'</h2>',pd.DataFrame(c['across_seeds']).to_html(index=False),'<img src="'+c['image']+'">']
    (OUT/'index.html').write_text('\n'.join(body)+'</html>',encoding='utf-8')
    print(means.to_string());print(pf[pf.scope=='last'].to_string(index=False));print(checks)


if __name__=='__main__':main()
