"""Fair tuned-vs-tuned and same-config comparisons, plus native chip review."""
import html
from tune import *
sys.path.insert(0,str(task.ROOT))
import review_task as gallery


def pair(candidate,control,z,b,y,**extra):
    cc,bb=z.argmax(1)==y,b.argmax(1)==y
    mc,mb=task.metrics(y,z),task.metrics(y,b)
    return dict(candidate=candidate,control=control,rescues=int((cc&~bb).sum()),harms=int((~cc&bb).sum()),
                changes=int((z.argmax(1)!=b.argmax(1)).sum()),delta_oa=mc['oa']-mb['oa'],delta_ba=mc['ba']-mb['ba'],
                delta_nll=mc['nll']-mb['nll'],**extra)


def main():
    rows,train,ids,rid=task.records()
    y=np.array([rows[i]['label'] for i in ids])
    entries={}
    for path in ROOT.glob('*/receipt.json'):
        r=json.loads(path.read_text())
        if r['seed']!=17:continue
        with np.load(path.parent/'selected.npz') as f:
            assert np.array_equal(f['ids'],ids)
            entries[path.parent.name]=(r,f['score'].copy())
    pairs=[]
    for name,(r,z) in entries.items():
        if r['design']!='class_gate':continue
        control=f'{r["position"]}_mean_gate_{"gain" if r["config"]=="sharp" else r["config"]}_17'
        if control in entries:
            pairs.append(pair(name,control,z,entries[control][1],y,comparison='matched_config_selected'))
            last=np.load(ROOT/name/f'epoch{r["epochs"]}.npz')['score']
            b=np.load(ROOT/control/f'epoch{r["epochs"]}.npz')['score']
            pairs.append(pair(name,control,last,b,y,comparison='matched_config_last'))
        for mode in ('off','uniform_context','spatial_mean'):
            p=ROOT/name/(mode+'.npz')
            if p.exists():pairs.append(pair(name,mode,z,np.load(p)['score'],y,comparison='same_weights'))
    best={}
    for design in ('class_gate','mean_gate'):
        pool=[(n,r,z) for n,(r,z) in entries.items() if r['design']==design]
        if not pool:continue
        n,r,z=max(pool,key=lambda t:rank(t[1]['selected']))
        best[design]=dict(run=n,position=r['position'],config=r['config'],epoch=r['selected']['epoch'],metrics=task.metrics(y,z))
    if len(best)<2:return
    name,control=best['class_gate']['run'],best['mean_gate']['run']
    z,b=entries[name][1],entries[control][1]
    pairs.append(pair(name,control,z,b,y,comparison='independently_tuned_best'))
    historical_mean=None
    if ROOT.name=='contrast_repair':
        previous=json.loads((ROOT.parent/'selection.json').read_text())['mean_gate']
        historical_mean=np.load(ROOT.parent/previous['run']/'selected.npz')['score']
        pairs.append(pair(name,'original_tuned_mean',z,historical_mean,y,comparison='strongest_prior_mean'))
    pd.DataFrame(pairs).to_csv(ROOT/'pairs.csv',index=False)
    task.write(ROOT/'selection.json',best)
    counts=[]
    for n,(r,s) in entries.items():
        for cls in range(4):
            m=y==cls
            counts.append(dict(run=n,label=cls,n=int(m.sum()),correct=int(((s.argmax(1)==y)&m).sum())))
    pd.DataFrame(counts).to_csv(ROOT/'class_counts.csv',index=False)
    orders=[json.loads((ROOT/n/'orders.json').read_text()) for n in entries]
    assert all(o==orders[0] for o in orders)
    base=np.load(OLD/'anchor.npz')['score']
    scores={'baseline':base,'tuned_class':z,'tuned_mean':b}
    if historical_mean is not None:
        scores['original_tuned_mean']=historical_mean
    for mode in ('off','uniform_context','spatial_mean','vv_only','vh_only'):
        scores[mode]=np.load(ROOT/name/(mode+'.npz'))['score']
    good,mean,old=z.argmax(1)==y,b.argmax(1)==y,base.argmax(1)==y
    groups=[('class_rescue_mean',good&~mean),('class_harm_mean',~good&mean),
            ('rescue_anchor',good&~old),('harm_anchor',~good&old),
            ('stable_correct',good&mean&old),('stable_wrong',~good&~mean&~old)]
    if historical_mean is not None:
        hm=historical_mean.argmax(1)==y
        groups=[('rescue_strong_mean',good&~hm),('harm_strong_mean',~good&hm)]+groups
    gap=abs(gallery.margins(z,y)-gallery.margins(b,y))
    seen,cases,status=set(),[],[]
    gallery.ROOT=ROOT
    a=task.arrays()
    for label,mask in groups:
        eligible=sorted(np.flatnonzero(mask),key=lambda i:-gap[i])
        pos=next((int(i) for i in eligible if int(i) not in seen),None)
        status.append(dict(group=label,n=len(eligible),selected=pos))
        if pos is None:continue
        seen.add(pos)
        case=gallery.draw_case(rows,a,rid,int(ids[pos]),label,scores,y,pos,focus='tuned_class')
        cases.append(case)
    task.write(ROOT/'cases.json',dict(selection=best,groups=status,cases=cases,orders_equal=True))
    events=[]
    for i,idx in enumerate(ids):
        events.append(dict(index=int(idx),id=rows[idx]['id'],mmsi=rows[idx]['mmsi'],label=int(y[i]),
                           class_pred=int(z[i].argmax()),mean_pred=int(b[i].argmax()),
                           class_margin=float(gallery.margins(z[i:i+1],y[i:i+1])[0]),
                           mean_margin=float(gallery.margins(b[i:i+1],y[i:i+1])[0])))
    pd.DataFrame(events).to_csv(ROOT/'events.csv',index=False)
    body=['<html><meta charset="utf-8"><title>M1 tuning and position</title>',
          '<style>body{font-family:Arial;margin:24px}img{max-width:100%}td,th{padding:5px;border:1px solid #bbb}table{border-collapse:collapse}</style>',
          '<h1>M1: jointly tune position and optimization</h1>',
          '<p>OSS2 source719 / historical development228 / fixed32 references. Development-selected results, NOT independent validation. M2/M3/head fixed; M3 message cached.</p>',
          '<pre>'+html.escape(json.dumps(best,indent=2))+'</pre>',
          pd.read_csv(ROOT/'summary.csv').to_html(index=False),pd.DataFrame(pairs).to_html(index=False)]
    for case in cases:
        body+=['<h2>'+html.escape(case['group']+' '+case['id'])+'</h2>','<img src="'+case['image']+'">',
               '<pre>'+html.escape(json.dumps(case['evidence'],indent=2))+'</pre>']
    (ROOT/'index.html').write_text('\n'.join(body+['</html>']),encoding='utf-8')
    print(json.dumps(best,indent=2),flush=True)
    print(pd.DataFrame(pairs)[lambda d:d.comparison!='same_weights'].to_string(index=False),flush=True)


if __name__=='__main__':main()
