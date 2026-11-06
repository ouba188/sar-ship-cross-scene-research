"""Source-only examples or completed four-arm development analysis."""
import argparse
import html
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from model_task import *


def margins(score,y):
    return score[np.arange(len(y)),y]-np.where(np.arange(score.shape[1])[None]==y[:,None],-np.inf,score).max(1)


def draw_case(rows,a,rid,idx,group,scores,y,position,focus='competitive'):
    row=rows[idx];true=row['label']
    chosen_score=scores.get(focus,scores['baseline'])[position]
    wrong=int(np.where(np.arange(4)==true,-np.inf,chosen_score).argmax())
    def desc(i):
        w=np.asarray(a['mass'][i]);d=np.asarray(a['desc'][i]);z=(w[...,None]*d).sum(1)/np.maximum(w.sum(1)[...,None],1e-20)
        return z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-8)
    q=desc(idx);sim={}
    for i in rid:
        if rows[i]['mmsi']==row['mmsi'] or orbit(rows[i])==orbit(row):continue
        valid=np.asarray(a['available'][idx])*np.asarray(a['available'][i])
        if valid.sum():sim[int(i)]=float(((q*desc(i)).sum(-1)*valid).sum()/valid.sum())
    same=sorted([i for i in sim if rows[i]['label']==true],key=lambda i:-sim[i])[:2]
    other=sorted([i for i in sim if rows[i]['label']==wrong],key=lambda i:-sim[i])[:1]
    ix=[idx]+same+other;roles=['Query']+[f'Same class ref {k+1}' for k in range(len(same))]+['Confusing class ref']*len(other)
    fig,axs=plt.subplots(2,len(ix),figsize=(3*len(ix),6),squeeze=False);records_out=[]
    for j,(i,role) in enumerate(zip(ix,roles)):
        r=rows[i];views=data.prepare(r);records_out.append(dict(role=role,record=r))
        for p,(pol,view) in enumerate(zip(('VV','VH'),views)):
            ax=axs[p,j];ax.axis('off');ax.set_title(f'{role}: {r["id"].removeprefix("pub2_opensarship2_")} / {pol}\n{CLASSES[r["label"]]}',fontsize=8)
            if view is None:ax.text(.5,.5,'Missing',ha='center');continue
            ax.imshow(view['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            if view['geometry'] is not None:
                cx,cy,l,w,t=view['geometry'][:5];box=np.array([[-l/2,-w/2],[l/2,-w/2],[l/2,w/2],[-l/2,w/2]])
                rot=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
                ax.add_patch(Polygon(box@rot.T+[cx,cy],fill=False,color='lime',linewidth=.6))
    evidence={name:dict(prediction=int(z[position].argmax()),margin=float(margins(z[position:position+1],np.array([true]))[0])) for name,z in scores.items()}
    text=' / '.join(f'{name}: {v["margin"]:+.3f}' for name,v in evidence.items() if name in ('baseline','none','original','aggregate','competitive',focus))
    fig.suptitle(group+' / '+row['id']+'\n'+text,fontsize=9);fig.tight_layout()
    filename=group+'_'+str(idx)+'.png';fig.savefig(ROOT/filename,dpi=145);plt.close(fig)
    return dict(group=group,id=row['id'],index=idx,label=true,image=filename,records=records_out,evidence=evidence,
        reference_selection='Nearest two same-class and nearest confusing-class legal fixed-bank cases under raw pooled cosine; NOT actual attention maximum',
        preprocessing='Native160 crop/pad, VV/VH separate; no image resizing')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',action='store_true');args=parser.parse_args()
    rows,train,hold,rid=records();a=arrays();baseline=np.load(ROOT/'baseline.npz')['score']
    cases=[]
    if args.source:
        ids=train;y=np.array([rows[i]['label'] for i in ids]);z=baseline[ids];m=margins(z,y);counts=[]
        for cls in range(4):
            mask=y==cls;counts.append(dict(label=cls,n=int(mask.sum()),wrong=int((mask&(m<0)).sum()),hard=int((mask&(m<1)).sum())))
            pool=np.flatnonzero(mask&(m<0))
            if len(pool):
                n=int(pool[np.argmin(abs(m[pool]))]);cases.append(draw_case(rows,a,rid,int(ids[n]),'source_error',{'baseline':z},y,n))
        pd.DataFrame(counts).to_csv(ROOT/'source_class_difficulty.csv',index=False)
        write(ROOT/'source_cases.json',cases);print(pd.DataFrame(counts).to_string(index=False));return
    ids=hold;y=np.array([rows[i]['label'] for i in ids]);scores={'baseline':baseline[ids]};summary=[];episodes=[]
    for arm in ('none','original','aggregate','competitive'):
        dest=ROOT/(arm+'_17');receipt=json.loads((dest/'receipt.json').read_text());episodes.append(json.loads((dest/'episodes.json').read_text()))
        with np.load(dest/'validation.npz') as z:
            assert np.array_equal(z['ids'],ids);scores[arm]=z['score']
        for name in ('initial_validation','same_weights_off','same_weights_aggregate'):
            if (dest/(name+'.npz')).exists():scores[arm+'_'+name]=np.load(dest/(name+'.npz'))['score']
        summary.append(dict(arm=arm,selected_epoch=receipt['selected_epoch'],**metrics(y,scores[arm])))
    assert all(e==episodes[0] for e in episodes)
    summary.insert(0,dict(arm='baseline',selected_epoch=-1,**metrics(y,scores['baseline'])))
    pd.DataFrame(summary).to_csv(ROOT/'summary.csv',index=False)
    comparisons=[(arm,'baseline') for arm in ('none','original','aggregate','competitive')]
    comparisons += [('original','none'),('aggregate','none'),('competitive','none'),('competitive','aggregate'),('competitive','competitive_same_weights_aggregate')]
    comparisons += [(arm,arm+'_same_weights_off') for arm in ('original','aggregate','competitive')]
    pairs=[]
    for n,k in comparisons:
        aa=scores[n].argmax(1);bb=scores[k].argmax(1);ma=metrics(y,scores[n]);mb=metrics(y,scores[k])
        pairs.append(dict(candidate=n,control=k,rescues=int(((aa==y)&(bb!=y)).sum()),harms=int(((aa!=y)&(bb==y)).sum()),
            changes=int((aa!=bb).sum()),delta_oa=ma['oa']-mb['oa'],delta_ba=ma['ba']-mb['ba'],
            max_margin_difference=float(abs(margins(scores[n],y)-margins(scores[k],y)).max())))
    pd.DataFrame(pairs).to_csv(ROOT/'pairs.csv',index=False)
    classes=[]
    for name,z in scores.items():
        for cls in range(4):
            mask=y==cls;classes.append(dict(arm=name,label=cls,n=int(mask.sum()),correct=int(((z.argmax(1)==y)&mask).sum())))
    pd.DataFrame(classes).to_csv(ROOT/'class_counts.csv',index=False)
    sp,ag,base=[scores[k].argmax(1)==y for k in ('competitive','aggregate','baseline')]
    groups=[('matching_rescue',sp&~ag),('matching_harm',~sp&ag),('base_rescue',sp&~base),('base_harm',~sp&base),('stable_correct',sp&ag&base),('stable_wrong',~sp&~ag&~base)]
    used=set();statuses=[];rank=abs(margins(scores['competitive'],y)-margins(scores['aggregate'],y))
    for name,mask in groups:
        pool=sorted(np.flatnonzero(mask),key=lambda i:-rank[i]);n=next((int(i) for i in pool if i not in used),None)
        statuses.append(dict(group=name,eligible=len(pool),selected=n))
        if n is not None:used.add(n);cases.append(draw_case(rows,a,rid,int(ids[n]),name,scores,y,n))
    write(ROOT/'cases.json',dict(cases=cases,groups=statuses,episodes_equal=True,episode_count=len(episodes[0]),scope='Exploratory validation228 selected by BA/OA; four-class OSS2 spatial-only; evaluation378 not read'))
    body=['<html><meta charset="utf-8"><title>OSS2 M3 source-hard experiment</title><style>body{font-family:Arial;margin:24px}img{max-width:100%;height:auto}td,th{padding:6px;border:1px solid #ccc}table{border-collapse:collapse}</style>',
        '<h1>OSS2 M3 source-hard experiment</h1><p>Four-class, spatial-only. Historical validation228; one seed; not independent confirmation. Evaluation378 not read.</p>',pd.DataFrame(summary).to_html(index=False),pd.DataFrame(pairs).to_html(index=False)]
    for case in cases:body.append('<h2>'+html.escape(case['group']+' '+case['id'])+'</h2><img src="'+case['image']+'">')
    body.append('</html>');(ROOT/'index.html').write_text('\n'.join(body),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False))


if __name__=='__main__':main()
