"""Inspect selected body changes and unselected mixed evidence on native chips."""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run as edge
from extract_tokens import geo, slots

HERE=Path(__file__).resolve().parent
OUT=HERE/'analysis'


def main():
    OUT.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table,x,mass,bank=edge.support.load()
    features=np.load(HERE/'features.npz')
    cache=np.load(HERE/'localcache.npz')
    pred=np.load(HERE/'predictions_17.npz')
    weights=np.load(HERE/'weights.npz')
    tokens=np.load(HERE.parent/'tokens.npz')
    y=table.class_id.to_numpy(int)
    ev=np.flatnonzero(table.role.eq('evaluation'))
    old=pred['anchor'][ev].argmax(-1)==y[ev]
    new=pred['body'][ev].argmax(-1)==y[ev]
    cases=[]
    for name,mask in [('body_rescued',~old&new),('body_harmed',old&~new),('CC',old&new),('WW',~old&~new)]:
        ids=ev[mask]
        if not len(ids): continue
        i=int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy())]); c=int(y[i])
        r=int(np.argmax(np.where(np.arange(8)==c,-np.inf,pred['body'][i])))
        parts=weights['body_17']*(features['body'][i,c]-features['body'][i,r])
        group=int(np.argmax(np.abs(parts)))
        trace=edge.trace_query(i,'mixed',x,mass,bank)
        refs={}
        for cl in (c,r):
            resp=trace[f'{edge.GROUPS[group]}_responsibility'][cl].sum((0,2))
            refs[cl]=bank[cl,np.argsort(-resp,kind='stable')[:2]].tolist()
        cases.append(dict(outcome=name,n=len(ids),i=i,obs_id=int(table.iloc[i].obs_id),true=c,rival=r,group=group,refs=refs))
    wanted={int(table.iloc[ix].obs_id) for ca in cases for ix in [ca['i'],*ca['refs'][ca['true']],*ca['refs'][ca['rival']]]}
    snaps={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    margins,records,evidence=[] ,[],[]
    for ca in cases:
        i,c,r=ca['i'],ca['true'],ca['rival']
        trace=edge.trace_query(i,'mixed',x,mass,bank)
        for g in range(4):
            name=edge.GROUPS[g]
            for cl in (c,r):
                resp=trace[name+'_responsibility'][cl]
                valid=trace['legal_edges'][0,g]
                pts=trace[name+'_four_points'][:,cl]
                expected=(pts*resp[None]).sum((1,2,3))/max(valid.sum(),1)/edge.TAU*.5*np.array([1,1,-1,-1])
                raw=float(trace['raw_scores'][cl,g])
                evidence.append(dict(obs_id=ca['obs_id'],group=name,class_id=cl,raw_mixed=raw,
                                     bb=float(expected[0]),nn=float(expected[1]),bn=float(expected[2]),nb=float(expected[3]),
                                     entropy_and_mass=raw-float(expected.sum()),
                                     body_raw=float(cache['body_raw'][i,cl,g]),additive_raw=float(cache['additive_raw'][i,cl,g])))
        fig,axes=plt.subplots(2,5,figsize=(20,10))
        fig.subplots_adjust(top=.83,bottom=.14,wspace=.15,hspace=.3)
        views=[(i,'QUERY')]+[(v,f'TRUE REF {n+1}') for n,v in enumerate(ca['refs'][c])]+[(v,f'RIVAL REF {n+1}') for n,v in enumerate(ca['refs'][r])]
        for p,pol in enumerate(('vv','vh')):
            for j,(ix,label) in enumerate(views):
                pix,fp,g,place=geo.native(table,snaps,ix,p)
                ax=axes[p,j]
                ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if tokens['valid'][ix,p]:
                    masks=slots(g,fp)
                    for (lo,hi),color in zip(((0,8),(8,16),(16,20)),('#24a878','#2c84c6','#d39222')):
                        f=masks[lo:hi].sum(0)
                        if f.min()<.5<f.max(): ax.contour(f,levels=[.5],colors=[color],linewidths=.7)
                row=table.iloc[ix]
                ax.set(title=f'{pol.upper()} {label}: obs {int(row.obs_id)}\n{row.class_name}; {row.port}\nlocal valid={int(tokens["valid"][ix,p])}',xlim=(-.5,159.5),ylim=(159.5,-.5),xticks=[0,80,159],yticks=[0,80,159])
                records.append(dict(query_obs=ca['obs_id'],obs_id=int(row.obs_id),pol=pol,path=row['image_'+pol],placement=place))
        lines=[]
        for arm in ('full','anchor','body','additive','mixed'):
            m=float(pred[arm][i,c]-pred[arm][i,r])
            margins.append(dict(obs_id=ca['obs_id'],outcome=ca['outcome'],arm=arm,prediction=int(pred[arm][i].argmax()),true=c,rival=r,margin=m))
            lines.append(f'{arm}: pred={int(pred[arm][i].argmax())}; margin={m:+.4f}')
        fig.suptitle(f"Edge operator | {ca['outcome']} vs anchor | obs {ca['obs_id']} | seed17 | group n={ca['n']}\nMixed selected step0: NO mixed effect on final prediction. Refs ranked by raw mixed response in {edge.GROUPS[ca['group']]}",fontsize=12)
        fig.text(.04,.018,'\n'.join(lines)+'\nNative160 crop/pad, no resize; VV/VH separate; layer1 RF=35. Regions are not pixel-pure.',fontsize=9)
        fig.savefig(OUT/f"case_{ca['outcome']}_{ca['obs_id']}.png",dpi=130)
        plt.close(fig)
    pd.DataFrame(margins).to_csv(OUT/'case_margins.csv',index=False)
    pd.DataFrame(evidence).to_csv(OUT/'four_point_evidence.csv',index=False)
    (OUT/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (OUT/'image_records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(pd.DataFrame(cases)[['outcome','n','obs_id','group']].to_string(index=False))
    print(pd.DataFrame(margins).to_string(index=False))
    print(pd.DataFrame(evidence).to_string(index=False))


if __name__=='__main__': main()
