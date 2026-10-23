"""Native-image review and additive head decomposition for joint evidence."""
import os
import sys
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
UP=ROOT.parent/'regional_utility_pilot/upstream'
sys.path.insert(0,str(ROOT))
from extract_tokens import geo, slots
PAIRS=((0,1),(0,2),(3,4),(3,5),(0,3),(1,4),(2,5))


def main(frozen=False):
    output=HERE/'frozen_marginal' if frozen else HERE
    comparison='anchor' if frozen else 'nonlinear'
    baseline='anchor' if frozen else 'full'
    offset=0 if frozen else 6
    table=pd.read_csv(UP/'table.csv',dtype={'mmsi':str})
    features=np.load(HERE/'features.npz')
    weights=np.load(output/'weights.npz')
    preds=np.load(output/'predictions_17.npz')
    native=np.load(ROOT/'tokens.npz')
    idx,bank=features['idx'],features['bank']
    truth=table.class_id.to_numpy()
    te=np.flatnonzero(table.iloc[idx].role.eq('evaluation'))
    old=preds[comparison][idx[te]].argmax(-1)==truth[idx[te]]
    new=preds['joint'][idx[te]].argmax(-1)==truth[idx[te]]
    cases=[]
    for name,mask in [('rescued',~old&new),('harmed',old&~new),('CC',old&new),('WW',~old&~new)]:
        ids=te[mask]
        if not len(ids):
            continue
        j=int(ids[np.argmin(table.iloc[idx[ids]].obs_id.to_numpy())])
        i,c=int(idx[j]),int(truth[idx[j]])
        rival=int(preds[comparison][i].argmax()) if name=='rescued' else int(np.argmax(np.where(np.arange(8)==c,-np.inf,preds['joint'][i])))
        difference=features['joint'][j,c]-features['joint'][j,rival]
        component=weights['joint_17']*(difference[6:] if frozen else difference)
        pair=int(np.argmax(np.abs(component[offset:])))
        g,h=PAIRS[pair]
        refs={}
        for cl in (c,rival):
            u=features['u'][j]
            ok=np.isfinite(u[g,cl])&np.isfinite(u[h,cl])
            reference=np.flatnonzero(ok)
            if len(reference):
                ranked=reference[np.argsort(-(u[g,cl,reference]+u[h,cl,reference]),kind='stable')[:2]]
            else:
                reference=np.flatnonzero(np.isfinite(u[0,cl]))
                ranked=reference[np.argsort(-u[0,cl,reference],kind='stable')[:2]]
            refs[cl]=bank[cl,ranked].tolist()
        cases.append(dict(outcome=name,n=int(len(ids)),j=j,i=i,obs_id=int(table.iloc[i].obs_id),true=c,rival=rival,pair=pair,refs=refs))
    out=output/'analysis'
    out.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(out/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wanted={int(table.iloc[ix].obs_id) for ca in cases for ix in [ca['i'],*ca['refs'][ca['true']],*ca['refs'][ca['rival']]]}
    snaps={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    rows,parts,records=[],[],[]
    for ca in cases:
        i,j,c,rival=[ca[k] for k in ('i','j','true','rival')]
        fig,axes=plt.subplots(2,5,figsize=(20,10))
        fig.subplots_adjust(top=.83,bottom=.14,wspace=.15,hspace=.3)
        views=[(i,'QUERY')]+[(v,f'TRUE REF {n+1}') for n,v in enumerate(ca['refs'][c])]+[(v,f'RIVAL REF {n+1}') for n,v in enumerate(ca['refs'][rival])]
        for p,pol in enumerate(('vv','vh')):
            for column,(ix,label) in enumerate(views):
                pix,fp,g,placement=geo.native(table,snaps,ix,p)
                masks=slots(g,fp)
                valid=bool(native['valid'][ix,p])
                ax=axes[p,column]
                ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if valid:
                    for (lo,hi),color in zip(((0,8),(8,16),(16,20)),('#24a878','#2c84c6','#d39222')):
                        field=masks[lo:hi].sum(0)
                        if field.min()<.5<field.max():
                            ax.contour(field,levels=[.5],colors=[color],linewidths=.7)
                row=table.iloc[ix]
                ax.set(title=f'{pol.upper()} {label}: obs {int(row.obs_id)}\n{row.class_name}; {row.port}\nlocal valid={int(valid)}',xlim=(-.5,159.5),ylim=(159.5,-.5),xticks=[0,80,159],yticks=[0,80,159])
                records.append(dict(query_obs=ca['obs_id'],obs_id=int(row.obs_id),pol=pol,path=row['image_'+pol],valid=valid,placement=placement))
        lines=[]
        arms=('full','anchor','nonlinear','joint','shuffle307') if frozen else ('full','marginal','nonlinear','joint','shuffle211')
        for arm in arms:
            margin=float(preds[arm][i,c]-preds[arm][i,rival])
            first=extra=0.
            if arm not in ('full','anchor'):
                difference=features[arm][j,c]-features[arm][j,rival]
                component=weights[arm+'_17']*(difference[6:] if frozen else difference)
                first=float(component[:offset].sum())
                extra=float(component[offset:].sum())
                closure=margin-float(preds[baseline][i,c]-preds[baseline][i,rival])-float(component.sum())
                assert abs(closure)<3e-6,closure
                for f,value in enumerate(component):
                    parts.append(dict(obs_id=ca['obs_id'],arm=arm,feature=f,contribution=float(value)))
            rows.append(dict(obs_id=ca['obs_id'],outcome=ca['outcome'],arm=arm,true=c,rival=rival,prediction=int(preds[arm][i].argmax()),margin=margin,marginal_delta=first,extra_delta=extra))
            lines.append(f'{arm}: pred={int(preds[arm][i].argmax())}; margin={margin:+.4f}; single={first:+.4f}; extra={extra:+.4f}')
        fig.suptitle(f"Reference dependence | frozen={frozen} | seed17 | {ca['outcome']} vs {comparison} | obs {ca['obs_id']} | group n={ca['n']}\n8refs/class. Display top2 joint-numerator refs for pair {ca['pair']}={PAIRS[ca['pair']]}, not all causal contribution.",fontsize=12)
        fig.text(.04,.018,'\n'.join(lines)+'\nNative160 crop/pad, no resize; VV/VH separate. Green body / blue side / amber end.',fontsize=9)
        fig.savefig(out/f"case_{ca['outcome']}_{ca['obs_id']}.png",dpi=130)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'case_margins.csv',index=False)
    pd.DataFrame(parts).to_csv(out/'head_contributions.csv',index=False)
    (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (out/'image_records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(pd.DataFrame(cases)[['outcome','n','obs_id','pair']].to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--frozen',action='store_true')
    main(parser.parse_args().frozen)
