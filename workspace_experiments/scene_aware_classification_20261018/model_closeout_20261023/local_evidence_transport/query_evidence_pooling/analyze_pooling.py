"""Inspect native query/reference images and exact regional margin changes."""
import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_pooling import ROOT, UP, REGIONS
sys.path.insert(0, str(ROOT))
from extract_tokens import geo, slots


def main():
    table = pd.read_csv(UP/'table.csv',dtype={'mmsi':str})
    local = np.load(HERE/'local_scores.npz')
    details = np.load(HERE/'details.npz')
    native = np.load(ROOT/'tokens.npz')
    preds = np.load(HERE/'predictions_17.npz')
    idx, bank = local['idx'], local['bank']
    y = table.class_id.to_numpy()
    te = np.flatnonzero(table.iloc[idx].role.eq('evaluation'))
    old = preds['joint'][idx[te]].argmax(-1)==y[idx[te]]
    new = preds['query_log'][idx[te]].argmax(-1)==y[idx[te]]
    cases = []
    pi = details['pi']
    logr = np.where(pi>0,np.log(np.maximum(pi,1e-30)),-1e4)
    share = np.exp(details['k']+logr[None,:,None]-details['token'][...,None])
    averaged = (share*details['a'][...,None,None]).sum(2)
    averaged = np.where(details['valid'][...,None],averaged,0).sum(1)
    for name,mask in [('rescued',~old&new),('harmed',old&~new),('CC',old&new),('WW',~old&~new)]:
        ids = te[mask]
        if not len(ids):
            continue
        j = int(ids[np.argmin(table.iloc[idx[ids]].obs_id.to_numpy())])
        i,c = int(idx[j]),int(y[idx[j]])
        rival = int(preds['joint'][i].argmax()) if name=='rescued' else int(np.argmax(np.where(np.arange(8)==c,-np.inf,preds['query_log'][i])))
        refs = {t:bank[t,np.argsort(-averaged[j,t],kind='stable')[:2]].tolist() for t in (c,rival)}
        cases.append(dict(outcome=name,n=int(len(ids)),j=j,i=i,obs_id=int(table.iloc[i].obs_id),true=c,rival=rival,refs=refs))
    out = HERE/'analysis'
    out.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(out/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wanted = {int(table.iloc[i].obs_id) for case in cases for i in [case['i'],*case['refs'][case['true']],*case['refs'][case['rival']]]}
    snapshots = {int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    rows, evidence, records = [], [], []
    for case in cases:
        i,j,c,rival = [case[k] for k in ('i','j','true','rival')]
        fig, axes = plt.subplots(2,5,figsize=(20,10))
        fig.subplots_adjust(top=.83,bottom=.14,wspace=.15,hspace=.3)
        views = [(i,'QUERY')]+[(v,f'TRUE REF {n+1}') for n,v in enumerate(case['refs'][c])]+[(v,f'RIVAL REF {n+1}') for n,v in enumerate(case['refs'][rival])]
        for p,pol in enumerate(('vv','vh')):
            for column,(ix,label) in enumerate(views):
                pix,fp,g,placement = geo.native(table,snapshots,ix,p)
                masks = slots(g,fp)
                valid = bool(native['valid'][ix,p])
                ax=axes[p,column]
                ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if valid:
                    for (lo,hi),color in zip(REGIONS,('#24a878','#2c84c6','#d39222')):
                        field=masks[lo:hi].sum(0)
                        if field.min()<.5<field.max():
                            ax.contour(field,levels=[.5],colors=[color],linewidths=.7)
                row=table.iloc[ix]
                ax.set(title=f'{pol.upper()} {label}: obs {int(row.obs_id)}\n{row.class_name}; {row.port}\nlocal valid={int(valid)}',xlim=(-.5,159.5),ylim=(159.5,-.5),xticks=[0,80,159],yticks=[0,80,159])
                records.append(dict(query_obs=case['obs_id'],obs_id=int(row.obs_id),pol=pol,path=row['image_'+pol],placement=placement,valid=valid))
        lines=[]
        for arm in ('full','joint','query_log','reference_coherent','shared_mixture'):
            margin=float(preds[arm][i,c]-preds[arm][i,rival])
            lines.append(f'{arm}: pred={int(preds[arm][i].argmax())}; true-rival={margin:+.5f}')
            rows.append(dict(obs_id=case['obs_id'],outcome=case['outcome'],arm=arm,prediction=int(preds[arm][i].argmax()),true=c,rival=rival,margin=margin))
        for g in range(6):
            for arm in ('joint','query_log','reference_coherent','shared_mixture'):
                r=details[arm+'_region'][j,g]
                evidence.append(dict(obs_id=case['obs_id'],group=g,arm=arm,valid=bool(details['valid'][j,g,c]),margin=float(r[c]-r[rival])))
        fig.suptitle(f"Query pooling | seed17 | {case['outcome']} | obs {case['obs_id']} | group n={case['n']}\n8 refs/class; top2 displayed by query_log reference responsibility, not a new retrieval.",fontsize=12)
        fig.text(.04,.018,'\n'.join(lines)+'\nNative160 crop/pad, no resize. Green body / blue side / amber ends. Not physical-causality evidence.',fontsize=9)
        fig.savefig(out/f"case_{case['outcome']}_{case['obs_id']}.png",dpi=130)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'case_margins.csv',index=False)
    pd.DataFrame(evidence).to_csv(out/'region_margins.csv',index=False)
    (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (out/'image_records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(pd.DataFrame(cases)[['outcome','n','obs_id']].to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    main()
