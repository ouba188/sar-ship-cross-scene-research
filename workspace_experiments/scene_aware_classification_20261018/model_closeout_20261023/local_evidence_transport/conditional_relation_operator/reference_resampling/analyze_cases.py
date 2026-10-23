"""Native query and actual reference views for the frozen reference intervention."""
import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reference_intervention_analysis',HERE/'run.py')
diag=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
exp=diag.exp


def main():
    out=HERE/'analysis'
    out.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(out/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from extract_tokens import geo,slots
    table,x,mass,old,idx,tr,va,te=exp.source_data()
    data=np.load(HERE/'predictions.npz')
    banktable=pd.read_csv(HERE/'banks.csv',dtype={'bank':str})
    banks={name:banktable[banktable.bank.eq(name)].sort_values(['class_id','slot'])['index'].to_numpy().reshape(8,8) for name in ('original','101')}
    weights=np.load(exp.HERE/'weights.npz')
    arm,seed='excess',43
    mb,mn,a=[weights[f'{arm}_{seed}_{k}'] for k in ('Mb','Mn','a')]
    divisor=pd.read_csv(exp.HERE/'scales.csv').query("arm=='excess'").sort_values('feature').divisor.to_numpy()
    y=table.class_id.to_numpy(int)
    previous=data['original_excess_43']
    new=data['101_excess_43']
    before=previous[te].argmax(-1)==y[idx[te]]
    after=new[te].argmax(-1)==y[idx[te]]
    cases,margins,parts,refs=[],[],[],[]
    for name,mask in [('rescued',~before&after),('harmed',before&~after),('CC',before&after),('WW',~before&~after)]:
        choices=te[mask]
        if not len(choices):
            continue
        loc=int(choices[np.argmin(table.iloc[idx[choices]].obs_id.to_numpy())])
        i=int(idx[loc]); c=int(y[i])
        score=new[loc] if name=='harmed' else previous[loc]
        r=int(np.argmax(np.where(np.arange(8)==c,-np.inf,score)))
        changes=[]
        trace={}
        for bankname,bank in banks.items():
            identity=exp.trace_query(i,arm,np.eye(32),np.eye(32),x,mass,bank)
            learned=exp.trace_query(i,arm,mb,mn,x,mass,bank)
            change=a*((learned['raw_scores'][c]-identity['raw_scores'][c])-(learned['raw_scores'][r]-identity['raw_scores'][r]))/divisor
            observed=data[f'{bankname}_{arm}_{seed}'][loc,c]-data[f'{bankname}_{arm}_{seed}'][loc,r]-data[f'full_{seed}'][loc,c]+data[f'full_{seed}'][loc,r]
            assert abs(change.sum()-observed)<3e-5
            changes.append(change)
            trace[bankname]=(identity,learned)
            parts.extend(dict(obs_id=int(table.iloc[i].obs_id),bank=bankname,group=group,increment=float(change[g])) for g,group in enumerate(exp.edge.GROUPS))
        g=int(np.argmax(abs(changes[1]-changes[0])))
        chosen=[]
        for bankname,bank in banks.items():
            identity,learned=trace[bankname]
            for cl in (c,r):
                valid=learned['reference_valid'][g,0,cl]
                delta=learned['reference_scores'][g,0,cl]-identity['reference_scores'][g,0,cl]
                contribution=a[g]*delta/divisor[g]/valid.sum()
                j=int(np.argmax(np.where(valid,abs(contribution),-np.inf)))
                chosen.append(int(bank[cl,j]))
                for k in np.flatnonzero(valid):
                    refs.append(dict(obs_id=int(table.iloc[i].obs_id),bank=bankname,group=exp.edge.GROUPS[g],class_id=cl,ref_obs=int(table.iloc[bank[cl,k]].obs_id),increment=float(contribution[k])))
        cases.append(dict(i=i,loc=loc,obs_id=int(table.iloc[i].obs_id),outcome=name,n=int(mask.sum()),true=c,rival=r,group=g,refs=chosen,seed=seed,bank_after='101',bank_choice='first predeclared, not best'))
        for label,v in [('full',data['full_43'][loc]),('original',previous[loc]),('101',new[loc])]:
            margins.append(dict(obs_id=int(table.iloc[i].obs_id),outcome=name,version=label,prediction=int(v.argmax()),margin=float(v[c]-v[r])))
    wanted={int(table.iloc[j].obs_id) for ca in cases for j in [ca['i'],*ca['refs']]}
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    tokens=np.load(exp.ROOT/'tokens.npz')
    provenance=[]
    for ca in cases:
        fig,axes=plt.subplots(2,5,figsize=(18,8),squeeze=False)
        labels=('QUERY','OLD TRUE REF','OLD RIVAL REF','NEW TRUE REF','NEW RIVAL REF')
        for col,(i,label) in enumerate(zip([ca['i'],*ca['refs']],labels)):
            for p,pol in enumerate(('vv','vh')):
                pixels,footprint,geometry,placement=geo.native(table,snapshots,int(i),p)
                ax=axes[p,col]
                ax.imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if tokens['valid'][i,p]:
                    masks=slots(geometry,footprint)
                    for (lo,hi),color in zip(((0,8),(8,16),(16,20)),('#24a878','#2c84c6','#d39222')):
                        region=masks[lo:hi].sum(0)
                        if region.min()<.5<region.max():
                            ax.contour(region,levels=[.5],colors=[color],linewidths=.6)
                row=table.iloc[i]
                ax.set(title=f"{pol.upper()} {label} {int(row.obs_id)}\n{row.class_name}\n{row.port}; valid={int(tokens['valid'][i,p])}",xlim=(-.5,159.5),ylim=(159.5,-.5),xticks=[0,80,159],yticks=[0,80,159])
                provenance.append(dict(query_obs=ca['obs_id'],obs_id=int(row.obs_id),pol=pol,path=row['image_'+pol],placement=placement))
        vals=[m for m in margins if m['obs_id']==ca['obs_id']]
        fig.suptitle(f"Frozen operator, only references changed: {ca['outcome']} obs {ca['obs_id']} (n={ca['n']}); {exp.edge.GROUPS[ca['group']]}\nOriginal bank -> first predeclared new bank101; references have largest absolute change, not attention weights")
        fig.text(.02,.02,'; '.join(m['version']+f": pred{m['prediction']}, margin{m['margin']:+.3f}" for m in vals)+'\nNative160 crop/pad, no resize; RF35; displayed pixels are not calibrated backscatter.',fontsize=9)
        fig.subplots_adjust(top=.83,bottom=.12,wspace=.27,hspace=.5)
        fig.savefig(out/f"case_{ca['outcome']}_{ca['obs_id']}.png",dpi=120)
        plt.close(fig)
    for name,rows in [('margins',margins),('group_contributions',parts),('reference_contributions',refs)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (out/'image_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(pd.DataFrame(margins).to_string(index=False))


if __name__=='__main__':
    exp.torch.set_num_threads(2)
    main()
