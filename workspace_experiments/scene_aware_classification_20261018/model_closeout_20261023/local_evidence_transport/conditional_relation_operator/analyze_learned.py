"""Selected learned operator: replay and native-image review, without refitting."""
import os
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIT = HERE/'learned'
OUT = HERE/'analysis_learned'
spec = importlib.util.spec_from_file_location('selected_conditional_learned',FIT/'run.py')
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def main():
    OUT.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from extract_tokens import geo, slots
    table,x,mass,bank = exp.edge.support.load()
    receipts = pd.read_csv(FIT/'training_receipts.csv')
    active = receipts[receipts.arm.eq('excess') & receipts.selected_step.gt(0)].seed.to_list()
    seed = int(active[0]) if active else 17
    pred = np.load(FIT/f'predictions_{seed}.npz')
    features = np.load(FIT/f'features_{seed}.npz')
    weights = np.load(FIT/'weights.npz')
    mb,mn,a = [weights[f'excess_{seed}_{key}'] for key in ('Mb','Mn','a')]
    divisor = pd.read_csv(FIT/'scales.csv').query("arm=='excess'").sort_values('feature').divisor.to_numpy()
    y = table.class_id.to_numpy(int)
    ev = np.flatnonzero(table.role.eq('evaluation'))
    old,new = pred['full'][ev].argmax(-1)==y[ev],pred['excess'][ev].argmax(-1)==y[ev]
    cases,margins,groups,refs,edges = [],[],[],[],[]
    for name,mask in [('rescued',~old&new),('harmed',old&~new),('CC',old&new),('WW',~old&~new)]:
        ids = ev[mask]
        if not len(ids):
            continue
        i = int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy())])
        c = int(y[i])
        score = pred['excess'][i] if name=='harmed' else pred['full'][i]
        r = int(np.argmax(np.where(np.arange(8)==c,-np.inf,score)))
        identity = exp.trace_query(i,'excess',np.eye(32),np.eye(32),x,mass,bank)
        learned = exp.trace_query(i,'excess',mb,mn,x,mass,bank)
        parts = a*(features['excess'][i,c]-features['excess'][i,r])
        observed = pred['excess'][i,c]-pred['excess'][i,r]-pred['full'][i,c]+pred['full'][i,r]
        scalar = a*((learned['raw_scores'][c]-identity['raw_scores'][c])-(learned['raw_scores'][r]-identity['raw_scores'][r]))/divisor
        err = max(abs(float(parts.sum()-observed)),float(np.max(np.abs(parts-scalar))))
        assert err<3e-5,err
        g = int(np.argmax(np.abs(parts)))
        chosen = []
        for cl in (c,r):
            rv = learned['reference_valid'][g,0,cl]
            change = learned['reference_scores'][g,0,cl]-identity['reference_scores'][g,0,cl]
            j = int(np.argmax(np.where(rv,np.abs(change),-np.inf)))
            chosen.append(int(bank[cl,j]))
            for rr in np.flatnonzero(rv):
                refs.append(dict(obs_id=int(table.iloc[i].obs_id),group=exp.edge.GROUPS[g],class_id=cl,ref_obs=int(table.iloc[bank[cl,rr]].obs_id),
                    identity=float(identity['reference_scores'][g,0,cl,rr]),learned=float(learned['reference_scores'][g,0,cl,rr]),
                    increment=float(a[g]*change[rr]/divisor[g]/rv.sum())))
            ri = cl*8+j
            for state,tr in [('identity',identity),('learned',learned)]:
                A = tr['A'][g,0,ri]
                qn,sn = tr['qn'][g,0],tr['sn'][g,ri]
                h = A@sn
                T = 2*np.sum(qn*h,-1)-np.sum(h*h,-1)
                ms = tr['ms'][g,ri]
                V = np.sum(sn*sn)/(ms-1)*(np.sum(A*A,-1)-A.sum(-1)**2/ms)
                for k in np.flatnonzero(tr['legal_edges'][0,g]):
                    edges.append(dict(obs_id=int(table.iloc[i].obs_id),ref_obs=int(table.iloc[bank[cl,j]].obs_id),state=state,group=exp.edge.GROUPS[g],edge=int(k),transfer=T[k],null=V[k],excess=T[k]+V[k]))
        cases.append(dict(i=i,obs_id=int(table.iloc[i].obs_id),outcome=name,n=int(mask.sum()),true=c,rival=r,group=g,refs=chosen,seed=seed,replay_error=err))
        groups.extend(dict(obs_id=int(table.iloc[i].obs_id),group=label,increment=float(parts[j])) for j,label in enumerate(exp.edge.GROUPS))
        for arm in ('full',*exp.ARMS):
            v = pred[arm][i]
            margins.append(dict(obs_id=int(table.iloc[i].obs_id),outcome=name,seed=seed,arm=arm,prediction=int(v.argmax()),margin=float(v[c]-v[r])))
    wanted = {int(table.iloc[j].obs_id) for ca in cases for j in [ca['i'],*ca['refs']]}
    snaps = {int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    tokens = np.load(HERE.parent/'tokens.npz')
    provenance = []
    for ca in cases:
        fig,axes = plt.subplots(2,3,figsize=(12,8),squeeze=False)
        for column,(i,label) in enumerate(zip([ca['i'],*ca['refs']],('QUERY','TRUE REF','RIVAL REF'))):
            for p,pol in enumerate(('vv','vh')):
                pix,fp,geom,placement = geo.native(table,snaps,int(i),p)
                ax = axes[p,column]
                ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if tokens['valid'][i,p]:
                    masks = slots(geom,fp)
                    for (lo,hi),color in zip(((0,8),(8,16),(16,20)),('#24a878','#2c84c6','#d39222')):
                        region = masks[lo:hi].sum(0)
                        if region.min()<.5<region.max():
                            ax.contour(region,levels=[.5],colors=[color],linewidths=.7)
                row = table.iloc[i]
                ax.set(title=f"{pol.upper()} {label} {int(row.obs_id)}\n{row.class_name}\n{row.port}; valid={int(tokens['valid'][i,p])}",xlim=(-.5,159.5),ylim=(159.5,-.5),xticks=[0,80,159],yticks=[0,80,159])
                provenance.append(dict(query_obs=ca['obs_id'],obs_id=int(row.obs_id),pol=pol,path=row['image_'+pol],placement=placement))
        fig.suptitle(f"Learned conditional transfer | {ca['outcome']} obs {ca['obs_id']}; group n={ca['n']}; seed{seed}\nRefs: largest absolute operator change in strongest changed group")
        vals = [v for v in margins if v['obs_id']==ca['obs_id']]
        fig.text(.02,.015,'; '.join(v['arm']+f": {v['margin']:+.3f}" for v in vals)+'\nNative160 crop/pad, NO resize; RF35; relative operator contribution, not physical attribution.',fontsize=8)
        fig.subplots_adjust(top=.83,bottom=.12,hspace=.5,wspace=.25)
        fig.savefig(OUT/f"case_{ca['outcome']}_{ca['obs_id']}.png",dpi=130)
        plt.close(fig)
    for name,rows in [('margins',margins),('group_contributions',groups),('reference_contributions',refs),('edge_scores',edges)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    (OUT/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (OUT/'image_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(pd.DataFrame(margins).to_string(index=False))


if __name__ == '__main__':
    exp.torch.set_num_threads(2)
    main()
