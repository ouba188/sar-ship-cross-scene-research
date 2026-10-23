"""Frozen full-stack cases with native images and local transfer-risk traces."""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run as experiment

HERE = Path(__file__).resolve().parent
OUT = HERE/'analysis'


def main():
    OUT.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from extract_tokens import geo, slots
    table,x,mass,bank = experiment.edge.support.load()
    b,n,legal,_ = experiment.edge.edges(x,mass)
    cache = np.load(HERE/'features.npz')
    idx = cache['idx']
    loc = {int(v):k for k,v in enumerate(idx)}
    y = table.class_id.to_numpy(int)
    seed, arm = 17, 'full__excess'
    pred = np.load(HERE/f'predictions_{seed}.npz')
    weights = np.load(HERE/'weights.npz')[f'{arm}_{seed}']
    divisor = pd.read_csv(HERE/'scales.csv').query("arm=='excess'").sort_values('feature').divisor.to_numpy()
    ev = idx[table.iloc[idx].role.to_numpy()=='evaluation']
    old,new = pred['full'][ev].argmax(-1)==y[ev],pred[arm][ev].argmax(-1)==y[ev]
    cases, margins, contributions, refs, edge_rows = [],[],[],[],[]
    for outcome,mask in [('rescued',~old&new),('harmed',old&~new),('CC',old&new),('WW',~old&~new)]:
        ids = ev[mask]
        if not len(ids):
            continue
        i = int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy())])
        c = int(y[i])
        score = pred[arm][i] if outcome=='harmed' else pred['full'][i]
        rival = int(np.argmax(np.where(np.arange(8)==c,-np.inf,score)))
        z = cache['excess'][loc[i]]
        parts = weights*(z[c]-z[rival])
        observed = pred[arm][i,c]-pred[arm][i,rival]-pred['full'][i,c]+pred['full'][i,rival]
        error = abs(float(parts.sum()-observed))
        assert error<5e-6
        group = int(np.argmax(np.abs(parts)))
        selected_refs = []
        for cl in (c,rival):
            values = cache['excess_refs'][loc[i],group,cl]
            rv = legal[bank[cl],group].sum(-1)>=2
            j = int(np.argmax(np.where(rv,np.abs(values*weights[group]/divisor[group]),-np.inf)))
            selected_refs.append(int(bank[cl,j]))
            for r in np.flatnonzero(rv):
                refs.append(dict(obs_id=int(table.iloc[i].obs_id),group=experiment.edge.GROUPS[group],class_id=cl,
                    ref_obs=int(table.iloc[bank[cl,r]].obs_id),raw=values[r],
                    score_contribution=weights[group]*values[r]/divisor[group]/rv.sum()))
        case = dict(i=i,obs_id=int(table.iloc[i].obs_id),outcome=outcome,n=int(mask.sum()),true=c,rival=rival,
                    group=group,refs=selected_refs,increment=float(parts.sum()),replay_error=error,seed=seed)
        cases.append(case)
        contributions.extend(dict(obs_id=case['obs_id'],group=g,increment=parts[k]) for k,g in enumerate(experiment.edge.GROUPS))
        for key in ('full','anchor','full__independent','full__transfer','full__excess','full__excess_shuffle101','full__excess_shuffle307'):
            v = pred[key][i]
            margins.append(dict(obs_id=case['obs_id'],outcome=outcome,arm=key,prediction=int(v.argmax()),margin=float(v[c]-v[rival])))
        fig,axes = plt.subplots(2,2,figsize=(11,8))
        for column,ri in enumerate(selected_refs):
            qkeep,rkeep = legal[i,group],legal[ri,group]
            trace = experiment.transfer(b[i,group,qkeep],n[i,group,qkeep],b[ri,group,rkeep],n[ri,group,rkeep])
            ax = axes[0,column]
            vmax = max(float(np.abs(trace['A']).max()),1e-8)
            image = ax.imshow(trace['A'],cmap='coolwarm',vmin=-vmax,vmax=vmax,aspect='auto')
            ax.set(title=f"{'TRUE' if column==0 else 'RIVAL'} ref {int(table.iloc[ri].obs_id)}",xlabel='reference body edge',ylabel='query body edge')
            fig.colorbar(image,ax=ax,shrink=.8)
            axes[1,column].plot(trace['transfer'],marker='o',label='real improvement over mean')
            axes[1,column].plot(trace['null'],marker='s',label='null prediction energy')
            axes[1,column].plot(trace['excess'],marker='x',label='excess over random pairing')
            axes[1,column].axhline(0,color='gray',linewidth=.5)
            axes[1,column].set(xlabel='query paired edge',ylabel='local feature-space score')
            axes[1,column].legend(fontsize=8)
            for k in range(len(trace['transfer'])):
                edge_rows.append(dict(obs_id=case['obs_id'],ref_obs=int(table.iloc[ri].obs_id),group=experiment.edge.GROUPS[group],edge=int(np.flatnonzero(qkeep)[k]),
                    transfer=trace['transfer'][k],null=trace['null'][k],excess=trace['excess'][k]))
        fig.suptitle(f"{outcome} obs {case['obs_id']}; {experiment.edge.GROUPS[group]}\nSigned ridge coefficients from body only; not attention probabilities")
        fig.tight_layout()
        fig.savefig(OUT/f"operator_{outcome}_{case['obs_id']}.png",dpi=130)
        plt.close(fig)
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
        fig.suptitle(f"Conditional transfer | {ca['outcome']} obs {ca['obs_id']}; group n={ca['n']}; seed17\nRefs: largest absolute contribution in strongest changed group, not independent trials")
        vals = [v for v in margins if v['obs_id']==ca['obs_id']]
        fig.text(.02,.015,'; '.join(v['arm'].replace('full__','')+f": {v['margin']:+.3f}" for v in vals)+'\nNative160 crop/pad, no resize; layer1 RF35. Body/neighbor regions are not pixel-isolated.',fontsize=8)
        fig.subplots_adjust(top=.83,bottom=.12,hspace=.5,wspace=.25)
        fig.savefig(OUT/f"case_{ca['outcome']}_{ca['obs_id']}.png",dpi=130)
        plt.close(fig)
    for name,rows in [('margins',margins),('group_contributions',contributions),('reference_contributions',refs),('edge_scores',edge_rows)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    (OUT/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (OUT/'image_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(pd.DataFrame(margins).to_string(index=False))


if __name__ == '__main__':
    main()
