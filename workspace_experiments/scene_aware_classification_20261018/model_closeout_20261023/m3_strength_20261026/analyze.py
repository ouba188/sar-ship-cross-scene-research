"""Trace unchanged shrinkage decisions and source-to-target correction effects."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run
from analyze_transfer import input_records, CLASSES
base=run.base
REPAIR=base.TRANSFER/'normalization_repair_20261026'


def main():
    out=HERE/'analysis'
    out.mkdir(exist_ok=True)
    paired,classes,checks,cases=[],[],[],[]
    for source,other in (('oss1','oss2'),('oss2','oss1')):
        direction=f'{source}-to-{other}'
        tables={'source':pd.read_csv(run.ROOT/f'{source}_m123_replication_20261026/stack_table.csv',dtype={'mmsi':str}),
            'cross':base.tr.read_table(base.TRANSFER/direction/'target_table.csv',labels=True)}
        allz={}
        for seed in base.SEEDS:
            for side,table in tables.items():
                z=base.tr.load_npz(HERE/direction/f'seed{seed}/{side}_predictions.npz')
                allz[seed,side]=z
                np.testing.assert_array_equal(z['obs_id'],table.obs_id)
                for arm in run.ENDPOINTS:
                    error=float(abs(z[arm]-z[arm+'_scaled']).max())
                    checks.append(dict(source=source,seed=seed,side=side,endpoint=arm,
                        max_score_change=error,changed_predictions=int((z[arm].argmax(1)!=z[arm+'_scaled'].argmax(1)).sum())))
                for role,ix in table.groupby('role').indices.items():
                    yy=table.iloc[ix].class_id.to_numpy(int)
                    for first,last in [('bare','frequency_scaled'),('frequency','frequency_scaled'),('B_fixed','B_fixed_scaled')]:
                        a,b=z[first][ix].argmax(1)==yy,z[last][ix].argmax(1)==yy
                        d=b.astype(float)-a
                        paired.append(dict(source=source,side=side,role=role,seed=seed,before=first,after=last,n=len(ix),
                            rescue=int((b&~a).sum()),harm=int((a&~b).sum()),OA_pp=100*d.mean(),
                            BA_pp=100*np.mean([d[yy==c].mean() for c in np.unique(yy)])))
                        if first=='bare':
                            for c in np.unique(yy):
                                mask=yy==c
                                classes.append(dict(source=source,side=side,role=role,seed=seed,class_name=CLASSES[c],n=int(mask.sum()),
                                    before_recall=100*a[mask].mean(),after_recall=100*b[mask].mean(),
                                    rescue=int((b&~a&mask).sum()),harm=int((a&~b&mask).sum()),
                                    baseline_correct=int((a&mask).sum()),
                                    harm_fraction_baseline_correct=float((a&~b&mask).sum()/max(1,(a&mask).sum()))))
        # Select an active seed from SOURCE scores only, not target success.
        seed=next(s for s in base.SEEDS if not np.array_equal(allz[s,'source']['frequency'],allz[s,'source']['bare']))
        table=tables['cross']
        y=table.class_id.to_numpy(int)
        z=allz[seed,'cross']
        cached=base.tr.load_npz(run.OLD/direction/f'seed{seed}/cross_predictions.npz')
        orig=base.tr.load_npz(REPAIR/direction/f'seed{seed}/predictions.npz')
        np.testing.assert_array_equal(orig['obs_id'],table.obs_id)
        combined=pd.read_csv(base.TRANSFER/direction/'table.csv',dtype={'mmsi':str})
        records=input_records(combined,base.tr.read_json(base.TRANSFER/direction/'protocol.json'))
        before,after=z['bare'],z['frequency_scaled']
        ac,bc=after.argmax(1)==y,before.argmax(1)==y
        masks=dict(rescue=ac&~bc,harm=bc&~ac,CC=ac&bc,WW=~ac&~bc)
        picks=[]
        for state,mask in masks.items():
            ids=np.flatnonzero(mask)
            if len(ids):
                picks.append((state,int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy())])))
        if len(picks)<4:
            used={i for _,i in picks}
            for state in ('CC','WW'):
                ids=np.flatnonzero(masks[state])
                for i in ids[np.argsort(table.iloc[ids].obs_id.to_numpy())]:
                    if int(i) not in used and len(picks)<4:
                        picks.append((state,int(i)));used.add(int(i))
        beta=base.tr.read_json(HERE/direction/f'seed{seed}/selection.json')['frequency']
        for position,(state,j) in enumerate(picks):
            true=int(y[j])
            rival=int(after[j].argmax()) if not ac[j] else int(before[j].argmax())
            if rival==true:
                rival=int(np.where(np.arange(4)==true,-np.inf,before[j]).argmax())
            refs=orig['refs'][j,[true,rival]].astype(int).ravel()
            rows=[int(orig['query_rows'][j]),*refs.tolist()]
            mb=float(before[j,true]-before[j,rival]);ma=float(after[j,true]-after[j,rival])
            w=cached['weight_frequency'][j]
            contribution=beta*w*np.array([cached['delta_vv'][j,true]-cached['delta_vv'][j,rival],
                cached['delta_vh'][j,true]-cached['delta_vh'][j,rival]])
            np.testing.assert_allclose(contribution.sum(),ma-mb,atol=2e-5,rtol=1e-5)
            fig,axes=plt.subplots(2,5,figsize=(15,6.7))
            prov=[]
            for col,row in enumerate(rows):
                record=records[row]
                for pol,v in enumerate(base.tr.extract.prepare(record)):
                    ax=axes[pol,col]
                    if v is not None:
                        ax.imshow(v['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                        if v['masks'][0].any():ax.contour(v['masks'][0],levels=[.5],colors=['#20a780'],linewidths=.6)
                        prov.append(dict(row=row,pol=('VV','VH')[pol],path=record[('vv','vh')[pol]]['image'],placement=v['placement']))
                    ax.set_title(f'{("VV","VH")[pol]} {("QUERY","TRUE REF 1","TRUE REF 2","RIVAL REF 1","RIVAL REF 2")[col]}\nobs {combined.iloc[row].original_obs_id}',fontsize=9)
                    ax.set(xticks=[],yticks=[])
            filename=f'{source}_{position}_{state}_{int(table.iloc[j].obs_id)}.png'
            fig.suptitle(f'{direction} seed{seed}, beta={beta}: bare -> M3; {state}, true {CLASSES[true]}\n{CLASSES[before[j].argmax()]} -> {CLASSES[after[j].argmax()]}; fixed margin {mb:+.4f} -> {ma:+.4f}')
            fig.text(.02,.02,f'Native160 crop/pad; actual frozen M3 references. VV/VH contributions {contribution.round(4)}. Scaling leaves this prediction unchanged.',fontsize=8)
            fig.subplots_adjust(top=.80,bottom=.09,hspace=.2,wspace=.1)
            fig.savefig(out/filename,dpi=115);plt.close(fig)
            cases.append(dict(source=source,seed=seed,beta=beta,status=state,obs_id=int(table.iloc[j].obs_id),true=true,rival=rival,
                before_pred=int(before[j].argmax()),after_pred=int(after[j].argmax()),before_margin=mb,after_margin=ma,
                weights=w.tolist(),polar_contributions=contribution.tolist(),
                local_true_rival=orig['local'][j,[true,rival]].tolist(),
                local_axis_order=['true/rival','VV/VH','body/side/end','transport_similarity/transport_mass'],
                refs_obs_id=combined.iloc[refs].original_obs_id.to_numpy(int).tolist(),
                image=filename,provenance=prov))
    pd.DataFrame(paired).to_csv(out/'paired.csv',index=False)
    pd.DataFrame(classes).to_csv(out/'classwise.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'identity.csv',index=False)
    base.write_json(out/'cases.json',cases)
    base.write_json(out/'receipt.json',dict(panels=len(cases),raw_review_pending=True,
        source_seed_selection='first nonzero M3 source endpoint',cases='minimum obsID per bare-to-M3 correctness state',
        scoped_result='selected strength versus original endpoint exact identities; no bootstrap needed to establish zero change',
        actual_M3_references=True))
    frame=pd.DataFrame(paired)
    print(frame.query("before=='bare' and (side=='cross' or role=='selection')").to_string(index=False),flush=True)


if __name__=='__main__':main()
