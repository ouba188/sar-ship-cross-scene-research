"""Paired uncertainty and actual native-query/reference decision tracing."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run
from analyze_transfer import input_records, CLASSES
tr=run.tr


def main():
    assert (HERE/'evaluation_complete.json').exists()
    out=HERE/'analysis'
    out.mkdir(exist_ok=True)
    intervals,cases,reference_effects=[],[],[]
    selections=pd.read_csv(HERE/'selections.csv')
    for source,target in (('oss1','oss2'),('oss2','oss1')):
        direction=f'{source}-to-{target}'
        crossroot=tr.HERE/direction
        table=tr.read_table(crossroot/'target_table.csv',labels=True)
        y=table.class_id.to_numpy(int)
        scores={s:tr.load_npz(HERE/direction/f'seed{s}/cross_predictions.npz') for s in run.SEEDS}
        prior=tr.load_npz(run.ROOT/f'paper_evidence_completion_20261026/classical_fit_only/{source}/cross_predictions.npz')
        np.testing.assert_array_equal(prior['obs_id'],table.obs_id)
        for before,after in (('old','frequency'),('bare','frequency'),('old','metric_only'),('ridge','frequency'),('raw_mlp','bare')):
            effects=[]
            for s,z in scores.items():
                b=prior['regional_ridge'] if before=='ridge' else prior[f'regional_mlp_seed{s}'] if before=='raw_mlp' else z[before]
                effects.append((z[after].argmax(1)==y).astype(float)-(b.argmax(1)==y))
            d=np.mean(effects,0)
            if not np.any(d):
                intervals.append(dict(source=source,before=before,after=after,OA_pp=0.,BA_pp=0.,OA_low=0.,OA_high=0.,BA_low=0.,BA_high=0.,groups=table.pass_group.nunique(),draws=0,BA_complete_draws=0,identity=True))
                continue
            groups=list(table.groupby('pass_group').indices.values())
            counts=np.array([np.bincount(y[ix],minlength=4) for ix in groups])
            sums=np.array([np.bincount(y[ix],weights=d[ix],minlength=4) for ix in groups])
            draws=np.random.default_rng(1026).integers(len(groups),size=(2000,len(groups)))
            ct=counts[draws].sum(1); sm=sums[draws].sum(1)
            keep=(ct>0).all(1)
            oa=100*sm.sum(1)/ct.sum(1); ba=100*(sm[keep]/ct[keep]).mean(1)
            intervals.append(dict(source=source,before=before,after=after,OA_pp=100*d.mean(),BA_pp=100*np.mean([d[y==c].mean() for c in range(4)]),
                OA_low=np.quantile(oa,.025),OA_high=np.quantile(oa,.975),BA_low=np.quantile(ba,.025),BA_high=np.quantile(ba,.975),groups=len(groups),draws=2000,BA_complete_draws=int(keep.sum())))
        # Source-selected activity, never target performance, fixes the case seed.
        active=selections.loc[selections.source.eq(source)&selections.metric_step.gt(0),'seed'].tolist()
        seed=int(active[0]) if active else 17
        z=scores[seed]
        old=tr.load_npz(run.OLD/direction/f'seed{seed}/cross_predictions.npz')
        orig=tr.load_npz(run.REPAIR/direction/f'seed{seed}/predictions.npz')
        np.testing.assert_array_equal(orig['refs'],z['refs'])
        decision=run.FrozenDecision(HERE/direction/f'seed{seed}',z,z['bare'],'cpu')
        before,after=z['old'],z['frequency']
        ac=after.argmax(1)==y; bc=before.argmax(1)==y
        masks=dict(rescue=ac&~bc,harm=bc&~ac,CC=ac&bc,WW=~ac&~bc)
        picks=[]
        for state,mask in masks.items():
            ids=np.flatnonzero(mask)
            if len(ids):picks.append((state,int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy())])))
        used={j for _,j in picks}
        for state in ('CC','WW'):
            ids=np.flatnonzero(masks[state])
            for j in ids[np.argsort(table.iloc[ids].obs_id.to_numpy())]:
                if len(picks)<4 and int(j) not in used:
                    picks.append((state,int(j)));used.add(int(j))
        combined=pd.read_csv(crossroot/'table.csv',dtype={'mmsi':str})
        records=input_records(combined,tr.read_json(crossroot/'protocol.json'))
        for pos,(status,j) in enumerate(picks):
            true=int(y[j]); rival=int(after[j].argmax()) if not ac[j] else int(before[j].argmax())
            if rival==true:rival=int(np.where(np.arange(4)==true,-np.inf,before[j]).argmax())
            with torch.no_grad():
                replay=decision(torch.tensor(z['local'][[j]]),np.array([j])).numpy()[0]
                oldlocal_newreadout=decision(torch.tensor(orig['local'][[j]]),np.array([j])).numpy()[0]
            np.testing.assert_allclose(replay,after[j],atol=2e-5,rtol=0)
            margin=lambda s:float(s[true]-s[rival])
            mb,ma,mc=margin(before[j]),margin(after[j]),margin(oldlocal_newreadout)
            vp=np.array([old['delta_vv'][j,true]-old['delta_vv'][j,rival],old['delta_vh'][j,true]-old['delta_vh'][j,rival]])*old['weight_frequency'][j]
            vn=np.array([z['delta_vv'][j,true]-z['delta_vv'][j,rival],z['delta_vh'][j,true]-z['delta_vh'][j,rival]])*z['weight_frequency'][j]
            np.testing.assert_allclose((vn-vp).sum(),ma-mb,atol=2e-5,rtol=1e-5)
            refs=z['refs'][j,[true,rival]].astype(int).ravel()
            rows=[int(orig['query_rows'][j]),*refs.tolist()]
            fig,axes=plt.subplots(2,5,figsize=(15,6.7))
            provenance=[]
            for col,row in enumerate(rows):
                record=records[row]
                for p,v in enumerate(tr.extract.prepare(record)):
                    ax=axes[p,col]
                    if v is not None:
                        ax.imshow(v['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                        if v['masks'][0].any():ax.contour(v['masks'][0],levels=[.5],colors=['#20a780'],linewidths=.6)
                        provenance.append(dict(row=row,pol=('VV','VH')[p],path=record[('vv','vh')[p]]['image'],placement=v['placement']))
                    ax.set_title(f'{("VV","VH")[p]} {("QUERY","TRUE REF 1","TRUE REF 2","RIVAL REF 1","RIVAL REF 2")[col]}\nobs {combined.iloc[row].original_obs_id}',fontsize=9)
                    ax.set(xticks=[],yticks=[])
            filename=f'{source}_{pos}_{status}_{int(table.iloc[j].obs_id)}.png'
            fig.suptitle(f'{direction} seed{seed}, old M3 -> aligned M3; {status}, true {CLASSES[true]}\n{CLASSES[before[j].argmax()]} -> {CLASSES[after[j].argmax()]}; margin {mb:+.4f} -> {ma:+.4f}')
            fig.text(.02,.02,f'Native160 crop/pad; same actual references. Local-change effect {ma-mc:+.4f}; readout-refit effect {mc-mb:+.4f}. Model-level decomposition, not physical causality.',fontsize=8)
            fig.subplots_adjust(top=.80,bottom=.09,hspace=.2,wspace=.1)
            fig.savefig(out/filename,dpi=115);plt.close(fig)
            cases.append(dict(source=source,seed=seed,status=status,obs_id=int(table.iloc[j].obs_id),true=true,rival=rival,
                before_pred=int(before[j].argmax()),after_pred=int(after[j].argmax()),before_margin=mb,after_margin=ma,
                local_effect_fixed_new_readout=ma-mc,readout_effect_fixed_old_local=mc-mb,
                old_polar_contribution=vp.tolist(),new_polar_contribution=vn.tolist(),old_weights=old['weight_frequency'][j].tolist(),new_weights=z['weight_frequency'][j].tolist(),
                old_local=orig['local'][j,[true,rival]].tolist(),new_local=z['local'][j,[true,rival]].tolist(),
                refs_obs_id=combined.iloc[refs].original_obs_id.to_numpy(int).tolist(),image=filename,provenance=provenance))
        # Separate matcher change from subsequent readout change at the whole-cohort level.
        with torch.no_grad():
            counter=np.concatenate([decision(torch.tensor(orig['local'][k:k+48]),np.arange(k,min(k+48,len(y)))).numpy() for k in range(0,len(y),48)])
        for name,score in (('old_local_new_readout',counter),('new_local_new_readout',after),('old_local_old_readout',before)):
            reference_effects.append(dict(source=source,seed=seed,arm=name,**run.m3.metrics(y,score)))
    pd.DataFrame(intervals).to_csv(out/'uncertainty.csv',index=False)
    pd.DataFrame(reference_effects).to_csv(out/'local_readout_cross.csv',index=False)
    tr.write_json(out/'cases.json',cases)
    tr.write_json(out/'receipt.json',dict(raw_review_pending=True,panels=len(cases),actual_references=True,case_seed='first metric step>0 by source selection; otherwise17',case_policy='minimum obsID per correctness state, fill CC/WW to4'))
    print(pd.DataFrame(intervals).to_string(index=False),flush=True)
    print(pd.DataFrame(reference_effects).to_string(index=False),flush=True)


if __name__=='__main__':main()
