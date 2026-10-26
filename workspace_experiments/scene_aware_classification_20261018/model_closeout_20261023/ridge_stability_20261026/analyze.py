"""Frozen reverse-transfer robustness comparisons, not seed or refit robustness for ridge."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'paper_evidence_completion_20261026'))
import run_classical as base
from analyze_transfer import input_records, CLASSES
TRANSFER=base.TRANSFER
CROSS=TRANSFER/'oss2-to-oss1'
REPAIR=TRANSFER/'normalization_repair_20261026/oss2-to-oss1'
NEW=ROOT/'raw_region_bypass_20261026'


def main():
    table=base.tr.read_table(CROSS/'target_table.csv',labels=True)
    y=table.class_id.to_numpy(int)
    prior=np.load(ROOT/'paper_evidence_completion_20261026/classical_fit_only/oss2/cross_predictions.npz')
    np.testing.assert_array_equal(prior['obs_id'],table.obs_id)
    ridge=prior['regional_ridge']
    scores={'ridge':ridge[None], 'raw_mlp':np.stack([prior[f'regional_mlp_seed{s}'] for s in base.SEEDS])}
    added=np.load(NEW/'oss2/cross_predictions.npz')
    np.testing.assert_array_equal(added['obs_id'],table.obs_id)
    scores['raw_summary']=np.stack([added[f'raw_summary_seed{s}'] for s in base.SEEDS])
    old,new=[],[]
    for seed in base.SEEDS:
        a=np.load(REPAIR/f'seed{seed}/predictions.npz')
        b=np.load(NEW/f'downstream/raw_summary/oss2-to-oss1/seed{seed}/cross_predictions.npz')
        np.testing.assert_array_equal(a['obs_id'],table.obs_id)
        np.testing.assert_array_equal(b['obs_id'],table.obs_id)
        old.append(a['frequency']); new.append(b['frequency'])
    scores['original_M123']=np.stack(old)
    scores['raw_summary_M3']=np.stack(new)
    metrics=[]
    correctness={k:(v.argmax(-1)==y[None]).astype(float) for k,v in scores.items()}
    for name,stack in scores.items():
        for i,s in enumerate(stack):
            metrics.append(dict(model=name,seed='deterministic' if name=='ridge' else base.SEEDS[i],**base.classification_metrics(y,s)))
    frame=pd.DataFrame(metrics)
    frame.to_csv(HERE/'metrics.csv',index=False)
    frame.groupby('model')[['OA','BA','NLL','Brier']].agg(['mean','std']).to_csv(HERE/'summary.csv')
    classes,groups,intervals,leave,paired=[],[],[],[],[]
    rc=correctness['ridge'][0]
    table.groupby('class_id').agg(n=('obs_id','size'),ships=('mmsi','nunique'),passes=('pass_group','nunique')).to_csv(HERE/'composition.csv')
    for name,correct in correctness.items():
        if name=='ridge':continue
        avg=correct.mean(0)
        d=rc-avg
        for seed,cs in zip(base.SEEDS,correct):
            paired.append(dict(model=name,seed=seed,ridge_only_correct=int(((rc==1)&(cs==0)).sum()),model_only_correct=int(((rc==0)&(cs==1)).sum()),ridge_minus_model_OA_pp=100*(rc-cs).mean(),ridge_minus_model_BA_pp=100*np.mean([(rc-cs)[y==c].mean() for c in range(4)])))
        for c in range(4):
            ids=y==c
            classes.append(dict(model=name,class_name=CLASSES[c],n=int(ids.sum()),ridge_recall=100*rc[ids].mean(),model_recall=100*avg[ids].mean(),difference_pp=100*d[ids].mean()))
        for group,ix in table.groupby('pass_group').indices.items():
            groups.append(dict(model=name,pass_group=group,n=len(ix),classes=len(np.unique(y[ix])),ridge_OA=100*rc[ix].mean(),model_OA=100*avg[ix].mean(),difference_pp=100*d[ix].mean()))
            keep=~table.pass_group.eq(group).to_numpy()
            complete=len(np.unique(y[keep]))==4
            leave.append(dict(model=name,removed=group,removed_n=len(ix),n=int(keep.sum()),ridge_minus_model_OA_pp=100*d[keep].mean(),all_four_classes=complete,ridge_minus_model_BA_pp=100*np.mean([d[keep&(y==c)].mean() for c in range(4)]) if complete else None))
        for field in ('pass_group','mmsi'):
            ix=list(table.groupby(field).indices.values())
            count=np.array([np.bincount(y[t],minlength=4) for t in ix])
            sums=np.array([np.bincount(y[t],weights=d[t],minlength=4) for t in ix])
            # Two separate cluster sensitivities; neither is a new training fold.
            rng=np.random.default_rng(1026)
            sample=rng.integers(len(ix),size=(4000,len(ix)))
            ct=count[sample].sum(1); sm=sums[sample].sum(1)
            valid=(ct>0).all(1)
            oa=100*sm.sum(1)/ct.sum(1)
            ba=100*(sm[valid]/ct[valid]).mean(1)
            intervals.append(dict(model=name,cluster=field,clusters=len(ix),OA_pp=100*d.mean(),OA_low=np.quantile(oa,.025),OA_high=np.quantile(oa,.975),BA_pp=100*np.mean([d[y==c].mean() for c in range(4)]),BA_low=np.quantile(ba,.025),BA_high=np.quantile(ba,.975),complete_draws=int(valid.sum()),draws=4000))
    for name,rows in [('classwise',classes),('passwise',groups),('uncertainty',intervals),('leave_one_pass',leave),('paired',paired)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    # Native cases compare the latest full candidate with ridge, keeping seed17 fixed.
    combined=pd.read_csv(CROSS/'table.csv',dtype={'mmsi':str})
    records=input_records(combined,base.tr.read_json(CROSS/'protocol.json'))
    cache=base.tr.load_npz(REPAIR/'seed17/predictions.npz')
    model=scores['raw_summary_M3'][0]
    mc=model.argmax(1)==y
    cases=[]
    for status,mask in dict(ridge_wins=(rc==1)&~mc,model_wins=(rc==0)&mc,CC=(rc==1)&mc,WW=(rc==0)&~mc).items():
        ids=np.flatnonzero(mask)
        if not len(ids):continue
        # Ascending observation ID is reproducible and avoids comparing differently scaled logits.
        j=int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy())])
        true=int(y[j])
        rival=int(model[j].argmax()) if model[j].argmax()!=true else int(ridge[j].argmax())
        if rival==true:
            rival=int(np.where(np.arange(4)==true,-np.inf,model[j]).argmax())
        rows=[int(cache['query_rows'][j]),*cache['refs'][j,[true,rival]].astype(int).ravel().tolist()]
        fig,axes=plt.subplots(2,5,figsize=(15,6.7))
        prov=[]
        for col,row in enumerate(rows):
            record=records[row]
            for p,v in enumerate(base.tr.extract.prepare(record)):
                ax=axes[p,col]
                if v is not None:
                    ax.imshow(v['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    if v['masks'][0].any():ax.contour(v['masks'][0],levels=[.5],colors=['#20a780'],linewidths=.6)
                    prov.append(dict(row=row,pol=('VV','VH')[p],path=record[('vv','vh')[p]]['image'],placement=v['placement']))
                ax.set_title(f'{("VV","VH")[p]} {("QUERY","TRUE 1","TRUE 2","RIVAL 1","RIVAL 2")[col]}\nobs {combined.iloc[row].original_obs_id}',fontsize=9)
                ax.set(xticks=[],yticks=[])
        filename=f'{status}_{int(table.iloc[j].obs_id)}.png'
        fig.suptitle(f'OSS2 -> OSS1: {status}; true {CLASSES[true]}\nRidge {CLASSES[ridge[j].argmax()]}; raw_summary + M3 seed17 {CLASSES[model[j].argmax()]}')
        fig.text(.02,.02,'Native160 crop/pad. References belong to M3 only; ridge does not retrieve them. No pixel intervention.',fontsize=9)
        fig.subplots_adjust(top=.80,bottom=.09,hspace=.2,wspace=.1)
        fig.savefig(HERE/filename,dpi=115);plt.close(fig)
        cases.append(dict(status=status,obs_id=int(table.iloc[j].obs_id),true=true,ridge_pred=int(ridge[j].argmax()),model_pred=int(model[j].argmax()),image=filename,ridge_scores=ridge[j].tolist(),model_scores=model[j].tolist(),refs_obs_id=combined.iloc[rows[1:]].original_obs_id.to_numpy(int).tolist(),provenance=prov))
    base.write_json(HERE/'cases.json',cases)
    base.write_json(HERE/'PROTOCOL.json',dict(no_training=True,ridge='fixed source-selected alpha1000, deterministic lsqr',comparison='paired same921queries; 3seed effects averaged, not prediction ensemble',clusters='pass and MMSI assessed separately, not simultaneous two-way bootstrap',scope='conditional test-cohort sensitivity, not source resampling/new-fold/seed stability of ridge',case_selection='seed17, ascending obsID in each disagreement state',raw_review_pending=True))
    print(frame.groupby('model')[['OA','BA']].agg(['mean','std']).to_string())
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__=='__main__':main()
