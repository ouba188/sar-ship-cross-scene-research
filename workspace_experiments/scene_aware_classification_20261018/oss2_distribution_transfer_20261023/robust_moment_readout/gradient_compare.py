"""Compare fixed original and asinh heads' source gradient energy, CPU only."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
OSS=HERE.parent
import run_robust as experiment
original=experiment.original
sys.path.insert(0,str(OSS/'geometry_coverage_diagnosis'))
from gradient_diagnose import gradients, FOCUS


def main():
    started=time.perf_counter()
    assert (HERE/'selection_frozen.json').exists()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table=pd.read_csv(OSS/'cohort.csv',dtype={'mmsi':str})
    with np.load(OSS/'moments.npz') as z:cache={k:z[k] for k in z.files}
    rows,summaries,classes,focus=[],[],[],[]
    for fold in range(6):
        d,_,_,om,od,scale=experiment.load_fold(fold,table,cache)
        tr=d['train'];f=table.iloc[tr].reset_index(drop=True)
        y=torch.tensor(f.class_id.to_numpy(),dtype=torch.long)
        counts=torch.bincount(y,minlength=8)
        coefficient=.5/len(tr)+.5/(8*counts[y].float())
        for seed in experiment.SEEDS:
            with np.load(OSS/'calibrated'/f'fold{fold}_seed{seed}.npz') as z:b=z['train_base'][tr]
            base=torch.tensor(b,dtype=torch.float32)
            for arm in experiment.ARMS:
                ck,selected=original.restore(HERE/'heads'/f'fold{fold}_{arm}_seed{seed}.pt')
                np.testing.assert_array_equal(ck['moment_scale'],scale)
                x,carrier,mu,sd=original.descriptor(arm,b,om[tr],od[tr],cache['valid'][tr],ck['mu'],ck['scale'])
                _,_,fm,fs=original.descriptor(arm,b,om[tr],od[tr],cache['valid'][tr])
                np.testing.assert_array_equal(mu,fm);np.testing.assert_array_equal(sd,fs)
                torch.manual_seed(seed)
                initial=original.make_model(arm).eval()
                with torch.no_grad():assert not original.delta(initial,arm,x,carrier).any()
                if ck['epoch']==0:
                    for k,v in initial.state_dict().items():assert torch.equal(v,ck['state'][k])
                for state,model in [('initial',initial),('selected',selected)]:
                    ce,score,energy,projection,branch,check=gradients(model,arm,x,base,y,coefficient)
                    share=energy/energy.sum();order=np.argsort(-energy,kind='stable')
                    ranks=np.empty(len(f),int);ranks[order]=np.arange(1,len(f)+1)
                    g=f[['sample_id','class_id','class_name','mmsi','pass_group']].copy()
                    meta=dict(fold=fold,seed=seed,arm=arm,state=state,selected_epoch=ck['epoch'])
                    for k,v in meta.items():g[k]=v
                    g['jointCE_coefficient']=coefficient.numpy();g['CE']=ce
                    g['weighted_loss']=ce*coefficient.numpy()
                    g['B_prediction']=base.argmax(1).numpy();g['prediction']=score.argmax(1)
                    g['gradient_sq'],g['gradient_sq_share'],g['gradient_rank']=energy,share,ranks
                    g['signed_aggregate_projection']=projection
                    for k,v in branch.items():g[k+'_gradient_sq']=v
                    rows.append(g)
                    maximum=g.iloc[order[0]]
                    summaries.append(dict(**meta,source_n=len(f),max_id=maximum.sample_id,
                        max_class=int(maximum.class_id),max_share=float(share[order[0]]),
                        max_CE=float(maximum.CE),max_B_prediction=int(maximum.B_prediction),
                        top5_share=float(share[order[:5]].sum()),
                        top1pct_share=float(share[order[:int(np.ceil(.01*len(f)))]].sum()),
                        effective_contributing_samples=float(1/(share*share).sum()),**check))
                    hit=g[g.sample_id.eq(FOCUS)]
                    focus.append(dict(hit.iloc[0],in_source=True) if len(hit) else
                        dict(**meta,sample_id=FOCUS,in_source=False))
                    for c,sub in g.groupby('class_id'):
                        classes.append(dict(**meta,class_id=c,n=len(sub),CE=float(sub.CE.mean()),
                            weighted_loss=float(sub.weighted_loss.sum()),
                            gradient_sq_share=float(sub.gradient_sq_share.sum())))
                print('gradient',fold,seed,arm,'done',flush=True)
    samples=pd.concat(rows,ignore_index=True)
    samples.to_csv(HERE/'gradient_samples.csv',index=False)
    summary=pd.DataFrame(summaries)
    summary.to_csv(HERE/'gradient_summary.csv',index=False)
    pd.DataFrame(classes).to_csv(HERE/'gradient_classes.csv',index=False)
    pd.DataFrame(focus).to_csv(HERE/'gradient_005904.csv',index=False)
    finalize(samples, summary, time.perf_counter()-started)


def finalize(samples, summary, gradient_seconds=None):
    olddir=OSS/'geometry_coverage_diagnosis'
    cols=['sample_id','fold','seed','arm','state','class_id','selected_epoch',
          'jointCE_coefficient','CE','weighted_loss','B_prediction','prediction',
          'gradient_sq','gradient_sq_share','gradient_rank','signed_aggregate_projection']
    old=pd.read_csv(olddir/'gradient_samples.csv',usecols=cols)
    keys=['sample_id','fold','seed','arm','state']
    paired=samples[cols].merge(old,on=keys,suffixes=('_robust','_original'),validate='one_to_one')
    assert len(paired)==len(samples)==len(old)
    for key in ['class_id','B_prediction']:
        np.testing.assert_array_equal(paired[key+'_robust'],paired[key+'_original'])
    # CSV round-trips float32 with short decimals; compare at the training dtype.
    np.testing.assert_array_equal(paired['jointCE_coefficient_robust'].to_numpy(np.float32),
                                  paired['jointCE_coefficient_original'].to_numpy(np.float32))
    paired.to_csv(HERE/'gradient_paired_original.csv',index=False)
    oldsummary=pd.read_csv(olddir/'gradient_summary.csv')
    statekeys=['fold','seed','arm','state']
    neff=old.groupby(statekeys).gradient_sq_share.agg(lambda a:1/(a*a).sum()).reset_index(name='effective_contributing_samples')
    oldsummary=oldsummary.merge(neff,on=statekeys,validate='one_to_one')
    compared=summary.merge(oldsummary,on=statekeys,suffixes=('_robust','_original'),validate='one_to_one')
    for key in ['max_share','top5_share','top1pct_share','effective_contributing_samples','sum_individual_gradient_sq','full_batch_gradient_sq']:
        compared['change_'+key]=compared[key+'_robust']-compared[key+'_original']
    compared.to_csv(HERE/'gradient_comparison.csv',index=False)
    pd.read_csv(HERE/'gradient_classes.csv').merge(pd.read_csv(olddir/'gradient_classes.csv'),on=statekeys+['class_id'],
        suffixes=('_robust','_original'),validate='one_to_one').to_csv(HERE/'gradient_class_comparison.csv',index=False)
    receipt=dict(heads=36,states=72,device='cpu',trained_steps=0,gradient_seconds=gradient_seconds,
        max_gradient_relative_error=float(summary.summed_gradient_relative_l2_error.max()),
        max_loss_error=float(summary.weighted_loss_replay_error.max()),
        original_gradient_source=str(olddir/'gradient_samples.csv'),
        original_source_rows_and_B_and_weights_match=True)
    (HERE/'gradient_summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
