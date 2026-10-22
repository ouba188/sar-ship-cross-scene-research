"""Separate sample-specific class dispersion from fixed class offsets and amplitude."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
from run_pool import EXP,NEW,LAMBDAS,metrics
import analyze_pool

ARMS=['full_std','source_mean','amplitude_only']


def main():
    t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    cached=np.load(PARENT/'delta_features.npz')
    np.testing.assert_array_equal(cached['instance_ids'],t.instance_id)
    all_delta=cached['delta'][:,:,list(cached['arms']).index('contrast_std')]
    y=t.class_id.to_numpy()
    candidates,choices,records,stats=[],[],[],[]
    fixed_records=[]
    original=pd.read_csv(PARENT/'predictions.csv')
    for fold in range(6):
        f=dict(np.load(NEW/f'fold{fold}_inputs.npz'))
        train,val,test=[f[k] for k in ['train','validation','test']]
        raw=all_delta[fold]
        delta=raw-raw.mean(1,keepdims=True)
        mu=delta[train].mean(0)
        length=np.linalg.norm(delta,axis=1)
        amp=length/max(float(length[train].mean()),1e-12)
        values={'full_std':delta,'source_mean':np.broadcast_to(mu,delta.shape),'amplitude_only':amp[:,None]*mu}
        cosine=(delta@mu)/np.maximum(length*np.linalg.norm(mu),1e-12)
        selected_full=float(original.loc[original.arm.eq('contrast_std')&original.fold.eq(fold),'selected_lambda'].iloc[0])
        for arm,d in values.items():
            score=f['base'][test]+selected_full*d[test]
            for j,i in enumerate(test):
                fixed_records.append(dict(instance_id=t.iloc[i].instance_id,fold=fold,arm=arm,class_id=int(y[i]),
                    pass_group=t.iloc[i].pass_group,mmsi=t.iloc[i].mmsi,selected_lambda=selected_full,
                    base_prediction=int(f['base'][i].argmax()),prediction=int(score[j].argmax()),
                    **{f's{k}':float(score[j,k]) for k in range(3)}))
        stats.append(dict(fold=fold,source_mean_norm=float(np.linalg.norm(mu)),source_mean_amplitude=float(length[train].mean()),
            source_cosine_mean=float(cosine[train].mean()),validation_cosine_mean=float(cosine[val].mean()),test_cosine_mean=float(cosine[test].mean())))
        for arm,d in values.items():
            options=[]
            for a in LAMBDAS:
                rr=metrics(y[val],f['base'][val]+a*d[val])['balanced_NLL']
                candidates.append(dict(fold=fold,arm=arm,coefficient=a,validation_NLL=rr))
                options.append((rr,a))
            best=options[0]
            for option in options[1:]:
                if option[0]<best[0]-1e-12:best=option
            rr,a=best
            score=f['base'][test]+a*d[test]
            if arm=='full_std':
                ref=original[original.arm.eq('contrast_std')&original.fold.eq(fold)].set_index('instance_id').loc[t.iloc[test].instance_id]
                assert a==ref.selected_lambda.iloc[0]
                np.testing.assert_array_equal(score.argmax(1),ref.prediction)
            choices.append(dict(fold=fold,arm=arm,selected_lambda=a,validation_NLL=rr))
            for j,i in enumerate(test):
                pred=int(score[j].argmax()); bp=int(f['base'][i].argmax());truth=int(y[i])
                net=int(pred==truth)-int(bp==truth)
                outcome='rescued' if net==1 else 'harmed' if net==-1 else 'CC' if pred==truth else 'WW'
                records.append(dict(t.iloc[i],fold=fold,arm=arm,selected_lambda=a,prediction=pred,base_prediction=bp,net=net,outcome=outcome,
                    **{f's{k}':float(score[j,k]) for k in range(3)},**{f'b{k}':float(f['base'][i,k]) for k in range(3)},**{f'delta{k}':float(d[i,k]) for k in range(3)}))
    p=pd.DataFrame(records)
    p.to_csv(HERE/'predictions.csv',index=False)
    pd.DataFrame(candidates).to_csv(HERE/'candidates.csv',index=False)
    pd.DataFrame(choices).to_csv(HERE/'selections.csv',index=False)
    pd.DataFrame(stats).to_csv(HERE/'source_shape_statistics.csv',index=False)
    pooled=[]
    for arm,g in p.groupby('arm'):
        pooled.append(dict(arm=arm,net=int(g.net.sum()),**metrics(g.class_id.to_numpy(),g[['s0','s1','s2']].to_numpy())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    fixed=pd.DataFrame(fixed_records)
    fixed.to_csv(HERE/'fixed_full_lambda_predictions.csv',index=False)
    rows=[]
    for arm,g in fixed.groupby('arm'):
        net=g.prediction.eq(g.class_id).astype(int)-g.base_prediction.eq(g.class_id).astype(int)
        rows.append(dict(arm=arm,net=int(net.sum()),**metrics(g.class_id.to_numpy(),g[['s0','s1','s2']].to_numpy())))
    pd.DataFrame(rows).to_csv(HERE/'fixed_full_lambda_pooled.csv',index=False)
    p.groupby(['arm','class_id']).agg(n=('net','size'),net=('net','sum')).to_csv(HERE/'class_effects.csv')
    analyze_pool.HERE=HERE
    analyze_pool.ARMS=ARMS
    analyze_pool.REVIEW_IDS=[]
    panels,missing=analyze_pool.render(t,p)
    print(pd.DataFrame(pooled).to_string(index=False))
    print('panels',panels,'missing',missing)


if __name__=='__main__':main()
