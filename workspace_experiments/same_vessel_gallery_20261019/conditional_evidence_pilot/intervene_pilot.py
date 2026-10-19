"""Replay selected fits and intervene on conditions without refitting the head."""
import run_pilot as p
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits


def main():
    f=pd.read_csv(p.ROOT/'cohort.csv',dtype={'mmsi':str})
    old=pd.read_csv(p.ROOT/'predictions.csv')
    selected=pd.read_csv(p.ROOT/'metrics.csv')
    y=f.class_id.to_numpy(); outputs=[]; decomposed=[]
    for fold in range(3):
        train,val,test=p.indices(f,fold)
        x=p.image_features(f,train,fold)
        c=p.standardize(f[['cos2','sin2','incidence']].to_numpy(float),train)
        rng=np.random.default_rng(p.SEED+100+fold)
        rng.normal(size=(x.shape[1],3))
        wrong=c.copy()
        for ii in (train,val,test):
            for _,part in f.iloc[ii].groupby(['port','platform','quarter']):
                jj=part.index.to_numpy(); wrong[jj]=c[rng.permutation(jj)]
        raw=p.product_features(x,c)
        mean=raw[train].mean(0); std=np.maximum(raw[train].std(0),1e-6)
        def transform(z): return (z-mean)/std/np.sqrt(raw.shape[1])
        alpha=selected.query('fold==@fold and arm=="interaction" and partition=="test"').iloc[0].alpha
        model=RidgeClassifier(alpha=alpha,solver='cholesky').fit(transform(raw)[train],y[train],sample_weight=p.weights(f,train))
        noangle=c.copy(); noangle[:,:2]=0
        noincidence=c.copy(); noincidence[:,2]=0
        variants={'actual':c,'mean_all':np.zeros_like(c),'mean_angle':noangle,'mean_incidence':noincidence,'mismatch':wrong}
        for name,cond in variants.items():
            z=transform(p.product_features(x[test],cond[test])); scores=model.decision_function(z)
            pred=scores.argmax(1); others=scores.copy(); others[np.arange(len(test)),y[test]]=-np.inf
            margins=scores[np.arange(len(test)),y[test]]-others.max(1)
            if name=='actual':
                reference=old.query('fold==@fold and arm=="interaction"').set_index('obs_id').loc[f.iloc[test].obs_id]
                assert np.array_equal(reference.prediction.to_numpy(),pred)
                assert np.allclose(reference.margin.to_numpy(),margins,atol=1e-6)
                rivals=others.argmax(1)
                differences=model.coef_[y[test]]-model.coef_[rivals]
                intercept=model.intercept_[y[test]]-model.intercept_[rivals]
                blocks=[(z[:,a:b]*differences[:,a:b]).sum(1) for a,b in ((0,48),(48,51),(51,195))]
                assert np.allclose(intercept+sum(blocks),margins,atol=1e-10)
                for k,i in enumerate(test):
                    decomposed.append(dict(obs_id=int(f.iloc[i].obs_id),fold=fold,true_class=int(y[i]),rival=int(rivals[k]),intercept=intercept[k],image_main=blocks[0][k],condition_main=blocks[1][k],interaction=blocks[2][k],margin=margins[k]))
            for k,i in enumerate(test):
                outputs.append(dict(obs_id=int(f.iloc[i].obs_id),fold=fold,variant=name,mmsi=f.iloc[i].mmsi,class_id=int(y[i]),prediction=int(pred[k]),correct=bool(pred[k]==y[i]),margin=margins[k]))
            print(fold,name,p.metrics(y[test],scores),flush=True)
    out=pd.DataFrame(outputs); out.to_csv(p.ROOT/'fixed_condition_predictions.csv',index=False)
    pd.DataFrame(decomposed).to_csv(p.ROOT/'score_decomposition.csv',index=False)
    summaries=[]
    for name,g in out.groupby('variant'):
        summaries.append(dict(variant=name,n=len(g),oa=g.correct.mean(),ba=g.groupby('class_id').correct.mean().mean()))
    summary=pd.DataFrame(summaries); summary.to_csv(p.ROOT/'fixed_condition_summary.csv',index=False)
    print(summary.to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
