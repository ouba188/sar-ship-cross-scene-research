"""Fit new layer1 directions on inherited source splits, never refit old stages."""
from common import *
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.special import logsumexp


def main():
    tick=time.perf_counter()
    table,valid,_=load_data()
    with np.load(HERE/'features_source.npz') as z:
        np.testing.assert_array_equal(z['instance_ids'],table.instance_id)
        h=z['layer1'].astype(float); computed=z['computed']
    roles=readers(); y=table.class_id.to_numpy()
    out=HERE/'readers'; out.mkdir(exist_ok=True)
    candidates=[]; checks=[]
    for r in roles:
        fit=r['fit']; assert computed[fit].all()
        part=table.iloc[fit]
        assert set(part.class_id)==set(range(8))
        if r['held'] is not None:
            for key in ['mmsi','product','pass_group']:
                assert not set(part[key]) & set(table.iloc[r['held']][key])
        scale=StandardScaler().fit(h[fit])
        pca=PCA(n_components=128,svd_solver='randomized',random_state=17,iterated_power=4).fit(scale.transform(h[fit]))
        post=StandardScaler().fit(pca.transform(scale.transform(h[fit])))
        def project(x): return post.transform(pca.transform(scale.transform(x)))
        weight=len(fit)/(8*part.groupby('mmsi').mmsi.transform('size').to_numpy()*part.groupby('class_id').mmsi.transform('nunique').to_numpy())
        zz=project(h[fit]); target=np.eye(8)[y[fit]]
        if r['held'] is None:
            va=splits(r['fold'])['validation']; assert computed[va].all()
            choices=[]
            for alpha in [1.,10.,100.]:
                head=Ridge(alpha=alpha).fit(zz,target,sample_weight=weight)
                score=head.predict(project(h[va]))*3
                losses=logsumexp(score,axis=1)-score[np.arange(len(va)),y[va]]
                risk=float(np.mean([losses[y[va]==c].mean() for c in range(8)]))
                candidates.append(dict(reader=r['name'],alpha=alpha,validation_balanced_NLL=risk))
                choices.append((risk,alpha,head))
            _,alpha,head=min(choices,key=lambda x:x[0])
            replay=va
        else:
            with np.load(out/f"fold{r['fold']}_outer.npz") as z: alpha=float(z['alpha'])
            head=Ridge(alpha=alpha).fit(zz,target,sample_weight=weight)
            replay=r['held']
        W=((head.coef_/post.scale_)@pca.components_)/scale.scale_*3
        intercept=head.predict(project(np.zeros((1,h.shape[1]))))[0]*3
        direct=head.predict(project(h[replay]))*3
        error=float(abs(h[replay]@W.T+intercept-direct).max())
        assert error<1e-8
        params=dict(W=W,intercept=intercept,fit=fit,alpha=alpha,fit_weights=weight,
            scale_mean=scale.mean_,scale_scale=scale.scale_,pca_mean=pca.mean_,pca_components=pca.components_,
            post_mean=post.mean_,post_scale=post.scale_,head_coef=head.coef_,head_intercept=head.intercept_)
        if r['held'] is not None: params['held']=r['held']
        np.savez_compressed(out/f"{r['name']}.npz",**params)
        checks.append(dict(reader=r['name'],fit=len(fit),replay=len(replay),alpha=alpha,max_error=error,classes=8))
        print(r['name'],'alpha',alpha,'replay',error,flush=True)
    pd.DataFrame(candidates).to_csv(HERE/'reader_alpha_candidates.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'reader_checks.csv',index=False)
    save_json(HERE/'readers_complete.json',dict(completed_utc=now(),readers=15,seconds=time.perf_counter()-tick,new_layer1=True,old_stage_fits=0))


if __name__=='__main__': main()
