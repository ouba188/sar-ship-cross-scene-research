"""Eight-class frozen layer3 reader and identity/pass-isolated OOF caches."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[1]
sys.path.insert(0, str(ROOT/'same_vessel_gallery_20261019/conditional_evidence_pilot'))
from run_pilot import indices
sys.path.insert(0, str(PARENT.parent/'context_strong_reader_20261022'))
from run_strong import View

CLASS_COUNT = 8
SEED = 20261023
SCALE = 3.
GROUPS = np.arange(2048).reshape(16, 128)


def load():
    f = pd.read_csv(PARENT/'cohort.csv', dtype={'mmsi': str})
    # The trusted local extraction stored string IDs in an object array.
    raw = dict(np.load(PARENT/'features.npz', allow_pickle=True))
    np.testing.assert_array_equal(raw['instance_ids'].astype(str), f.instance_id)
    assert f.instance_id.is_unique and f.obs_id.is_unique
    assert len(f) == 15251 and not f.obs_id.eq(31237).any()
    assert set(f.class_id) == set(range(CLASS_COUNT))
    f['pass_group'] = f['product'].map(
        lambda s: '_'.join([s.split('_')[0], s.split('_')[6], s.split('_')[7]]))
    assert raw['valid'].shape == (len(f), 2)
    assert raw['regional_valid'].shape == (len(f), 2, 2)
    for name, width in [('layer3',1024), ('stem',128), ('fg',128), ('bg',128)]:
        assert raw[name].shape == (len(f), width*2)
        assert np.isfinite(raw[name]).all()
        flags = raw['valid'] if name in ('layer3','stem') else raw['regional_valid'][:,:,0 if name == 'fg' else 1]
        for pol in range(2):
            assert not raw[name][~flags[:,pol],pol*width:(pol+1)*width].any()
    return f, raw


def weights(f, fit):
    part = f.iloc[fit]
    count = part.groupby('mmsi').mmsi.transform('size').to_numpy()
    ships = part.groupby('class_id').mmsi.transform('nunique').to_numpy()
    assert set(part.class_id) == set(range(CLASS_COUNT))
    w = len(fit)/(CLASS_COUNT*count*ships)
    np.testing.assert_allclose([w[part.class_id.eq(c)].sum() for c in range(CLASS_COUNT)], len(fit)/CLASS_COUNT)
    return w


def risk(y, s):
    loss = logsumexp(s, axis=1)-s[np.arange(len(y)),y]
    return float(np.mean([loss[y == c].mean() for c in range(CLASS_COUNT)]))


def metrics(y, s):
    pred = s.argmax(1)
    return dict(n=len(y), OA=float((pred == y).mean()*100),
        BA=float(np.mean([(pred[y == c] == c).mean() for c in range(CLASS_COUNT)])*100),
        balanced_NLL=risk(y,s))


def isolated(f, a, b):
    for key in ('mmsi','product','pass_group'):
        assert not set(f.iloc[a][key]) & set(f.iloc[b][key]), key


def prepare_splits(f, raw):
    rows, memberships, assignments, folds = [], [], [], []
    def record(fold, inner, role, ix):
        part = f.iloc[ix]
        rows.append(dict(fold=fold, inner=inner, role=role, n=len(ix),
            mmsi=part.mmsi.nunique(), passes=part.pass_group.nunique(),
            date_min=part.time.min(), date_max=part.time.max(),
            vv_valid=int(raw['valid'][ix,0].sum()), vh_valid=int(raw['valid'][ix,1].sum()),
            **{f'class_{c}':int(part.class_id.eq(c).sum()) for c in range(CLASS_COUNT)}))
        memberships.extend(dict(fold=fold, inner=inner, role=role, row_index=int(i),
            instance_id=f.iloc[i].instance_id) for i in ix)
    for fold in range(3):
        train, val, test = indices(f, fold)
        for a,b in ((train,val),(train,test),(val,test)):
            isolated(f,a,b)
        assert pd.to_datetime(f.iloc[train].time).lt('2025-09-01').all()
        assert pd.to_datetime(f.iloc[val].time).between('2025-09-01','2025-09-30 23:59:59').all()
        assert pd.to_datetime(f.iloc[test].time).ge('2025-10-01').all()
        for role, ix in [('train',train),('validation',val),('test',test)]:
            record(fold,-1,role,ix)
        groups = np.random.default_rng(SEED).permutation(sorted(f.iloc[train].pass_group.unique()))
        alloc = {g:j%4 for j,g in enumerate(groups)}
        bins = f.pass_group.map(alloc).fillna(-1).to_numpy(dtype=np.int64)
        assignments.extend(dict(fold=fold, pass_group=g, inner=b) for g,b in alloc.items())
        inners = []
        for b in range(4):
            held = train[bins[train] == b]
            fit = train[(bins[train] != b) & ~f.iloc[train].mmsi.isin(f.iloc[held].mmsi).to_numpy()]
            isolated(f,fit,held)
            record(fold,b,'fit',fit)
            record(fold,b,'held',held)
            inners.append((fit,held))
        np.testing.assert_array_equal(np.sort(np.concatenate([h for _,h in inners])),train)
        folds.append((train,val,test,inners))
    coverage = pd.DataFrame(rows)
    coverage.to_csv(HERE/'split_coverage.csv',index=False)
    pd.DataFrame(memberships).to_csv(HERE/'split_membership.csv',index=False)
    pd.DataFrame(assignments).to_csv(HERE/'pass_assignment.csv',index=False)
    print(coverage.to_string(index=False),flush=True)
    bad = coverage.loc[coverage[[f'class_{c}' for c in range(CLASS_COUNT)]].min(axis=1).eq(0)]
    if len(bad):
        raise RuntimeError('Actual split lacks a class; see split_coverage.csv. No split changes made.')
    return folds


def projection_parameters(view, prefix=''):
    return {prefix+k:v for k,v in dict(scale_mean=view.scale.mean_, scale_scale=view.scale.scale_,
        pca_mean=view.pca.mean_, pca_components=view.pca.components_,
        post_mean=view.post.mean_, post_scale=view.post.scale_).items()}


def conditional(raw, fit, take):
    # Shared coordinates are learned from whole stem only, without labels.
    scale = StandardScaler().fit(raw['stem'][fit])
    pca = PCA(n_components=64, svd_solver='full').fit(scale.transform(raw['stem'][fit]))
    post = StandardScaler().fit(pca.transform(scale.transform(raw['stem'][fit])))
    projected = {k:post.transform(pca.transform(scale.transform(raw[k][take]))) for k in ('stem','fg','bg')}
    flags = {'stem':raw['valid'][take].any(1),
        'fg':raw['regional_valid'][take,:,0].any(1), 'bg':raw['regional_valid'][take,:,1].any(1)}
    def normalize(z, valid):
        z = z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)*np.sqrt(32.)
        z[~valid] = 0
        return z.astype(np.float32)
    out = dict(whole64=normalize(projected['stem'],flags['stem']),
        fg64=normalize(projected['fg'],flags['fg']))
    out['dual64'] = np.concatenate([normalize(projected[k][:,:32],flags[k]) for k in ('fg','bg')],axis=1)/np.sqrt(2.)
    out['dual64'] = out['dual64'].astype(np.float32)
    for name, valid in [('whole64',flags['stem']),('fg64',flags['fg']),('dual64',flags['fg'] & flags['bg'])]:
        assert np.isfinite(out[name]).all()
        np.testing.assert_allclose(np.linalg.norm(out[name][valid],axis=1), np.sqrt(32.), atol=3e-6)
    params = dict(cond_scale_mean=scale.mean_,cond_scale_scale=scale.scale_,cond_pca_mean=pca.mean_,
        cond_pca_components=pca.components_,cond_post_mean=post.mean_,cond_post_scale=post.scale_)
    return out, params


def linear_cache(h, fit, take, view, head, w):
    W = ((head.coef_.astype(np.float64)/view.post.scale_) @ view.pca.components_.astype(np.float64))/view.scale.scale_*SCALE
    h0 = h[fit].mean(0,dtype=np.float64)
    h0w = np.average(h[fit].astype(np.float64),axis=0,weights=w)
    b = head.predict(view.transform(h0[None]))[0]*SCALE
    bw = head.predict(view.transform(h0w[None]))[0]*SCALE
    direct = head.predict(view.transform(h[take].astype(np.float64)))*SCALE
    score32 = head.predict(view.transform(h[take]))*SCALE
    q = np.stack([(h[take][:,cols].astype(np.float64)-h0[cols]) @ W[:,cols].T for cols in GROUPS],axis=1)
    error = float(np.max(np.abs(q.sum(1)+b-direct)))
    assert error < 1e-7,error
    np.testing.assert_allclose(h[take].astype(np.float64)@W.T+(b-W@h0),direct,atol=1e-7)
    np.testing.assert_allclose(bw,b+W@(h0w-h0),atol=1e-7)
    check = dict(decomposition_max_error=error,float32_score_max_error=float(np.max(np.abs(score32-direct))),
        float32_top1_differences=int((score32.argmax(1)!=direct.argmax(1)).sum()))
    model = dict(W=W,h0=h0,h0_weighted=h0w,b=b,b_weighted=bw,intercept=b-W@h0,
        head_coef=head.coef_,head_intercept=head.intercept_,fit=fit,fit_weights=w,
        groups=GROUPS,**projection_parameters(view))
    return direct,q,model,check


def prediction_frame(f, ix, scores, fold, role, inner=-1):
    out=f.iloc[ix].copy()
    out['row_index']=ix
    out['outer_fold']=fold
    out['role']=role
    out['inner']=inner
    out['prediction']=scores.argmax(1)
    for c in range(CLASS_COUNT):
        out[f's{c}']=scores[:,c]
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    started=time.perf_counter()
    f,raw=load()
    folds=prepare_splits(f,raw)
    if args.preflight:
        return
    (HERE/'models').mkdir(exist_ok=True)
    h,y=raw['layer3'],f.class_id.to_numpy()
    candidates,records,checks,predictions,oof_predictions=[],[],[],[],[]
    allrows=np.arange(len(f))
    for fold,(train,val,test,inners) in enumerate(folds):
        stamp=time.perf_counter()
        view=View(h,train,'layer3')
        z=view.transform(h)
        w=weights(f,train)
        choices=[]
        for alpha in (1.,10.,100.):
            head=Ridge(alpha=alpha).fit(z[train],np.eye(CLASS_COUNT)[y[train]],sample_weight=w)
            loss=risk(y[val],head.predict(view.transform(h[val].astype(np.float64)))*SCALE)
            candidates.append(dict(fold=fold,alpha=alpha,validation_balanced_NLL=loss))
            choices.append((loss,alpha,head))
        loss,alpha,head=min(choices,key=lambda x:x[0])
        base,q,model,check=linear_cache(h,train,allrows,view,head,w)
        cond,params=conditional(raw,train,allrows)
        np.savez_compressed(HERE/'models'/f'fold{fold}_outer.npz',**model,**params,alpha=alpha)
        checks.append(dict(fold=fold,inner=-1,**check))
        oof=np.zeros_like(base)
        oq=np.zeros_like(q)
        ob=np.zeros_like(base)
        ocond={k:np.zeros_like(v) for k,v in cond.items()}
        oof_valid=np.zeros(len(f),bool)
        inner_bin=np.full(len(f),-1,np.int64)
        for inner,(fit,held) in enumerate(inners):
            iv=View(h,fit,'layer3')
            iw=weights(f,fit)
            ih=Ridge(alpha=alpha).fit(iv.transform(h[fit]),np.eye(CLASS_COUNT)[y[fit]],sample_weight=iw)
            s,g,im,ic=linear_cache(h,fit,held,iv,ih,iw)
            cp,pm=conditional(raw,fit,held)
            np.savez_compressed(HERE/'models'/f'fold{fold}_inner{inner}.npz',**im,**pm,held=held,alpha=alpha)
            oof[held],oq[held],ob[held]=s,g,im['b']
            for k in cond:
                ocond[k][held]=cp[k]
            oof_valid[held]=True
            inner_bin[held]=inner
            checks.append(dict(fold=fold,inner=inner,**ic))
            oof_predictions.append(prediction_frame(f,held,s,fold,'source_oof',inner))
            print(f'fold {fold} inner {inner}: fit={len(fit)} held={len(held)} classes=8 error={ic["decomposition_max_error"]:.3g}',flush=True)
        np.testing.assert_array_equal(np.flatnonzero(oof_valid),train)
        np.testing.assert_allclose(oq.sum(1)+ob,oof,atol=1e-7)
        out=dict(instance_ids=f.instance_id.to_numpy(dtype=str),obs_ids=f.obs_id.to_numpy(),
            train=train,validation=val,test=test,base=base,oof=oof,q=q,oq=oq,b=model['b'],ob=ob,
            W=model['W'],h0=model['h0'],h0_weighted=model['h0_weighted'],b_weighted=model['b_weighted'],
            oof_valid=oof_valid,inner_bin=inner_bin,valid=raw['valid'],regional_valid=raw['regional_valid'],
            selected_alpha=alpha,logit_scale=SCALE,class_count=CLASS_COUNT,groups=GROUPS,
            **cond,**{'oof_'+k:v for k,v in ocond.items()})
        assert all(np.isfinite(v).all() for v in out.values() if np.asarray(v).dtype.kind in 'fc')
        np.savez_compressed(HERE/f'fold{fold}_inputs.npz',**out)
        for role,ix,s in [('validation',val,base[val]),('test',test,base[test]),('source_oof',train,oof[train])]:
            records.append(dict(fold=fold,role=role,alpha=alpha,**metrics(y[ix],s)))
            if role!='source_oof':
                predictions.append(prediction_frame(f,ix,s,fold,role))
        print(f'fold {fold} alpha={alpha:g} valNLL={loss:.6f} test={metrics(y[test],base[test])} seconds={time.perf_counter()-stamp:.1f}',flush=True)
        pd.DataFrame(candidates).to_csv(HERE/'alpha_candidates.csv',index=False)
        pd.DataFrame(records).to_csv(HERE/'metrics.csv',index=False)
        pd.DataFrame(checks).to_csv(HERE/'linear_checks.csv',index=False)
        pd.concat(predictions,ignore_index=True).to_csv(HERE/'predictions.csv',index=False)
        pd.concat(oof_predictions,ignore_index=True).to_csv(HERE/'oof_predictions.csv',index=False)
    p=pd.concat(predictions,ignore_index=True)
    t=p[p.role.eq('test')]
    assert t.instance_id.is_unique
    expected=np.flatnonzero(pd.to_datetime(f.time).ge('2025-10-01'))
    np.testing.assert_array_equal(np.sort(t.row_index.to_numpy()),expected)
    pooled=metrics(t.class_id.to_numpy(),t[[f's{c}' for c in range(CLASS_COUNT)]].to_numpy())
    pd.DataFrame([dict(reader='new_frozen_layer3_PCA128_Ridge8',**pooled)]).to_csv(HERE/'pooled.csv',index=False)
    classrows=[]
    for (role,fold,c),part in p.groupby(['role','outer_fold','class_id']):
        classrows.append(dict(role=role,fold=fold,class_id=c,class_name=part.class_name.iloc[0],n=len(part),
            recall=float(part.prediction.eq(c).mean()*100),vh_valid=int(raw['valid'][part.row_index,1].sum())))
    for c,part in t.groupby('class_id'):
        classrows.append(dict(role='test',fold='pooled',class_id=c,class_name=part.class_name.iloc[0],n=len(part),
            recall=float(part.prediction.eq(c).mean()*100),vh_valid=int(raw['valid'][part.row_index,1].sum())))
    pd.DataFrame(classrows).to_csv(HERE/'class_metrics.csv',index=False)
    t.assign(correct=t.prediction.eq(t.class_id)).groupby(['port','pass_group']).agg(n=('correct','size'),
        correct=('correct','sum'),OA=('correct',lambda x:x.mean()*100)).reset_index().to_csv(HERE/'acquisition_metrics.csv',index=False)
    coverage=f.groupby(['class_id','class_name']).agg(n=('instance_id','size'),mmsi=('mmsi','nunique')).reset_index()
    coverage['vv_valid']=[int(raw['valid'][f.class_id.eq(c),0].sum()) for c in coverage.class_id]
    coverage['vh_valid']=[int(raw['valid'][f.class_id.eq(c),1].sum()) for c in coverage.class_id]
    coverage.to_csv(HERE/'cohort_coverage.csv',index=False)
    summary=dict(rows=len(f),mmsi=f.mmsi.nunique(),products=f['product'].nunique(),passes=f.pass_group.nunique(),
        class_count=CLASS_COUNT,vv_valid=int(raw['valid'][:,0].sum()),vh_valid=int(raw['valid'][:,1].sum()),
        evaluated_test_rows=len(t),test_mmsi=t.mmsi.nunique(),test_vh_valid=int(raw['valid'][t.row_index,1].sum()),
        outer_models=3,inner_models=12,ridge_fits_including_candidates=21,attention_heads=0,
        seconds=time.perf_counter()-started,pooled=pooled,
        max_decomposition_error=max(c['decomposition_max_error'] for c in checks),
        float32_top1_differences=sum(c['float32_top1_differences'] for c in checks),
        evaluation='New frozen layer3 baseline; not old A+D. Float32 fit, float64 linear-map evaluation.',
        oof='Source-only shuffled passes seed20261023 roundrobin4; held identities excluded; inner refit all preprocessing.',
        projection='Whole-stem-only StandardScaler/PCA64(full)/postscale; whole64 and fg64 L2 sqrt32; dual first32 per region /sqrt2.',
        missing='Invalid polarization raw block stays zero; any-valid-polarization region retained; flags saved.')
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
