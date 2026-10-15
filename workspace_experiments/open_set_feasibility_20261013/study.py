"""Frozen-feature class-holdout feasibility, not a new open-set neural module."""

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
WORK=OUT.parent
ARCH=Path('E:/Hermes/paper_tracking/sensitivity')
SOURCE=WORK/'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916'
sys.path.insert(0,str(ARCH/'oof_reliability_check'))
from unified_head_core import coefficients
PORTS=('Los Angeles','Port Klang','Santos')
SEED=20261013
METHODS=('msp','source_knn','support_knn','msp_support')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def load(path,key):
    with np.load(path,allow_pickle=False) as z:return z[key]


def ids(values):
    return pd.Series(values).astype(str).str.replace(r'\.0$','',regex=True).to_numpy()


def clean(frame, identity_col, id_col):
    d=frame.copy();d['identity']=ids(d[identity_col]);d['sample_id']=d[id_col].astype(str)
    good=d.identity.str.fullmatch(r'[1-9][0-9]{8}').fillna(False)
    bad=d[good].groupby('identity').class_id.nunique();bad=set(bad[bad>1].index)
    d['eligible']=good&~d.identity.isin(bad)
    return d


def inputs():
    source=clean(pd.read_csv(SOURCE/'manifest.csv',dtype={'mmsi':str},keep_default_na=False),'mmsi','instance_id')
    source['feature_row']=np.arange(len(source))
    raw=np.load(SOURCE/'pooled_features.npy',mmap_mode='r')
    c4=np.load(ARCH/'c4_base_trial_20261009/pooled_c4.npy',mmap_mode='r')
    extra=np.load(ARCH/'neural_complement_pilot_20261007/frozen_replay/source_extra.npy',mmap_mode='r')
    permitted=np.arange(len(source));tasks={}
    for port in PORTS:
        cache=ARCH/'learned_frequency_residual_20261009/tasks/self24ports'/port/'target.npz'
        permitted=np.intersect1d(permitted,load(cache,'source_rows'))
        rows=load(cache,'rows')
        frame=source.iloc[rows].copy().reset_index(drop=True)
        tasks['ports/'+port]=dict(frame=frame,arrays=(raw,c4,extra))
    for name in ('f0921','oss2'):
        old=ARCH/('f0921_transfer_20261007' if name=='f0921' else 'opensarship2_transfer_20261007')
        permitted=np.intersect1d(permitted,load(old/'source_split.npz','source_rows'))
        target=ARCH/('f0921_detector_recrop_20261007' if name=='f0921' else 'opensarship2_transfer_20261007')
        idcol='mmsi' if name=='f0921' else 'mmsi_or_ship_id'
        frame=clean(pd.read_csv(target/'target_manifest.csv',dtype={idcol:str},keep_default_na=False),idcol,'sample_id')
        frame['feature_row']=np.arange(len(frame))
        if name=='f0921':
            r=np.load(target/'paired_r50.npy',mmap_mode='r');e=np.load(target/'vh_ssl.npy',mmap_mode='r')
        else:
            r=load(target/'features.npz','paired_r50');e=load(target/'features.npz','vh_ssl')
            np.testing.assert_array_equal(load(target/'features.npz','sample_ids'),frame.sample_id)
        c=np.load(ARCH/'cross_dataset_c4_20261009'/name/'pooled_c4.npy',mmap_mode='r')
        np.testing.assert_array_equal(np.load(ARCH/'cross_dataset_c4_20261009'/name/'sample_ids.npy'),frame.sample_id)
        tasks[name]=dict(frame=frame,arrays=(r,c,e))
    target_ids=set().union(*(set(t['frame'].loc[t['frame'].eligible,'identity']) for t in tasks.values()))
    allowed=source.eligible&~source.identity.isin(target_ids)&~source.port.isin(PORTS)
    permitted=permitted[allowed.iloc[permitted].to_numpy()]
    rng=np.random.default_rng(SEED);chosen=[]
    for cl in range(8):
        group=source.iloc[permitted];group=group[group.class_id==cl]
        ships=rng.permutation(group.identity.unique())[:512]
        chosen.extend(int(rng.choice(group.index[group.identity==s])) for s in ships)
    chosen=np.array(sorted(chosen),int)
    assert len(np.unique(source.class_id.iloc[chosen]))==8
    np.save(OUT/'source_rows.npy',chosen)
    source.iloc[chosen].to_csv(OUT/'source_manifest.csv',index=False)
    enrollment={};capacity=[]
    for name,t in tasks.items():
        frame=t['frame'];good=frame[frame.eligible];rng=np.random.default_rng(SEED)
        ships=np.sort(good.identity.unique());shuffle=rng.permutation(ships)
        support_ids=set(shuffle[:int(np.ceil(.3*len(shuffle)))])
        pools=good[good.identity.isin(support_ids)];queries=good[~good.identity.isin(support_ids)]
        q=[]
        for cl in range(8):
            sub=queries[queries.class_id==cl]
            selected=rng.permutation(sub.identity.unique())[:128]
            q.extend(int(rng.choice(sub.index[sub.identity==s])) for s in selected)
        q=np.array(sorted(q),int);t['query']=q
        trials=[]
        for unknown in range(8):
            pool=pools[pools.class_id!=unknown]
            usable=len(pool.identity.unique())>=64 and (frame.class_id.iloc[q]==unknown).any()
            capacity.append(dict(task=name,unknown=unknown,query_unknown=int((frame.class_id.iloc[q]==unknown).sum()),
                                 query_known=int((frame.class_id.iloc[q]!=unknown).sum()),support_ships=pool.identity.nunique(),usable=usable))
            if not usable:continue
            supports=[]
            for draw in range(5):
                rr=np.random.default_rng(SEED+unknown*101+draw)
                sids=rr.choice(pool.identity.unique(),64,replace=False)
                s=np.array([int(rr.choice(pool.index[pool.identity==s])) for s in sids])
                assert not set(frame.identity.iloc[s])&set(frame.identity.iloc[q])
                assert not (frame.class_id.iloc[s]==unknown).any()
                supports.append(s.tolist())
            trials.append(dict(unknown=unknown,supports=supports))
        enrollment[name]=dict(query=q.tolist(),trials=trials)
        dest=OUT/'tasks'/name;dest.mkdir(parents=True,exist_ok=True)
        union=np.unique(np.r_[q,np.array([s for tr in trials for s in tr['supports']]).ravel()])
        t['union']=union
        frame.iloc[union].assign(local_row=union).to_csv(dest/'manifest.csv',index=False)
    write(OUT/'ENROLLMENT.json',enrollment)
    pd.DataFrame(capacity).to_csv(OUT/'capacity.csv',index=False)
    write(OUT/'SOURCE_SCOPE.json',dict(source_n=len(chosen),classes=source.iloc[chosen].class_id.value_counts().sort_index().to_dict(),
        tasks=list(tasks),known_source_target_identity_overlap=0,source_ports=sorted(source.iloc[chosen].port.unique()),
        frozen_encoder_exposure_not_reset=True,compact_reader_not_full_reference=True))
    return source,chosen,(raw,c4,extra),tasks,enrollment


def raw_take(arrays,frame,rows):
    ix=frame.feature_row.to_numpy()[rows]
    return tuple(np.asarray(a[ix],np.float32) for a in arrays)


def encode(arrays,model):
    r,c,e=arrays
    x=(model['pca'].transform(r)-model['center'])/model['scale']
    x4=(model['pca'].transform(c)-model['center'])/model['scale']
    xe=np.c_[x,model['ssl'].transform(e)]
    xe=(xe-model['ec'])/model['es']
    return dict(x=np.asarray(xe,float),x4=np.asarray(x4,float),
        pe=model['expert'].decision_function(xe),pb=model['base'].decision_function(x4),
        k=np.c_[xe/np.sqrt(192),np.ones(len(r))],
        b=np.c_[x4/np.sqrt(128),np.ones(len(r))],
        direction=xe/np.maximum(np.linalg.norm(xe,axis=1,keepdims=True),1e-12))


def fit(arrays,y,unknown):
    take=np.ones(len(y),bool) if unknown<0 else y!=unknown
    r,c,e=(a[take] for a in arrays);yy=y[take]
    pca=PCA(n_components=128,svd_solver='randomized',random_state=0).fit(r)
    ssl=PCA(n_components=64,svd_solver='randomized',random_state=0).fit(e)
    x=pca.transform(r);center=x.mean(0);scale=x.std(0)+1e-6
    xx=(x-center)/scale;x4=(pca.transform(c)-center)/scale
    joint=np.c_[xx,ssl.transform(e)];ec=joint.mean(0);es=joint.std(0)+1e-6
    joint=(joint-ec)/es
    expert=RidgeClassifier(alpha=1.,class_weight='balanced').fit(joint,yy)
    base=RidgeClassifier(alpha=1.,class_weight='balanced').fit(x4,yy)
    classes=np.arange(8) if unknown<0 else np.delete(np.arange(8),unknown)
    np.testing.assert_array_equal(expert.classes_,classes)
    model=dict(pca=pca,ssl=ssl,center=center,scale=scale,ec=ec,es=es,expert=expert,base=base,classes=classes)
    source=encode((r,c,e),model)
    rng=np.random.default_rng(SEED);bank=[]
    for cl in classes:
        pool=np.flatnonzero(yy==cl);bank.extend(rng.choice(pool,min(128,len(pool)),replace=False))
    model['source_bank']=source['direction'][bank]
    model['fit_class_counts']={int(c):int((yy==c).sum()) for c in classes}
    return model


def nearest(q,s,k):
    d=np.maximum(0.,1.-q@s.T)
    return np.partition(d,k-1,axis=1)[:,:k].mean(1)


def predict(f,s,q,labels,classes,loo=False):
    yy=np.searchsorted(classes,labels)
    np.testing.assert_array_equal(classes[yy],labels)
    kb=f['b'][s]@f['b'][s].T;ke=f['k'][s]@f['k'][s].T
    ab=coefficients(kb,f['pb'][s],yy,8.)
    ae=coefficients(ke,f['pe'][s],yy,8.)
    result=.5*softmax(f['pb'][q]+(f['b'][q]@f['b'][s].T)@ab,axis=1)
    result+=.5*softmax(f['pe'][q]+(f['k'][q]@f['k'][s].T)@ae,axis=1)
    if not loo:return result
    calibration=[]
    for i in range(len(s)):
        keep=np.arange(len(s))!=i
        a=coefficients(kb[np.ix_(keep,keep)],f['pb'][s[keep]],yy[keep],8.)
        z=coefficients(ke[np.ix_(keep,keep)],f['pe'][s[keep]],yy[keep],8.)
        calibration.append(.5*softmax(f['pb'][s[i]]+kb[i,keep]@a)
                           +.5*softmax(f['pe'][s[i]]+ke[i,keep]@z))
    return result,np.stack(calibration)


def anomaly(f,s,q,prob,cal,bank):
    dq=nearest(f['direction'][q],f['direction'][s],3)
    distances=np.maximum(0.,1.-f['direction'][s]@f['direction'][s].T)
    np.fill_diagonal(distances,np.inf)
    ds=np.partition(distances,2,axis=1)[:,:3].mean(1)
    aq=np.c_[1.-prob.max(1),nearest(f['direction'][q],bank,5),dq]
    ac=np.c_[1.-cal.max(1),nearest(f['direction'][s],bank,5),ds]
    mean=ac[:,[0,2]].mean(0);scale=np.maximum(ac[:,[0,2]].std(0),1e-6)
    return np.c_[aq,((aq[:,[0,2]]-mean)/scale).mean(1)],np.c_[ac,((ac[:,[0,2]]-mean)/scale).mean(1)]


def main():
    started=time.perf_counter()
    source,rows,arr,tasks,enrollment=inputs()
    source_values=raw_take(arr,source,rows);y=source.class_id.to_numpy()[rows]
    checkpoints=OUT/'fits';checkpoints.mkdir(exist_ok=True)
    full=fit(source_values,y,-1)
    with (checkpoints/'eight_class.pkl').open('wb') as h:pickle.dump(full,h)
    full_targets={}
    for name,t in tasks.items():
        values=raw_take(t['arrays'],t['frame'],t['union'])
        t['values']=values;full_targets[name]=encode(values,full)
    controls=[];closed=[];support_rows=[];partial=[]
    for unknown in range(8):
        model=fit(source_values,y,unknown)
        with (checkpoints/f'without_{unknown}.pkl').open('wb') as h:pickle.dump(model,h)
        for name,t in tasks.items():
            trial=next((tr for tr in enrollment[name]['trials'] if tr['unknown']==unknown),None)
            if trial is None:continue
            f=encode(t['values'],model);union=t['union'];q=np.searchsorted(union,t['query'])
            yy=t['frame'].class_id.to_numpy();query_y=yy[t['query']]
            all_prob=[];all_scores=[];all_cal=[];all_cal_scores=[];all_thresholds=[];all_eight=[]
            for draw,support in enumerate(trial['supports']):
                s=np.searchsorted(union,support);sy=yy[support]
                prob,cal=predict(f,s,q,sy,model['classes'],loo=True)
                scores,cs=anomaly(f,s,q,prob,cal,model['source_bank'])
                thresholds=np.quantile(cs,[.90,.95],axis=0)
                p8=predict(full_targets[name],s,q,sy,full['classes'])
                all_prob.append(prob);all_cal.append(cal);all_scores.append(scores);all_cal_scores.append(cs)
                all_thresholds.append(thresholds);all_eight.append(p8)
                support_rows.append(dict(task=name,unknown=unknown,draw=draw,classes=len(np.unique(sy)),
                    counts=np.bincount(sy,minlength=8).tolist(),rows=support))
                if draw==0:
                    # Independent primal weighted-ridge check of one real leave-one-out case.
                    from sklearn.linear_model import Ridge
                    keep=np.arange(len(s))!=0;xs=f['k'][s[keep]];ys=np.searchsorted(model['classes'],sy[keep])
                    counts=np.bincount(ys,minlength=7);weights=len(ys)/(np.count_nonzero(counts)*counts[ys])
                    target=2*np.eye(7)[ys]-1-f['pe'][s[keep]]
                    primal=Ridge(alpha=8.,fit_intercept=False).fit(xs,target,sample_weight=weights)
                    alpha=coefficients(xs@xs.T,f['pe'][s[keep]],ys,8.)
                    a=primal.predict(f['k'][s[:1]])[0];b=f['k'][s[0]]@xs.T@alpha
                    np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-8)
                    controls.append(dict(task=name,unknown=unknown,primal_max_error=float(abs(a-b).max())))
            dest=OUT/'tasks'/name/f'unknown_{unknown}.npz'
            np.savez_compressed(dest,probability=np.array(all_prob),score=np.array(all_scores),
                calibration_probability=np.array(all_cal),calibration_score=np.array(all_cal_scores),
                thresholds=np.array(all_thresholds),eight_probability=np.array(all_eight),
                query=t['query'],labels=query_y,classes=model['classes'],
                supports=np.array(trial['supports']),support_labels=yy[np.array(trial['supports'])],
                methods=np.array(METHODS))
            print('FROZEN',name,'unknown',unknown,'query',len(q),'sec',round(time.perf_counter()-started,1),flush=True)
        write(OUT/'PROGRESS.json',dict(last_unknown=unknown,seconds=time.perf_counter()-started))
    pd.DataFrame(controls).to_csv(OUT/'primal_checks.csv',index=False)
    write(OUT/'support_manifest.json',support_rows)
    write(OUT/'PREDICTIONS_FROZEN.json',dict(tasks=len(tasks),contexts=len(support_rows),neural_training=False,
        source_readout_fits=18,source_geometries=9,seconds=time.perf_counter()-started,
        threshold='Known target support LOO 90/95 percentiles only',query_label_selection=False))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
