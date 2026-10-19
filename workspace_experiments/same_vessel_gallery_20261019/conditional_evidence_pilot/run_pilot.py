"""Fixed-feature diagnostic: geometry prior versus content-conditioned evidence."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parent.parent/'same_vessel_condition_study_20261010'
FEATURE_ROOT=Path('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
R18=Path('E:/Hermes/paper_tracking/sensitivity/neural_complement_pilot_20261007/frozen_replay/source_extra.npy')
ARMS=('image','condition_only','concat','interaction','content_interaction','mismatched_interaction')
SEED=20261019


def dump(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def standardize(x,train):
    return (x-x[train].mean(0))/np.maximum(x[train].std(0),1e-6)


def cohort():
    source=pd.read_csv(STUDY/'observations.csv',dtype={'mmsi':str}).set_index('obs_id')
    text=(ROOT.parent/'gallery_data.js').read_text(encoding='utf-8')
    d=json.loads(text.removeprefix('window.GALLERY_DATA=').rstrip(';\n'))
    rows=[]
    for o in d['observations'].values():
        p=o['images']['vv']['priors']; a=p.get('ship_axis')
        if not a or p.get('radar_kind')!='orbit_los' or p.get('radar_incidence') is None: continue
        if p['radar_axis'] is None or not np.isfinite(p['radar_incidence']): continue
        original=source.loc[o['id']]
        assert original.mmsi==o['mmsi'] and original.class_id==o['class_id'] and original.instance_id==o['instance_id']
        delta=a['angle']-p['radar_axis']
        rows.append(dict(obs_id=o['id'],instance_id=o['instance_id'],manifest_row=int(original.manifest_row),mmsi=o['mmsi'],class_id=o['class_id'],class_name=o['category'],fold=int(original.fold),port=o['port'],product=o['product'],time=o['time'],platform=o['platform'],cos2=np.cos(2*delta),sin2=np.sin(2*delta),incidence=float(p['radar_incidence']),axis_deg=a['angle']*180/np.pi,los_axis_deg=p['radar_axis']*180/np.pi,elongation=a['length_px']/a['width_px'],match_status=p['match_status'],image_vv=original.image_relpath_vv,image_vh=original.image_relpath_vh))
    f=pd.DataFrame(rows).sort_values('obs_id').reset_index(drop=True)
    f['quarter']=pd.to_datetime(f.time).dt.quarter
    f.to_csv(ROOT/'cohort.csv',index=False)
    dump('COHORT_SCOPE.json',dict(n=len(f),ships=f.mmsi.nunique(),products=f['product'].nunique(),classes=f.class_name.value_counts().to_dict(),ports=f.port.value_counts().to_dict(),geometry='Exact-product annotation orbit LOS; axial relative encoding; annotation incidence',gallery_summary=d['summary'],read_time=time.strftime('%Y-%m-%dT%H:%M:%S')))
    return f


def indices(f,fold):
    identities=f.loc[f.fold.ne(fold),['mmsi','class_id']].drop_duplicates()
    rng=np.random.default_rng(SEED+fold)
    valships=set()
    for _,part in identities.groupby('class_id'):
        ids=np.sort(part.mmsi.to_numpy()); valships.update(rng.permutation(ids)[:max(1,round(.2*len(ids)))])
    dates=pd.to_datetime(f.time)
    train=np.flatnonzero(f.fold.ne(fold)&~f.mmsi.isin(valships)&dates.lt('2025-09-01'))
    val=np.flatnonzero(f.mmsi.isin(valships)&dates.ge('2025-09-01')&dates.lt('2025-10-01'))
    test=np.flatnonzero(f.fold.eq(fold)&dates.ge('2025-10-01'))
    for a,b in ((train,val),(train,test),(val,test)):
        assert not set(f.iloc[a].mmsi)&set(f.iloc[b].mmsi)
        assert not set(f.iloc[a]['product'])&set(f.iloc[b]['product'])
    assert set(f.iloc[train].class_id)==set(range(8))
    return train,val,test


def image_features(f,train,fold):
    blocks=[]; rows=f.manifest_row.to_numpy()
    for path,rank in ((FEATURE_ROOT/'pooled_features.npy',32),(R18,16)):
        raw=np.load(path,mmap_mode='r'); assert len(raw)==38091
        x=np.array(raw[rows],dtype=np.float32)
        x=standardize(x,train)
        pca=PCA(rank,svd_solver='randomized',random_state=SEED+fold,iterated_power=3).fit(x[train])
        blocks.append(standardize(pca.transform(x),train))
    return np.concatenate(blocks,axis=1)


def weights(f,train):
    part=f.iloc[train]
    count=part.groupby('mmsi').mmsi.transform('size').to_numpy()
    nships=part.groupby('class_id').mmsi.transform('nunique').to_numpy()
    return len(train)/(8*count*nships)


def metrics(y,scores):
    pred=scores.argmax(1); prob=softmax(scores,axis=1)
    recalls=[np.mean(pred[y==c]==c) for c in range(8) if np.any(y==c)]
    return dict(n=len(y),classes=len(recalls),oa=float(np.mean(pred==y)),ba=float(np.mean(recalls)),brier=float(np.mean(np.sum((prob-np.eye(8)[y])**2,axis=1))))


def product_features(x,c):
    return np.concatenate((x,c,(x[:,:,None]*c[:,None,:]).reshape(len(x),-1)),axis=1)


def main():
    started=time.perf_counter(); f=cohort(); y=f.class_id.to_numpy(); results=[]; predictions=[]; selection=[]; coverage=[]; changed=[]; partitions=[]; class_coverage=[]
    for fold in range(3):
        train,val,test=indices(f,fold)
        for name,ii in (('train',train),('val',val),('test',test)):
            part=f.iloc[ii]
            coverage.append(dict(fold=fold,partition=name,n=len(ii),ships=part.mmsi.nunique(),products=part['product'].nunique(),classes=part.class_id.nunique(),class_counts=json.dumps(part.class_id.value_counts().to_dict())))
            for c,g in part.groupby('class_id'):
                class_coverage.append(dict(fold=fold,partition=name,class_id=c,n=len(g),ships=g.mmsi.nunique(),products=g['product'].nunique()))
            partitions.extend(dict(fold=fold,partition=name,obs_id=int(i)) for i in part.obs_id)
        print('fold',fold,'sizes',len(train),len(val),len(test),flush=True)
        x=image_features(f,train,fold)
        c=standardize(f[['cos2','sin2','incidence']].to_numpy(float),train)
        rng=np.random.default_rng(SEED+100+fold)
        h=standardize(np.tanh(x@rng.normal(size=(x.shape[1],3))/np.sqrt(x.shape[1])),train)
        wrong=c.copy(); donor=np.arange(len(f))
        for name,ii in (('train',train),('val',val),('test',test)):
            singleton=0
            for _,part in f.iloc[ii].groupby(['port','platform','quarter']):
                jj=part.index.to_numpy(); donor[jj]=rng.permutation(jj); wrong[jj]=c[donor[jj]]
                if len(jj)==1: singleton+=1
            delta=np.deg2rad(f.axis_deg.to_numpy()-f.los_axis_deg.to_numpy())
            gaps=np.abs((delta[ii]-delta[donor[ii]]+np.pi/2)%np.pi-np.pi/2)*180/np.pi
            igap=np.abs(f.incidence.to_numpy()[ii]-f.incidence.to_numpy()[donor[ii]])
            changed.append(dict(fold=fold,partition=name,changed_fraction=float(np.mean(np.max(np.abs(wrong[ii]-c[ii]),axis=1)>1e-6)),donor_changed_fraction=float(np.mean(donor[ii]!=ii)),singleton_fraction=singleton/len(ii),axis_gap_median=float(np.median(gaps)),axis_gap_mean=float(gaps.mean()),incidence_gap_median=float(np.median(igap)),incidence_gap_mean=float(igap.mean())))
        matrices=dict(image=x,condition_only=c,concat=np.concatenate((x,c),axis=1),interaction=product_features(x,c),content_interaction=product_features(x,h),mismatched_interaction=product_features(x,wrong))
        w=weights(f,train)
        for arm,features in matrices.items():
            features=standardize(features,train)/np.sqrt(features.shape[1])
            candidates=[]
            for alpha in (.1,1.,10.):
                model=RidgeClassifier(alpha=alpha,solver='cholesky').fit(features[train],y[train],sample_weight=w)
                m=metrics(y[val],model.decision_function(features[val])); candidates.append((m,alpha,model))
                selection.append(dict(fold=fold,arm=arm,alpha=alpha,**m))
            m,alpha,model=max(candidates,key=lambda v:(v[0]['ba'],-v[0]['brier']))
            for name,ii in (('train',train),('val',val),('test',test)):
                scores=model.decision_function(features[ii]); results.append(dict(fold=fold,arm=arm,partition=name,alpha=alpha,dimension=features.shape[1],**metrics(y[ii],scores)))
                if name=='test':
                    p=scores.argmax(1); other=scores.copy(); other[np.arange(len(ii)),y[ii]]=-np.inf
                    margin=scores[np.arange(len(ii)),y[ii]]-other.max(1)
                    prob=softmax(scores,axis=1); risks=((prob-np.eye(8)[y[ii]])**2).sum(1)
                    for k,i in enumerate(ii):
                        predictions.append(dict(fold=fold,arm=arm,obs_id=int(f.iloc[i].obs_id),mmsi=f.iloc[i].mmsi,port=f.iloc[i].port,product=f.iloc[i]['product'],class_id=int(y[i]),prediction=int(p[k]),correct=bool(p[k]==y[i]),margin=float(margin[k]),brier=float(risks[k])))
            print(fold,arm,'alpha',alpha,'test',results[-1]['oa'],results[-1]['ba'],flush=True)
        pd.DataFrame(results).to_csv(ROOT/'metrics.csv',index=False)
        pd.DataFrame(predictions).to_csv(ROOT/'predictions.csv',index=False)
        pd.DataFrame(selection).to_csv(ROOT/'validation_selection.csv',index=False)
        pd.DataFrame(coverage).to_csv(ROOT/'coverage.csv',index=False)
        pd.DataFrame(class_coverage).to_csv(ROOT/'class_coverage.csv',index=False)
        pd.DataFrame(partitions).to_csv(ROOT/'partitions.csv',index=False)
        pd.DataFrame(changed).to_csv(ROOT/'mismatch_coverage.csv',index=False)
    dump('RECEIPT.json',dict(status='complete',seconds=time.perf_counter()-started,folds=3,arms=list(ARMS),neural_training=False,scope='Frozen-representation supervised probe; no target support adaptation; not a causal or full-model claim'))


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
