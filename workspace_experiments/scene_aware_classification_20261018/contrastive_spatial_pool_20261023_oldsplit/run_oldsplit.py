"""Old EXP split: fixed-layer3 reader and prescribed spatial std replication."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys
sys.dont_write_bytecode=True
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models import resnet50

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
EXP=BASE/'context_expansion_20261022'
STRONG=BASE/'context_strong_reader_20261022'
CHECKPOINT=Path('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')
sys.path.insert(0,str(STRONG))
import run_strong as strong
sys.path.insert(0,str(BASE/'contrastive_spatial_pool_20261023'))
import run_pool as reference
metrics=reference.metrics
ARMS=['contrast_std']
LAMBDAS=[0.,.1,.25,.5,1.]
REVIEW_IDS=['pair_0032624','pair_0053959','pair_0030906','pair_0030918']

def pool(v):
    r=v-v.mean(-2,keepdim=True)
    return r.std(-1,correction=0).unsqueeze(-2),r

def prepare(t):
    started=time.perf_counter()
    deep=np.load(STRONG/'deep_features.npz')
    assert np.array_equal(deep['instance_ids'],t.instance_id)
    raw=deep['layer3']
    historical=pd.read_csv(STRONG/'baseline_predictions.csv')
    historical=historical[historical.kind.eq('layer3')]
    chosen=pd.read_csv(STRONG/'baseline_candidates.csv')
    chosen=chosen[chosen.kind.eq('layer3')].set_index('fold')
    y=t.class_id.to_numpy(); ids=t.mmsi.to_numpy()
    folds=[]; origins=[]; checks=[]; candidates=[]
    for f in range(6):
        split=np.load(EXP/f'fold{f}_inputs.npz')
        train,val,test=[split[k] for k in ['train','validation','test']]
        for a,b in [(train,val),(train,test),(val,test)]:
            assert not set(ids[a])&set(ids[b])
            assert not set(t.iloc[a].pass_group)&set(t.iloc[b].pass_group)
        view=strong.View(raw,train,'layer3')
        z=view.transform(raw)
        best=None
        for alpha in [1.,10.,100.]:
            head=strong.fit_head(z,y,ids,train,alpha)
            score=head.predict(z)*3
            vr=strong.risk(y[val],score[val])
            candidates.append(dict(fold=f,alpha=alpha,val_balanced_NLL=vr))
            if best is None or vr<best[0]:
                best=vr,alpha,head,score
        vr,alpha,head,score=best
        assert alpha==chosen.loc[f,'alpha']
        assert abs(vr-chosen.loc[f,'val_nll'])<1e-5
        old=historical[historical.fold.eq(f)].set_index('instance_id').loc[t.iloc[test].instance_id]
        expected=old[['s0','s1','s2']].to_numpy()
        err=float(abs(score[test]-expected).max())
        top_changes=int(np.sum(score[test].argmax(1)!=old.prediction.to_numpy()))
        assert err<3e-4 and top_changes==0,(f,err,top_changes)
        W=((head.coef_.astype(float)/view.post.scale_)@view.pca.components_.astype(float))/view.scale.scale_*3
        # Full affine backprojection, including both scaler means and PCA centering.
        offset=(-view.scale.mean_/view.scale.scale_-view.pca.mean_)@view.pca.components_.astype(float).T
        intercept=3*(head.intercept_.astype(float)+((offset-view.post.mean_)/view.post.scale_)@head.coef_.astype(float).T)
        affine=raw.astype(float)@W.T+intercept
        affine_error=float(abs(affine-score).max())
        assert affine_error<1e-4
        base=score.astype(float)
        base[test]=expected
        origin=dict(W=W,raw_intercept=intercept,h0=raw[train].mean(0,dtype=float),fit=train)
        frozen=dict(base=base,train=train,validation=val,test=test,alpha=alpha,instance_ids=t.instance_id.to_numpy(str))
        np.savez_compressed(HERE/f'fold{f}_reader.npz',**frozen,**origin)
        folds.append(frozen);origins.append(origin)
        checks.append(dict(fold=f,alpha=alpha,source_n=len(train),validation_n=len(val),test_n=len(test),
            score_max_error=err,top1_changes=top_changes,affine_max_error=affine_error,
            validation_risk_error=float(abs(vr-chosen.loc[f,'val_nll'])),test_A='exact historical fixed-layer3 scores'))
        print('reader verified',checks[-1],flush=True)
    coverage=np.bincount(np.concatenate([z['test'] for z in folds]),minlength=len(t))
    assert (coverage==1).all()
    pooled=metrics(historical.class_id.to_numpy(),historical[['s0','s1','s2']].to_numpy())
    assert abs(pooled['OA']-81.51009657594382)<1e-10
    assert abs(pooled['BA']-80.12973869349547)<1e-10
    pd.DataFrame(checks).to_csv(HERE/'baseline_verification.csv',index=False)
    pd.DataFrame(candidates).to_csv(HERE/'alpha_candidates.csv',index=False)
    pd.DataFrame([pooled]).to_csv(HERE/'historical_baseline.csv',index=False)
    print('BASELINE REPRODUCED',pooled,flush=True)
    return folds,origins,time.perf_counter()-started

@torch.inference_mode()
def extract(t,folds,origins):
    started=time.perf_counter()
    deep=np.load(STRONG/'deep_features.npz')
    assert np.array_equal(deep['instance_ids'],t.instance_id)
    oldh=deep['layer3']
    h=np.zeros_like(oldh)
    delta=np.zeros((6,len(t),1,3),dtype=np.float64)
    counts=np.zeros((len(t),2),int)
    weights=np.stack([o['W'].reshape(3,2,1024).transpose(1,0,2) for o in origins])
    w=torch.tensor(weights,dtype=torch.float64,device='cuda')
    model=resnet50(weights=None)
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
    model=model.cuda().float().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    buckets=defaultdict(list)
    for i,row in t.iterrows():
        for p,pol in enumerate(['vv','vh']):
            path=EXP/'readout/native_targets'/f'{row.instance_id}_{pol}.png'
            with Image.open(path) as im:
                width,height=im.size
                assert im.mode=='L'
            assert [width,height]==json.loads(row[f'target_window_{pol}'])[2:]
            buckets[(height,width)].append((i,p,path))
    saved_maps={}
    verify_rows=[];gap_rows=[];batch=0
    last=time.perf_counter()
    for shape,items in sorted(buckets.items(),key=lambda kv:(-len(kv[1]),kv[0])):
        for start in range(0,len(items),16):
            chunk=items[start:start+16]
            indices=np.array([x[0] for x in chunk]);polarizations=np.array([x[1] for x in chunk])
            pixels=[]
            for _,_,path in chunk:
                with Image.open(path) as im:
                    pixels.append(np.array(im,copy=True))
            x=torch.tensor(np.stack(pixels),dtype=torch.float32,device='cuda')[:,None].expand(-1,3,-1,-1)/255
            x=(x-mean)/std
            wp=w[:,torch.tensor(polarizations,device='cuda')]
            pooled_features=[]
            for rotation in range(4):
                z=x.rot90(rotation,(-2,-1)).contiguous()
                z=model.maxpool(model.relu(model.bn1(model.conv1(z))))
                z=model.layer3(model.layer2(model.layer1(z)))
                pooled_features.append(z.mean((-2,-1)))
                response=torch.einsum('fbck,bkp->fbcp',wp,z.flatten(2).double())
                v=response-response.mean(-1,keepdim=True)
                values,r=pool(v)
                row=dict(batch=batch,rotation=rotation,spatial_mean_error=float(v.mean(-1).abs().max()),class_mean_error=float(r.mean(-2).abs().max()))
                assert max(row["spatial_mean_error"],row["class_mean_error"])<1e-10
                verify_rows.append(row)
                values=values.cpu().numpy()
                # A shape bucket can contain both polarizations of the same target.
                for f in range(6):
                    np.add.at(delta[f],indices,values[f]/4)
                for b,(i,p,_) in enumerate(chunk):
                    ident=t.iloc[i].instance_id
                    if ident in REVIEW_IDS:
                        f=next(f for f,zf in enumerate(folds) if i in zf['test'])
                        prefix=f'{ident}_{["vv","vh"][p]}_rot{rotation}'
                        saved_maps[prefix+'_v']=v[f,b].cpu().numpy().reshape(3,*z.shape[-2:])
                        saved_maps[prefix+'_r']=r[f,b].cpu().numpy().reshape(3,*z.shape[-2:])
                        saved_maps[prefix+'_delta']=values[f,b]
                        saved_maps[prefix+'_native_shape']=np.array(shape)
                        saved_maps[prefix+'_fold']=np.array(f)
            features=torch.stack(pooled_features).mean(0).cpu().numpy()
            for b,(i,p,path) in enumerate(chunk):
                cols=slice(p*1024,(p+1)*1024)
                actual,expected=features[b],oldh[i,cols]
                np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=5e-5)
                h[i,cols]=actual;counts[i,p]+=1
                gap_rows.append(dict(instance_id=t.iloc[i].instance_id,polarization=['vv','vh'][p],height=shape[0],width=shape[1],
                    max_abs_error=float(abs(actual-expected).max()),rmse=float(np.sqrt(np.mean((actual-expected)**2))),
                    top_feature_same=bool(actual.argmax()==expected.argmax())))
            batch+=1
            if time.perf_counter()-last>20:
                print(json.dumps(dict(images=int(counts.sum()),total=2*len(t),batches=batch,seconds=time.perf_counter()-started)),flush=True)
                last=time.perf_counter()
    assert (counts==1).all() and np.isfinite(delta).all()
    pd.DataFrame(gap_rows).to_csv(HERE/'gap_verification.csv',index=False)
    pd.DataFrame(verify_rows).to_csv(HERE/'invariants.csv',index=False)
    np.savez_compressed(HERE/'delta_features.npz',instance_ids=t.instance_id.to_numpy(str),arms=np.array(ARMS),delta=delta)
    np.savez_compressed(HERE/'validation_maps.npz',**saved_maps)
    reader=[]
    for f,(frozen,origin) in enumerate(zip(folds,origins)):
        cached_affine=oldh.astype(float)@origin['W'].T+origin['raw_intercept']
        fresh_affine=h.astype(float)@origin['W'].T+origin['raw_intercept']
        for role in ['validation','test']:
            ix=frozen[role]
            err=float(abs(fresh_affine[ix]-frozen['base'][ix]).max())
            # Fail on material map/normalization mismatch; do not replace A.
            assert err<1e-4
            reader.append(dict(fold=f,role=role,n=len(ix),max_cached_affine_error=float(abs(cached_affine[ix]-frozen['base'][ix]).max()),
                max_new_affine_error=err,max_new_vs_cached_affine_error=float(abs(fresh_affine[ix]-cached_affine[ix]).max()),
                new_top1_changes=int(np.sum(fresh_affine[ix].argmax(1)!=frozen['base'][ix].argmax(1)))))
    pd.DataFrame(reader).to_csv(HERE/'reader_verification.csv',index=False)
    return delta,dict(extract_seconds=time.perf_counter()-started,shape_buckets=len(buckets),batches=batch,
        gap_max_abs_error=max(r['max_abs_error'] for r in gap_rows),max_reader_error=max(r['max_new_affine_error'] for r in reader),
        reader_top1_changes=sum(r['new_top1_changes'] for r in reader),gpu_peak_bytes=torch.cuda.max_memory_allocated())


def main():
    start=time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')
    t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    assert len(t)==5695
    folds,origins,prepare_seconds=prepare(t)
    if '--prepare-only' in sys.argv:
        return
    delta,cost=extract(t,folds,origins)
    reference.HERE=HERE
    reference.ARMS=ARMS
    reference.LAMBDAS=LAMBDAS
    selection_seconds=reference.select_evaluate(t,folds,delta)
    summary=dict(status='complete',targets=len(t),mmsi=t.mmsi.nunique(),passes=t.pass_group.nunique(),
        classes=3,arms=ARMS,lambdas=LAMBDAS,population_std=True,reader_refits=18,
        reader_kind='fixed layer3, not source-selected mixed reader',source_only_selection=True,
        prepare_seconds=prepare_seconds,selection_seconds=selection_seconds,
        total_seconds=time.perf_counter()-start,checkpoint=str(CHECKPOINT),
        baseline_test_scores='historical exact',gpu=torch.cuda.get_device_name(0),**cost)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
