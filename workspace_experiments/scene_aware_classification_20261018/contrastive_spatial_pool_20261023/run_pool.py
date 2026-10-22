"""Frozen mean-preserving spatial pooling; no training, fitting or test tuning."""
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
from scipy.special import logsumexp
from sklearn.metrics import f1_score

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
EXP=BASE/'context_expansion_20261022'
STRONG=BASE/'context_strong_reader_20261022'
NEW=BASE/'target_anchored_context_20261023/partition_replication'
CHECKPOINT=Path('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')
ARMS=['class_residual','contrast_residual','contrast_std']
LAMBDAS=[0.,.1,.25,.5,1.]
REVIEW_IDS=['pair_0032624','pair_0053959','pair_0030906','pair_0030918']


def metrics(y,score):
    score=np.asarray(score,dtype=np.float64)
    pred=score.argmax(1)
    logp=score-logsumexp(score,axis=1,keepdims=True)
    nll=-logp[np.arange(len(y)),y]
    brier=((np.exp(logp)-np.eye(3)[y])**2).sum(1)
    return dict(OA=100*float(np.mean(pred==y)),
        BA=100*float(np.mean([np.mean(pred[y==c]==c) for c in np.unique(y)])),
        macro_F1=100*float(f1_score(y,pred,labels=range(3),average='macro',zero_division=0)),
        NLL=float(nll.mean()),balanced_NLL=float(np.mean([nll[y==c].mean() for c in np.unique(y)])),
        Brier=float(brier.mean()),balanced_Brier=float(np.mean([brier[y==c].mean() for c in np.unique(y)])),
        **{f'recall_{c}':100*float(np.mean(pred[y==c]==c)) if np.any(y==c) else np.nan for c in range(3)})


def pool(v):
    r=v-v.mean(-2,keepdim=True)
    return torch.stack([(v.softmax(-1)*v).sum(-1),
                        (r.softmax(-1)*r).sum(-1),r.std(-1,correction=0)],dim=-2),r


def numpy_pool(v):
    r=v-v.mean(-2,keepdims=True)
    def soft(x):
        return np.exp(x-logsumexp(x,axis=-1,keepdims=True))
    return np.stack([(soft(v)*v).sum(-1),(soft(r)*r).sum(-1),r.std(-1,ddof=0)],axis=-2)


def verify(v,batch,rotation):
    pooled,r=pool(v)
    row=dict(batch=batch,rotation=rotation,v_spatial_mean=float(v.mean(-1).abs().max()),
        r_spatial_mean=float(r.mean(-1).abs().max()),r_class_mean=float(r.mean(-2).abs().max()),
        uniform_delta=float(torch.stack([v.mean(-1),r.mean(-1)]).abs().max()))
    assert max(row[k] for k in ['v_spatial_mean','r_spatial_mean','r_class_mean','uniform_delta'])<1e-10
    return row


def special_checks(v):
    v=v[:,:2]
    out,_=pool(v)
    field=torch.linspace(-.8,.8,v.shape[-1],dtype=v.dtype,device=v.device)
    changed,_=pool(v+field)
    perm=torch.randperm(v.shape[-1],generator=torch.Generator().manual_seed(20261023)).to(v.device)
    shuffled,_=pool(v[...,perm])
    constant=torch.arange(3,dtype=v.dtype,device=v.device)[None,None,:,None].expand_as(v)
    zero,_=pool(constant-constant.mean(-1,keepdim=True))
    expected=numpy_pool(v.cpu().numpy())
    row=dict(common_field_contrast_error=float((out[...,1:,:]-changed[...,1:,:]).abs().max()),
        common_field_class_change=float((out[...,0,:]-changed[...,0,:]).abs().max()),
        joint_permutation_error=float((out-shuffled).abs().max()),constant_response_error=float(zero.abs().max()),
        numpy_pool_error=float(np.max(abs(out.cpu().numpy()-expected))))
    assert max(row[k] for k in row if k!='common_field_class_change')<1e-10
    return row


def load_frozen(t):
    folds,origins,sources=[],[],[]
    for f in range(6):
        with np.load(NEW/f'fold{f}_inputs.npz') as z:
            item={k:z[k] for k in ['base','train','validation','test','alpha','instance_ids']}
        with np.load(NEW/'origins'/f'fold{f}_outer.npz') as z:
            origin={k:z[k] for k in ['W','h0','fit','raw_intercept']}
        assert np.array_equal(item['instance_ids'],t.instance_id)
        assert np.array_equal(item['train'],origin['fit'])
        folds.append(item);origins.append(origin)
        sources.append(dict(fold=f,alpha=float(item['alpha']),source_n=len(item['train']),validation_n=len(item['validation']),
            test_n=len(item['test']),inputs=str(NEW/f'fold{f}_inputs.npz'),origin=str(NEW/'origins'/f'fold{f}_outer.npz'),
            W='saved original-channel PCA/scaler/ridge backprojection incl factor3',refit=False))
    pd.DataFrame(sources).to_csv(HERE/'sources.csv',index=False)
    coverage=np.bincount(np.concatenate([z['test'] for z in folds]),minlength=len(t))
    assert (coverage==1).all()
    return folds,origins


@torch.inference_mode()
def extract(t,folds,origins):
    started=time.perf_counter()
    deep=np.load(STRONG/'deep_features.npz')
    assert np.array_equal(deep['instance_ids'],t.instance_id)
    oldh=deep['layer3']
    h=np.zeros_like(oldh)
    delta=np.zeros((6,len(t),3,3),dtype=np.float64)
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
        for start in range(0,len(items),32):
            chunk=items[start:start+32]
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
                row=verify(v,batch,rotation)
                if start==0 and rotation==0:
                    row.update(special_checks(v))
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


def select_evaluate(t,folds,delta):
    started=time.perf_counter()
    y=t.class_id.to_numpy()
    candidates,selections,frames,records=[],[],[],[]
    for f,frozen in enumerate(folds):
        val,test=frozen['validation'],frozen['test']
        base=frozen['base'].astype(float)
        for j,arm in enumerate(ARMS):
            best_value=np.inf;best_lambda=None
            for lam in LAMBDAS:
                val_score=base[val]+lam*delta[f,val,j]
                vr=metrics(y[val],val_score)['balanced_NLL']
                candidates.append(dict(fold=f,arm=arm,lambda_value=lam,val_balanced_NLL=vr,validation_n=len(val)))
                if vr<best_value-1e-12:
                    best_value,best_lambda=vr,lam
            selections.append(dict(fold=f,arm=arm,selected_lambda=best_lambda,val_balanced_NLL=best_value))
            score=base[test]+best_lambda*delta[f,test,j]
            pred,bp=score.argmax(1),base[test].argmax(1)
            frame=t.iloc[test][['instance_id','class_id','mmsi','port','pass_group','neighbors']].copy()
            frame['fold']=f;frame['arm']=arm;frame['selected_lambda']=best_lambda
            frame['prediction']=pred;frame['base_prediction']=bp
            frame['net']=(pred==y[test]).astype(int)-(bp==y[test]).astype(int)
            frame['outcome']=np.select([frame.net.eq(1),frame.net.eq(-1),pred==y[test]],['rescued','harmed','CC'],default='WW')
            for c in range(3):
                frame[f's{c}']=score[:,c];frame[f'b{c}']=base[test,c];frame[f'delta{c}']=delta[f,test,j,c]
            if best_lambda==0:
                assert np.array_equal(score,base[test])
            frames.append(frame)
            records.append(dict(fold=f,arm=arm,selected_lambda=best_lambda,n=len(test),rescued=int(frame.net.eq(1).sum()),
                harmed=int(frame.net.eq(-1).sum()),net=int(frame.net.sum()),**metrics(y[test],score)))
    p=pd.concat(frames,ignore_index=True)
    assert p.groupby('arm').size().eq(len(t)).all() and not p.duplicated(['arm','instance_id']).any()
    pooled=[]
    for arm,g in p.groupby('arm',sort=False):
        pooled.append(dict(arm=arm,n=len(g),rescued=int(g.net.eq(1).sum()),harmed=int(g.net.eq(-1).sum()),net=int(g.net.sum()),
            **metrics(g.class_id.to_numpy(),g[['s0','s1','s2']].to_numpy())))
    g=p[p.arm==ARMS[0]]
    pooled.insert(0,dict(arm='A',n=len(g),rescued=0,harmed=0,net=0,**metrics(g.class_id.to_numpy(),g[['b0','b1','b2']].to_numpy())))
    for name,obj in [('predictions',p),('lambda_candidates',pd.DataFrame(candidates)),('selections',pd.DataFrame(selections)),
                     ('metrics',pd.DataFrame(records)),('pooled',pd.DataFrame(pooled))]:
        obj.to_csv(HERE/f'{name}.csv',index=False)
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    return time.perf_counter()-started


def main():
    start=time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')
    t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    assert len(t)==5695
    folds,origins=load_frozen(t)
    delta,cost=extract(t,folds,origins)
    selection_seconds=select_evaluate(t,folds,delta)
    summary=dict(status='complete',targets=len(t),mmsi=t.mmsi.nunique(),passes=t.pass_group.nunique(),ports=t.port.nunique(),
        classes=3,arms=ARMS,lambdas=LAMBDAS,temperature=1,population_std=True,trained_models=0,reader_refits=0,
        lambda_selections=18,selection_seconds=selection_seconds,total_seconds=time.perf_counter()-start,
        checkpoint=str(CHECKPOINT),gpu=torch.cuda.get_device_name(0),fp32_maps_fp64_readout=True,
        no_h0_in_delta=True,baseline_unchanged=True,no_full_F_saved=True,**cost)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    pd.DataFrame([dict(stage='extraction_and_verification',seconds=cost['extract_seconds']),
                  dict(stage='selection_and_evaluation',seconds=selection_seconds)]).to_csv(HERE/'timing.csv',index=False)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
