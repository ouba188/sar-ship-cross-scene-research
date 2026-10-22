"""Eight-class frozen weighted spatial std transfer. No fits or NN training."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'2')
import sys
sys.dont_write_bytecode=True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import resnet50
from scipy.special import logsumexp
from sklearn.metrics import f1_score

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
BASELINE=PARENT/'baseline'
sys.path.insert(0,str(PARENT))
from prepare_features import crop_pad,masks_for,WEIGHTS,OLD,pool as original_pool

LAMBDAS=[0.,.1,.25,.5,1.]
CLASSES=8


def metrics(y,s):
    logp=s-logsumexp(s,axis=1,keepdims=True)
    nll=-logp[np.arange(len(y)),y]
    brier=((np.exp(logp)-np.eye(CLASSES)[y])**2).sum(1)
    pred=s.argmax(1)
    present=np.unique(y)
    return dict(n=len(y),OA=100*float((pred==y).mean()),
        BA=100*float(np.mean([(pred[y==c]==c).mean() for c in present])),
        macro_F1=100*float(f1_score(y,pred,labels=range(CLASSES),average='macro',zero_division=0)),
        NLL=float(nll.mean()),balanced_NLL=float(np.mean([nll[y==c].mean() for c in present])),
        Brier=float(brier.mean()),balanced_Brier=float(np.mean([brier[y==c].mean() for c in present])),
        **{f'recall_{c}':100*float((pred[y==c]==c).mean()) if c in present else np.nan for c in range(CLASSES)})


def weighted_std(v,weight):
    r=v-v.mean(-2,keepdim=True)
    normalized=weight/weight.sum(-1,keepdim=True)
    center=(r*normalized).sum(-1,keepdim=True)
    result=torch.sqrt(((r-center).square()*normalized).sum(-1).clamp_min(0))
    return result,r


def invariant_checks(v,weight,full=False):
    values,r=weighted_std(v,weight)
    norm=weight/weight.sum(-1,keepdim=True)
    row=dict(v_weighted_mean=float((v*norm).sum(-1).abs().max()),
        r_weighted_mean=float((r*norm).sum(-1).abs().max()),r_class_mean=float(r.mean(-2).abs().max()))
    assert max(row.values())<1e-10
    if full:
        vv=v[:,:2];ww=weight[:,:2]
        expected,_=weighted_std(vv,ww)
        perm=torch.randperm(v.shape[-1],generator=torch.Generator().manual_seed(20261023)).to(v.device)
        back,_=weighted_std(vv[...,perm],ww[...,perm])
        field=torch.linspace(-1,1,v.shape[-1],dtype=v.dtype,device=v.device)
        added,_=weighted_std(vv+field,ww)
        class_offsets=torch.arange(1,9,dtype=v.dtype,device=v.device)[None,None,:,None]*123.0
        changed,_=weighted_std(vv+(ww==0)*class_offsets,ww)
        constant=torch.arange(8,dtype=v.dtype,device=v.device)[None,None,:,None].expand_as(vv)
        flat,_=weighted_std(constant,ww)
        vn=vv.cpu().numpy();wn=ww.cpu().numpy();wn=wn/wn.sum(-1,keepdims=True)
        rn=vn-vn.mean(-2,keepdims=True)
        rn-=np.sum(rn*wn,axis=-1,keepdims=True)
        independent=np.sqrt(np.sum(rn*rn*wn,axis=-1))
        row.update(joint_map_mask_permutation_error=float((back-expected).abs().max()),
            common_field_error=float((added-expected).abs().max()),zero_weight_candidate_error=float((changed-expected).abs().max()),
            constant_response_error=float(flat.abs().max()),numpy_error=float(np.max(abs(independent-expected.cpu().numpy()))),
            zero_weight_cells=int((ww==0).sum()))
        assert max(row[k] for k in row if k!='zero_weight_cells')<1e-10
    return row


def load_data():
    frame=pd.read_csv(PARENT/'cohort.csv',dtype={'mmsi':str})
    frame['pass_group']=frame['product'].map(lambda s:'_'.join([s.split('_')[0],s.split('_')[6],s.split('_')[7]]))
    raw=np.load(PARENT/'features.npz',allow_pickle=True)
    valid=raw['valid'];h=raw['layer3']
    assert len(frame)==15251 and np.array_equal(raw['instance_ids'].astype(str),frame.instance_id)
    snapshots={s['obs_id']:s for s in json.loads((OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    coverage=pd.read_csv(PARENT/'coverage.csv').set_index(['instance_id','pol'])
    folds,origins,source=[],[],[]
    for k in range(3):
        with np.load(BASELINE/f'fold{k}_inputs.npz') as z:
            fold={key:z[key] for key in ['base','train','validation','test','instance_ids','valid','selected_alpha']}
        with np.load(BASELINE/'models'/f'fold{k}_outer.npz') as z:
            origin={key:z[key] for key in ['W','intercept','fit']}
        assert np.array_equal(fold['instance_ids'],frame.instance_id)
        assert np.array_equal(fold['valid'],valid) and np.array_equal(fold['train'],origin['fit'])
        folds.append(fold);origins.append(origin)
        source.append(dict(fold=k,alpha=float(fold['selected_alpha']),source_n=len(fold['train']),validation_n=len(fold['validation']),
            test_n=len(fold['test']),inputs=str(BASELINE/f'fold{k}_inputs.npz'),W_source=str(BASELINE/'models'/f'fold{k}_outer.npz')))
    test=np.concatenate([f['test'] for f in folds])
    assert len(test)==4744 and len(np.unique(test))==len(test)
    pd.DataFrame(source).to_csv(HERE/'sources.csv',index=False)
    return frame,valid,h,snapshots,coverage,folds,origins


@torch.inference_mode()
def extract(frame,valid,oldh,snapshots,coverage,folds,origins):
    started=time.perf_counter()
    n=len(frame)
    h=np.zeros_like(oldh)
    means=np.zeros((3,n,2,8),np.float64)
    deltas=np.zeros_like(means)
    w=torch.tensor(np.stack([o['W'].reshape(8,2,1024).transpose(1,0,2) for o in origins]),device='cuda')
    model=resnet50(weights=None)
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
    model=model.cuda().float().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    test=np.concatenate([f['test'] for f in folds])
    sample_indices=set([0,1,int(test.min()),int(next(i for i in test if not valid[i,1]))])
    checks=[];gap=[];windows=[];maps={};last=time.perf_counter();batches=0;seen=np.zeros_like(valid,dtype=int)
    for first in range(0,n,16):
        images,footprints,addresses=[],[],[]
        for i in range(first,min(first+16,n)):
            row=frame.iloc[i]
            for p,pol in enumerate(['vv','vh']):
                view=snapshots[int(row.obs_id)]['images'][pol]
                with Image.open(row[f'image_{pol}']) as im:
                    image=np.array(im.convert('L'))
                assert image.shape==(view['h'],view['w'])
                pixels,masks,place=crop_pad(image,masks_for(view),160)
                old=coverage.loc[(row.instance_id,pol)]
                actual_valid=bool(masks[0].any() and not bool(old.disabled_alignment))
                assert actual_valid==bool(valid[i,p])==bool(old.valid)
                for key,value in place.items():
                    assert value==old[key],(row.instance_id,pol,key,value,old[key])
                support=masks.sum(0)
                windows.append(dict(instance_id=row.instance_id,row_index=i,pol=pol,valid=actual_valid,
                    disabled_alignment=bool(old.disabled_alignment),native_height=image.shape[0],native_width=image.shape[1],
                    footprint_pixels=float(support.sum()),footprint_fraction=float(support.mean()),**place))
                if not actual_valid:
                    continue
                images.append(pixels[None].astype(np.float32)/255)
                footprints.append(support[None].astype(np.float32))
                addresses.append((i,p))
        x=torch.tensor(np.stack(images),device='cuda')
        footprint=torch.tensor(np.stack(footprints),device='cuda')
        pol=torch.tensor([p for _,p in addresses],device='cuda')
        wp=w[:,pol]
        pooled=[]
        for rotation in range(4):
            xx=x.rot90(rotation,(-2,-1)).expand(-1,3,-1,-1)
            mask=footprint.rot90(rotation,(-2,-1))
            z=model.relu(model.bn1(model.conv1((xx-mean)/std)))
            z=model.layer3(model.layer2(model.layer1(model.maxpool(z))))
            pooled.append(original_pool(z,mask,maximum=False))
            weight=F.interpolate(mask,size=z.shape[-2:],mode='area').flatten(2).double()[None]
            norm=weight/weight.sum(-1,keepdim=True)
            response=torch.einsum('fbck,bkp->fbcp',wp,z.flatten(2).double())
            mu=(response*norm).sum(-1,keepdim=True)
            v=response-mu
            delta,r=weighted_std(v,weight)
            check=invariant_checks(v,weight,full=(first==0 or (first%2048==0 and rotation==0)))
            check.update(batch=batches,rotation=rotation)
            checks.append(check)
            mm,dd=mu[...,0].cpu().numpy(),delta.cpu().numpy()
            for b,(i,p) in enumerate(addresses):
                means[:,i,p]+=mm[:,b]/4
                deltas[:,i,p]+=dd[:,b]/4
                if i in sample_indices:
                    prefix=f'row{i}_{["vv","vh"][p]}_rot{rotation}'
                    maps[prefix+'_v']=v[:,b].cpu().numpy().reshape(3,8,10,10)
                    maps[prefix+'_r']=r[:,b].cpu().numpy().reshape(3,8,10,10)
                    maps[prefix+'_weight']=weight[0,b].cpu().numpy().reshape(10,10)
        fresh=torch.stack(pooled).mean(0).cpu().numpy()
        for b,(i,p) in enumerate(addresses):
            cols=slice(p*1024,(p+1)*1024)
            np.testing.assert_allclose(fresh[b],oldh[i,cols],atol=1e-4,rtol=1e-4)
            h[i,cols]=fresh[b];seen[i,p]+=1
            gap.append(dict(instance_id=frame.iloc[i].instance_id,row_index=i,pol=['vv','vh'][p],
                max_abs_error=float(abs(fresh[b]-oldh[i,cols]).max()),rmse=float(np.sqrt(np.mean((fresh[b]-oldh[i,cols])**2)))))
        batches+=1
        if time.perf_counter()-last>25:
            print(json.dumps(dict(targets=min(first+16,n),total=n,valid_images=int(seen.sum()),seconds=time.perf_counter()-started)),flush=True)
            last=time.perf_counter()
    assert np.array_equal(seen,valid.astype(int))
    assert not means[:,~valid].any() and not deltas[:,~valid].any()
    assert np.isfinite(deltas).all()
    np.savez_compressed(HERE/'std_features.npz',instance_ids=frame.instance_id.to_numpy(str),valid=valid,
                        mean_pol=means,delta_pol=deltas,delta=deltas.sum(2))
    np.savez_compressed(HERE/'validation_maps.npz',**maps)
    pd.DataFrame(windows).to_csv(HERE/'window_coverage.csv',index=False)
    pd.DataFrame(gap).to_csv(HERE/'gap_verification.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'invariants.csv',index=False)
    reader=[]
    for k,(fold,origin) in enumerate(zip(folds,origins)):
        old_affine=oldh.astype(float)@origin['W'].T+origin['intercept']
        fresh_affine=h.astype(float)@origin['W'].T+origin['intercept']
        for role in ['train','validation','test']:
            ix=fold[role]
            error=float(abs(fresh_affine[ix]-fold['base'][ix]).max())
            assert error<1e-4
            reader.append(dict(fold=k,role=role,n=len(ix),cached_affine_error=float(abs(old_affine[ix]-fold['base'][ix]).max()),
                new_affine_error=error,top1_changes=int(np.sum(fresh_affine[ix].argmax(1)!=fold['base'][ix].argmax(1)))))
    pd.DataFrame(reader).to_csv(HERE/'reader_verification.csv',index=False)
    parts=[];source_stats=[]
    for k,fold in enumerate(folds):
        for p,pol in enumerate(['vv','vh']):
            part=pd.DataFrame(dict(instance_id=frame.instance_id,fold=k,pol=pol,valid=valid[:,p]))
            for c in range(8):
                part[f'mean{c}']=means[k,:,p,c];part[f'delta{c}']=deltas[k,:,p,c]
            parts.append(part)
        for role in ['train','validation','test']:
            ix=fold[role]
            for c in range(8):
                source_stats.append(dict(fold=k,role=role,class_direction=c,n=len(ix),
                    mean_response=float(means[k,ix,:,c].sum(1).mean()),
                    delta_mean=float(deltas[k,ix,:,c].sum(1).mean()),delta_sd=float(deltas[k,ix,:,c].sum(1).std()),
                    delta_p05=float(np.quantile(deltas[k,ix,:,c].sum(1),.05)),delta_p95=float(np.quantile(deltas[k,ix,:,c].sum(1),.95))))
    pd.concat(parts,ignore_index=True).to_csv(HERE/'mean_delta.csv',index=False)
    pd.DataFrame(source_stats).to_csv(HERE/'source_statistics.csv',index=False)
    return deltas.sum(2),dict(extract_seconds=time.perf_counter()-started,batches=batches,valid_images=int(valid.sum()),
        gap_max_error=max(r['max_abs_error'] for r in gap),max_reader_error=max(r['new_affine_error'] for r in reader),
        reader_top1_changes=sum(r['top1_changes'] for r in reader),gpu_peak_bytes=torch.cuda.max_memory_allocated())


def evaluate(frame,folds,delta):
    start=time.perf_counter()
    candidates=[];selected=[];frames=[];records=[]
    y=frame.class_id.to_numpy()
    for k,fold in enumerate(folds):
        val,test=fold['validation'],fold['test'];base=fold['base']
        best=np.inf;lam=None
        for value in LAMBDAS:
            risk=metrics(y[val],base[val]+value*delta[k,val])['balanced_NLL']
            candidates.append(dict(fold=k,lambda_value=value,val_balanced_NLL=risk))
            if risk<best-1e-12:
                best,lam=risk,value
        selected.append(dict(fold=k,selected_lambda=lam,val_balanced_NLL=best))
        score=base[test]+lam*delta[k,test]
        bp,pred=base[test].argmax(1),score.argmax(1)
        p=frame.iloc[test][['instance_id','obs_id','class_id','class_name','port','mmsi','product','pass_group','time']].copy()
        p['row_index']=test;p['fold']=k;p['selected_lambda']=lam
        p['base_prediction']=bp;p['prediction']=pred
        p['net']=(pred==y[test]).astype(int)-(bp==y[test]).astype(int)
        p['outcome']=np.select([p.net.eq(1),p.net.eq(-1),pred==y[test]],['rescued','harmed','CC'],default='WW')
        for c in range(8):
            p[f'b{c}']=base[test,c];p[f's{c}']=score[:,c];p[f'delta{c}']=delta[k,test,c]
        if lam==0:
            assert np.array_equal(score,base[test])
        frames.append(p)
        for arm,s in [('A',base[test]),('contrast_std',score)]:
            records.append(dict(fold=k,arm=arm,selected_lambda=0 if arm=='A' else lam,
                rescued=0 if arm=='A' else int(p.net.eq(1).sum()),harmed=0 if arm=='A' else int(p.net.eq(-1).sum()),
                net=0 if arm=='A' else int(p.net.sum()),**metrics(y[test],s)))
    p=pd.concat(frames,ignore_index=True)
    assert len(p)==4744 and p.instance_id.is_unique
    pooled=[]
    for arm,prefix in [('A','b'),('contrast_std','s')]:
        pooled.append(dict(arm=arm,rescued=0 if arm=='A' else int(p.net.eq(1).sum()),
            harmed=0 if arm=='A' else int(p.net.eq(-1).sum()),net=0 if arm=='A' else int(p.net.sum()),
            **metrics(p.class_id.to_numpy(),p[[f'{prefix}{c}' for c in range(8)]].to_numpy())))
    for name,obj in [('lambda_candidates',pd.DataFrame(candidates)),('selections',pd.DataFrame(selected)),
                     ('predictions',p),('metrics',pd.DataFrame(records)),('pooled',pd.DataFrame(pooled))]:
        obj.to_csv(HERE/f'{name}.csv',index=False)
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    return time.perf_counter()-start


def main():
    started=time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')
    frame,valid,h,snapshots,coverage,folds,origins=load_data()
    delta,cost=extract(frame,valid,h,snapshots,coverage,folds,origins)
    seconds=evaluate(frame,folds,delta)
    summary=dict(status='complete',rows=len(frame),test_rows=4744,classes=8,valid_vv=int(valid[:,0].sum()),
        valid_vh=int(valid[:,1].sum()),invalid_vh=int((~valid[:,1]).sum()),trained_models=0,reader_refits=0,
        input='original 160 crop/pad; area-downsampled valid footprint; no image resize',
        pooling='weighted population spatial std, pol sum, C4 mean, class-centered local evidence',
        baseline_unchanged=True,no_h0_in_delta=True,no_full_F_saved=True,selection_seconds=seconds,
        total_seconds=time.perf_counter()-started,**cost)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    pd.DataFrame([dict(stage='extraction_and_verification',seconds=cost['extract_seconds']),
                  dict(stage='selection_and_evaluation',seconds=seconds)]).to_csv(HERE/'timing.csv',index=False)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
