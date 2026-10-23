"""Frozen eight-class source/outer weighted signed moments, native footprint."""
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

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
BASELINE=PARENT/'baseline'
sys.path.insert(0,str(PARENT))
from prepare_features import crop_pad,masks_for,WEIGHTS,OLD,pool as original_pool


def weighted_moments(response,weight):
    norm=weight/weight.sum(-1,keepdim=True)
    mu=(response*norm).sum(-1,keepdim=True)
    v=response-mu
    r=v-v.mean(-2,keepdim=True)
    r=r-(r*norm).sum(-1,keepdim=True)
    sigma=(r.square()*norm).sum(-1).clamp_min(0).sqrt()
    kappa=(r*r.abs()*norm).sum(-1)/sigma.clamp_min(1e-12)
    return torch.stack([sigma,kappa],-1),mu[...,0],r


def main():
    start=time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')
    t=pd.read_csv(PARENT/'cohort.csv',dtype={'mmsi':str})
    z=np.load(PARENT/'features.npz',allow_pickle=True)
    valid,oldh=z['valid'],z['layer3']
    assert np.array_equal(z['instance_ids'].astype(str),t.instance_id)
    snapshots={s['obs_id']:s for s in json.loads((OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    coverage=pd.read_csv(PARENT/'coverage.csv').set_index(['instance_id','pol'])
    folds=[];origins=[];names=[];sources=[]
    for f in range(3):
        zz=np.load(BASELINE/f'fold{f}_inputs.npz')
        d={k:zz[k] for k in ['base','oof','train','validation','test','instance_ids','inner_bin']}
        assert np.array_equal(d['instance_ids'],t.instance_id)
        folds.append(d)
        count=np.zeros(len(t),int)
        for tag in ['outer']+[f'inner{i}' for i in range(4)]:
            name=f'fold{f}_{tag}'
            with np.load(BASELINE/'models'/f'{name}.npz') as reader:
                o={k:reader[k] for k in ['W','intercept','fit']+(['held'] if tag!='outer' else [])}
            assert o['W'].shape==(8,2048)
            assert (np.bincount(t.iloc[o['fit']].class_id,minlength=8)>0).all()
            if 'held' in o:
                count[o['held']]+=1
                assert not set(t.iloc[o['fit']].mmsi)&set(t.iloc[o['held']].mmsi)
                assert not set(t.iloc[o['fit']]['product'])&set(t.iloc[o['held']]['product'])
            else:
                assert np.array_equal(o['fit'],d['train'])
            origins.append(o);names.append(name)
            sources.append(dict(reader=name,fold=f,fit=len(o['fit']),held=len(o.get('held',[])),path=str(BASELINE/'models'/f'{name}.npz')))
        assert (count[d['train']]==1).all() and count.sum()==len(d['train'])
    pd.DataFrame(sources).to_csv(HERE/'sources.csv',index=False)
    n=len(t);nread=len(names)
    moment=np.zeros((nread,n,2,8,2),float)
    means=np.zeros((nread,n,2,8),float)
    freshh=np.zeros_like(oldh);seen=np.zeros_like(valid,int)
    w=torch.tensor(np.stack([o['W'].reshape(8,2,1024).transpose(1,0,2) for o in origins]),device='cuda',dtype=torch.float64)
    model=resnet50(weights=None)
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
    model=model.cuda().float().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    gap=[];maps={};last=time.perf_counter();batch=0
    with torch.inference_mode():
        for first in range(0,n,16):
            images=[];footprints=[];addresses=[]
            for i in range(first,min(first+16,n)):
                row=t.iloc[i]
                for p,pol in enumerate(['vv','vh']):
                    view=snapshots[int(row.obs_id)]['images'][pol]
                    with Image.open(row[f'image_{pol}']) as im:
                        image=np.array(im.convert('L'))
                    assert image.shape==(view['h'],view['w'])
                    pixels,masks,place=crop_pad(image,masks_for(view),160)
                    old=coverage.loc[(row.instance_id,pol)]
                    assert bool(masks[0].any() and not old.disabled_alignment)==bool(valid[i,p])
                    assert all(value==old[key] for key,value in place.items())
                    if not valid[i,p]:
                        continue
                    images.append(pixels[None].astype(np.float32)/255)
                    footprints.append(masks.sum(0)[None].astype(np.float32))
                    addresses.append((i,p))
            x=torch.tensor(np.stack(images),device='cuda')
            support=torch.tensor(np.stack(footprints),device='cuda')
            wp=w[:,torch.tensor([p for _,p in addresses],device='cuda')]
            gaps=[]
            for rotation in range(4):
                xx=x.rot90(rotation,(-2,-1)).expand(-1,3,-1,-1)
                mask=support.rot90(rotation,(-2,-1))
                zz=model.relu(model.bn1(model.conv1((xx-mean)/std)))
                zz=model.layer3(model.layer2(model.layer1(model.maxpool(zz))))
                gaps.append(original_pool(zz,mask,maximum=False))
                weight=F.interpolate(mask,size=zz.shape[-2:],mode='area').flatten(2).double()[None]
                response=torch.einsum('fbck,bkp->fbcp',wp,zz.flatten(2).double())
                out,mu,r=weighted_moments(response,weight)
                assert (out[...,1].abs()<=out[...,0]+1e-10).all()
                if batch==0:
                    reverse,_,_=weighted_moments(-response,weight)
                    assert torch.allclose(reverse[...,0],out[...,0],atol=1e-12,rtol=1e-12)
                    assert torch.allclose(reverse[...,1],-out[...,1],atol=1e-12,rtol=1e-12)
                    maps[f'rot{rotation}_r']=r[:,:2].cpu().numpy()
                    maps[f'rot{rotation}_weight']=weight[:,:2].cpu().numpy()
                    maps[f'rot{rotation}_moments']=out[:,:2].cpu().numpy()
                out=out.cpu().numpy();mu=mu.cpu().numpy()
                for b,(i,p) in enumerate(addresses):
                    moment[:,i,p]+=out[:,b]/4
                    means[:,i,p]+=mu[:,b]/4
            gh=torch.stack(gaps).mean(0).cpu().numpy()
            for b,(i,p) in enumerate(addresses):
                cols=slice(p*1024,(p+1)*1024)
                err=float(abs(gh[b]-oldh[i,cols]).max())
                assert err<1e-4
                freshh[i,cols]=gh[b];seen[i,p]+=1
                gap.append(dict(row_index=i,instance_id=t.iloc[i].instance_id,pol=['vv','vh'][p],max_error=err))
            batch+=1
            if time.perf_counter()-last>25:
                print('extracted',min(first+16,n),'/',n,'seconds',round(time.perf_counter()-start,2),flush=True)
                last=time.perf_counter()
    assert np.array_equal(seen,valid.astype(int))
    assert np.isfinite(moment).all() and not moment[:,~valid].any()
    oldsigma=np.load(PARENT/'spatial_std_transfer/std_features.npz')['delta_pol']
    checks=[]
    for j,(name,o) in enumerate(zip(names,origins)):
        f=int(name[4]);d=folds[f]
        ix=o['held'] if 'held' in o else np.arange(n)
        expected=d['oof'] if 'held' in o else d['base']
        score=means[j,ix].sum(1)+o['intercept']
        err=float(abs(score-expected[ix]).max())
        assert err<1e-4
        row=dict(reader=name,n=len(ix),mean_logit_error=err,top1_changes=int(np.sum(score.argmax(1)!=expected[ix].argmax(1))))
        assert row['top1_changes']==0
        if 'held' not in o:
            row['outer_sigma_error']=float(abs(moment[j,:,:,:,0]-oldsigma[f]).max())
            assert row['outer_sigma_error']<1e-8
        checks.append(row)
    np.savez_compressed(HERE/'signed_features.npz',reader_names=np.array(names),instance_ids=t.instance_id.to_numpy(str),
        valid=valid,moments_pol=moment,moments=moment.sum(2),mean_pol=means)
    np.savez_compressed(HERE/'validation_maps.npz',**maps)
    pd.DataFrame(checks).to_csv(HERE/'reader_checks.csv',index=False)
    pd.DataFrame(gap).to_csv(HERE/'gap_checks.csv',index=False)
    summary=dict(status='complete',readers=nread,targets=n,valid_vv=int(valid[:,0].sum()),valid_vh=int(valid[:,1].sum()),
        images=int(valid.sum()),batches=batch,seconds=time.perf_counter()-start,reader_fits=0,
        gap_max_error=max(v['max_error'] for v in gap),mean_logit_max_error=max(v['mean_logit_error'] for v in checks),
        outer_sigma_max_error=max(v.get('outer_sigma_error',0) for v in checks),gpu_peak_bytes=torch.cuda.max_memory_allocated())
    (HERE/'extraction.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
