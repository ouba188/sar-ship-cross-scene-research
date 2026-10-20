"""Native-scale, endpoint-symmetric structural probe; no angle inputs to heads."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.models import resnet50
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
LOCAL=ROOT.parent/'local_condition_module'
sys.path.insert(0,str(LOCAL))
sys.path.insert(0,str(ROOT.parent/'conditional_evidence_pilot'))
from native_input import crop_pad
from prepare import masks_for, WEIGHTS
import run_pilot as prior
ARMS=('whole','axis','transverse','scrambled')


def regions(image,view,obs_id):
    raw=masks_for(view)
    pixels,m,placement=crop_pad(image,raw,160)
    axis=view['priors'].get('ship_axis')
    partitions=[]
    for offset in (0.,np.pi/2):
        slots=np.zeros((3,160,160),np.float32)
        if axis and m[0].any():
            yy,xx=np.indices((160,160))
            cx,cy=axis['center']
            cx+=placement['pad_left']-placement['crop_x']
            cy+=placement['pad_top']-placement['crop_y']
            angle=axis['angle']+offset
            length=axis['length_px'] if offset==0 else axis['width_px']
            coordinate=(xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle)
            # The same native pixels are assigned to thirds; none are rescaled.
            slots[0]=m[0]*(coordinate < -length/6)
            slots[1]=m[0]*(np.abs(coordinate)<=length/6)
            slots[2]=m[0]*(coordinate > length/6)
        partitions.append(slots)
    scrambled=partitions[0].copy()
    yy,xx=np.nonzero(m[0])
    if len(xx):
        rng=np.random.default_rng(20261020+int(obs_id))
        scrambled[:,yy,xx]=scrambled[:,yy,xx][:,rng.permutation(len(xx))]
    masks=np.concatenate((m,*partitions,scrambled))
    # These shape controls are shared by all arms, so axial results cannot win
    # merely by introducing target size or extra occupied-region counts.
    shape=[float(axis['length_px']) if axis else 0.,float(axis['width_px']) if axis else 0.]
    for slots in (m,*partitions,scrambled):
        count=slots.sum((1,2))
        if slots is m: shape.extend(count.tolist())
        else: shape.extend([float(count[1]),float((count[0]+count[2])/2),float(abs(count[0]-count[2]))])
    return pixels,masks,np.asarray(shape,np.float32),placement


def pool_response(stem,x,m):
    mean=torch.tensor([.485,.456,.406],device=x.device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=x.device)[None,:,None,None]
    output=[]
    # Average the full readout over the exact C4 orbit, not interpolated rotations.
    for k in range(4):
        image=torch.rot90(x,k,(-2,-1)).expand(-1,3,-1,-1)
        mask=F.avg_pool2d(torch.rot90(m,k,(-2,-1)),4)
        features=F.avg_pool2d(stem((image-mean)/std),2)
        pooled=torch.einsum('bchw,brhw->brc',features,mask)/mask.sum((2,3)).clamp_min(1e-8)[:,:,None]
        output.append(pooled)
    return torch.stack(output).mean(0)


def symmetric(slots):
    return np.concatenate((slots[...,1,:],(slots[...,0,:]+slots[...,2,:])/2,
                           np.abs(slots[...,0,:]-slots[...,2,:])),axis=-1)


def extract():
    start=time.perf_counter(); torch.set_num_threads(2)
    f=pd.read_csv(prior.ROOT/'cohort.csv',dtype={'mmsi':str})
    snapshots=json.loads((LOCAL/'view_snapshot.json').read_text(encoding='utf-8'))
    assert [s['obs_id'] for s in snapshots]==f.obs_id.tolist()
    conflict=pd.read_csv(LOCAL/'crossview_alignment.csv').set_index('obs_id').loc[f.obs_id,'disable_new_vh'].to_numpy(bool)
    model=resnet50(weights=None)
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
    stem=torch.nn.Sequential(model.conv1,model.bn1,model.relu).cuda().eval()
    del model
    pooled=np.zeros((len(f),2,12,64),np.float32)
    shapes=np.zeros((len(f),2,14),np.float32)
    coverage=[]; invariance=[]
    with torch.inference_mode():
        for first in range(0,len(f),24):
            images=[]; masks=[]; addresses=[]
            for i in range(first,min(first+24,len(f))):
                for p,pol in enumerate(('vv','vh')):
                    if f.iloc[i].obs_id==31237 or (p==1 and conflict[i]): continue
                    v=snapshots[i]['images'][pol]
                    raw=np.asarray(Image.open(f.iloc[i]['image_'+pol]).convert('L'))
                    assert raw.shape==(v['h'],v['w'])
                    pix,mask,shape,place=regions(raw,v,f.iloc[i].obs_id)
                    assert place['native_target_pixels']==place['retained_target_pixels']
                    shapes[i,p]=shape
                    images.append(torch.from_numpy(pix.copy())[None].float()/255)
                    masks.append(torch.from_numpy(mask)); addresses.append((i,p))
                    coverage.append(dict(obs_id=int(f.iloc[i].obs_id),pol=pol,width=v['w'],height=v['h'],**place))
            if not images: continue
            x=torch.stack(images).cuda(); m=torch.stack(masks).cuda()
            desc=pool_response(stem,x,m)
            for b,(i,p) in enumerate(addresses): pooled[i,p]=desc[b].cpu().numpy()
            if first==0:
                alt=pool_response(stem,torch.rot90(x[:4],1,(-2,-1)),torch.rot90(m[:4],1,(-2,-1)))
                invariance.append(dict(test='joint_90deg_content_masks',max_abs=float((alt-desc[:4]).abs().max())))
                assert torch.allclose(alt,desc[:4],atol=2e-5,rtol=2e-5)
            if first%480==0: print('extract',first,'/',len(f),'seconds',round(time.perf_counter()-start,1),flush=True)
    base=np.concatenate((pooled[:,:,:3].reshape(len(f),-1),shapes.reshape(len(f),-1)),axis=1)
    blocks={a:symmetric(pooled[:,:,3+3*j:6+3*j]).reshape(len(f),-1) for j,a in enumerate(ARMS[1:])}
    assert np.array_equal(symmetric(pooled[:,:,3:6]),symmetric(pooled[:,:,3:6][:,:,::-1]))
    np.savez_compressed(ROOT/'descriptors.npz',base=base,**blocks)
    pd.DataFrame(coverage).to_csv(ROOT/'coverage.csv',index=False)
    pd.DataFrame(invariance).to_csv(ROOT/'invariance.csv',index=False)
    (ROOT/'EXTRACTION.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,n=len(f),canvas=160,
        no_resize=True,head_tail_exchange_exact=True,rotation='C4 only, not arbitrary angle invariance',
        shape='native pixels, not physical metres',angle_input_to_classifier=False),indent=2),encoding='utf-8')


def evaluate():
    start=time.perf_counter(); f=pd.read_csv(prior.ROOT/'cohort.csv',dtype={'mmsi':str})
    y=f.class_id.to_numpy(); d=np.load(ROOT/'descriptors.npz')
    excluded=pd.read_csv(LOCAL/'EXCLUSIONS.csv').obs_id.to_numpy()
    metrics=[]; selection=[]; predictions=[]; tests={}; coefficients={}
    # Fit deterministic heads. Three identity folds are not three training seeds.
    for fold in range(3):
        train,val,test=prior.indices(f,fold)
        train,val,test=[ii[~f.iloc[ii].obs_id.isin(excluded).to_numpy()] for ii in (train,val,test)]
        w=prior.weights(f,train)
        for arm in ARMS:
            raw=d['base'] if arm=='whole' else np.concatenate((d['base'],d[arm]),axis=1)
            mean=raw[train].mean(0); std=np.maximum(raw[train].std(0),1e-6)
            x=(raw-mean)/std/np.sqrt(raw.shape[1]); candidates=[]
            for alpha in (.1,1.,10.):
                model=RidgeClassifier(alpha=alpha,solver='cholesky').fit(x[train],y[train],sample_weight=w)
                met=prior.metrics(y[val],model.decision_function(x[val]))
                selection.append(dict(fold=fold,arm=arm,alpha=alpha,**met))
                candidates.append((met,alpha,model))
            _,alpha,model=max(candidates,key=lambda z:(z[0]['ba'],-z[0]['brier']))
            scores=model.decision_function(x[test]); tests[(fold,arm)]=(test,scores)
            np.savez_compressed(ROOT/f'head_f{fold}_{arm}.npz',mean=mean,std=std,coef=model.coef_,intercept=model.intercept_,dimension=raw.shape[1])
            metrics.append(dict(fold=fold,arm=arm,alpha=alpha,**prior.metrics(y[test],scores)))
            other=scores.copy(); other[np.arange(len(test)),y[test]]=-np.inf
            margins=scores[np.arange(len(test)),y[test]]-other.max(1)
            for b,i in enumerate(test):
                predictions.append(dict(fold=fold,arm=arm,obs_id=int(f.iloc[i].obs_id),mmsi=f.iloc[i].mmsi,
                    port=f.iloc[i].port,product=f.iloc[i]['product'],class_id=int(y[i]),prediction=int(scores[b].argmax()),
                    correct=bool(scores[b].argmax()==y[i]),margin=float(margins[b])))
            print('fit',fold,arm,metrics[-1],flush=True)
    pd.DataFrame(metrics).to_csv(ROOT/'metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(ROOT/'validation_selection.csv',index=False)
    p=pd.DataFrame(predictions); p.to_csv(ROOT/'predictions.csv',index=False)
    rows=[]
    for arm in ARMS:
        yy=np.concatenate([y[tests[(fold,arm)][0]] for fold in range(3)])
        ss=np.concatenate([tests[(fold,arm)][1] for fold in range(3)])
        rows.append(dict(arm=arm,**prior.metrics(yy,ss)))
    pd.DataFrame(rows).to_csv(ROOT/'summary.csv',index=False)
    pairs=[]
    for reference in ('whole','transverse','scrambled'):
        a=p[p.arm.eq('axis')].set_index('obs_id'); b=p[p.arm.eq(reference)].set_index('obs_id').loc[a.index]
        pairs.append(dict(reference=reference,rescue=int((a.correct&~b.correct).sum()),harm=int((~a.correct&b.correct).sum())))
    pd.DataFrame(pairs).to_csv(ROOT/'paired.csv',index=False)
    for group in ('class_id','port'):
        p.groupby(['arm',group]).agg(n=('correct','size'),oa=('correct','mean')).reset_index().to_csv(ROOT/f'by_{group}.csv',index=False)
    (ROOT/'RECEIPT.json').write_text(json.dumps(dict(status='complete',fit_seconds=time.perf_counter()-start,
        fits=12,folds=3,training_seeds='not applicable to deterministic ridge',new_neural_training=False,
        scope='Native frozen stem structural probe, no directed bow/stern labels or support adaptation'),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('extract','evaluate'))
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        extract() if args.stage=='extract' else evaluate()
