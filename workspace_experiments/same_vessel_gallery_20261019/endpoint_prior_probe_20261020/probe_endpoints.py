"""Strong-end canonicalization without claiming anatomical bow/stern truth."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models import resnet50
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'native_axis_probe_20261020'
sys.path.insert(0,str(PREV))
import probe as old
ARMS=('axis','hard_strong','soft_strong','random_flip','symmetric_content','soft_random_flip','soft_absdiff')


def endpoint_strength(pixels,slots):
    peaks=[]; means=[]; saturation=[]; counts=[]
    for k in (0,2):
        a=pixels[slots[k]>0].astype(float)
        counts.append(len(a))
        if len(a):
            peaks.append(float(np.sort(a)[-max(1,int(np.ceil(.2*len(a)))):].mean()))
            means.append(float(a.mean())); saturation.append(float(np.mean(a==255)))
        else: peaks.append(0.); means.append(0.); saturation.append(0.)
    peak_tie=peaks[0]==peaks[1]
    used=means if peak_tie else peaks
    contrast=(used[1]-used[0])/max(used[1]+used[0],1.) if min(counts)>0 else 0.
    return float(contrast),dict(end0_peak=peaks[0],end2_peak=peaks[1],end0_mean=means[0],end2_mean=means[1],
        end0_saturation=saturation[0],end2_saturation=saturation[1],end0_pixels=counts[0],end2_pixels=counts[1],
        peak_tie=peak_tie,contrast=contrast)


def directed(slots,contrast):
    delta=slots[...,2,:]-slots[...,0,:]
    hard=np.sign(contrast)[...,None]*delta
    return hard,hard*np.abs(contrast)[...,None],delta**2


def extract():
    start=time.perf_counter(); torch.set_num_threads(2)
    f=pd.read_csv(old.prior.ROOT/'cohort.csv',dtype={'mmsi':str})
    snapshots=json.loads((old.LOCAL/'view_snapshot.json').read_text(encoding='utf-8'))
    assert [s['obs_id'] for s in snapshots]==f.obs_id.tolist()
    conflict=pd.read_csv(old.LOCAL/'crossview_alignment.csv').set_index('obs_id').loc[f.obs_id,'disable_new_vh'].to_numpy(bool)
    model=resnet50(weights=None); model.load_state_dict(torch.load(old.WEIGHTS,map_location='cpu',weights_only=True))
    stem=torch.nn.Sequential(model.conv1,model.bn1,model.relu).cuda().eval(); del model
    slots=np.zeros((len(f),2,3,64),np.float32); contrast=np.zeros((len(f),2),np.float32)
    rows=[]; rotation=[]
    with torch.inference_mode():
        for first in range(0,len(f),24):
            images=[]; masks=[]; addresses=[]
            for i in range(first,min(first+24,len(f))):
                for p,pol in enumerate(('vv','vh')):
                    if f.iloc[i].obs_id==31237 or (p==1 and conflict[i]): continue
                    v=snapshots[i]['images'][pol]
                    raw=np.asarray(Image.open(f.iloc[i]['image_'+pol]).convert('L'))
                    pixels,m,_,placement=old.regions(raw,v,f.iloc[i].obs_id)
                    c,stats=endpoint_strength(pixels,m[3:6]); contrast[i,p]=c
                    rows.append(dict(obs_id=int(f.iloc[i].obs_id),pol=pol,**stats))
                    images.append(torch.from_numpy(pixels.copy())[None].float()/255)
                    masks.append(torch.from_numpy(m[3:6])); addresses.append((i,p))
            if not images: continue
            x=torch.stack(images).cuda(); m=torch.stack(masks).cuda()
            desc=old.pool_response(stem,x,m)
            for b,(i,p) in enumerate(addresses): slots[i,p]=desc[b].cpu().numpy()
            if first==0:
                alt=old.pool_response(stem,torch.rot90(x[:4],1,(-2,-1)),torch.rot90(m[:4],1,(-2,-1)))
                rotation.append(dict(test='C4_slots',max_abs=float((alt-desc[:4]).abs().max())))
                assert torch.allclose(alt,desc[:4],atol=2e-5,rtol=2e-5)
            if first%960==0: print('extract',first,'/',len(f),round(time.perf_counter()-start,1),flush=True)
    previous=np.load(PREV/'descriptors.npz')
    replay=old.symmetric(slots).reshape(len(f),-1)
    error=float(np.max(np.abs(replay-previous['axis'])))
    assert np.allclose(replay,previous['axis'],atol=2e-5,rtol=2e-5)
    hard,soft,nonlinear=directed(slots,contrast)
    reverse=directed(slots[:,:,::-1],-contrast)
    for a,b in zip((hard,soft,nonlinear),reverse): assert np.array_equal(a,b)
    random=np.random.default_rng(20261020).choice([-1.,1.],size=(len(f),2,1))*hard
    np.savez_compressed(ROOT/'endpoint_descriptors.npz',hard_strong=hard.reshape(len(f),-1),
        soft_strong=soft.reshape(len(f),-1),random_flip=random.reshape(len(f),-1),
        symmetric_content=nonlinear.reshape(len(f),-1),slots=slots,contrast=contrast)
    pd.DataFrame(rows).to_csv(ROOT/'endpoint_strength.csv',index=False)
    pd.DataFrame(rotation).to_csv(ROOT/'invariance.csv',index=False)
    (ROOT/'EXTRACTION.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,
        symmetric_replay_max_abs=error,endpoint_exchange_exact=True,scope='PNG strong end, not anatomical stern'),indent=2),encoding='utf-8')


def evaluate():
    start=time.perf_counter(); f=pd.read_csv(old.prior.ROOT/'cohort.csv',dtype={'mmsi':str}); y=f.class_id.to_numpy()
    previous=np.load(PREV/'descriptors.npz')
    extras=dict(np.load(ROOT/'endpoint_descriptors.npz'))
    extras['soft_random_flip']=(extras['random_flip'].reshape(len(f),2,64)*np.abs(extras['contrast'])[:,:,None]).reshape(len(f),-1)
    extras['soft_absdiff']=np.abs(extras['soft_strong'])
    axis=np.concatenate((previous['base'],previous['axis']),axis=1)
    excluded=pd.read_csv(old.LOCAL/'EXCLUSIONS.csv').obs_id.to_numpy()
    metrics=[]; selection=[]; predictions=[]; outputs={}; interventions=[]
    for fold in range(3):
        train,val,test=old.prior.indices(f,fold)
        train,val,test=[ii[~f.iloc[ii].obs_id.isin(excluded).to_numpy()] for ii in (train,val,test)]
        for arm in ARMS:
            raw=axis if arm=='axis' else np.concatenate((axis,extras[arm]),axis=1)
            mean=raw[train].mean(0); std=np.maximum(raw[train].std(0),1e-6)
            x=(raw-mean)/std/np.sqrt(raw.shape[1]); candidates=[]
            for alpha in (.1,1.,10.):
                model=RidgeClassifier(alpha=alpha,solver='cholesky').fit(x[train],y[train],sample_weight=old.prior.weights(f,train))
                met=old.prior.metrics(y[val],model.decision_function(x[val]))
                candidates.append((met,alpha,model)); selection.append(dict(fold=fold,arm=arm,alpha=alpha,**met))
            _,alpha,model=max(candidates,key=lambda a:(a[0]['ba'],-a[0]['brier']))
            scores=model.decision_function(x[test]); outputs[(fold,arm)]=(test,scores)
            if arm=='axis':
                ref=pd.read_csv(PREV/'predictions.csv').query('fold==@fold and arm=="axis"').set_index('obs_id').loc[f.iloc[test].obs_id]
                assert np.array_equal(scores.argmax(1),ref.prediction.to_numpy())
            metrics.append(dict(fold=fold,arm=arm,alpha=alpha,**old.prior.metrics(y[test],scores)))
            np.savez_compressed(ROOT/f'head_f{fold}_{arm}.npz',mean=mean,std=std,coef=model.coef_,intercept=model.intercept_,dimension=raw.shape[1])
            other=scores.copy(); other[np.arange(len(test)),y[test]]=-np.inf
            margin=scores[np.arange(len(test)),y[test]]-other.max(1)
            for j,i in enumerate(test):
                predictions.append(dict(fold=fold,arm=arm,obs_id=int(f.iloc[i].obs_id),mmsi=f.iloc[i].mmsi,port=f.iloc[i].port,
                    product=f.iloc[i]['product'],class_id=int(y[i]),prediction=int(scores[j].argmax()),
                    correct=bool(scores[j].argmax()==y[i]),margin=float(margin[j])))
            if arm in ('hard_strong','soft_strong'):
                for intervention,scale in (('reverse',-1.),('zero',0.)):
                    altered=raw[test].copy(); altered[:,axis.shape[1]:]*=scale
                    alt=model.decision_function((altered-mean)/std/np.sqrt(raw.shape[1]))
                    interventions.append(dict(fold=fold,arm=arm,intervention=intervention,
                        flips=int(np.sum(alt.argmax(1)!=scores.argmax(1))),**old.prior.metrics(y[test],alt)))
                    np.savez_compressed(ROOT/f'intervention_f{fold}_{arm}_{intervention}.npz',obs_id=f.iloc[test].obs_id.to_numpy(),scores=alt)
            print('fit',fold,arm,metrics[-1],flush=True)
    p=pd.DataFrame(predictions); p.to_csv(ROOT/'predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(ROOT/'metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(ROOT/'validation_selection.csv',index=False)
    pd.DataFrame(interventions).to_csv(ROOT/'interventions.csv',index=False)
    summary=[]
    for arm in ARMS:
        yy=np.concatenate([y[outputs[(fold,arm)][0]] for fold in range(3)])
        ss=np.concatenate([outputs[(fold,arm)][1] for fold in range(3)])
        summary.append(dict(arm=arm,**old.prior.metrics(yy,ss)))
    pd.DataFrame(summary).to_csv(ROOT/'summary.csv',index=False)
    base=p[p.arm.eq('axis')].set_index('obs_id'); paired=[]
    for arm in ARMS[1:]:
        alt=p[p.arm.eq(arm)].set_index('obs_id').loc[base.index]
        paired.append(dict(arm=arm,rescue=int((alt.correct&~base.correct).sum()),harm=int((~alt.correct&base.correct).sum())))
    pd.DataFrame(paired).to_csv(ROOT/'paired.csv',index=False)
    for group in ('class_id','port'):
        p.groupby(['arm',group]).agg(n=('correct','size'),oa=('correct','mean')).reset_index().to_csv(ROOT/f'by_{group}.csv',index=False)
    (ROOT/'RECEIPT.json').write_text(json.dumps(dict(status='complete',seconds=time.perf_counter()-start,
        fits=len(ARMS)*3,identity_folds=3,no_true_bow_stern_labels=True,neural_training=False),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('extract','evaluate'))
    args=parser.parse_args()
    with threadpool_limits(limits=2): extract() if args.stage=='extract' else evaluate()
