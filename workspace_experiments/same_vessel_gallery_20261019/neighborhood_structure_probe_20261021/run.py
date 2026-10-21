"""Fixed-head and matched-training tests of neighborhood information."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('level_previous',ROOT.parent/'neighborhood_level_split_20261021/run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
ARMS=('half_amplitude','tile_shuffle','global_shuffle','flat')


def transform(x,r,arm,oid,pol,seed):
    mu=float(x[r].mean())
    centered=(x-mu)*r
    z=.5+.5*centered
    if arm=='half_amplitude':
        return .5+.25*centered
    if arm=='flat':
        return np.full_like(x,.5)
    rng=np.random.default_rng(np.random.SeedSequence([20261021,int(oid),int(pol),int(seed)]))
    if arm=='global_shuffle':
        z[r]=rng.permutation(z[r])
    elif arm=='tile_shuffle':
        for a in range(0,x.shape[0],16):
            for b in range(0,x.shape[1],16):
                block=z[a:a+16,b:b+16]; mask=r[a:a+16,b:b+16]
                block[mask]=rng.permutation(block[mask])
    else:
        raise ValueError(arm)
    return z


def feature_key(arm,seed):
    return f'{arm}_s{seed}' if 'shuffle' in arm else arm


def extract():
    start=time.perf_counter(); f,views,disabled=old.control.load_data()
    cache=np.load(old.ROOT/'features.npz')
    stem=old.control.make_stem()
    keys=['half_amplitude','flat']+[f'{a}_s{s}' for a in ('tile_shuffle','global_shuffle') for s in range(3)]
    arrays={k:np.zeros((len(f),2,128),np.float32) for k in keys}
    checks=[]; checked_views=0
    with torch.inference_mode():
        for first in range(0,len(f),12):
            images=[]; regions=[]; addresses=[]; originals=[]; orig_regions=[]; orig_addresses=[]
            for i in range(first,min(first+12,len(f))):
                if int(f.iloc[i].obs_id)==31237:
                    continue
                row=f.iloc[i]
                for p,pol in enumerate(('vv','vh')):
                    pix,masks,_,_,place=old.control.prepare_view(row,views[i]['images'][pol],pol)
                    ok=bool(masks[0].any() and masks[2].any() and not(p==1 and disabled[i]))
                    assert ok==bool(cache['usable'][i,p])
                    assert place['input_scale']==1 and place['native_target_pixels']==place['retained_target_pixels']
                    if not ok:continue
                    x=pix[0]; r=masks[2].astype(bool); mu=float(x[r].mean())
                    assert abs(mu-cache['means'][i,p])<2e-7
                    original=.5+.5*(x-mu)*r
                    if first==0:
                        originals.append(original[None]);orig_regions.append(r[None]);orig_addresses.append((i,p))
                    for k in keys:
                        if '_s' in k and k.rsplit('_s',1)[1].isdigit():
                            arm,s=k.rsplit('_s',1); s=int(s)
                        else:arm,s=k,0
                        z=transform(x,r,arm,row.obs_id,p,s)
                        assert np.all(z[~r]==.5)
                        if 'shuffle' in arm:
                            np.testing.assert_array_equal(np.sort(z[r]),np.sort(original[r]))
                            if arm=='tile_shuffle':
                                for a in range(0,160,16):
                                    for b in range(0,160,16):
                                        rr=r[a:a+16,b:b+16]
                                        np.testing.assert_array_equal(np.sort(z[a:a+16,b:b+16][rr]),
                                                                      np.sort(original[a:a+16,b:b+16][rr]))
                        elif arm=='half_amplitude':
                            np.testing.assert_allclose((z[r]-.5)*2,original[r]-.5,atol=1e-7,rtol=0)
                        else:assert np.all(z==.5)
                        images.append(z[None]);regions.append(r[None]);addresses.append((k,i,p))
                    checked_views+=1
            if not images:continue
            x=torch.tensor(np.stack(images),device='cuda'); r=torch.tensor(np.stack(regions),device='cuda',dtype=torch.float32)
            feat=old.descriptor(stem,x,r).cpu().numpy()
            for j,(k,i,p) in enumerate(addresses):arrays[k][i,p]=feat[j]
            if originals:
                out=old.descriptor(stem,torch.tensor(np.stack(originals),device='cuda'),
                    torch.tensor(np.stack(orig_regions),device='cuda',dtype=torch.float32)).cpu().numpy()
                for j,(i,p) in enumerate(orig_addresses):
                    err=float(np.abs(out[j]-cache['centered'][i,p]).max()); assert err<2e-5
                    checks.append(dict(test='original_descriptor_replay',obs_id=int(f.iloc[i].obs_id),pol=p,max_abs=err))
            if first%480==0:print('extract',first,len(f),round(time.perf_counter()-start,1),flush=True)
    np.savez_compressed(ROOT/'features.npz',**arrays,means=cache['means'],usable=cache['usable'])
    pd.DataFrame(checks).to_csv(ROOT/'descriptor_replay.csv',index=False)
    (ROOT/'EXTRACTION.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,rows=len(f),
        checked_views=checked_views,histogram_checks='all usable views, all shuffle realizations',
        native_scale=1,keys=keys),indent=2),encoding='utf-8')


def matrices(cache,key,tr,normalization=None):
    if normalization is None:
        data={'raw':cache[key],'centered':cache[key],'means':cache['means'],'usable':cache['usable']}
        result,stats=old.standardize(data,tr)
        return result['separated'],stats
    usable=cache['usable'].astype(bool)
    z=(cache[key]-normalization['centered_mean'])/normalization['centered_std']
    z*=usable[:,:,None]
    level=(cache['means']-normalization['level_mean'])/normalization['level_std']
    level*=usable
    return np.concatenate((z.reshape(len(z),-1),level,usable.astype(np.float32)),1),None


def train():
    start=time.perf_counter();torch.set_num_threads(2)
    f,splits=old.relation.load_data(); y=f.class_id.to_numpy(); yy=torch.tensor(y,device='cuda')
    cache=np.load(ROOT/'features.npz'); original_cache=np.load(old.ROOT/'features.npz')
    raw=np.load(old.five.AXIS/'descriptors.npz');raw_base=np.concatenate((raw['base'],raw['axis']),1)
    raw_five=np.load(old.strip.PREV/'keep_refs.npy',mmap_mode='r').reshape(len(f),60,64)
    metrics=[];selections=[];replays=[]
    for fold,(tr,val,test) in enumerate(splits):
        norm=np.load(old.ROOT/f'normalization_f{fold}.npz')
        original,_=old.standardize(original_cache,tr)
        weights=torch.tensor(old.prior.weights(f,tr),device='cuda',dtype=torch.float32)
        for seed in range(3):
            base_np=old.strip.frozen_scores(fold,seed,raw_base,raw_five);base=torch.tensor(base_np,device='cuda')
            prior_scores=np.load(old.ROOT/f'scores_f{fold}_baseline_s{seed}.npz')
            np.testing.assert_array_equal(test,prior_scores['rows']);np.testing.assert_allclose(base_np[test],prior_scores['scores'],atol=1e-6,rtol=0)
            saved=torch.load(old.ROOT/f'head_f{fold}_separated_s{seed}.pt',map_location='cpu',weights_only=False)
            fixed=old.Residual().cuda().eval();fixed.load_state_dict(saved['state'])
            ref=np.load(old.ROOT/f'scores_f{fold}_separated_s{seed}.npz')
            with torch.no_grad():check=base_np[test]+fixed(torch.tensor(original['separated'][test],device='cuda')).cpu().numpy()
            err=float(np.abs(check-ref['scores']).max());assert err<2e-5
            replays.append(dict(fold=fold,seed=seed,max_abs=err))
            np.savez_compressed(ROOT/f'scores_f{fold}_reference_s{seed}.npz',rows=test,scores=ref['scores'],base=base_np[test])
            for kind,scores in (('baseline',base_np[test]),('reference',ref['scores'])):
                metrics.append(dict(fold=fold,seed=seed,arm=kind,mode='reference',partition='test',step=0 if kind=='baseline' else saved['step'],
                                    **old.control.metrics(y[test],scores)))
            for arm in ARMS:
                key=feature_key(arm,seed)
                fixed_x,_=matrices(cache,key,tr,norm)
                with torch.no_grad():fixed_score=base_np[test]+fixed(torch.tensor(fixed_x[test],device='cuda')).cpu().numpy()
                np.savez_compressed(ROOT/f'scores_f{fold}_{arm}_fixed_s{seed}.npz',rows=test,scores=fixed_score,
                                    base=base_np[test],reference=ref['scores'])
                metrics.append(dict(fold=fold,seed=seed,arm=arm,mode='fixed',partition='test',step=saved['step'],
                                    **old.control.metrics(y[test],fixed_score)))
                xx,stats=matrices(cache,key,tr)
                np.savez_compressed(ROOT/f'normalization_f{fold}_{arm}_s{seed}.npz',**stats)
                xx=torch.tensor(xx,device='cuda');torch.manual_seed(20261021+seed)
                head=old.Residual().cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
                rng=np.random.default_rng(20261021+seed+31*fold);best=None;chosen=0;state=None
                for step in range(151):
                    if step in (0,50,100,150):
                        with torch.no_grad():score=(base[val]+head(xx[val])).cpu().numpy()
                        met=old.control.metrics(y[val],score);selections.append(dict(fold=fold,seed=seed,arm=arm,step=step,**met))
                        rank=(met['ba'],-met['brier'])
                        if best is None or rank>best:
                            best=rank;chosen=step;state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                        if step==0:np.testing.assert_array_equal(score,base_np[val])
                    if step==150:break
                    take=rng.integers(len(tr),size=128);ii=tr[take]
                    target=2*F.one_hot(yy[ii],8).float()-1
                    loss=((base[ii]+head(xx[ii])-target).square().mean(-1)*weights[take]).mean()
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                with torch.no_grad():
                    for part,ii in (('terminal_train',tr),('terminal_val',val)):
                        scores=(base[ii]+head(xx[ii])).cpu().numpy()
                        metrics.append(dict(fold=fold,seed=seed,arm=arm,mode='matched',partition=part,step=150,**old.control.metrics(y[ii],scores)))
                head.load_state_dict(state)
                torch.save(dict(state=state,step=chosen),ROOT/f'head_f{fold}_{arm}_s{seed}.pt')
                with torch.no_grad():residual=head(xx).cpu().numpy()
                for part,ii in (('train',tr),('val',val),('test',test)):
                    metrics.append(dict(fold=fold,seed=seed,arm=arm,mode='matched',partition=part,step=chosen,**old.control.metrics(y[ii],base_np[ii]+residual[ii])))
                np.savez_compressed(ROOT/f'scores_f{fold}_{arm}_matched_s{seed}.npz',rows=test,
                    scores=base_np[test]+residual[test],base=base_np[test],reference=ref['scores'],
                    source_constant=base_np[test]+residual[tr].mean(0))
                print('fit',fold,seed,arm,chosen,metrics[-1]['oa'],metrics[-1]['ba'],flush=True)
            pd.DataFrame(metrics).to_csv(ROOT/'metrics.csv',index=False)
            pd.DataFrame(selections).to_csv(ROOT/'validation_selection.csv',index=False)
            pd.DataFrame(replays).to_csv(ROOT/'reference_replay.csv',index=False)
    (ROOT/'TRAINING.json').write_text(json.dumps(dict(status='complete',new_fits=36,folds=3,paired_runs=3,
        steps=150,seconds=time.perf_counter()-start,selection='source BA then Brier',
        fixed_head_tests=36,shuffle_realizations=3),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('extract','train'))
    args=parser.parse_args()
    extract() if args.stage=='extract' else train()
