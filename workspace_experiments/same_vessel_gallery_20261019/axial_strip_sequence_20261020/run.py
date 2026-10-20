"""Native-scale oriented strips on top of a frozen five-point residual model."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent / 'sampling_preservation_20261020'
spec = importlib.util.spec_from_file_location('fivepoint', PREV / 'run.py')
five = importlib.util.module_from_spec(spec); spec.loader.exec_module(five)
old, prior, LOCAL = five.old, five.prior, five.LOCAL
ARMS = ('strip_mean', 'strip_conv', 'strip_shuffle')
LENGTH = 49
STEP = 4.


def strip_layout(view, placement, rotation=0):
    axis = view['priors'].get('ship_axis')
    if not axis or not placement['target_present']:
        return np.zeros((2, 3, LENGTH, 2), np.float32), np.zeros((2, LENGTH), np.float32), [0, 0]
    cx, cy = axis['center']
    center = np.array([cx + placement['pad_left'] - placement['crop_x'],
                       cy + placement['pad_top'] - placement['crop_y']], float)
    for _ in range(rotation): center = np.array([center[1], 159 - center[0]])
    geom, _ = five.geometry(view)
    ship_angle = geom[0] - rotation * np.pi / 2
    angles = [ship_angle, geom[1] - rotation * np.pi / 2 if geom[3] else 0.]
    coords, validity, halves = [], [], []
    for angle in angles:
        delta = ship_angle - angle
        a = abs(np.cos(delta)) * axis['length_px'] / 2 + abs(np.sin(delta)) * axis['width_px'] / 2
        b = abs(np.sin(delta)) * axis['length_px'] / 2 + abs(np.cos(delta)) * axis['width_px'] / 2
        half = int(np.ceil((a + 8) / STEP))
        if half > LENGTH // 2: raise ValueError(f'Uncovered target: half={half}; increase padded sequence length')
        t = np.arange(-LENGTH // 2 + 1, LENGTH // 2 + 1) * STEP
        along = np.array([np.cos(angle), np.sin(angle)])
        across = np.array([-np.sin(angle), np.cos(angle)])
        coords.append(center + t[None, :, None] * along + np.array([0, -b-4, b+4])[:, None, None] * across)
        validity.append((np.abs(t) <= half * STEP).astype(np.float32)); halves.append(half)
    return np.asarray(coords, np.float32), np.asarray(validity), halves


def symmetric_permutation(half, seed):
    order = np.arange(LENGTH)
    rng = np.random.default_rng(seed)
    dest = np.arange(1, half + 1)
    src = rng.permutation(dest) * rng.choice([-1, 1], size=half)
    mid = LENGTH // 2
    order[mid + dest] = mid + src
    order[mid - dest] = mid - src
    return order


def sample_strips(features, masks, coords, valid):
    b, c, h, w = features.shape
    # conv1 stride2/pad3 then avg_pool2 gives native receptive-field centers 4*i+1.
    fxy = (coords - 1.) / 4.
    grid = 2 * (fxy + .5) / features.new_tensor([w, h]) - 1
    content = F.grid_sample(features, grid.reshape(b, 6, LENGTH, 2), align_corners=False)
    content = content.permute(0, 2, 3, 1).reshape(b, 2, 3, LENGTH, c)
    grid_native = 2 * (coords + .5) / 160 - 1
    area = F.avg_pool2d(masks, 3, stride=1, padding=1)
    mass = F.grid_sample(area, grid_native.reshape(b, 6, LENGTH, 2), align_corners=False)
    mass = mass.permute(0, 2, 3, 1).reshape(b, 2, 3, LENGTH, 2)
    mass = mass * valid[:, :, None, :, None]
    return content, mass


def extract():
    start = time.perf_counter(); torch.set_num_threads(2)
    f = pd.read_csv(prior.ROOT / 'cohort.csv', dtype={'mmsi': str})
    views = json.loads((LOCAL / 'view_snapshot.json').read_text(encoding='utf-8'))
    assert [v['obs_id'] for v in views] == f.obs_id.tolist()
    disabled = pd.read_csv(LOCAL / 'crossview_alignment.csv').set_index('obs_id')
    excluded = set(pd.read_csv(LOCAL / 'EXCLUSIONS.csv').obs_id)
    shape = (len(f), 2, 4, 2, 3, LENGTH)
    values = np.lib.format.open_memmap(ROOT/'tokens.npy', mode='w+', dtype=np.float16, shape=shape+(64,))
    masses = np.lib.format.open_memmap(ROOT/'masses.npy', mode='w+', dtype=np.float16, shape=shape+(2,))
    permutations = np.lib.format.open_memmap(ROOT/'permutations.npy', mode='w+', dtype=np.int16,
                                            shape=(len(f), 2, 4, 2, LENGTH))
    values[:] = 0; masses[:] = 0; permutations[:] = np.arange(LENGTH)
    backbone = resnet50(weights=None)
    backbone.load_state_dict(torch.load(old.WEIGHTS, map_location='cpu', weights_only=True))
    stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu).cuda().eval(); del backbone
    norm = torch.tensor([.485,.456,.406], device='cuda')[None,:,None,None]
    std = torch.tensor([.229,.224,.225], device='cuda')[None,:,None,None]
    coverage = []
    with torch.inference_mode():
        for first in range(0,len(f),16):
            images=[]; masks=[]; entries=[]; layouts=[[] for _ in range(4)]; windows=[[] for _ in range(4)]
            for i in range(first,min(first+16,len(f))):
                row=f.iloc[i]
                for p, pol in enumerate(('vv','vh')):
                    v=views[i]['images'][pol]
                    skip = row.obs_id in excluded or (p == 1 and disabled.loc[row.obs_id,'disable_new_vh'])
                    raw=np.asarray(Image.open(row['image_'+pol]).convert('L'))
                    pix, mm, placement=old.crop_pad(raw,old.masks_for(v),160)
                    axis=v['priors'].get('ship_axis')
                    skip = bool(skip or not axis or not placement['target_present'])
                    g,_=five.geometry(v)
                    coverage.append(dict(obs_id=int(row.obs_id),pol=pol,disabled=skip,ship_valid=bool(axis),
                        radar_valid=bool(g[3]),length_px=float(axis['length_px']) if axis else 0,
                        width_px=float(axis['width_px']) if axis else 0,**placement))
                    if skip: continue
                    assert placement['retained_target_pixels']==placement['native_target_pixels']
                    images.append(torch.from_numpy(pix.copy())[None].float()/255)
                    masks.append(torch.from_numpy(np.stack((mm[0],mm[1]+mm[2])))); entries.append((i,p))
                    for k in range(4):
                        co,valid,halves=strip_layout(v,placement,k)
                        layouts[k].append(co); windows[k].append(valid)
                        for ref in range(2):
                            permutations[i,p,k,ref]=symmetric_permutation(halves[ref],20261020+int(row.obs_id)*13+p*3+ref)
            if not entries: continue
            x=torch.stack(images).cuda(); m=torch.stack(masks).cuda()
            for k in range(4):
                xx=torch.rot90(x,k,(-2,-1)).expand(-1,3,-1,-1)
                ff=F.avg_pool2d(stem((xx-norm)/std),2)
                co=torch.tensor(np.stack(layouts[k]),device='cuda'); valid=torch.tensor(np.stack(windows[k]),device='cuda')
                vv,ww=sample_strips(ff,torch.rot90(m,k,(-2,-1)),co,valid)
                vv=vv.cpu().numpy(); ww=ww.cpu().numpy()
                for j,(i,p) in enumerate(entries): values[i,p,k]=vv[j]; masses[i,p,k]=ww[j]
            if first % 512 == 0: print('extract',first,len(f),round(time.perf_counter()-start,1),flush=True)
    for cache in (values,masses,permutations): cache.flush()
    pd.DataFrame(coverage).to_csv(ROOT/'coverage.csv',index=False)
    (ROOT/'EXTRACTION.json').write_text(json.dumps(dict(status='complete',seconds=time.perf_counter()-start,
        shape=shape,channels=64,canvas=160,step_px=STEP,resize=False,cache_dtype='float16'),indent=2),encoding='utf-8')


class SymmetricConv(nn.Module):
    def __init__(self, spatial):
        super().__init__(); self.center=nn.Linear(8,8); self.spatial=spatial
        self.side=nn.Linear(8,8,bias=False) if spatial else None

    def forward(self,h,m):
        if not self.spatial: return F.gelu(self.center(h)) * (m>0).to(h.dtype)
        weighted=h*m
        left=F.pad(weighted[...,:-1,:],(0,0,1,0)); right=F.pad(weighted[...,1:,:],(0,0,0,1))
        ml=F.pad(m[...,:-1,:],(0,0,1,0)); mr=F.pad(m[...,1:,:],(0,0,0,1))
        num=F.linear(weighted,self.center.weight)+self.side(left+right)
        out=num/((m+ml+mr)/3).clamp_min(1e-4)+self.center.bias
        return F.gelu(out)*(m>0).to(h.dtype)


class StripResidual(nn.Module):
    def __init__(self,spatial):
        super().__init__(); self.content=nn.Linear(64,8)
        self.layers=nn.ModuleList([SymmetricConv(spatial),SymmetricConv(spatial)])
        self.output=nn.Linear(2*2*2*2*8,8,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,x,m):
        # B, polarization, rotation, reference, lane, length, channels.
        h=F.gelu(self.content(x))
        h=h.unsqueeze(-3).expand(*h.shape[:-2],2,*h.shape[-2:])
        w=m.transpose(-1,-2).unsqueeze(-1)
        h=h*(w>0).to(h.dtype)
        for layer in self.layers: h=layer(h,w)
        pooled=(h*w).sum(-2)/w.sum(-2).clamp_min(1e-6)
        # Lateral lanes and axis endpoints are unordered; no reliable bow/stern.
        pooled=torch.stack((pooled[...,0,:,:],pooled[...,1:,:,:].mean(-3)),dim=-3)
        pooled=pooled.mean(2)
        return self.output(pooled.flatten(1))


def shuffle(x,m,perm):
    index=perm[:,:,:,:,None,:,None]
    return torch.gather(x,-2,index.expand(*x.shape[:-1],x.shape[-1])),torch.gather(m,-2,index.expand(*m.shape[:-1],m.shape[-1]))


def frozen_scores(fold,seed,raw_base,raw_five):
    base=np.concatenate([five.baseline(raw_base[i:i+512],fold).astype(np.float32) for i in range(0,len(raw_base),512)])
    head=torch.load(PREV/'mse_aligned'/f'head_f{fold}_keep_refs_s{seed}.pt',map_location='cpu',weights_only=False)
    model=five.Residual().cuda(); model.load_state_dict(head['state']); model.eval()
    with torch.inference_mode():
        for i in range(0,len(base),512):
            z=torch.tensor((raw_five[i:i+512]-head['mean'])/head['std'],device='cuda')
            base[i:i+512]+=model(z).cpu().numpy()
    del model
    return base


def moments(x,m,idx):
    total=torch.zeros(64,device='cuda',dtype=torch.float64); squares=total.clone(); count=0.
    for start in range(0,len(idx),64):
        xx=torch.tensor(np.asarray(x[idx[start:start+64]]),device='cuda',dtype=torch.float32)
        ww=torch.tensor(np.asarray(m[idx[start:start+64]]),device='cuda',dtype=torch.float32).sum(-1,keepdim=True)
        dims=tuple(range(xx.ndim-1)); total+=(xx*ww).sum(dims).double(); squares+=(xx.square()*ww).sum(dims).double(); count+=ww.sum().double()
    mean=total/count; std=(squares/count-mean.square()).clamp_min(1e-8).sqrt()
    return mean.float(),std.float()


def batch_data(x,m,p,ii,mean,std,arm):
    xx=torch.tensor(np.asarray(x[ii]),device='cuda',dtype=torch.float32)
    mm=torch.tensor(np.asarray(m[ii]),device='cuda',dtype=torch.float32)
    if arm=='strip_shuffle': xx,mm=shuffle(xx,mm,p[ii])
    return (xx-mean)/std,mm


def predict(model,x,m,p,ii,mean,std,arm,base):
    result=[]
    with torch.inference_mode():
        for start in range(0,len(ii),128):
            jj=ii[start:start+128]; xx,mm=batch_data(x,m,p,jj,mean,std,arm)
            result.append((base[jj]+model(xx,mm)).cpu().numpy())
    return np.concatenate(result)


def train(resume=False):
    started=time.perf_counter(); torch.set_num_threads(2)
    f=pd.read_csv(prior.ROOT/'cohort.csv',dtype={'mmsi':str}); y=f.class_id.to_numpy()
    excluded=pd.read_csv(LOCAL/'EXCLUSIONS.csv').obs_id.to_numpy()
    cache=np.load(ROOT/'tokens.npy',mmap_mode='r'); mass=np.load(ROOT/'masses.npy',mmap_mode='r')
    x=cache; m=mass
    p=torch.tensor(np.load(ROOT/'permutations.npy'),device='cuda',dtype=torch.int64)
    d=np.load(five.AXIS/'descriptors.npz'); raw_base=np.concatenate((d['base'],d['axis']),1)
    raw_five=np.load(PREV/'keep_refs.npy',mmap_mode='r').reshape(len(f),60,64)
    oldpred=pd.read_csv(PREV/'mse_aligned/predictions.csv'); oldpred=oldpred[oldpred.arm.eq('keep_refs')]
    def previous(name):
        path=ROOT/name
        return pd.read_csv(path,dtype={'mmsi':str}).to_dict('records') if resume and path.exists() else []
    metrics=previous('metrics.csv'); selections=previous('validation_selection.csv')
    predictions=previous('predictions.csv'); interventions=previous('shuffle_interventions.csv')
    finished={(int(r['fold']),int(r['seed']),r['arm']) for r in metrics if r['partition']=='test'}
    targets=torch.tensor(y,device='cuda')
    for fold in range(3):
        if all((fold,seed,arm) in finished for seed in range(3) for arm in ('frozen_fivepoint',)+ARMS): continue
        tr,val,test=prior.indices(f,fold)
        tr,val,test=[ii[~f.iloc[ii].obs_id.isin(excluded).to_numpy()] for ii in (tr,val,test)]
        mean,std=moments(x,m,tr)
        weights=torch.tensor(prior.weights(f,tr),device='cuda',dtype=torch.float32)
        for seed in range(3):
            if all((fold,seed,arm) in finished for arm in ('frozen_fivepoint',)+ARMS): continue
            base_np=frozen_scores(fold,seed,raw_base,raw_five); base=torch.tensor(base_np,device='cuda')
            expected=oldpred[oldpred.fold.eq(fold)&oldpred.seed.eq(seed)].set_index('obs_id').loc[f.iloc[test].obs_id]
            assert np.array_equal(base_np[test].argmax(1),expected.prediction.to_numpy())
            for arm in ('frozen_fivepoint',)+ARMS:
                if (fold,seed,arm) in finished: continue
                if arm!='frozen_fivepoint':
                    torch.manual_seed(20261020+seed); torch.cuda.manual_seed_all(20261020+seed)
                    rng=np.random.default_rng(20261020+seed+31*fold)
                    model=StripResidual(arm!='strip_mean').cuda()
                    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                    best=None; state=None; chosen=0
                    for step in range(151):
                        if step in (0,50,100,150):
                            model.eval(); scores=predict(model,x,m,p,val,mean,std,arm,base); met=prior.metrics(y[val],scores)
                            selections.append(dict(fold=fold,seed=seed,arm=arm,step=step,**met))
                            key=(met['ba'],-met['brier'])
                            if best is None or key>best:
                                best=key; chosen=step; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                            if step==0: assert np.array_equal(scores,base_np[val])
                        if step==150: break
                        take=rng.integers(len(tr),size=128); ii=tr[take]
                        xx,mm=batch_data(x,m,p,ii,mean,std,arm)
                        model.train(); opt.zero_grad(set_to_none=True)
                        scores=base[ii]+model(xx,mm)
                        wanted=2*F.one_hot(targets[ii],8).float()-1
                        loss=((scores-wanted).square().mean(-1)*weights[take]).mean()
                        if not torch.isfinite(loss): raise RuntimeError((fold,seed,arm,step))
                        loss.backward(); opt.step()
                    for name,ii in (('terminal_train',tr),('terminal_val',val)):
                        metrics.append(dict(fold=fold,seed=seed,arm=arm,step=150,partition=name,
                            **prior.metrics(y[ii],predict(model,x,m,p,ii,mean,std,arm,base))))
                    model.load_state_dict(state); model.eval()
                    torch.save(dict(state=state,mean=mean.cpu(),std=std.cpu(),step=chosen),ROOT/f'head_f{fold}_{arm}_s{seed}.pt')
                else: chosen=0
                for name,ii in (('train',tr),('val',val),('test',test)):
                    scores=base_np[ii] if arm=='frozen_fivepoint' else predict(model,x,m,p,ii,mean,std,arm,base)
                    metrics.append(dict(fold=fold,seed=seed,arm=arm,step=chosen,partition=name,**prior.metrics(y[ii],scores)))
                    if name=='test':
                        np.savez_compressed(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz',rows=ii,scores=scores,base=base_np[ii])
                        for j,i in enumerate(ii):
                            predictions.append(dict(obs_id=int(f.iloc[i].obs_id),fold=fold,seed=seed,arm=arm,step=chosen,
                                mmsi=f.iloc[i].mmsi,port=f.iloc[i].port,product=f.iloc[i]['product'],class_id=int(y[i]),
                                prediction=int(scores[j].argmax()),base_prediction=int(base_np[i].argmax()),correct=bool(scores[j].argmax()==y[i])))
                if arm=='strip_conv':
                    ss=predict(model,x,m,p,test,mean,std,'strip_shuffle',base)
                    original=np.load(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz')['scores']
                    interventions.append(dict(fold=fold,seed=seed,step=chosen,flips=int(np.sum(ss.argmax(1)!=original.argmax(1))),
                        score_l1=float(np.abs(ss-original).mean()),**prior.metrics(y[test],ss)))
                print('fit',fold,seed,arm,chosen,metrics[-1],flush=True)
                if arm!='frozen_fivepoint': del model,opt,xx,mm,scores,loss
                pd.DataFrame(selections).to_csv(ROOT/'validation_selection.csv',index=False)
                pd.DataFrame(predictions).to_csv(ROOT/'predictions.csv',index=False)
                pd.DataFrame(interventions).to_csv(ROOT/'shuffle_interventions.csv',index=False)
                # A test metrics row is the resume marker; publish it last.
                pd.DataFrame(metrics).to_csv(ROOT/'metrics.csv',index=False)
    (ROOT/'RECEIPT.json').write_text(json.dumps(dict(status='complete',seconds=time.perf_counter()-started,
        fits=27,steps=150,seeds=3,folds=3,base='frozen MSE keep_refs',resumed=resume,
        timing_scope='this invocation only; prior interrupted fits retained' if resume else 'all fits',
        parameters={a:sum(v.numel() for v in StripResidual(a!='strip_mean').parameters()) for a in ARMS},
        trained='two symmetric sequence convolutions over oriented strip tokens; backbone and fivepoint reader frozen',
        scope='development cohort; no full support adaptation'),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('extract','train')); parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    extract() if args.stage=='extract' else train(resume=args.resume)
