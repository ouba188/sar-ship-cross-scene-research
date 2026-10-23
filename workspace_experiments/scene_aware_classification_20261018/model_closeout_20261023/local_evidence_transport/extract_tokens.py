"""Preserve native-scale local evidence before region aggregation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
import time
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet50

HERE = Path(__file__).resolve().parent
UP = HERE.parent/'regional_utility_pilot/upstream'
ROOT = HERE.parents[1]/'eight_class_evidence_20261023'
spec = importlib.util.spec_from_file_location('token_geometry', ROOT/'obb_dynamic_readout/common.py')
geo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo)


def slots(g, footprint):
    if g is None:
        return np.zeros((20, 160, 160))
    cx, cy, length, width, angle = g[:5]
    u = ((geo.XX-cx)*np.cos(angle)+(geo.YY-cy)*np.sin(angle))/length
    v = (-(geo.XX-cx)*np.sin(angle)+(geo.YY-cy)*np.cos(angle))/width
    region = geo.soft_regions(g)*footprint
    ix = np.clip(np.floor((u+.5)*4).astype(int), 0, 3)
    iy = (v>=0).astype(int)
    end = (u>=0).astype(int)
    masks = [region[0]*(ix==a)*(iy==b) for a in range(4) for b in range(2)]
    masks += [region[2]*(ix==a)*(iy==b) for a in range(4) for b in range(2)]
    masks += [region[1]*(end==a)*(iy==b) for a in range(2) for b in range(2)]
    out = np.stack(masks)
    # This detects unintended field-of-view enlargement or missing edge pixels.
    assert np.max(np.abs(out.sum(0)-region.sum(0)))<1e-10
    return out


def cell_mass(mask, stride):
    if stride==4:
        return geo.cell_mass(mask)
    edges = np.r_[-.5, np.arange(stride/2,160-stride/2,stride),159.5]
    points = edges+.5
    idx = np.minimum(np.floor(points).astype(int),159)
    frac = points-idx
    def integral(a):
        prefix = np.concatenate([np.zeros((*a.shape[:-1],1)),np.cumsum(a,axis=-1)],-1)
        values = prefix[...,idx]+a[...,idx]*frac
        return np.diff(values,axis=-1)
    return integral(integral(mask).swapaxes(-1,-2)).swapaxes(-1,-2)


def main():
    tick = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    table = pd.read_csv(UP/'table.csv',dtype={'mmsi':str})
    valid = np.load(UP/'valid.npz')['valid'].astype(bool)
    wanted = set(table.obs_id)
    snapshots = {s['obs_id']:s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if s['obs_id'] in wanted}
    n = len(table)
    features = dict(stem=np.zeros((n,2,20,64),np.float32),layer1=np.zeros((n,2,20,256),np.float32))
    mass = np.zeros((n,2,20),np.float32)
    neff = np.zeros((n,2,2,20),np.float32)
    geometry = np.zeros((n,2,6),np.float32)
    seen = np.zeros((n,2),bool)
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(geo.WEIGHTS,map_location='cpu',weights_only=True))
    model = model.cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std = torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    last = tick
    with torch.inference_mode():
        for first in range(0,n,8):
            addresses,images,masks = [],[],[]
            for i in range(first,min(first+8,n)):
                for p in range(2):
                    if not valid[i,p]:
                        continue
                    pix,fp,g,place = geo.native(table,snapshots,i,p)
                    if g is None:
                        continue
                    geometry[i,p] = g
                    m = slots(g,fp)
                    mass[i,p] = m.sum((-1,-2))
                    masks.append(m)
                    images.append(pix[None].astype(np.float32)/255)
                    addresses.append((i,p))
            if not addresses:
                continue
            x = torch.tensor(np.stack(images),device='cuda')
            eff = {key:[] for key in features}
            for k in range(4):
                xx = x.rot90(k,(-2,-1)).expand(-1,3,-1,-1)
                stem = model.relu(model.bn1(model.conv1((xx-mean)/std)))
                layer1 = model.layer1(model.maxpool(stem))
                for name,z,stride in [('stem',stem,2),('layer1',layer1,4)]:
                    w = np.stack([cell_mass(np.rot90(m,k,(-2,-1)),stride) for m in masks]).reshape(len(masks),20,-1)
                    sums = w.sum(-1)
                    eff[name].append(sums**2/np.maximum((w*w).sum(-1),1e-20))
                    norm = torch.tensor(w/np.maximum(sums[...,None],1e-20),device='cuda',dtype=torch.float32)
                    desc = torch.einsum('btu,bdu->btd',norm,z.flatten(2)).cpu().numpy()
                    for b,(i,p) in enumerate(addresses):
                        features[name][i,p] += desc[b]/4
            for b,(i,p) in enumerate(addresses):
                seen[i,p] = True
                for l,name in enumerate(features):
                    neff[i,p,l] = np.min(eff[name],axis=0)[b]
            if time.perf_counter()-last>20:
                print('extract',first+len(set(i for i,p in addresses)),'/',n,round(time.perf_counter()-tick,1),flush=True)
                last = time.perf_counter()
    assert all(np.isfinite(f).all() for f in features.values())
    np.savez_compressed(HERE/'tokens.npz',**features,mass=mass,neff=neff,geometry=geometry,valid=seen,instance_ids=table.instance_id.to_numpy(dtype=str),regions=np.r_[np.zeros(8,int),np.full(8,2),np.ones(4,int)])
    rows=[]
    for l,name in enumerate(features):
        for r,ids in [('body',np.arange(8)),('side',np.arange(8,16)),('end',np.arange(16,20))]:
            a = neff[:,:,l,ids][seen]
            rows.append(dict(layer=name,region=r,median_neff=float(np.median(a)),fraction_neff_lt2=float((a<2).mean()),fraction_neff_lt4=float((a<4).mean())))
    pd.DataFrame(rows).to_csv(HERE/'token_resolution.csv',index=False)
    receipt=dict(n=n,valid_pol=int(seen.sum()),seconds=time.perf_counter()-tick,slots=20,rf=dict(stem=7,layer1=35),raw_scope='same body/end/side masks as frozen OBB',new_weights=0)
    (HERE/'extraction.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(receipt,flush=True)


if __name__=='__main__':
    main()
