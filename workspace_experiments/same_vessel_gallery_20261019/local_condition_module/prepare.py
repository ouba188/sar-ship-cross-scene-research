"""Snapshot local geometry and frozen early responses; preserve chip aspect ratio."""
import os
import argparse
os.environ['OMP_NUM_THREADS']='2'
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision.models import resnet50
from native_input import crop_pad, place_native

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'conditional_evidence_pilot'
WEIGHTS=Path('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')


def masks_for(view):
    h,w=view['h'],view['w']; p=view['priors']; target=np.zeros((h,w),bool); other=target.copy()
    for j,d in enumerate(p.get('detections',[])):
        if not d.get('shape_usable') or d['confidence']<.25: continue
        canvas=Image.new('L',(w,h)); ImageDraw.Draw(canvas).polygon([tuple(v) for v in d['corners']],fill=1)
        mask=np.asarray(canvas)>0
        if j==p['target_index']: target|=mask
        else: other|=mask
    other &= ~target
    return np.stack((target,other,~(target|other))).astype(np.float32)


def place(image,masks):
    h,w=image.shape; scale=96/max(h,w); nw=max(1,round(w*scale)); nh=max(1,round(h*scale))
    left=(96-nw)//2; top=(96-nh)//2
    x=torch.zeros(1,96,96); m=torch.zeros(3,96,96)
    x[:,top:top+nh,left:left+nw]=F.interpolate(torch.from_numpy(image.copy())[None,None].float()/255,size=(nh,nw),mode='bilinear',align_corners=False)[0]
    m[:,top:top+nh,left:left+nw]=F.interpolate(torch.from_numpy(masks)[None],size=(nh,nw),mode='nearest')[0]
    return x,m


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--legacy-resize',action='store_true',help='Reproduce historical resized inputs in a separate directory')
    args=parser.parse_args()
    canvas=96 if args.legacy_resize else 160
    grid=canvas//4
    output=ROOT/('legacy_resize_replay' if args.legacy_resize else 'native_crop_pad')
    output.mkdir(exist_ok=True)
    started=time.perf_counter(); torch.set_num_threads(2)
    f=pd.read_csv(PREV/'cohort.csv',dtype={'mmsi':str})
    text=(ROOT.parent/'gallery_data.js').read_text(encoding='utf-8')
    gallery=json.loads(text.removeprefix('window.GALLERY_DATA=').rstrip(';\n'))['observations']; del text
    snapshots=[]
    for row in f.itertuples():
        o=gallery[str(row.obs_id)]; assert o['instance_id']==row.instance_id
        snapshots.append(dict(obs_id=row.obs_id,images=o['images']))
    del gallery; gc.collect()
    # Check actual target coverage before allocating feature caches or loading the GPU.
    if not args.legacy_resize:
        failures=[]
        for snapshot in snapshots:
            for pol in ('vv','vh'):
                view=snapshot['images'][pol]
                try:
                    crop_pad(np.zeros((view['h'],view['w']),np.uint8),masks_for(view),canvas)
                except ValueError as exc:
                    failures.append(dict(obs_id=snapshot['obs_id'],pol=pol,reason=str(exc)))
        if failures:
            pd.DataFrame(failures).to_csv(output/'oversize_targets.csv',index=False)
            raise ValueError(f'{len(failures)} targets do not fit: see oversize_targets.csv; no cache overwritten')
    (output/'view_snapshot.json').write_text(json.dumps(snapshots),encoding='utf-8')
    n=len(f)
    maps=np.lib.format.open_memmap(output/'maps.npy',mode='w+',dtype=np.float16,shape=(n,2,64,grid,grid))
    masks=np.lib.format.open_memmap(output/'masks.npy',mode='w+',dtype=np.float32,shape=(n,2,3,grid,grid))
    conditions=np.zeros((n,2,3),np.float32); valid=np.zeros_like(conditions); rows=[]
    model=resnet50(weights=None); model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
    stem=torch.nn.Sequential(model.conv1,model.bn1,model.relu).eval().cuda(); del model
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    with torch.inference_mode():
        for first in range(0,n,32):
            batch=[]; region=[]
            for i in range(first,min(n,first+32)):
                for polidx,pol in enumerate(('vv','vh')):
                    view=snapshots[i]['images'][pol]; p=view['priors']
                    im=np.asarray(Image.open(f.iloc[i]['image_'+pol]).convert('L'))
                    assert im.shape==(view['h'],view['w'])
                    native_masks=masks_for(view)
                    if args.legacy_resize:
                        x,m=place(im,native_masks); placement={}
                    else:
                        x,m=place_native(im,native_masks,canvas)
                        _,_,placement=crop_pad(im,native_masks,canvas)
                    batch.append(x); region.append(m)
                    axis=p.get('ship_axis'); radar=p.get('radar_axis'); inc=p.get('radar_incidence')
                    if axis and radar is not None and p.get('radar_kind')=='orbit_los':
                        delta=axis['angle']-radar; conditions[i,polidx,:2]=[np.cos(2*delta),np.sin(2*delta)]; valid[i,polidx,:2]=1
                    if inc is not None and np.isfinite(inc) and p.get('radar_kind')=='orbit_los':
                        conditions[i,polidx,2]=inc; valid[i,polidx,2]=1
                    rows.append(dict(obs_id=int(f.iloc[i].obs_id),pol=pol,target_pixels=float(m[0].sum()),other_pixels=float(m[1].sum()),rest_pixels=float(m[2].sum()),axis_valid=int(valid[i,polidx,0]),incidence_valid=int(valid[i,polidx,2]),match_status=p.get('match_status'),width=view['w'],height=view['h']))
                    rows[-1].update(placement)
            x=torch.stack(batch).cuda().expand(-1,3,-1,-1)
            response=F.avg_pool2d(stem((x-mean)/std),2)
            mass=F.avg_pool2d(torch.stack(region),4)
            count=len(batch)//2
            maps[first:first+count]=response.cpu().numpy().reshape(count,2,64,grid,grid)
            masks[first:first+count]=mass.numpy().reshape(count,2,3,grid,grid)
            if first%1024==0: print('prepared',first,'/',n,flush=True)
    maps.flush(); masks.flush()
    np.savez(output/'conditions.npz',values=conditions,valid=valid)
    pd.DataFrame(rows).to_csv(output/'view_coverage.csv',index=False)
    info=dict(status='complete',n=n,seconds=time.perf_counter()-started,weights=str(WEIGHTS),features='frozen R50 conv1/bn/relu, area downsample24',canvas=96,source='fixed cohort, current view snapshot',notes='Masks applied after frozen stem cannot undo its receptive-field mixing')
    info['input_policy']='legacy_resize' if args.legacy_resize else 'native_target_crop_zero_pad_no_resampling'
    info['canvas']=canvas
    info['features']=f'frozen R50 conv1/bn/relu, fixed 2x2 response pooling, grid {grid}'
    (output/'PREPARATION.json').write_text(json.dumps(info,indent=2),encoding='utf-8'); print(info)


if __name__=='__main__': main()
