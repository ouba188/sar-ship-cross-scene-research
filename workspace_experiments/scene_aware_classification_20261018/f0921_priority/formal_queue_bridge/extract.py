"""Frozen per-channel OBB masks and matched R50/R18 features on restored PNGs."""
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import resnet18,resnet50
from ultralytics import YOLO

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'prepool_local_probe'))
from extract_local import spatial_grid,pool


def normalize(x):
    return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]


def mask_from_boxes(records,anchor,size=224):
    target=np.zeros((size,size),np.uint8);other=np.zeros_like(target)
    used=[];ambiguous=0
    for pol in [0,1]:
        current=[r for r in records if r['pol']==pol]
        inside=[r for r in current if cv2.pointPolygonTest(np.array(r['corners'],np.float32),tuple(map(float,anchor)),False)>=0]
        selected=inside[0] if len(inside)==1 else None
        ambiguous+=int(len(inside)>1)
        for r in current:
            dest=target if r is selected else other
            cv2.fillPoly(dest,[np.rint(r['corners']).astype(np.int32)],1)
        if selected is not None:used.append(pol)
    other=other*(1-target)
    return np.stack([target,other,1-np.maximum(target,other)]),dict(target_polarizations=len(used),ambiguous_polarizations=ambiguous)


def main():
    torch.set_num_threads(2)
    meta=pd.read_csv(HERE/'cohort.csv');start=time.perf_counter()
    images=np.stack([np.stack([np.asarray(Image.open(r.png_vv)),np.asarray(Image.open(r.png_vh))]) for r in meta.itertuples()])
    original={p:np.load(f'E:/Hermes/paper_tracking/f0921_chips_clean_npy/F0921_{p}.npy',mmap_mode='r') for p in ['VV','VH']}
    for p,name in enumerate(['VV','VH']):
        np.testing.assert_array_equal(images[:,p],original[name][meta.clean_row.to_numpy()])
    if (HERE/'detections.json').exists():
        records=json.loads((HERE/'detections.json').read_text())
        assert sorted({r['row'] for r in records})==list(range(len(meta)))
    else:
        model=YOLO('E:/Docms/SARlabelimg/550k.pt',task='obb')
        records=[];flat=images.reshape(-1,224,224)
        for start_i in range(0,len(flat),4):
            canvases=[]
            for a in flat[start_i:start_i+4]:
                canvas=np.zeros((1024,1024,3),np.uint8);canvas[400:624,400:624]=a[:,:,None];canvases.append(canvas)
            output=model.predict(canvases,imgsz=1024,conf=.25,iou=.3,device=0,verbose=False,save=False,max_det=300)
            for j,result in enumerate(output):
                k=start_i+j;corners=result.obb.xyxyxyxy.cpu().numpy()-400
                conf=result.obb.conf.cpu().numpy()
                boxes=[]
                for polygon,c in zip(corners,conf):
                    if polygon[:,0].max()<0 or polygon[:,1].max()<0 or polygon[:,0].min()>223 or polygon[:,1].min()>223:continue
                    boxes.append(dict(corners=polygon.tolist(),confidence=float(c),pol=k%2))
                records.append(dict(row=k//2,pol=k%2,boxes=boxes))
            if start_i%80==0:print('DETECT',min(start_i+4,len(flat)),'/',len(flat),flush=True)
        (HERE/'detections.json').write_text(json.dumps(records))
        del model
        torch.cuda.empty_cache()
    regions=[];availability=[]
    for row in meta.itertuples():
        boxes=sum([r['boxes'] for r in records if r['row']==row.row],[])
        anchor=np.array([row.anchor_x,row.anchor_y])/row.side*224
        mask,info=mask_from_boxes(boxes,anchor)
        small=np.stack([np.asarray(Image.fromarray(m).resize((128,128),Image.Resampling.NEAREST)) for m in mask])
        assert np.all(small.sum(0)==1)
        regions.append(small)
        availability.append(dict(row=row.row,boxes=len(boxes),target_fraction=float(mask[0].mean()),
                                 other_fraction=float(mask[1].mean()),**info))
    regions=np.stack(regions)
    np.savez_compressed(HERE/'masks.npz',regions=regions,clean_rows=meta.clean_row.to_numpy())
    pd.DataFrame(availability).to_csv(HERE/'availability.csv',index=False)
    mask_seconds=time.perf_counter()-start
    r50=resnet50(weights=None).eval().cuda()
    r50.load_state_dict(torch.load('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth',map_location='cpu',weights_only=True))
    r18=resnet18(weights=None);r18.fc=torch.nn.Identity()
    ckpt=PROJECT/'scene_cross_dataset_redesign_20261014/cross_dataset/source_only_r18_best.pt'
    r18.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)['backbone']);r18=r18.eval().cuda()
    grid2=spatial_grid(2,2).cuda();grid12=spatial_grid(3,4).cuda()
    out={k:[] for k in ['global','r18','grid2','grid12','postpool','prepool','valid12','mass12']}
    error=0.
    with torch.inference_mode():
        for first in range(0,len(meta),8):
            pair=torch.tensor(images[first:first+8],device='cuda',dtype=torch.float32)/255
            n=len(pair)
            x=F.interpolate(pair,size=(128,128),mode='bilinear',align_corners=False,antialias=True)
            x=normalize(x.flatten(0,1)[:,None].expand(-1,3,-1,-1))
            fmap=r50.layer1(r50.maxpool(r50.relu(r50.bn1(r50.conv1(x)))))
            global_x=r50.avgpool(r50.layer4(r50.layer3(r50.layer2(fmap)))).flatten(1)
            role=torch.tensor(regions[first:first+8],device='cuda',dtype=torch.float32)
            masks=(grid2[None,:,None]*role[:,None]).flatten(1,2).repeat_interleave(2,0)
            pre,mass=pool(fmap,masks)
            original,omass=pool(fmap,grid2[None].expand(2*n,-1,-1,-1))
            regular,_=pool(fmap,grid12[None].expand(2*n,-1,-1,-1))
            reconstructed=(pre.reshape(2*n,4,3,-1)*mass.reshape(2*n,4,3,1)).sum(2)/omass[...,None]
            error=max(error,float((reconstructed-original).abs().max()))
            torch.testing.assert_close(reconstructed,original,atol=1e-5,rtol=1e-5)
            post=original.repeat_interleave(3,1)*(mass>0)[...,None]
            for name,value in [('grid2',original),('grid12',regular),('prepool',pre),('postpool',post)]:
                out[name].append(value.reshape(n,2,value.shape[1],value.shape[2]).cpu().numpy())
            out['global'].append(global_x.reshape(n,2,-1).cpu().numpy())
            v=F.interpolate(pair,size=(112,112),mode='bilinear',align_corners=False,antialias=True)
            v=normalize(torch.cat([v,v.mean(1,keepdim=True)],1))
            out['r18'].append(r18(v).cpu().numpy())
            out['mass12'].append(mass.reshape(n,2,-1)[:,0].cpu().numpy())
            out['valid12'].append((mass.reshape(n,2,-1)[:,0]>0).cpu().numpy())
    np.savez_compressed(HERE/'features.npz',clean_rows=meta.clean_row.to_numpy(),**{k:np.concatenate(v) for k,v in out.items()})
    receipt=dict(observations=len(meta),mask_seconds=mask_seconds,feature_seconds=time.perf_counter()-start-mask_seconds,
                 all_png_npy_pixels_equal=True,max_partition_reconstruction_error=error,
                 r18_checkpoint=str(ckpt),r18_input='112 RGB(VV,VH,mean)',r50_input='128 independent grayscale polarizations',
                 detector_canvas=1024,detector_conf=.25,detector_iou=.3,backbone_training=False)
    (HERE/'extraction.json').write_text(json.dumps(receipt,indent=2))
    print(pd.DataFrame(availability).target_polarizations.value_counts().to_string());print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
