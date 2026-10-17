"""Two adapters applied to identical raw detector outputs; labels are not inputs."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from ultralytics import YOLO

HERE=Path(__file__).resolve().parent


def project(cx,cy,w,h,theta,width,height,sort_sides):
    if sort_sides:
        w,h=max(w,h),min(w,h)
    co,si=abs(math.cos(theta)),abs(math.sin(theta))
    bw,bh=max(8.,1.25*(co*w+si*h)),max(8.,1.25*(si*w+co*h))
    box=np.array([math.floor(cx-bw/2),math.floor(cy-bh/2),
                  math.ceil(cx+bw/2),math.ceil(cy+bh/2)],dtype=int)
    box[[0,2]]=box[[0,2]].clip(0,width)
    box[[1,3]]=box[[1,3]].clip(0,height)
    return box


def main():
    torch.set_num_threads(2)
    inventory=json.loads((HERE/'inventory.json').read_text())
    frame=pd.read_csv(HERE/'inventory.csv')
    model=YOLO(inventory['detector_model'],task='obb')
    jobs=[(r,pol) for r in frame.to_dict('records') for pol in ('vv','vh')]
    path=HERE/'detections.jsonl'
    completed={}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            item=json.loads(line); completed[(item['global_row'],item['pol'])]=item
    missing=[job for job in jobs if (job[0]['global_row'],job[1]) not in completed]
    started=time.perf_counter()
    with path.open('a',encoding='utf-8') as out:
        for start in range(0,len(missing),16):
            batch=missing[start:start+16]
            images=[]; context=[]
            for row,pol in batch:
                with Image.open(row['image_relpath_'+pol]) as im:
                    native=np.array(im.convert('L'))
                h,w=native.shape
                assert (h,w)==(int(row['native_h_'+pol]),int(row['native_w_'+pol]))
                assert max(h,w)<=128
                ox,oy=(1024-w)//2,(1024-h)//2
                canvas=np.zeros((1024,1024,3),np.uint8)
                canvas[oy:oy+h,ox:ox+w]=native[:,:,None]
                images.append(canvas)
                old=np.array([row['bbox_'+k+'_'+pol] for k in ('x0','y0','x1','y1')],float)
                context.append((native,ox,oy,old))
            results=model.predict(source=images,imgsz=1024,conf=.08,iou=.30,
                                  device=0,half=False,verbose=False,save=False)
            for (row,pol),result,(native,ox,oy,old) in zip(batch,results,context):
                h,w=native.shape; center=(old[:2]+old[2:])/2
                raw=result.obb.xywhr.detach().cpu().numpy().copy()
                scores=result.obb.conf.detach().cpu().numpy()
                raw[:,:2]-=[ox,oy]
                long=raw[:,2:4].max(1); short=raw[:,2:4].min(1)
                radius=max(16.,.25*float((old[2:]-old[:2]).max()))
                candidates=np.flatnonzero((np.linalg.norm(raw[:,:2]-center,axis=1)<=radius)&
                    (raw[:,0]>=0)&(raw[:,0]<w)&(raw[:,1]>=0)&(raw[:,1]<h)&
                    (long>=4)&(long<=140)&(short>=1.5)&(short<=70)&(long/np.maximum(short,1)<=25))
                accepted=bool(len(candidates)); boxes={}
                record=dict(global_row=int(row['global_row']),pol=pol,native_h=h,native_w=w,
                    accepted=accepted,center_radius=radius,raw_detections=len(raw))
                if accepted:
                    pick=int(candidates[np.argmax(scores[candidates])]); xywhr=raw[pick]
                    record.update(raw_xywhr=xywhr.tolist(),confidence=float(scores[pick]),
                                  width_lt_height=bool(xywhr[2]<xywhr[3]))
                    for arm,sort_sides in (('wrong',True),('fixed',False)):
                        boxes[arm]=project(*map(float,xywhr),w,h,sort_sides)
                    # Corners supplied by the detector are the independent geometric reference.
                    corners=result.obb.xyxyxyxy[pick].detach().cpu().numpy()-[ox,oy]
                    center_raw=xywhr[:2]
                    ext=np.maximum(8.,np.ptp(corners,axis=0)*1.25)
                    reference=np.r_[np.floor(center_raw-ext/2),np.ceil(center_raw+ext/2)]
                    reference[[0,2]]=reference[[0,2]].clip(0,w)
                    reference[[1,3]]=reference[[1,3]].clip(0,h)
                    # Small float32 corner rounding can cross an integer boundary.
                    assert np.abs(boxes['fixed']-reference).max()<=1
                    if not record['width_lt_height']:
                        np.testing.assert_array_equal(boxes['fixed'],boxes['wrong'])
                else:
                    boxes={a:old.astype(int) for a in ('wrong','fixed')}
                    record.update(raw_xywhr=None,confidence=None,width_lt_height=False)
                yy,xx=np.indices(native.shape); bright=native>=200
                record['bright_count']=int(bright.sum())
                for arm,box in [('historical',old),*boxes.items()]:
                    x0,y0,x1,y1=box
                    inside=(xx+.5>=x0)&(xx+.5<x1)&(yy+.5>=y0)&(yy+.5<y1)
                    record[arm+'_box']=box.tolist()
                    record[arm+'_bright_outside']=int((bright&~inside).sum())
                record['adapter_changes_box']=bool(np.any(boxes['wrong']!=boxes['fixed']))
                out.write(json.dumps(record)+'\n')
            out.flush()
            if start%160==0 or start+16>=len(missing):
                print('DETECT',min(start+16,len(missing)),'/',len(missing),
                      'seconds',round(time.perf_counter()-started,1),flush=True)
    records=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(records)==len(jobs)
    pd.DataFrame(records).to_csv(HERE/'detections.csv',index=False)
    summary=dict(polarization_images=len(records),accepted=sum(r['accepted'] for r in records),
        changed=sum(r['adapter_changes_box'] for r in records),
        native_pixels_unscaled=True,unknown_surroundings='zero padding to1024',
        full_scene_replay=False,seconds_this_run=time.perf_counter()-started,
        criterion='same center radius and same highest confidence; original size filter',
        label_used=False,failed_detection_policy=inventory['missing_detection_policy'])
    (HERE/'DETECTION_DONE.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
