"""Native-chip OBB axes; no AIS headings or query labels are model inputs."""
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from ultralytics import YOLO
from common import HERE, MANIFEST


def main():
    torch.set_num_threads(2)
    frame=pd.read_csv(MANIFEST)
    model=YOLO('E:/Docms/SARlabelimg/550k.pt',task='obb')
    jobs=[(i,pol,path) for i,row in frame.iterrows() for pol,path in
          [('vv',row.chip_path_vv),('vh',row.chip_path_vh)]]
    results_all=[]
    started=time.perf_counter()
    for start in range(0,len(jobs),16):
        batch=jobs[start:start+16];images=[];sizes=[]
        for _,_,path in batch:
            with Image.open(path) as im:
                x=np.asarray(im.convert('L'))
            h,w=x.shape
            side=max(1024,int(np.ceil(max(h,w)/32)*32))
            ox,oy=(side-w)//2,(side-h)//2
            canvas=np.zeros((side,side,3),np.uint8)
            canvas[oy:oy+h,ox:ox+w]=x[:,:,None]
            images.append(canvas);sizes.append((h,w,ox,oy))
        outputs=model.predict(images,imgsz=1024,conf=.08,iou=.3,device=0,verbose=False,save=False)
        for (i,pol,_),out,(h,w,ox,oy) in zip(batch,outputs,sizes):
            raw=out.obb.xywhr.cpu().numpy().copy();conf=out.obb.conf.cpu().numpy()
            raw[:,:2]-=[ox,oy]
            radius=max(8.,.3*min(h,w))
            long=raw[:,2:4].max(1);short=raw[:,2:4].min(1)
            ok=(np.linalg.norm(raw[:,:2]-[(w-1)/2,(h-1)/2],axis=1)<=radius)
            ok&=(raw[:,0]>=0)&(raw[:,0]<w)&(raw[:,1]>=0)&(raw[:,1]<h)
            ok&=(long>=4)&(short>=1.5)&(long<=1.5*max(h,w))
            candidates=np.flatnonzero(ok)
            rec=dict(global_row=i,pol=pol,native_h=h,native_w=w,detected=bool(len(candidates)))
            if len(candidates):
                j=int(candidates[np.argmax(conf[candidates])]);cx,cy,bw,bh,theta=raw[j]
                # Width/height swaps must carry their coupled angle.
                axis=float((theta+(np.pi/2 if bh>bw else 0))%np.pi)
                rec.update(cx=float(cx),cy=float(cy),length=float(max(bw,bh)),
                           width=float(min(bw,bh)),ship_axis=axis,confidence=float(conf[j]),
                           raw_xywhr=raw[j].tolist(),axis_valid=bool(max(bw,bh)/min(bw,bh)>=1.2))
            else:
                rec.update(cx=(w-1)/2,cy=(h-1)/2,length=max(4.,min(h,w)*.25),
                           width=max(3.,min(h,w)*.15),ship_axis=0.,confidence=0.,
                           raw_xywhr=None,axis_valid=False)
            results_all.append(rec)
        if start%160==0:
            print('OBB',start+len(batch),'/',len(jobs),round(time.perf_counter()-started,1),flush=True)
    (HERE/'detection_views.json').write_text(json.dumps(results_all),encoding='utf-8')
    table=pd.DataFrame(results_all)
    rows=[]
    for i,group in table.groupby('global_row',sort=True):
        vv=group[group.pol.eq('vv')].iloc[0];vh=group[group.pol.eq('vh')].iloc[0]
        use=vv if vv.axis_valid or not vh.axis_valid else vh
        r=use.to_dict();r['sample_id']=frame.iloc[i].sample_id
        r['views_same_size']=bool(vv.native_h==vh.native_h and vv.native_w==vh.native_w)
        r['axis_disagreement_deg']=float(np.rad2deg(np.arccos(np.clip(abs(np.cos(vv.ship_axis-vh.ship_axis)),0,1)))) if vv.axis_valid and vh.axis_valid else None
        rows.append(r)
    pd.DataFrame(rows).drop(columns='raw_xywhr').to_csv(HERE/'detections.csv',index=False)
    summary=dict(samples=len(frame),valid_axes=sum(x['axis_valid'] for x in rows),
                 seconds=time.perf_counter()-started,ais_heading_used=False,
                 detector='E:/Docms/SARlabelimg/550k.pt',detector_pretraining_sample_provenance='not audited; shared frozen localizer',
                 input='native uint8 on zero-padded canvas; not full-scene detection',
                 selection='VV first, VH fallback if VV lacks elongated detection')
    (HERE/'DETECTION_DONE.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(summary,flush=True)


if __name__=='__main__':
    main()
