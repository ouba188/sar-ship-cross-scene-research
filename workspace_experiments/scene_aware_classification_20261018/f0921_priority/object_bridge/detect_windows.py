"""Frozen inference with exact existing 45FE tile predictions reused."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from tile_policy import select_tiles

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'chip_quality_20261017'))
from detect_chips import decode_result


def main(output_dir=HERE, tile_policy=None):
    HERE=Path(output_dir)
    torch.set_num_threads(2)
    data=json.loads((HERE/'INPUTS.json').read_text(encoding='utf-8'))
    selected_tiles,skipped_tiles=select_tiles(data['tiles'],tile_policy)
    cache=Path(__file__).resolve().parent.parent/'scene_45fe_context'
    old=json.loads((cache/'objects.json').read_text())
    old_product=json.loads((cache/'INPUTS.json').read_text())['product']
    old_config=json.loads((cache/'DETECTION_SUMMARY.json').read_text())
    assert old_config['confidence']==.25 and old_config['iou']==.3 and old_config['imgsz']==1024
    objects=[]
    model=None;started=time.perf_counter();computed=0
    for i,tile in enumerate(selected_tiles):
        if tile['cached_detection']:
            assert tile['product']==old_product
            records=[dict(o) for o in old if o['tile_id']==tile['legacy_tile_id']]
        else:
            if model is None:model=YOLO('E:/Docms/SARlabelimg/550k.pt',task='obb')
            gray=np.asarray(Image.open(tile['path']).convert('L'))
            pred=model.predict(np.repeat(gray[:,:,None],3,axis=2),imgsz=1024,conf=.25,iou=.3,
                               device=0,verbose=False,save=False,max_det=1000)[0]
            records=decode_result(pred,1024,1024,0,0);computed+=1
        for j,r in enumerate(records):
            origin=np.asarray(tile['origin'])
            r.update(product=tile['product'],object_id=tile['tile_id']+'_'+str(j),
                tile_id=tile['tile_id'],pol=tile['pol'],source=tile['source'],
                source_center=(np.asarray(r['xywhr'][:2])+origin).tolist(),
                source_corners=(np.asarray(r['corners'])+origin).tolist())
            objects.append(r)
        if i%20==0 or i+1==len(selected_tiles):print(json.dumps(dict(tiles_done=i+1,total=len(selected_tiles),detections=len(objects))),flush=True)
    (HERE/'detections.json').write_text(json.dumps(objects,indent=2),encoding='utf-8')
    summary=dict(tiles=len(selected_tiles),computed_tiles=computed,reused_tiles=len(selected_tiles)-computed,
        detections=len(objects),seconds=time.perf_counter()-started,weights='E:/Docms/SARlabelimg/550k.pt',
        confidence=.25,iou=.3,imgsz=1024,scope=data['scope'])
    if tile_policy is not None:
        summary.update(tile_policy=str(tile_policy),skipped_land_tiles=[t['tile_id'] for t in skipped_tiles],original_tiles=len(data['tiles']))
    (HERE/'DETECTION_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=HERE)
    parser.add_argument('--tile-policy',type=Path,help='Optional land_mask/tile_policy.csv; original pixels stay untouched')
    args=parser.parse_args();main(args.output_dir,args.tile_policy)
