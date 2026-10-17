"""Extract matched local descriptors from each source-only frozen R18 stem."""
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from common import HERE,PIXELS,MANIFEST,folds
from local_reader import ARMS,regions
from general_domain_ssl import backbone,normalize,rgb


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    m=pd.read_csv(MANIFEST);d=pd.read_csv(HERE/'detections.csv').set_index('global_row')
    g=pd.read_csv(HERE/'geometry.csv').set_index('global_row').loc[np.arange(len(m))]
    np.testing.assert_array_equal(g.sample_id.to_numpy(str),m.sample_id.to_numpy(str))
    masks={a:[] for a in ARMS};positions={a:[] for a in ARMS};bodies=[];records=[]
    for i in range(len(m)):
        row=d.loc[i].to_dict();geo=g.loc[i]
        for arm in ARMS:
            body,mask,pos,cov=regions(row,float(geo.radar_axis_rad) if geo.radar_valid else 0.,
                  float(geo.incidence_rad) if geo.incidence_valid else 0.,
                  bool(geo.radar_valid),bool(geo.incidence_valid),arm)
            masks[arm].append(mask.reshape(12,-1));positions[arm].append(pos)
            records.append(dict(global_row=i,arm=arm,mean_coverage=float(cov.mean()),
                empty_tokens=int((pos[:,3]==0).sum()),body_pixels=int(body.sum())))
        bodies.append(body)
    bodies=np.stack(bodies)
    masks={a:np.stack(v) for a,v in masks.items()}
    positions={a:np.stack(v) for a,v in positions.items()}
    pd.DataFrame(records).to_csv(HERE/'region_coverage.csv',index=False)
    pixels=np.load(PIXELS,mmap_mode='r')
    np.testing.assert_array_equal(np.load(PIXELS.with_name('oss1_ids.npy')).astype(str),m.sample_id.to_numpy(str))
    timings=[]
    for fold,cache,_,_ in folds():
        output=HERE/fold;output.mkdir(exist_ok=True)
        model,_=backbone('r18')
        checkpoint=cache/'source_only_r18_best.pt'
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['backbone'])
        stem=torch.nn.Sequential(model.conv1,model.bn1,model.relu).cuda().eval()
        stem.requires_grad_(False)
        values={a:[] for a in ARMS};body_values=[];start_time=time.perf_counter()
        for start in range(0,len(m),32):
            end=min(start+32,len(m))
            pair=torch.from_numpy(np.array(pixels[start:end],np.float32)).cuda()/255
            image=normalize(rgb(F.interpolate(pair,(112,112),mode='bilinear',align_corners=False,antialias=True)))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                feature=stem(image)
            feature=feature.float().flatten(2).transpose(1,2)
            body=torch.from_numpy(bodies[start:end].reshape(end-start,-1)).cuda()
            avg=(feature*body[:,:,None]).sum(1)/body.sum(1)[:,None]
            peak=feature.masked_fill(~body[:,:,None],-torch.inf).amax(1)
            body_values.append(torch.cat((avg,peak),1).cpu().numpy())
            for arm in ARMS:
                values[arm].append(torch.bmm(torch.from_numpy(masks[arm][start:end]).cuda(),feature).cpu().numpy())
        output_arrays={'body':np.concatenate(body_values),'valid':
            (d.axis_valid.to_numpy(bool)&d.views_same_size.to_numpy(bool)),'sample_ids':m.sample_id.to_numpy(str)}
        for arm in ARMS:
            output_arrays[arm+'_tokens']=np.concatenate(values[arm])
            output_arrays[arm+'_positions']=positions[arm]
        np.savez_compressed(output/'local_features.npz',**output_arrays)
        seconds=time.perf_counter()-start_time
        timings.append(dict(fold=fold,seconds=seconds,checkpoint=str(checkpoint)))
        print('FEATURES',fold,round(seconds,2),flush=True)
        del stem,model
    (HERE/'FEATURES_DONE.json').write_text(json.dumps(timings,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
