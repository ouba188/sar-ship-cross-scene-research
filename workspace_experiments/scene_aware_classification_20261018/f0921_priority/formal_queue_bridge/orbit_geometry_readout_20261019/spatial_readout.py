"""Ship-axis positions crossed with signed ground LOS, without image rotation."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
OLD = PARENT / 'axis_radar_readout_20261019'


def joint_windows(x, y, ship, half_length, half_width, reliability, radar, incidence):
    u = (x*np.cos(ship) + y*np.sin(ship)) / half_length
    v = (-x*np.sin(ship) + y*np.cos(ship)) / half_width
    along = .5 * (1 + np.tanh(reliability * (np.abs(u) - np.abs(v))))
    projection = (x*np.cos(radar) + y*np.sin(radar)) * np.sin(incidence) / half_width
    toward = .5 * (1 + np.tanh(projection))
    return np.stack([along*toward, along*(1-toward),
                     (1-along)*toward, (1-along)*(1-toward)])


def masks_for(axis, roles, radar, incidence):
    size = roles.shape[-1]
    yy, xx = np.mgrid[:size, :size]
    x = (xx+.5)*224/size-.5-axis.center_x
    y = (yy+.5)*224/size-.5-axis.center_y
    windows = joint_windows(x,y,axis.ship_axis_rad,axis.half_length,axis.half_width,
                            axis.axis_reliability,radar,np.radians(incidence))
    fg_bg = np.stack([roles[0],roles[1]+roles[2]])
    return (windows[:,None]*fg_bg[None]).reshape(8,size,size).astype(np.float32)


def main():
    import cv2
    import torch
    import torch.nn.functional as F
    from torchvision.models import resnet50
    from PIL import Image
    sys.path.insert(0,str(PARENT.parent/'prepool_local_probe'))
    from extract_local import pool, spatial_grid

    start=time.perf_counter()
    torch.set_num_threads(2)
    meta=pd.read_csv(PARENT/'cohort.csv')
    axes=pd.read_csv(OLD/'axes.csv')
    geometry=pd.read_csv(HERE/'orbit_geometry.csv')
    np.testing.assert_array_equal(meta.row,geometry.row)
    np.testing.assert_array_equal(meta.row,axes.row)
    boxes={}
    for record in json.loads((PARENT/'detections.json').read_text()):
        boxes.setdefault(record['row'],[]).extend(record['boxes'])
    sizes=[]
    for row in meta.itertuples():
        anchor=np.array([row.anchor_x,row.anchor_y])*224/row.side
        dims=[]
        for pol in (0,1):
            selected=[b for b in boxes[row.row] if b['pol']==pol and
                      cv2.pointPolygonTest(np.asarray(b['corners'],np.float32),tuple(map(float,anchor)),False)>=0]
            if len(selected)==1:
                p=np.asarray(selected[0]['corners'])
                edges=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
                dims.append((edges.max()/2,edges.min()/2))
        sizes.append(np.mean(dims,axis=0) if dims else (np.nan,np.nan))
    axes[['half_length','half_width']]=sizes
    valid=(geometry.orbit_valid.to_numpy(bool)&(axes.axis_reliability.to_numpy()>1e-8)
           &np.isfinite(axes.half_width)&(axes.half_width>0)).to_numpy()
    axes['readout_valid']=valid
    axes.to_csv(HERE/'axes.csv',index=False)
    roles=np.load(PARENT/'masks.npz')['regions']
    old=np.load(PARENT/'features.npz')
    model=resnet50(weights=None).eval().cuda()
    model.load_state_dict(torch.load('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth',
                                    map_location='cpu',weights_only=True))
    output={k:[] for k in ['image','orbit']}
    masses={k:[] for k in output}
    grid=spatial_grid(2,2).cuda()
    replay,reconstruction=0.,0.
    with torch.inference_mode():
        for start_row in range(0,len(meta),8):
            rows=meta.iloc[start_row:start_row+8]
            images=np.stack([np.stack([np.asarray(Image.open(r.png_vv)),np.asarray(Image.open(r.png_vh))])
                             for r in rows.itertuples()])
            x=torch.tensor(images,device='cuda',dtype=torch.float32)/255
            x=F.interpolate(x,size=(128,128),mode='bilinear',align_corners=False,antialias=True)
            x=x.flatten(0,1)[:,None].expand(-1,3,-1,-1)
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            fmap=model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
            restored,_=pool(fmap,grid[None].expand(len(x),-1,-1,-1))
            expected=torch.tensor(old['grid2'][start_row:start_row+len(rows)].reshape(len(x),4,-1),device='cuda')
            torch.testing.assert_close(restored,expected,atol=2e-5,rtol=1e-5)
            replay=max(replay,float((restored-expected).abs().max()))
            for ref in output:
                masks=[]
                for i in rows.row:
                    if valid[i]:
                        angle=geometry.iloc[i].los_angle_image_rad if ref=='orbit' else 0.
                        mask=masks_for(axes.iloc[i],roles[i],angle,geometry.iloc[i].orbit_incidence_deg)
                    else:
                        mask=np.zeros((8,roles.shape[-2],roles.shape[-1]),np.float32)
                    masks.append(mask)
                tensor=torch.tensor(np.stack(masks),device='cuda').repeat_interleave(2,0)
                tokens,mass=pool(fmap,tensor)
                output[ref].append(tokens.reshape(len(rows),2,8,-1).cpu().numpy())
                masses[ref].append(mass.reshape(len(rows),2,8)[:,0].cpu().numpy())
                combined=(tokens.reshape(len(x),4,2,-1)*mass.reshape(len(x),4,2,1)).sum(1)
                total=mass.reshape(len(x),4,2).sum(1)
                combined/=total[...,None].clamp_min(1e-8)
                direct_masks=np.stack([roles[rows.row,0],roles[rows.row,1]+roles[rows.row,2]],1)
                direct_masks*=valid[rows.row,None,None,None]
                direct,_=pool(fmap,torch.tensor(direct_masks,device='cuda',dtype=torch.float32).repeat_interleave(2,0))
                torch.testing.assert_close(combined,direct,atol=1e-5,rtol=1e-5)
                reconstruction=max(reconstruction,float((combined-direct).abs().max()))
    values={ref:np.concatenate(items) for ref,items in output.items()}
    values.update({ref+'_mass':np.concatenate(items) for ref,items in masses.items()})
    np.savez_compressed(HERE/'spatial_features.npz',clean_rows=meta.clean_row,valid=valid,**values)
    receipt=dict(seconds=time.perf_counter()-start,observations=len(meta),valid=int(valid.sum()),
                 grid_replay_error=replay,partition_reconstruction_error=reconstruction,
                 detector_training=False,backbone_training=False)
    (HERE/'extraction.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
