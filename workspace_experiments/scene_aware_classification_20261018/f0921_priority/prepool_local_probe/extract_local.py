"""Matched ownership-before-pooling and ordinary spatial tokens."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet50

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'region_relation_probe'
WEIGHTS=Path('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')


def spatial_grid(rows,cols,size=128):
    mask=torch.zeros(rows*cols,size,size)
    ys=np.linspace(0,size,rows+1,dtype=int);xs=np.linspace(0,size,cols+1,dtype=int)
    for r in range(rows):
        for c in range(cols):mask[r*cols+c,ys[r]:ys[r+1],xs[c]:xs[c+1]]=1
    return mask


def ownership_masks(regions):
    roles=torch.stack([regions[:,0],regions[:,1],regions[:,2]+regions[:,3]],1)
    grid=spatial_grid(2,2,regions.shape[-1]).to(regions.device)
    return (grid[None,:,None]*roles[:,None]).flatten(1,2)


def pool(feature,masks):
    weights=F.interpolate(masks,size=feature.shape[-2:],mode='area').flatten(2)
    mass=weights.sum(-1)
    token=(weights@feature.flatten(2).transpose(1,2))/mass[...,None].clamp_min(1e-8)
    return token,mass


def main():
    start=time.perf_counter();torch.set_num_threads(2)
    data=np.load(PREV/'inputs.npz');images=data['images'];regions=data['regions']
    model=resnet50(weights=None).eval().cuda()
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    grid2=spatial_grid(2,2).cuda();grid12=spatial_grid(3,4).cuda()
    out={name:[] for name in ['grid12','postpool','prepool','valid12','mass12','grid2']}
    errors=[]
    with torch.inference_mode():
        for first in range(0,len(images),8):
            x=torch.tensor(images[first:first+8],dtype=torch.float32,device='cuda')/255
            n=len(x);x=x.flatten(0,1)[:,None].expand(-1,3,-1,-1)
            x=(x-mean)/std
            fmap=model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
            region=torch.tensor(regions[first:first+8],dtype=torch.float32,device='cuda')
            masks=ownership_masks(region).repeat_interleave(2,0)
            pre,mass=pool(fmap,masks)
            original,original_mass=pool(fmap,grid2[None].expand(2*n,-1,-1,-1))
            regular,_=pool(fmap,grid12[None].expand(2*n,-1,-1,-1))
            reconstructed=(pre.reshape(2*n,4,3,-1)*mass.reshape(2*n,4,3,1)).sum(2)/original_mass[...,None]
            errors.append(float((reconstructed-original).abs().max().cpu()))
            torch.testing.assert_close(reconstructed,original,atol=1e-5,rtol=1e-5)
            post=original.repeat_interleave(3,1)*(mass>0)[...,None]
            for name,val in [('grid12',regular),('prepool',pre),('postpool',post),('grid2',original)]:
                out[name].append(val.reshape(n,2,val.shape[1],val.shape[2]).cpu().numpy())
            out['mass12'].append(mass.reshape(n,2,-1)[:,0].cpu().numpy())
            out['valid12'].append((mass.reshape(n,2,-1)[:,0]>0).cpu().numpy())
    saved={k:np.concatenate(v) for k,v in out.items()}
    old=np.load(PREV/'features.npz')['grid'].reshape(len(images),2,4,-1)
    np.testing.assert_allclose(saved['grid2'],old,atol=1e-5,rtol=1e-5)
    np.savez_compressed(HERE/'local_features.npz',**saved)
    info=dict(seconds=time.perf_counter()-start,observations=len(images),device=torch.cuda.get_device_name(),
              weights=str(WEIGHTS),max_partition_reconstruction_error=max(errors),
              grid2_feature_replay=True,neural_training=False,spatial_shape=[32,32],channels=256)
    (HERE/'extraction.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
