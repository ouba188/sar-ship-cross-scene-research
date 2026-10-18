"""Read frozen R50 features once for matched geometric/ownership readouts."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet50

HERE=Path(__file__).resolve().parent
WEIGHTS=Path('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')


def pool(feature,masks):
    weights=F.interpolate(masks,size=feature.shape[-2:],mode='area').flatten(2)
    denominator=weights.sum(-1,keepdim=True)
    return weights@feature.flatten(2).transpose(1,2)/denominator.clamp_min(1e-8)


def main():
    tic=time.perf_counter();torch.set_num_threads(2)
    data=np.load(HERE/'inputs.npz')
    images=data['images'];regions=data['regions']
    model=resnet50(weights=None).eval().cuda()
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
    grid=torch.zeros((1,4,128,128))
    for k,(r,c) in enumerate([(0,0),(0,1),(1,0),(1,1)]):grid[0,k,r*64:(r+1)*64,c*64:(c+1)*64]=1
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    output={key:[] for key in ['global','grid','region','area']}
    with torch.inference_mode():
        for start in range(0,len(images),8):
            x=torch.tensor(images[start:start+8],dtype=torch.float32,device='cuda')/255
            n=len(x);x=x.flatten(0,1)[:,None].expand(-1,3,-1,-1)
            x=(x-mean)/std
            spatial=model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
            mask=torch.tensor(regions[start:start+8],dtype=torch.float32,device='cuda').repeat_interleave(2,0)
            plain=pool(spatial,grid.to('cuda').expand(2*n,-1,-1,-1))
            organized=pool(spatial,mask)
            last=model.layer4(model.layer3(model.layer2(spatial)))
            global_x=model.avgpool(last).flatten(1)
            output['global'].append(global_x.reshape(n,2,-1).cpu().numpy())
            output['grid'].append(plain.reshape(n,2,-1).cpu().numpy())
            output['region'].append(organized.reshape(n,2,-1).cpu().numpy())
            areas=mask.mean((-2,-1))
            output['area'].append(areas.reshape(n,2,-1).cpu().numpy())
    np.savez_compressed(HERE/'features.npz',**{k:np.concatenate(v) for k,v in output.items()})
    info=dict(seconds=time.perf_counter()-tic,weights=str(WEIGHTS),n=len(images),device=torch.cuda.get_device_name(),same_features_same_pixels=True)
    (HERE/'extraction.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
