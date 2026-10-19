"""Cache frozen pre-cell-pooling feature maps; no change to prior experiments."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torchvision.models import resnet50
from PIL import Image

HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent
OLD=BRIDGE/'learned_geometry_crosspass_20261019'


def main():
    start=time.perf_counter()
    torch.set_num_threads(2)
    meta=pd.read_csv(OLD/'cohort.csv')
    previous=np.load(BRIDGE/'features.npz')
    model=resnet50(weights=None).cuda().eval()
    model.load_state_dict(torch.load('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth',map_location='cpu',weights_only=True))
    maps=np.empty((len(meta),2,256,32,32),np.float32)
    error=0.
    with torch.inference_mode():
        for begin in range(0,len(meta),8):
            rows=meta.iloc[begin:begin+8]
            a=np.stack([[np.asarray(Image.open(r.png_vv)),np.asarray(Image.open(r.png_vh))] for r in rows.itertuples()])
            x=torch.tensor(a,device='cuda',dtype=torch.float32)/255
            x=F.interpolate(x,(128,128),mode='bilinear',align_corners=False,antialias=True)
            x=x.flatten(0,1)[:,None].expand(-1,3,-1,-1)
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            f=model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
            grid=F.adaptive_avg_pool2d(f,(2,2)).flatten(2).transpose(1,2)
            expected=torch.tensor(previous['grid2'][begin:begin+len(rows)].reshape(len(x),4,256),device='cuda')
            torch.testing.assert_close(grid,expected,rtol=1e-5,atol=2e-5)
            error=max(error,float((grid-expected).abs().max()))
            maps[begin:begin+len(rows)]=f.reshape(len(rows),2,256,32,32).cpu().numpy()
    np.save(HERE/'maps.npy',maps)
    roles=np.load(BRIDGE/'masks.npz')['regions']
    masks=np.stack([roles[:,0],roles[:,1]+roles[:,2]],1)
    masks=F.interpolate(torch.tensor(masks,dtype=torch.float32),(32,32),mode='area').numpy()
    np.save(HERE/'masks32.npy',masks)
    receipt=dict(finished=True,observations=len(meta),feature_shape=list(maps.shape),dtype='float32',
                 old_grid_replay_error=error,seconds=time.perf_counter()-start)
    (HERE/'extraction.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
