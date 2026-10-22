"""Read local signed moments with each saved source-only outer/inner classifier."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys
import time
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import resnet50

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'contrastive_spatial_pool_20261023'))
from run_pool import EXP,STRONG,NEW,CHECKPOINT


def moments(v):
    v=v-v.mean(-1,keepdim=True)
    v=v-v.mean(-2,keepdim=True)
    sigma=v.square().mean(-1).sqrt()
    signed=(v*v.abs()).mean(-1)/sigma.clamp_min(1e-12)
    return torch.stack([sigma,signed],dim=-1)


@torch.inference_mode()
def main(reference_std=None, batch_size=32):
    start=time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    paths=sorted((NEW/'origins').glob('fold*.npz'))
    origins=[dict(np.load(p)) for p in paths]
    folds=[dict(np.load(NEW/f'fold{f}_inputs.npz')) for f in range(6)]
    checks=[]
    h_old=np.load(STRONG/'deep_features.npz')['layer3']
    for f,d in enumerate(folds):
        np.testing.assert_array_equal(d['instance_ids'],t.instance_id)
        covered=np.zeros(len(t),int)
        for path,o in zip(paths,origins):
            if not path.stem.startswith(f'fold{f}_'):continue
            held=o.get('held',np.r_[d['validation'],d['test']])
            ref=d['oof'] if 'held' in o else d['base']
            score=h_old[held].astype(float)@o['W'].T+o['raw_intercept']
            error=float(abs(score-ref[held]).max())
            assert error<1e-4
            if 'held' in o:
                assert not set(t.iloc[o['fit']].mmsi)&set(t.iloc[held].mmsi)
                assert not set(t.iloc[o['fit']].pass_group)&set(t.iloc[held].pass_group)
                covered[held]+=1
            checks.append(dict(reader=path.stem,held=len(held),mean_logit_error=error))
        assert (covered[d['train']]==1).all() and covered.sum()==len(d['train'])
    w=torch.tensor(np.stack([o['W'].reshape(3,2,1024).transpose(1,0,2) for o in origins]),device='cuda',dtype=torch.float64)
    model=resnet50(weights=None)
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
    model=model.cuda().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    buckets=defaultdict(list)
    for i,row in t.iterrows():
        for p,pol in enumerate(['vv','vh']):
            path=EXP/'readout/native_targets'/f'{row.instance_id}_{pol}.png'
            with Image.open(path) as im: shape=im.size
            assert list(shape)==json.loads(row[f'target_window_{pol}'])[2:]
            buckets[shape].append((i,p,path))
    result=np.zeros((len(paths),len(t),3,2),float)
    counts=np.zeros((len(t),2),int)
    gap_error=0.;last=time.perf_counter()
    for shape,items in sorted(buckets.items(),key=lambda item:(-len(item[1]),item[0])):
        for begin in range(0,len(items),batch_size):
            batch=items[begin:begin+batch_size]
            ix=np.array([a[0] for a in batch]);pol=np.array([a[1] for a in batch])
            pixels=[]
            for _,_,path in batch:
                with Image.open(path) as im:pixels.append(np.array(im,copy=True))
            x=torch.tensor(np.stack(pixels),device='cuda',dtype=torch.float32)[:,None].expand(-1,3,-1,-1)/255
            x=(x-mean)/std
            wp=w[:,torch.tensor(pol,device='cuda')]
            gap=[]
            for rot in range(4):
                z=x.rot90(rot,(-2,-1)).contiguous()
                z=model.maxpool(model.relu(model.bn1(model.conv1(z))))
                z=model.layer3(model.layer2(model.layer1(z)))
                gap.append(z.mean((-2,-1)))
                v=torch.einsum('fbck,bkp->fbcp',wp,z.flatten(2).double())
                pooled=moments(v)
                assert (pooled[...,1].abs()<=pooled[...,0]+1e-10).all()
                if begin==0 and rot==0:
                    negative=moments(-v)
                    assert torch.allclose(negative[...,0],pooled[...,0],atol=1e-10,rtol=1e-10)
                    assert torch.allclose(negative[...,1],-pooled[...,1],atol=1e-10,rtol=1e-10)
                values=pooled.cpu().numpy()/4
                for j in range(len(paths)):np.add.at(result[j],ix,values[j])
            fresh=torch.stack(gap).mean(0).cpu().numpy()
            for b,(i,p,_) in enumerate(batch):
                gap_error=max(gap_error,float(abs(fresh[b]-h_old[i,p*1024:(p+1)*1024]).max()))
                counts[i,p]+=1
            if time.perf_counter()-last>20:
                print(f'images={counts.sum()}/{2*len(t)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
                last=time.perf_counter()
    assert (counts==1).all() and gap_error<1e-4
    old=(np.load(HERE.parent/'contrastive_spatial_pool_20261023/delta_features.npz')['delta'][:,:,2]
         if reference_std is None else reference_std)
    for f in range(6):
        j=[p.stem for p in paths].index(f'fold{f}_outer')
        np.testing.assert_allclose(result[j,:,:,0],old[f],atol=1e-8,rtol=1e-8)
    np.savez_compressed(HERE/'signed_features.npz',reader_names=np.array([p.stem for p in paths]),instance_ids=t.instance_id.to_numpy(str),moments=result)
    pd.DataFrame(checks).to_csv(HERE/'reader_checks.csv',index=False)
    summary=dict(readers=len(paths),observations=len(t),images=int(counts.sum()),gap_error=gap_error,
                 seconds=time.perf_counter()-start,trained_backbones=0,reader_refits=0)
    (HERE/'extraction.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(summary,flush=True)


if __name__=='__main__':main()
