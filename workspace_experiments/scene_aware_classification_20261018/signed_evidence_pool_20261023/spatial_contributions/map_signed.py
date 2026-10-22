"""Exact fixed-head signed-pool margin accounting, not causal pixel attribution."""
import sys
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import resnet50
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from train_signed import NEW,EXP,Correction,descriptors
from extract_signed import CHECKPOINT
IDS=['pair_0032439','pair_0033132','pair_0030906','pair_0030918']


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=resnet50(weights=None)
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
    model=model.cuda().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    buckets=defaultdict(list)
    for ti,tr in t.iterrows():
        for polname in ['vv','vh']:
            shape=tuple(json.loads(tr[f'target_window_{polname}'])[2:])
            buckets[shape].append((ti,polname))
    cached=np.load(HERE.parent/'signed_features.npz');names=list(cached['reader_names'])
    p=pd.read_csv(HERE.parent/'signed_only_retrain/predictions.csv')
    p=p[p.arm.eq('signed_only')&p.seed.eq(17)].set_index('instance_id')
    rows=[];saved={}
    for ident in IDS:
        i=int(np.flatnonzero(t.instance_id.eq(ident))[0]);row=t.iloc[i];pr=p.loc[ident];fold=int(pr.fold)
        d=dict(np.load(NEW/f'fold{fold}_inputs.npz'))
        origin=np.load(NEW/'origins'/f'fold{fold}_outer.npz')
        mom=cached['moments'][names.index(f'fold{fold}_outer')]
        ck=torch.load(HERE.parent/'signed_only_retrain/models'/f'fold{fold}_signed_only_seed17.pt',weights_only=False)
        head=Correction('signed_only');head.load_state_dict(ck['state']);head.eval()
        desc=descriptors(d['base'],mom)[i:i+1]
        xhead=torch.tensor((desc-ck['mu'])/ck['scale'],dtype=torch.float32)
        coeff=head.net(xhead).tanh().squeeze().numpy().astype(float)
        base=d['base'][i].astype(float);truth=int(row.class_id)
        rival=max((c for c in range(3) if c!=truth),key=lambda c:base[c])
        maps={};raws={};sum_delta=0.;newmom=np.zeros((3,2))
        for pol,polname in enumerate(['vv','vh']):
            with Image.open(EXP/'readout/native_targets'/f'{ident}_{polname}.png') as im:raw=np.array(im)
            raws[polname]=raw
            # Match the extraction batch: single-image fp32 convolution uses a
            # different kernel and otherwise changes small signed moments.
            items=buckets[(raw.shape[1],raw.shape[0])]
            loc=items.index((i,polname));batch=items[(loc//32)*32:(loc//32+1)*32]
            pixels=[]
            for bi,bpol in batch:
                with Image.open(EXP/'readout/native_targets'/f'{t.iloc[bi].instance_id}_{bpol}.png') as im:
                    pixels.append(np.array(im,copy=True))
            x=torch.tensor(np.stack(pixels),device='cuda',dtype=torch.float32)[:,None].expand(-1,3,-1,-1)/255
            x=(x-mean)/sd
            w=torch.tensor(origin['W'][:,pol*1024:(pol+1)*1024],dtype=torch.float64,device='cuda')
            for rotation in range(4):
                z=x.rot90(rotation,(-2,-1)).contiguous()
                z=model.maxpool(model.relu(model.bn1(model.conv1(z))))
                z=model.layer3(model.layer2(model.layer1(z)))
                z=z[loc%32:loc%32+1]
                v=(w@z[0].flatten(1).double())
                v=v-v.mean(-1,keepdim=True);v=v-v.mean(0,keepdim=True)
                sigma=v.square().mean(-1).sqrt()
                local=v*v.abs()/sigma[:,None].clamp_min(1e-12)/v.shape[-1]
                signed=local.sum(-1)
                newmom+=torch.stack([sigma,signed],dim=-1).cpu().numpy()/4
                part=(coeff[truth]*local[truth]-coeff[rival]*local[rival])/4
                grid=part.cpu().numpy().reshape(z.shape[-2:]);maps[(polname,rotation)]=grid
                saved[f'{ident}_{polname}_rot{rotation}']=grid
                sum_delta+=float(grid.sum())
        np.testing.assert_allclose(newmom,mom[i],atol=3e-6,rtol=3e-6)
        actual=float(pr[f'delta{truth}']-pr[f'delta{rival}'])
        error=abs(sum_delta-actual);assert error<3e-6
        rows.append(dict(instance_id=ident,fold=fold,seed=17,class_id=truth,rival=rival,outcome=pr.outcome,
            base_margin=float(base[truth]-base[rival]),pooled_margin_delta=actual,map_sum=sum_delta,error=error,
            true_coefficient=coeff[truth],rival_coefficient=coeff[rival],moment_replay_error=float(abs(newmom-mom[i]).max())))
        fig=plt.figure(figsize=(12,9),layout='constrained')
        gs=fig.add_gridspec(3,4,height_ratios=[1.65,1,1])
        for polname,columns in [('vv',slice(0,2)),('vh',slice(2,4))]:
            ax=fig.add_subplot(gs[0,columns]);raw=raws[polname];ax.imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            points=np.array(json.loads(row.target_points))-np.array(json.loads(row[f'target_window_{polname}']))[:2]
            points=np.r_[points,points[:1]];ax.plot(points[:,0],points[:,1],color='lime',linewidth=.8)
            ax.set_title(f'{polname.upper()} native {raw.shape[1]}x{raw.shape[0]} pixels');ax.set_xlabel('native x');ax.set_ylabel('native y')
        limit=max(float(abs(a).max()) for a in maps.values())
        for ipol,polname in enumerate(['vv','vh']):
            for rotation in range(4):
                ax=fig.add_subplot(gs[ipol+1,rotation]);grid=maps[(polname,rotation)]
                im=ax.imshow(grid,cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
                ax.set_title(f'{polname.upper()} rot{rotation*90}; sum={grid.sum():+.4f}',fontsize=10)
                ax.set_xlabel('feature-grid column');ax.set_ylabel('feature-grid row')
        fig.colorbar(im,ax=fig.axes[2:],shrink=.7,label='Contribution to fixed true-rival margin')
        fig.suptitle(f'{ident} | {row.port} | {pr.outcome} | fixed pair {truth}-{rival}\n'
            f'learned coefficients {coeff[truth]:+.3f}/{coeff[rival]:+.3f}; total increment={sum_delta:+.6f}\n'
            'Native C4 grids, not aligned pixel masks. Receptive fields cross OBB; weights held fixed for accounting.',fontsize=11)
        fig.savefig(HERE/f'{ident}_signed_contributions.png',dpi=130);plt.close(fig)
    pd.DataFrame(rows).to_csv(HERE/'exact_contributions.csv',index=False)
    np.savez_compressed(HERE/'contribution_maps.npz',**saved)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
