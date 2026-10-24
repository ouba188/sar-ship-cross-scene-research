"""Source-only measurement of pair cancellation on actual sampled inputs."""
import math
import train_hybrid as run
import numpy as np
import pandas as pd
import torch
tr=run.tr


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    data=tr.Data();reader=dict(np.load(tr.HERE/'reader.npz'))
    mu=torch.tensor(reader['mu'],device='cuda',dtype=torch.float32)
    scale=torch.tensor(reader['scale'],device='cuda',dtype=torch.float32)
    # Three source observations per class, independent of predictions/outcomes.
    selection=data.table.iloc[data.tr].sort_values('obs_id').groupby('class_id',sort=True).head(3)
    selection[['obs_id','class_id','mmsi','pass_group','port']].to_csv(run.HERE/'source_probe_cases.csv',index=False)
    rows=[]
    radii=torch.tensor([1.,2.,4.,1.,2.,4.],device='cuda')
    for j in selection.index:
        b=data.batch([j]);h,w=b['x'].shape[-2:]
        x=((b['x']-mu[None,:,:,None,None])/scale[None,:,:,None,None]).flatten(0,1)
        valid=b['valid'][:,:,None].expand(1,2,3,h,w).reshape(6,1,h,w)
        region=tr.typed_regions(b).flatten(0,1)
        theta=b['theta'][:,:,None].expand(1,2,3).reshape(-1)
        for direction,angle in enumerate((0.,math.pi/2,math.pi/4,-math.pi/4)):
            dx,dy=radii*(theta+angle).cos(),radii*(theta+angle).sin()
            vp,vm=tr.ops.sample_offset(valid,dx,dy),tr.ops.sample_offset(valid,-dx,-dy)
            plus=tr.ops.sample_offset(x*valid,dx,dy)/vp.clamp_min(1e-12)
            minus=tr.ops.sample_offset(x*valid,-dx,-dy)/vm.clamp_min(1e-12)
            weight=valid*(vp>.999)*(vm>.999)*(region.sum(1,keepdim=True)>0)
            for t in range(6):
                for name,channels in [('semantic7',slice(0,7)),('native_band1',slice(7,8))]:
                    a=(plus-x)[t,channels];bminus=(minus-x)[t,channels]
                    mask=weight[t]
                    total=float((((a.square()+bminus.square())*.5)*mask).sum())
                    retained=float((((a+bminus)*.5).square()*mask).sum())
                    if total>1e-12:
                        ratio=retained/total
                        assert -.00001<=ratio<=1.00001
                        rows.append(dict(obs_id=int(data.table.iloc[j].obs_id),class_id=int(data.table.iloc[j].class_id),
                            polarization=('VV','VH')[t//3],band=t%3,direction=direction,channels=name,
                            paired_difference_energy=retained,endpoint_difference_energy=total,
                            retained_energy_fraction=ratio,valid_pixels=int(mask.sum())))
    f=pd.DataFrame(rows);f.to_csv(run.HERE/'source_sampling_probe.csv',index=False)
    f.groupby(['polarization','band','channels']).retained_energy_fraction.agg(['count','mean','median']).to_csv(run.HERE/'source_sampling_summary.csv')
    print(f.groupby('channels').retained_energy_fraction.agg(['count','mean','median']).to_string(),flush=True)


if __name__=='__main__':main()
