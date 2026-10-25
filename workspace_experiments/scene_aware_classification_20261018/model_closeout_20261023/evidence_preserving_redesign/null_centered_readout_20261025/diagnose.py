"""Frozen-head decomposition; no model fitting or selection."""
import train as run
import numpy as np
import pandas as pd
import torch

@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=run.Data();rows=[];harms=[];coverage=[]
    stats=run.load('null_diagnostic_stats',run.HERE.parent/'typed_m3_20261025/analyze.py')
    for seed in run.SEEDS:
        z,x,a,null=run.inputs(data,seed);empty=x.clone();empty[...,:12]=torch.tensor(null[:12],device='cuda')
        base=z['base'];oldlate=np.load(run.PREVIOUS/'results'/f'late_average_{seed}.npz')['score']
        d={p:run.old.restore(seed,p)(x).cpu().numpy() for p in ('vv','vh')}
        d0={p:run.old.restore(seed,p)(empty).cpu().numpy() for p in ('vv','vh')}
        centered=base+.5*(d['vv']-d0['vv']+d['vh']-d0['vh'])
        gated=base+.5*(d['vv']+d['vh']*a[:,1,None].cpu().numpy())
        np.testing.assert_allclose(base+.5*(d['vv']+d['vh']),oldlate,atol=1e-6)
        # VV never lacks evidence here; matched masked VV training must replay prior VV.
        np.testing.assert_allclose(np.load(run.HERE/'results'/f'mask_vv_{seed}.npz')['score'],base+d['vv'],atol=1e-6)
        oldfull=np.load(run.HERE/'results'/f'old_full_{seed}.npz')['score']
        for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
            yy=data.table.iloc[ids].class_id.to_numpy(int)
            for group,mask in (('all',np.ones(len(ids),bool)),('paired',z['query_valid'][ids].all(1)),('VH_missing',~z['query_valid'][ids,1])):
                for name,s in (('fixed_null_center',centered),('fixed_missing_VH_zero',gated)):
                    rows.append(dict(seed=seed,role=role,group=group,arm=name,
                        prediction_changes=int((s[ids][mask].argmax(1)!=oldlate[ids][mask].argmax(1)).sum()))|
                        stats.effect(s[ids][mask],oldlate[ids][mask],yy[mask])|
                        {k:v for k,v in run.metrics(yy[mask],s[ids][mask]).items() if k!='n'})
            if seed==run.SEEDS[0]:
                frame=data.table.iloc[ids].copy();frame['missing_VH']=~z['query_valid'][ids,1]
                for c,f in frame.groupby('class_id'):
                    coverage.append(dict(role=role,class_id=c,class_name=f.class_name.iloc[0],n=len(f),missing=int(f.missing_VH.sum())))
            bad=(oldfull[ids].argmax(1)==yy)&(oldlate[ids].argmax(1)!=yy)&~z['query_valid'][ids,1]
            for k in np.flatnonzero(bad):
                j=int(ids[k]);r=data.table.iloc[j];y=int(yy[k]);rival=int(oldlate[j].argmax())
                diff=lambda s:float(s[j,y]-s[j,rival])
                harms.append(dict(seed=seed,role=role,obs_id=int(r.obs_id),mmsi=r.mmsi,pass_group=r.pass_group,port=r.port,
                    class_name=r.class_name,true=y,rival=rival,base_margin=diff(base),old_margin=diff(oldfull),
                    prior_late_margin=diff(oldlate),VV_delta=diff(d['vv']),VH_delta=diff(d['vh']),
                    VH_zero_margin=diff(gated),VH_zero_prediction=int(gated[j].argmax())))
    pd.DataFrame(rows).to_csv(run.HERE/'frozen_readout_diagnostics.csv',index=False)
    pd.DataFrame(harms).to_csv(run.HERE/'missing_VH_harm_trace.csv',index=False)
    pd.DataFrame(coverage).to_csv(run.HERE/'missing_VH_class_coverage.csv',index=False)
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby(['group','arm'])[['OA_pp','BA_pp','prediction_changes']].mean().to_string())

if __name__=='__main__':main()
