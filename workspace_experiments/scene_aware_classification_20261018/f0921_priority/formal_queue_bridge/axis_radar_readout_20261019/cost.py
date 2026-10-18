"""Cached per-task adaptation latency; excludes image IO, CNN and detection."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_axis as ev
from run_condition import geometry_kernels

HERE=Path(__file__).resolve().parent


def main():
    f=dict(np.load(HERE.parent/'features.npz'))
    d=dict(np.load(HERE/'direction_features.npz'))
    axes=pd.read_csv(HERE/'axes.csv')
    geo=pd.read_csv(HERE/'geometry.csv')
    meta=pd.read_csv(HERE.parent/'cohort.csv')
    task=next(t for t in json.loads((HERE.parent/'episodes.json').read_text()) if t['budget']==32)
    source,q,s=np.array(task['source']),np.array(task['query']),np.array(task['supports'][0])
    y=meta.label.to_numpy()
    tokens,valid=ev.roi.merged_regions(f['prepool'],f['mass12'])
    joint=geometry_kernels(axes,geo)[2]
    missing=axes.axis_reliability.to_numpy()<1e-8
    available=d['radar_valid']
    data=[]
    with threadpool_limits(limits=1):
        g=ev.bridge.global_descriptor(f['global'],source)
        b=ev.bridge.global_descriptor(f['r18'],source)
        kg,kb=g@g.T,b@b.T
        prior=ev.bridge.prior(kg,source,y,7)
        expert,_=ev.bridge.adapted(kb,ev.bridge.prior(kb,source,y,7),s,y)
        grid=ev.bridge.cosine(f['grid2'].reshape(len(meta),2,-1),s).astype(float)
        for repeat in range(25):
            for arm in ['fg_bg_global','ship_soft','ship_radar','conditioned_ship']:
                start=time.perf_counter()
                fg,_=ev.bridge.token_descriptor(tokens,valid,s)
                kfg=ev.bridge.kernel(fg)
                local=kfg
                if arm!='fg_bg_global':
                    ship,_=ev.bridge.token_descriptor(d['ship'],d['ship_mass']>1e-8,s)
                    ship[missing]=np.tile(fg[missing],(1,1,4,1))/2
                    ks=ev.bridge.kernel(ship)
                    local=.5*kfg+.5*ks
                    if arm=='ship_radar':
                        ok=(d['radar_mass']>1e-8)&available[:,None]&~missing[:,None]
                        radar,_=ev.bridge.token_descriptor(d['radar'],ok,s)
                        radar[~available|missing]=ship[~available|missing]
                        local=.5*kfg+.25*ks+.25*ev.bridge.kernel(radar)
                    if arm=='conditioned_ship':
                        local=.5*kfg+.25*ks+.25*ks*joint
                score,_=ev.bridge.adapted(.5*kg+.25*grid+.25*local,prior,s,y)
                prob=ev.bridge.late(score[q],expert[q])
                if repeat>=5:
                    data.append(dict(arm=arm,repeat=repeat,milliseconds=1000*(time.perf_counter()-start)))
        saved=pd.concat([pd.read_csv(HERE/'predictions.csv',dtype={'fold':str}),
                         pd.read_csv(HERE/'condition_predictions.csv',dtype={'fold':str})])
        expected=saved[saved.arm.eq(arm)&saved.fold.eq(task['fold'])&saved.budget.eq(32)&saved.draw.eq(0)].set_index('row').loc[q,ev.roi.PROBS]
        np.testing.assert_allclose(prob,expected,atol=2e-12,rtol=0)
    frame=pd.DataFrame(data)
    frame.to_csv(HERE/'cached_latency_runs.csv',index=False)
    result=frame.groupby('arm').milliseconds.agg(['median','min','max'])
    result.to_csv(HERE/'cached_latency.csv')
    print(result.to_string())


if __name__=='__main__':
    main()
