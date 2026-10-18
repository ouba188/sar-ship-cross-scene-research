"""Prespecified explanatory cases and cached readout cost, not model selection."""
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from controls import HERE,PARENT,bridge,merged_regions


def main():
    pred = pd.read_csv(HERE/'predictions.csv')
    known = pred[(pred.budget.eq(32)) & ((pred.row.eq(80)&pred.draw.eq(0)) | (pred.row.eq(236)&pred.draw.eq(1)))]
    known.to_csv(HERE/'previous_rescues_all_arms.csv',index=False)
    paired = pd.read_csv(HERE/'paired_events.csv')
    changed = paired[paired.arm.eq('target_only') & paired.comparator.isin(['roi_global','roi_repeat_target','fg_bg_global']) & (paired.rescue|paired.harm)]
    changed.to_csv(HERE/'changed_events.csv',index=False)
    arms=['global_grid','target_only','roi_global','fg_bg_global','roi_repeat_target']
    meta=pd.read_csv(PARENT/'cohort.csv');y=meta.label.to_numpy()
    feat=dict(np.load(PARENT/'features.npz'))
    task=next(t for t in json.loads((PARENT/'episodes.json').read_text()) if t['budget']==32)
    src=np.array(task['source']);s=np.array(task['supports'][0]);q=np.array(task['query']);ids=np.r_[s,q]
    support=np.arange(len(s));all_y=y[ids]
    g=bridge.global_descriptor(feat['global'],src);b=bridge.global_descriptor(feat['r18'],src)
    kg=g@g.T;kb=b@b.T
    f={name:feat[name][ids] for name in ['grid2','postpool','prepool','mass12','valid12']}
    merged,valid=merged_regions(f['prepool'],f['mass12'])
    repeated=f['postpool'].copy();repeated[:,:,::3]=merged[:,:,:1]
    timing=[]
    with threadpool_limits(limits=1):
        pa=bridge.prior(kg,src,y,7)[ids];pb=bridge.prior(kb,src,y,7)[ids]
        ka=kg[np.ix_(ids,ids)];ke=kb[np.ix_(ids,ids)]
        for repeat in range(25):
            for arm in (arms if repeat%2==0 else arms[::-1]):
                tic=time.perf_counter()
                grid=bridge.cosine(f['grid2'].reshape(len(ids),2,-1),support)
                k=.5*ka+.5*grid
                if arm!='global_grid':
                    if arm=='target_only':
                        pre,_=bridge.token_descriptor(f['prepool'],f['valid12'],support)
                        z,_=bridge.token_descriptor(f['postpool'],f['valid12'],support)
                        z[:,:,::3]=pre[:,:,::3]
                    elif arm=='roi_repeat_target':
                        z,_=bridge.token_descriptor(repeated,f['valid12'],support)
                    else:
                        n=1 if arm=='roi_global' else 2
                        z,_=bridge.token_descriptor(merged[:,:,:n],valid[:,:n],support)
                    k=.5*ka+.25*grid+.25*bridge.kernel(z)
                a,_=bridge.adapted(k,pa,support,all_y)
                e,_=bridge.adapted(ke,pb,support,all_y)
                result=bridge.late(a[len(s):],e[len(s):])
                seconds=time.perf_counter()-tic
                if repeat>=5:timing.append(dict(arm=arm,repeat=repeat,seconds=seconds,queries=len(q),supports=len(s)))
    pd.DataFrame(timing).to_csv(HERE/'cached_latency.csv',index=False)
    print(pd.DataFrame(timing).groupby('arm').seconds.agg(['median','mean']).to_string())
    print(known[['row','arm','prediction','margin','ap']].to_string(index=False))


if __name__=='__main__':main()
