"""Cached readout/solve cost on one preselected B32 episode, not end-to-end speed."""
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run import HERE,global_descriptor,prior,adapted,late,cosine,token_descriptor,kernel


def main():
    feat=dict(np.load(HERE/'features.npz'));meta=pd.read_csv(HERE/'cohort.csv');y=meta.label.to_numpy()
    tasks=json.loads((HERE/'episodes.json').read_text());task=next(t for t in tasks if t['budget']==32)
    src=np.array(task['source']);s=np.array(task['supports'][0]);q=np.array(task['query']);ids=np.r_[s,q]
    local_s=np.arange(len(s));all_y=y[ids]
    g=global_descriptor(feat['global'],src);b=global_descriptor(feat['r18'],src)
    kg=g@g.T;kb=b@b.T
    with threadpool_limits(limits=1):
        pa=prior(kg,src,y,7)[ids];pb=prior(kb,src,y,7)[ids]
        ka=kg[np.ix_(ids,ids)];ke=kb[np.ix_(ids,ids)];records=[]
        for repeat in range(25):
            for arm in (['global_grid','target_only'] if repeat%2==0 else ['target_only','global_grid']):
                tic=time.perf_counter()
                old=cosine(feat['grid2'][ids].reshape(len(ids),2,-1),local_s)
                k=.5*ka+.5*old
                if arm=='target_only':
                    pre,_=token_descriptor(feat['prepool'][ids],feat['valid12'][ids],local_s)
                    post,_=token_descriptor(feat['postpool'][ids],feat['valid12'][ids],local_s)
                    post[:,:,::3,:]=pre[:,:,::3,:]
                    k=.5*ka+.25*old+.25*kernel(post)
                a,_=adapted(k,pa,local_s,all_y);b,_=adapted(ke,pb,local_s,all_y)
                out=late(a[len(s):],b[len(s):])
                elapsed=time.perf_counter()-tic
                if repeat>=5:records.append(dict(arm=arm,repeat=repeat,seconds=elapsed,queries=len(q),supports=len(s)))
    pd.DataFrame(records).to_csv(HERE/'cached_latency.csv',index=False)
    print(pd.DataFrame(records).groupby('arm').seconds.agg(['median','mean']).to_string())


if __name__=='__main__':main()
