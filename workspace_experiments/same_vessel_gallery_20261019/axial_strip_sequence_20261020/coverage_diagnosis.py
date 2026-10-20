"""Actual valid token counts and held-out short/long coverage, no new fitting."""
import numpy as np
import pandas as pd
from run import ROOT, prior, LOCAL


def main():
    f=pd.read_csv(prior.ROOT/'cohort.csv',dtype={'mmsi':str})
    excluded=set(pd.read_csv(LOCAL/'EXCLUSIONS.csv').obs_id)
    m=np.load(ROOT/'masses.npy',mmap_mode='r')
    coverage=pd.read_csv(ROOT/'coverage.csv').set_index(['obs_id','pol'])
    records=[]
    for fold in range(3):
        for part,ii in zip(('train','val','test'),prior.indices(f,fold)):
            ii=np.array([i for i in ii if f.iloc[i].obs_id not in excluded])
            for pol,p in enumerate(('vv','vh')):
                for ref,r in enumerate(('ship','radar_or_fallback')):
                    # Original rotation, center line, target-mask occupancy.
                    weights=np.asarray(m[ii,pol,0,ref,0,:,0],dtype=np.float32)
                    count=(weights>.05).sum(-1)
                    triples=((weights[:,:-2]>.05)&(weights[:,1:-1]>.05)&(weights[:,2:]>.05)).sum(-1)
                    sub=coverage.loc[[(int(v),p) for v in f.iloc[ii].obs_id]]
                    records.append(dict(fold=fold,partition=part,pol=p,reference=r,n=len(ii),
                        disabled=int(sub.disabled.sum()),no_target_tokens=int((count==0).sum()),
                        median_tokens=float(np.median(count)),p25_tokens=float(np.quantile(count,.25)),
                        p75_tokens=float(np.quantile(count,.75)),lt3_tokens=int((count<3).sum()),
                        no_valid_triples=int((triples==0).sum()),median_triples=float(np.median(triples))))
    pd.DataFrame(records).to_csv(ROOT/'sequence_coverage.csv',index=False)
    print(pd.DataFrame(records).query("partition=='test'").to_string(index=False))


if __name__=='__main__': main()
