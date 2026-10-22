"""Compare fixed seed17 retraining to its original two-path parent on native chips."""
import sys
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent/'contrastive_spatial_pool_20261023'))
import analyze_pool
from run_pool import EXP

if __name__=='__main__':
    t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    p=pd.read_csv(HERE/'predictions.csv')
    p=p[p.seed.eq(17)|p.arm.eq('scalar_signed')].copy()
    q=pd.read_csv(HERE.parent/'predictions.csv')
    q=q[q.seed.eq(17)&q.arm.eq('conditional_signed')]
    p=pd.concat([p,q],ignore_index=True)
    p['selected_lambda']=1.
    analyze_pool.HERE=HERE
    analyze_pool.ARMS=['scalar_signed','signed_only','conditional_signed']
    analyze_pool.REVIEW_IDS=['pair_0031218','pair_0031577','pair_0052252']
    panels,missing=analyze_pool.render(t,p)
    print('panels',panels,'missing',missing)
