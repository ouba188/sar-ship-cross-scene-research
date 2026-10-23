"""Fixed-joint-head dependence removal and marginal-preserving replacement."""
import sys
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run import UP, metrics


def main(frozen=False):
    output=HERE/'frozen_marginal' if frozen else HERE
    baseline='anchor' if frozen else 'full'
    features=np.load(HERE/'features.npz')
    weights=np.load(output/'weights.npz')
    idx=features['idx']
    table=pd.read_csv(UP/'table.csv')
    te=np.flatnonzero(table.iloc[idx].role.eq('evaluation'))
    y=table.iloc[idx].class_id.to_numpy()
    cases=json.loads((output/'analysis/cases.json').read_text())
    rows,case_rows=[],[]
    for seed in (17,29,43):
        saved=np.load(output/f'predictions_{seed}.npz')
        base=saved[baseline][idx]
        coef=weights[f'joint_{seed}']
        for arm in ('joint','removed','shuffle101','shuffle211','shuffle307'):
            x=features['joint'].copy() if arm=='removed' else features[arm]
            if arm=='removed':
                x[:,:,6:]=0
            if frozen:
                x=x[:,:,6:]
            change=np.einsum('ncf,f->nc',x,coef)
            change-=change.mean(-1,keepdims=True)
            score=base+change
            if arm=='joint':
                np.testing.assert_allclose(score,saved['joint'][idx],rtol=1e-5,atol=2e-6)
            correct=score[te].argmax(-1)==y[te]
            old=saved['joint'][idx[te]].argmax(-1)==y[te]
            rows.append(dict(seed=seed,arm=arm,rescued=int((correct&~old).sum()),harmed=int((~correct&old).sum()),**metrics(y[te],score[te])))
            if seed==17:
                for case in cases:
                    j,c,r=[case[k] for k in ('j','true','rival')]
                    case_rows.append(dict(obs_id=case['obs_id'],arm=arm,prediction=int(score[j].argmax()),margin=float(score[j,c]-score[j,r])))
    pd.DataFrame(rows).to_csv(output/'fixed_head_interventions.csv',index=False)
    pd.DataFrame(case_rows).to_csv(output/'analysis/fixed_head_case_margins.csv',index=False)
    print(pd.DataFrame(rows).groupby('arm')[['OA','BA','NLL','Brier']].mean().to_string())
    print(pd.DataFrame(case_rows).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--frozen',action='store_true')
    main(parser.parse_args().frozen)
