"""Describe reference sensitivity without fitting or selecting a new model."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
old=np.load(HERE.parent/'predictions.npz')
new=np.load(HERE/'predictions_excess_43.npz')
table=pd.read_csv(HERE.parents[3]/'regional_utility_pilot/upstream/table.csv')
idx=old['idx']
np.testing.assert_array_equal(idx,new['idx'])
roles=table.iloc[idx].role.to_numpy()
rows=[]
for role in ('utility_train','selection','evaluation'):
    mask=roles==role
    for method,data,fullkey in [('old_fixed',old,'full_43'),('episodic',new,'full')]:
        key='original_excess_43' if method=='old_fixed' else 'episodic_original'
        orig=data[key][mask]-data[fullkey][mask]
        for bank in ('101','307','509','711','907'):
            key=f'{bank}_excess_43' if method=='old_fixed' else f'episodic_{bank}'
            changed=data[key][mask]-data[fullkey][mask]
            rows.append(dict(role=role,method=method,bank=bank,
                correlation=float(np.corrcoef(orig.ravel(),changed.ravel())[0,1]),
                original_rms=float(np.sqrt(np.mean(orig*orig))),
                new_rms=float(np.sqrt(np.mean(changed*changed))),
                change_rms=float(np.sqrt(np.mean((changed-orig)**2))),
                # Fit scalar only to measure collinearity, never deploy it.
                unexplained_fraction=float(1-(np.sum(changed*orig)**2)/(np.sum(changed*changed)*np.sum(orig*orig)))))
pd.DataFrame(rows).to_csv(HERE/'stability_comparison.csv',index=False)
print(pd.DataFrame(rows).groupby(['method','role']).mean(numeric_only=True).to_string())
