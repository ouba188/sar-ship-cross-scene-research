"""Reuse loss diagnostics and account for source/target regional effects, no fits."""
from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('learning', HERE.parent / 'analysis/learning_diagnosis.py')
learning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learning)
learning.HERE = HERE
learning.ROOT = HERE
learning.main()

source = pd.read_csv(HERE/'source_groups.csv')
target = pd.read_csv(HERE/'test_groups.csv')
keys = ['seed','left','right','field','group']
joined = source.merge(target, on=keys, suffixes=('_source','_target'), validate='one_to_one')
joined.to_csv(HERE/'source_target_groups.csv', index=False)

# Reproduce the actual observation-keyed assignments without touching any model.
import sys
sys.path.insert(0,str(HERE.parent))
import run_heads as old
table,_,_ = old.load_data()
rows=[]
for fold in range(3):
    for role,ix in old.splits(fold).items():
        permutations=[''.join(map(str,np.random.default_rng([4711,int(o)]).permutation(3))) for o in table.iloc[ix].obs_id]
        for perm,count in pd.Series(permutations).value_counts().items():
            rows.append(dict(fold=fold,role=role,permutation=perm,count=int(count),fraction=float(count/len(ix))))
pd.DataFrame(rows).to_csv(HERE/'permutation_counts.csv',index=False)

# Independently verify persisted full scores against the previously frozen result.
keys=['fold','seed','instance_id']
score=[f's{i}' for i in range(8)]
new=pd.read_csv(HERE/'test_predictions.csv',usecols=keys+['arm','epoch']+score,float_precision='round_trip')
prior=pd.read_csv(HERE.parent/'test_predictions.csv',usecols=keys+['arm','epoch']+score,float_precision='round_trip')
prior=prior[prior.arm.eq('dynamic')].set_index(keys)
checks=[]
for (fold,seed),g in new[new.arm.eq('dynamic')].groupby(['fold','seed']):
    other=prior.loc[pd.MultiIndex.from_frame(g[keys])]
    np.testing.assert_array_equal(g.epoch.to_numpy(),other.epoch.to_numpy())
    error=float(abs(g[score].to_numpy()-other[score].to_numpy()).max())
    assert error < 1e-6
    checks.append(dict(fold=fold,seed=seed,n=len(g),full_replay_error=error,epoch_exact=True))
pd.DataFrame(checks).to_csv(HERE/'test_full_replay_checks.csv',index=False)
