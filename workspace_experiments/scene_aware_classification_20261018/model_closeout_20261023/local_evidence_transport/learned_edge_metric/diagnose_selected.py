"""Selected-state cache-only W-to-I substitution; no fitting or selection."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('learned_metric_diagnostic', HERE / 'run.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
metrics = experiment.edge.dep.metrics
table = pd.read_csv(HERE.parent.parent / 'regional_utility_pilot/upstream/table.csv')
y = table.class_id.to_numpy(dtype=int)
roles = table.role.to_numpy()
fixed = np.load(HERE.parent / 'edge_difference/features.npz')
weights = np.load(HERE / 'weights.npz')
receipts = pd.read_csv(HERE / 'training_receipts.csv')
rows, pairs, contributions = [], [], []

for seed in experiment.SEEDS:
    predictions = np.load(HERE / f'predictions_{seed}.npz')
    features = np.load(HERE / f'features_{seed}.npz')
    anchor = predictions['anchor']
    rival_score = anchor.copy()
    rival_score[np.arange(len(y)), y] = -np.inf
    rival = rival_score.argmax(-1)
    for arm in experiment.ARMS:
        a = weights[f'{arm}_{seed}_a']
        z0, z1 = fixed[arm], features[arm]
        d0 = (z0 * a).sum(-1)
        d0 -= d0.mean(-1, keepdims=True)
        variants = dict(anchor=anchor, identity_same_a=anchor + d0,
                        actual_W_same_a=predictions[arm])
        step = int(receipts.query('seed == @seed and arm == @arm').selected_step.iloc[0])
        for role in ('utility_train', 'selection', 'evaluation'):
            ids = np.flatnonzero(roles == role)
            for name, scores in variants.items():
                m = metrics(y[ids], scores[ids])
                rows.append(dict(seed=seed, arm=arm, selected_step=step, role=role,
                                 variant=name, joint_CE=.5*(m['NLL']+m['balanced_NLL']), **m))
            for old, new in [('anchor', 'identity_same_a'), ('identity_same_a', 'actual_W_same_a'),
                             ('anchor', 'actual_W_same_a')]:
                before = variants[old][ids].argmax(-1) == y[ids]
                after = variants[new][ids].argmax(-1) == y[ids]
                pairs.append(dict(seed=seed, arm=arm, role=role, reference=old, variant=new,
                                  rescued=int((after & ~before).sum()), harmed=int((before & ~after).sum()),
                                  net=int(after.sum()-before.sum())))
            # Fixed anchor rival isolates continuous contributions from rival switching.
            for g, group in enumerate(experiment.edge.GROUPS):
                base = (z0[ids,y[ids],g]-z0[ids,rival[ids],g])*a[g]
                actual = (z1[ids,y[ids],g]-z1[ids,rival[ids],g])*a[g]
                for name, values in [('identity_same_a', base), ('actual_W_same_a', actual),
                                     ('W_change_at_same_a', actual-base)]:
                    contributions.append(dict(seed=seed,arm=arm,role=role,group=group,variant=name,
                                              coefficient=float(a[g]),mean=float(values.mean()),
                                              q10=float(np.quantile(values,.1)),median=float(np.median(values)),
                                              q90=float(np.quantile(values,.9))))

pd.DataFrame(rows).to_csv(HERE/'selected_W_identity_metrics.csv',index=False)
pd.DataFrame(pairs).to_csv(HERE/'selected_W_identity_pairs.csv',index=False)
pd.DataFrame(contributions).to_csv(HERE/'selected_W_identity_contributions.csv',index=False)
result = pd.DataFrame(rows)
print(result.query("arm == 'mixed'")[['seed','selected_step','role','variant','OA','BA','J','joint_CE']].to_string(index=False))
print('\nEVALUATION MEANS')
print(result.query("role == 'evaluation'").groupby(['arm','variant'])[['OA','BA','NLL','Brier']].mean().to_string())
print('\nPAIRED NETS')
print(pd.DataFrame(pairs).pivot(index=['arm','seed','role'],columns=['reference','variant'],values='net').to_string())
