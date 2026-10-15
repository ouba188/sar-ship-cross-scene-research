"""Exploratory attribution of the three positive fishing-holdout cases, no tuning."""

import numpy as np
import pandas as pd
from run import OUT, METHODS, core
from analyze import weights


def main():
    rows=[]; boats=[]
    pairs=[('f0921','ova'),('oss2','fixed_union'),('ports/Port Klang','joint')]
    for task,method in pairs:
        matrices=[]; totals=[]
        for draw in range(5):
            z=np.load(OUT/'target'/task/f'u1_d{draw}_random.npz')
            y=z['labels'];pred=z['pred'];known,w,u=weights(y,1)
            i=METHODS.index(method);j=METHODS.index('support_knn')
            new=z['score'][:,i]>z['support_thresholds'][i]
            old=z['score'][:,j]>z['support_thresholds'][j]
            correct=pred==y
            delta=100*(.2*u*(new.astype(int)-old)-.8*w*correct*(new.astype(int)-old))
            value=100*(.2*u*new-.8*w*correct*new)
            matrices.append(delta);totals.append(value)
            for cl in np.unique(y):
                mask=y==cl
                rows.append(dict(task=task,method=method,draw=draw,class_id=int(cl),n=int(mask.sum()),
                    unknown=bool(cl==1),known_correct_freed=int((mask&known&correct&old&~new).sum()),
                    known_correct_newly_rejected=int((mask&known&correct&~old&new).sum()),
                    unknown_extra_detected=int((mask&~known&~old&new).sum()),
                    unknown_newly_missed=int((mask&~known&old&~new).sum()),delta_u20=float(delta[mask].sum())))
        contribution=np.stack(matrices).mean(0);standalone=np.stack(totals).mean(0)
        rng=np.random.default_rng(20261014);sampled=[];absolute=[]
        for _ in range(1000):
            ix=np.concatenate([rng.choice(np.flatnonzero(y==cl),(y==cl).sum(),replace=True) for cl in np.unique(y)])
            sampled.append(contribution[ix].sum());absolute.append(standalone[ix].sum())
        boats.append(dict(task=task,method=method,unknown_ships=int((y==1).sum()),known_ships=int(known.sum()),
            delta_u20=float(contribution.sum()),delta_low=float(np.quantile(sampled,.025)),delta_high=float(np.quantile(sampled,.975)),
            delta_positive_draws=int((np.stack(matrices).sum(1)>0).sum()),u20=float(standalone.sum()),
            u20_low=float(np.quantile(absolute,.025)),u20_high=float(np.quantile(absolute,.975))))
    pd.DataFrame(rows).to_csv(OUT/'scoped_events.csv',index=False)
    pd.DataFrame(boats).to_csv(OUT/'scoped_bootstrap.csv',index=False)
    print(pd.DataFrame(boats).to_string(index=False))
    print(pd.DataFrame(rows).groupby(['task','method']).sum(numeric_only=True)[['known_correct_freed','known_correct_newly_rejected','unknown_extra_detected','unknown_newly_missed']].to_string())
    core.write(OUT/'SCOPED_ANALYSIS.json',dict(posthoc_method_by_dataset=True,
        fishing_holdout_previously_known_scope=True,deployable_selector=False,
        bootstrap='1000 paired query-ship resamples within true class; fixed fits and supports'))


if __name__=='__main__':main()
