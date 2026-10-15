"""Posthoc computational contribution probe, not a physical intervention or refit."""

import pickle
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from run import OUT, core
from analyze import weights, readout


def main():
    with (OUT/'frozen_gates.pkl').open('rb') as h:p=pickle.load(h)[1]
    records=[]
    names=['ova','source_distance','support_distance','count']
    for task in ('f0921','oss2','ports/Port Klang'):
        for draw in range(5):
            z=np.load(OUT/'target'/task/f'u1_d{draw}_random.npz')
            q=(z['inputs']-p['scaler'].mean_)/p['scaler'].scale_
            c=(z['cal_inputs']-p['scaler'].mean_)/p['scaler'].scale_
            known,w,u=weights(z['labels'],1)
            for feature in [-1,0,1,2,3]:
                qq=q.copy();cc=c.copy()
                if feature>=0:qq[:,feature]=0;cc[:,feature]=0
                score=expit(qq@p['gate'].coef_[0]+p['gate'].intercept_[0])
                cal=expit(cc@p['gate'].coef_[0]+p['gate'].intercept_[0])
                threshold=np.quantile(cal,.95)
                result=readout(score,threshold,z['labels'],z['pred'],1)
                records.append(dict(task=task,draw=draw,removed='none' if feature<0 else names[feature],
                    auroc=100*roc_auc_score(~known,score,sample_weight=w+u),**result))
    frame=pd.DataFrame(records);frame.to_csv(OUT/'component_probe.csv',index=False)
    summary=frame.groupby(['task','removed'])[['auroc','known_false_reject','unknown_tpr','utility_20']].mean()
    summary.to_csv(OUT/'component_summary.csv');print(summary.to_string())
    core.write(OUT/'COMPONENT_PROBE.json',dict(posthoc=True,unknown_class=1,
        operation='set one standardized gate input to its source-training mean; recalibrate using support LOO only',
        no_retraining=True,not_physical_causality=True))


if __name__=='__main__':main()
