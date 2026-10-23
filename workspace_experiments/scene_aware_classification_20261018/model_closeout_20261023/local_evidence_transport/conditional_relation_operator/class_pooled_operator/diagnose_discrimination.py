"""Source-only comparison of class evidence before/after the conditional inverse."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('discrimination_impl',HERE/'run.py')
run=importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
exp=run.exp


def main():
    table,x,mass,bank,idx,tr,va,te=exp.source_data()
    ids=idx[tr]
    labels=table.class_id.to_numpy(int)[ids]
    # Hold the transforms fixed across score formulas; no new fit or selection.
    states={'identity':(np.eye(32),np.eye(32))}
    weights=np.load(HERE/'weights.npz')
    states['selected_excess43']=(weights['excess_43_Mb'],weights['excess_43_Mn'])
    rows=[]
    group_rows=[]
    full=np.load(exp.HERE/'predictions_43.npz')['full'][ids]
    oldcorrect=full.argmax(-1)==labels
    for arm in ('independent','covariance','transfer','excess','excess_shuffle101'):
        reader=run.Reader(x,mass,bank,arm)
        reader.initialize(ids)
        for state,(mb,mn) in states.items():
            z,raw=run.epi.diag.relation(reader,ids,torch.tensor(mb,device='cuda',dtype=torch.float32),torch.tensor(mn,device='cuda',dtype=torch.float32))
            valid=reader.counts[:,ids].cpu().numpy().T>=2
            for g,group in enumerate(exp.edge.GROUPS):
                true=z[np.arange(len(ids)),labels,g]
                gap=true-(z[:,:,g].sum(-1)-true)/7
                rival=z[:,:,g].copy()
                rival[np.arange(len(ids)),labels]=-np.inf
                boundary=true-rival.max(-1)
                for c in range(8):
                    for outcome,mask in [('all',np.ones(len(ids),bool)),('full_correct',oldcorrect),('full_wrong',~oldcorrect)]:
                        keep=(labels==c)&valid[:,g]&mask
                        if not keep.any():
                            continue
                        rows.append(dict(arm=arm,state=state,group=group,class_id=c,subset=outcome,n=int(keep.sum()),
                            true_minus_mean_wrong=float(gap[keep].mean()),true_minus_best_wrong=float(boundary[keep].mean()),
                            top1=float((z[keep,:,g].argmax(-1)==labels[keep]).mean())))
            combined=z.sum(-1)/np.maximum(valid.sum(-1,keepdims=True),1)
            group_rows.append(dict(arm=arm,state=state,**exp.dep.metrics(labels,combined)))
    frame=pd.DataFrame(rows)
    frame.to_csv(HERE/'source_evidence_classes.csv',index=False)
    summary=frame.groupby(['arm','state','group','subset'])[['true_minus_mean_wrong','true_minus_best_wrong','top1']].mean()
    summary.to_csv(HERE/'source_evidence_summary.csv')
    pd.DataFrame(group_rows).to_csv(HERE/'source_evidence_direct.csv',index=False)
    print(summary.xs('all',level='subset').to_string())
    print(pd.DataFrame(group_rows).to_string(index=False))


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
