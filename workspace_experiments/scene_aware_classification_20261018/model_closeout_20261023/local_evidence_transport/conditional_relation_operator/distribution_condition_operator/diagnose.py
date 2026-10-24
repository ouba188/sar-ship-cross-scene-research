"""Matched output-representation diagnostic without changing fitted parameters."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('distribution_diag_run',HERE/'run.py')
run=importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
exp=run.exp


def main():
    table,x,mass,bank,idx,tr,va,te=exp.source_data()
    labels=table.class_id.to_numpy(int)
    full=np.load(exp.HERE/'predictions_43.npz')['full']
    w=np.load(HERE/'weights.npz')
    states={'identity':(np.eye(32),np.eye(32)),
        'selected_excess43':(w['excess_43_Mb'],w['excess_43_Mn'])}
    rows=[]
    for output in ('linear','rbf'):
        reader=run.Reader(x,mass,bank,'excess',output_kernel=output)
        reader.initialize(idx[tr])
        for state,(mb,mn) in states.items():
            matrices=[torch.tensor(v,device='cuda',dtype=torch.float32) for v in (mb,mn)]
            for role,locs in (('utility_train',tr),('selection',va)):
                with torch.no_grad():
                    for start in range(0,len(locs),128):
                        ids=idx[locs[start:start+128]]
                        z,_,valid,d=reader.scores(ids,*matrices,trace=True)
                        z=z.cpu().numpy(); v=valid.cpu().numpy()
                        energy=(d['Lqq'].diagonal(dim1=-2,dim2=-1).sum(-1)/d['mq']).cpu().numpy().T
                        T=d['T'].cpu().numpy().transpose(1,2,0)
                        truth=labels[ids]
                        strongest=np.argmax(np.where(np.arange(8)[None,:]==truth[:,None],-np.inf,full[ids]),axis=1)
                        for g,group in enumerate(exp.edge.GROUPS):
                            pos=np.arange(len(ids)); true=z[pos,truth,g]
                            rival=np.where(np.arange(8)[None,:]==truth[:,None],-np.inf,z[:,:,g]).max(-1)
                            for j,i in enumerate(ids):
                                if not v[j,truth[j],g]:
                                    continue
                                rows.append(dict(output=output,state=state,role=role,group=group,class_id=int(truth[j]),
                                    obs_id=int(table.iloc[i].obs_id),full_correct=bool(full[i].argmax()==truth[j]),
                                    mean_gap=float(true[j]-(z[j,:,g].sum()-true[j])/7),best_gap=float(true[j]-rival[j]),
                                    full_rival_gap=float(true[j]-z[j,strongest[j],g]),top1=bool(z[j,:,g].argmax()==truth[j]),
                                    true_reconstruction_skill=float(T[j,truth[j],g]/max(energy[j,g],1e-12))))
    frame=pd.DataFrame(rows)
    cols=['mean_gap','best_gap','full_rival_gap','top1','true_reconstruction_skill']
    frame.to_csv(HERE/'source_mechanism_samples.csv',index=False)
    classes=frame.groupby(['output','state','role','group','class_id'])[cols].mean()
    classes.to_csv(HERE/'source_mechanism_classes.csv')
    summary=classes.groupby(['output','state','role','group'])[cols].mean()
    summary.to_csv(HERE/'source_mechanism_summary.csv')
    print(summary.to_string())


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
