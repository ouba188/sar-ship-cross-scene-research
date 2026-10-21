"""Posthoc scalar-family reachability, never used for training or selection."""
import numpy as np
import pandas as pd
import torch
import run as exp

ROOT=exp.ROOT


def main():
    f,splits=exp.old.relation.load_data();allrows=[];summary=[];dependency=[]
    for fold,(tr,val,test) in enumerate(splits):
        geometry=np.load(ROOT/f'geometry_f{fold}.npz')
        for seed in range(3):
            data=np.load(ROOT/f'inputs_f{fold}_s{seed}.npz')
            a=data['base'][test];r=data['residual'][test];y=f.iloc[test].class_id.to_numpy()
            b=a[np.arange(len(y)),y,None]-a;d=r[np.arange(len(y)),y,None]-r
            b[np.arange(len(y)),y]=1;d[np.arange(len(y)),y]=0
            lo=np.zeros(len(y));hi=np.full(len(y),2.);feasible=np.ones(len(y),bool)
            for c in range(8):
                positive=d[:,c]>1e-12;negative=d[:,c]<-1e-12;zero=~(positive|negative)
                lo[positive]=np.maximum(lo[positive],(-b[positive,c]+1e-6)/d[positive,c])
                hi[negative]=np.minimum(hi[negative],(-b[negative,c]+1e-6)/d[negative,c])
                feasible[zero]&=b[zero,c]>0
            feasible &= lo<hi
            w=(lo+hi)/2;w[~feasible]=1
            verify=(a+w[:,None]*r).argmax(1)==y
            assert np.array_equal(verify[feasible],np.ones(feasible.sum(),bool))
            old_correct=(a+r).argmax(1)==y
            joint=np.load(ROOT/f'scores_f{fold}_joint_s{seed}.npz');jcorrect=joint['scores'].argmax(1)==y
            potential=feasible&~old_correct
            distance=np.maximum(np.maximum(lo-1,1-hi),0)
            summary.append(dict(fold=fold,seed=seed,n=len(test),reference_correct=int(old_correct.sum()),
                reachable_correct=int((feasible|old_correct).sum()),potential_rescues=int(potential.sum()),
                actual_rescues=int((jcorrect&~old_correct).sum()),actual_harms=int((~jcorrect&old_correct).sum()),
                potential_weight_distance_median=float(np.median(distance[potential])) if potential.any() else 0,
                potential_needs_over_quarter=int((potential&(distance>.25)).sum())))
            for j,i in enumerate(test):
                allrows.append(dict(obs_id=int(f.iloc[i].obs_id),fold=fold,seed=seed,class_id=int(y[j]),
                    reference_correct=bool(old_correct[j]),reachable=bool(feasible[j]|old_correct[j]),
                    rescue_possible=bool(potential[j]),joint_correct=bool(jcorrect[j]),
                    nearest_weight_distance=float(distance[j]) if feasible[j] else np.nan))
            saved=torch.load(ROOT/f'head_f{fold}_joint_s{seed}.pt',map_location='cpu',weights_only=False)
            gate=exp.Gate('joint').eval();gate.load_state_dict(saved['state'])
            content=torch.tensor(data['content'][test]);g=torch.tensor(geometry['geometry'][test]);s=torch.tensor(geometry['sham'][test])
            with torch.no_grad():wg=gate(content,g).numpy();ws=gate(content,s).numpy()
            np.testing.assert_allclose(wg,joint['weight'],atol=1e-6,rtol=0)
            dependency.append(dict(fold=fold,seed=seed,step=saved['step'],mean_geometry_weight_change=float(np.abs(wg-ws).mean()),
                max_geometry_weight_change=float(np.abs(wg-ws).max()),
                geometry_changes_prediction=int(((a+wg[:,None]*r).argmax(1)!=(a+ws[:,None]*r).argmax(1)).sum())))
    pd.DataFrame(summary).to_csv(ROOT/'oracle_family_summary.csv',index=False,lineterminator='\n')
    pd.DataFrame(allrows).to_csv(ROOT/'oracle_family_rows.csv',index=False,lineterminator='\n')
    pd.DataFrame(dependency).to_csv(ROOT/'geometry_dependency.csv',index=False,lineterminator='\n')
    print(pd.DataFrame(summary).groupby('seed')[['reference_correct','reachable_correct','potential_rescues','actual_rescues','actual_harms','potential_needs_over_quarter']].sum().to_string())
    print(pd.DataFrame(dependency).to_string(index=False))


if __name__=='__main__':main()
