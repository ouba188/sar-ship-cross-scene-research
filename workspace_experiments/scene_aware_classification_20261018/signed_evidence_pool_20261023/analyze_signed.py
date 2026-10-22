"""Fixed-head decomposition, probability/decision separation and native-chip review."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train_signed import HERE,EXP,NEW,ARMS,Correction,descriptors,metrics
sys.path.insert(0,str(HERE.parent/'contrastive_spatial_pool_20261023'))
import analyze_pool


def main():
    torch.set_num_threads(2)
    t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    p=pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    cache=np.load(HERE/'signed_features.npz');names=list(cache['reader_names'])
    source=[];parts=[];individual=[]
    for fold in range(6):
        d=dict(np.load(NEW/f'fold{fold}_inputs.npz'))
        mo=cache['moments'][names.index(f'fold{fold}_outer')]
        te=d['test'];base=d['base'][te].astype(float);y=t.iloc[te].class_id.to_numpy()
        xraw=descriptors(d['base'],mo)
        for seed in [17,29,43]:
            ck=torch.load(HERE/'models'/f'fold{fold}_conditional_signed_seed{seed}.pt',weights_only=False)
            model=Correction('conditional_signed');model.load_state_dict(ck['state']);model.eval()
            x=torch.tensor((xraw-ck['mu'])/ck['scale'],dtype=torch.float32)
            m=torch.tensor(mo,dtype=torch.float32)
            with torch.no_grad():
                coeff=model.net(x[te]).tanh().numpy()
                components=coeff*mo[te]
                delta=components.sum(-1)
                full=model(x[te],m[te]).numpy()
                np.testing.assert_allclose(delta-delta.mean(1,keepdims=True),full,atol=2e-6,rtol=2e-6)
                no_signed_x=x[te].clone()
                # Set raw signed descriptors to zero, including their source normalization.
                no_signed_x[...,[3,4]]=torch.tensor(-ck['mu'][[3,4]]/ck['scale'][[3,4]],dtype=torch.float32)
                no_signed_m=m[te].clone();no_signed_m[...,1]=0
                removed=model(no_signed_x,no_signed_m).numpy()
            arrays={'full':full,'std_path_only':components[...,0],'signed_path_only':components[...,1],
                    'signed_input_and_path_removed':removed}
            for mode,dd in arrays.items():
                score=base+dd
                for j,i in enumerate(te):
                    individual.append(dict(instance_id=t.iloc[i].instance_id,fold=fold,seed=seed,mode=mode,class_id=int(y[j]),
                        base_prediction=int(base[j].argmax()),prediction=int(score[j].argmax()),
                        **{f's{k}':float(score[j,k]) for k in range(3)}))
            masked=base.copy();masked[np.arange(len(y)),y]=-np.inf;wrong=masked.argmax(1)
            for j,i in enumerate(te):
                parts.append(dict(instance_id=t.iloc[i].instance_id,fold=fold,seed=seed,class_id=int(y[j]),wrong_class=int(wrong[j]),
                    std_pair_delta=float(components[j,y[j],0]-components[j,wrong[j],0]),
                    signed_pair_delta=float(components[j,y[j],1]-components[j,wrong[j],1]),
                    std_true_coefficient=float(coeff[j,y[j],0]),signed_true_coefficient=float(coeff[j,y[j],1])))
            for role in ['validation','test']:
                for c in range(3):
                    ix=d[role][t.iloc[d[role]].class_id.to_numpy()==c]
                    source.append(dict(fold=fold,seed=seed,role=role,class_id=c,n=len(ix),
                        mean_absolute_z=float(x[ix].abs().mean()),max_absolute_z=float(x[ix].abs().max())))
    ip=pd.DataFrame(individual);ip.to_csv(HERE/'path_predictions.csv',index=False)
    rows=[]
    for (mode,seed),g in ip.groupby(['mode','seed']):
        net=g.prediction.eq(g.class_id).astype(int)-g.base_prediction.eq(g.class_id).astype(int)
        rows.append(dict(mode=mode,seed=seed,net=int(net.sum()),**metrics(g.class_id.to_numpy(),g[['s0','s1','s2']].to_numpy())))
    pd.DataFrame(rows).to_csv(HERE/'path_pooled.csv',index=False)
    pd.DataFrame(parts).to_csv(HERE/'path_contributions.csv',index=False)
    pd.DataFrame(source).to_csv(HERE/'descriptor_shift.csv',index=False)
    # A fixed seed is reviewed, not the best-performing seed.
    review=p[p.seed.eq(17)|p.arm.eq('linear_distribution')].copy()
    review['selected_lambda']=1.
    analyze_pool.HERE=HERE;analyze_pool.ARMS=ARMS
    analyze_pool.REVIEW_IDS=['pair_0031577','pair_0031218','pair_0052252']
    panels,missing=analyze_pool.render(t,review)
    print(pd.DataFrame(rows).to_string(index=False))
    print('panels',panels,'missing',missing)


if __name__=='__main__':main()
