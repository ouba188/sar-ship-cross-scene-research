"""Replace learned residual by its source-training mean; no refitting or oracle."""
import numpy as np
import pandas as pd
import torch
from run import ROOT, LOCAL, StripResidual, prior, predict


def main():
    torch.set_num_threads(2)
    f=pd.read_csv(prior.ROOT/'cohort.csv',dtype={'mmsi':str}); y=f.class_id.to_numpy()
    excluded=set(pd.read_csv(LOCAL/'EXCLUSIONS.csv').obs_id)
    x=np.load(ROOT/'tokens.npy',mmap_mode='r'); m=np.load(ROOT/'masses.npy',mmap_mode='r')
    perm=torch.tensor(np.load(ROOT/'permutations.npy'),device='cuda',dtype=torch.int64)
    zeros=torch.zeros(len(f),8,device='cuda'); rows=[]; classes=[]; predictions=[]
    for fold in range(3):
        tr,_,test=prior.indices(f,fold)
        tr=np.array([i for i in tr if f.iloc[i].obs_id not in excluded])
        test=np.array([i for i in test if f.iloc[i].obs_id not in excluded])
        for seed in range(3):
            head=torch.load(ROOT/f'head_f{fold}_strip_conv_s{seed}.pt',map_location='cpu',weights_only=False)
            data=np.load(ROOT/f'scores_f{fold}_strip_conv_s{seed}.npz')
            np.testing.assert_array_equal(test,data['rows'])
            if head['step']:
                model=StripResidual(True).cuda(); model.load_state_dict(head['state']); model.eval()
                residual=predict(model,x,m,perm,tr,head['mean'].cuda(),head['std'].cuda(),'strip_conv',zeros)
                average=residual.mean(0)
                del model,residual
            else: average=np.zeros(8,np.float32)
            current=data['scores']-data['base']; ss=data['base']+average
            # Top1 is unaffected by a per-query common class-score shift.
            centered=current-current.mean(-1,keepdims=True); mean_c=average-average.mean()
            denominator=float(np.mean(centered**2))
            relative=float(np.mean((centered-mean_c)**2)/denominator) if denominator>0 else 0.
            rows.append(dict(fold=fold,seed=seed,step=head['step'],
                flips_vs_actual=int(np.sum(ss.argmax(1)!=data['scores'].argmax(1))),
                residual_rms=float(np.sqrt(denominator)),source_constant_relative_error=relative,
                **prior.metrics(y[test],ss)))
            for c in range(8):
                classes.append(dict(fold=fold,seed=seed,class_id=c,source_mean=float(average[c]),
                    test_mean=float(current[:,c].mean()),test_std=float(current[:,c].std())))
            for j,i in enumerate(test):
                predictions.append(dict(obs_id=int(f.iloc[i].obs_id),fold=fold,seed=seed,class_id=int(y[i]),
                    prediction=int(ss[j].argmax()),actual_prediction=int(data['scores'][j].argmax()),
                    base_prediction=int(data['base'][j].argmax()),correct=bool(ss[j].argmax()==y[i])))
            print('diagnosis',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'source_constant_metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(ROOT/'source_constant_classes.csv',index=False)
    p=pd.DataFrame(predictions); p.to_csv(ROOT/'source_constant_predictions.csv',index=False)
    summary=[]
    for seed,part in p.groupby('seed'):
        summary.append(dict(seed=seed,oa=part.correct.mean(),ba=part.groupby('class_id').correct.mean().mean(),
            flips_vs_actual=int(part.prediction.ne(part.actual_prediction).sum()),
            net=int(part.correct.sum()-part.base_prediction.eq(part.class_id).sum())))
    pd.DataFrame(summary).to_csv(ROOT/'source_constant_summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__': main()
