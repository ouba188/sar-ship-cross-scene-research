"""Source-only loss replacement and scalar score-scale diagnosis; no test scoring."""
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from run import ROOT, AXIS, LOCAL, prior, Residual, baseline, predict


def main():
    start=time.perf_counter(); torch.set_num_threads(2)
    f=pd.read_csv(prior.ROOT/'cohort.csv',dtype={'mmsi':str}); y=f.class_id.to_numpy()
    excluded=pd.read_csv(LOCAL/'EXCLUSIONS.csv').obs_id.to_numpy()
    d=np.load(AXIS/'descriptors.npz'); old=np.concatenate((d['base'],d['axis']),1)
    rows=[]; curves=[]; scales=[]
    for fold in range(3):
        train,val,_=prior.indices(f,fold)
        train,val=[ii[~f.iloc[ii].obs_id.isin(excluded).to_numpy()] for ii in (train,val)]
        # Scores are constructed only for source partitions; target data is not evaluated.
        raw=np.concatenate((old[train],old[val]))
        base_np=baseline(raw,fold).astype(np.float32)
        n=len(train); yy=np.concatenate((y[train],y[val])); w=prior.weights(f,train)
        def ce(logscale):
            ss=base_np[:n]*np.exp(logscale)
            return np.average(logsumexp(ss,axis=1)-ss[np.arange(n),yy[:n]],weights=w)
        fitted=minimize_scalar(ce,bounds=(-4,4),method='bounded')
        scale=float(np.exp(fitted.x))
        for name,sl in (('train',slice(0,n)),('val',slice(n,None))):
            for kind,mult in (('baseline',1.),('positive_scale',scale)):
                scales.append(dict(fold=fold,partition=name,kind=kind,scale=mult,**prior.metrics(yy[sl],base_np[sl]*mult)))
        for arm in ('early_both','late_both'):
            cache=np.load(ROOT/(arm+'.npy'),mmap_mode='r')
            src=np.concatenate((cache[train],cache[val])).reshape(len(raw),60,64)
            mean=src[:n].mean(0); std=np.maximum(src[:n].std(0),1e-6)
            x=torch.tensor((src-mean)/std,device='cuda'); base=torch.tensor(base_np,device='cuda')
            target=2*F.one_hot(torch.tensor(yy,device='cuda'),8).float()-1
            weight=torch.tensor(w,dtype=torch.float32,device='cuda')
            torch.manual_seed(20261020); torch.cuda.manual_seed_all(20261020)
            rng=np.random.default_rng(20261020+31*fold)
            model=Residual().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            best=None; beststep=0; state=None
            for step in range(151):
                if step in (0,50,100,150):
                    model.eval(); ss=predict(model,x[n:],base[n:]); met=prior.metrics(yy[n:],ss)
                    curves.append(dict(fold=fold,arm=arm,step=step,**met))
                    key=(met['ba'],-met['brier'])
                    if best is None or key>best:
                        best=key;beststep=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if step==150: break
                ii=rng.integers(n,size=128); model.train();opt.zero_grad(set_to_none=True)
                error=(base[ii]+model(x[ii])-target[ii]).square().mean(-1)
                loss=(error*weight[ii]).mean();loss.backward();opt.step()
            for phase in ('terminal','selected'):
                if phase=='selected':model.load_state_dict(state)
                model.eval()
                for name,sl in (('train',slice(0,n)),('val',slice(n,None))):
                    ss=predict(model,x[sl],base[sl])
                    rows.append(dict(fold=fold,arm=arm,phase=phase,partition=name,step=beststep if phase=='selected' else 150,**prior.metrics(yy[sl],ss)))
            print('source only',fold,arm,beststep,rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'source_mse_diagnosis.csv',index=False)
    pd.DataFrame(curves).to_csv(ROOT/'source_mse_selection.csv',index=False)
    pd.DataFrame(scales).to_csv(ROOT/'source_scale_diagnosis.csv',index=False)
    (ROOT/'SOURCE_DIAGNOSIS.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,fits=6,seed=0,
        scope='source train/val only, no target evaluation',change='CE replaced by +/-1 squared regression, matching frozen Ridge target'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
