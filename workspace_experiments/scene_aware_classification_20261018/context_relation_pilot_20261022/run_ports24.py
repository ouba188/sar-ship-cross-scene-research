"""Matched source-only A/B/C pilots; no target support labels or test tuning."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent / 'ports24'


def weights(y, identity):
    w = np.zeros(len(y))
    for c in range(3):
        hit = y == c
        counts = pd.Series(identity[hit]).value_counts()
        if len(counts):
            w[hit] = [1 / (len(counts) * counts[x]) for x in identity[hit]]
    return w / max(w.mean(), 1e-8)


def risk(y, score):
    p = torch.softmax(torch.tensor(score,dtype=torch.float64),1).numpy()
    return np.mean([np.mean(-np.log(np.maximum(p[y==c,c],1e-12))) for c in np.unique(y)])


def metrics(y, score):
    pred=score.argmax(1)
    return dict(OA=float(np.mean(y==pred)*100), BA=float(np.mean([np.mean(pred[y==c]==c) for c in np.unique(y)])*100),
                macro_F1=float(f1_score(y,pred,labels=range(3),average='macro',zero_division=0)*100),
                **{f'recall_{c}':float(np.mean(pred[y==c]==c)*100) if np.any(y==c) else None for c in range(3)})


class Baseline:
    def __init__(self, h, y, ids, alpha):
        self.scale=StandardScaler().fit(h)
        self.pca=PCA(n_components=min(32,len(h)-1),svd_solver='full').fit(self.scale.transform(h))
        self.post=StandardScaler().fit(self.pca.transform(self.scale.transform(h)))
        z=self.project(h)
        self.head=Ridge(alpha=alpha).fit(z,np.eye(3)[y],sample_weight=weights(y,ids))

    def project(self,h):
        return self.post.transform(self.pca.transform(self.scale.transform(h)))

    def predict(self,h):
        return self.head.predict(self.project(h))*3


class Residual(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(dim,32),torch.nn.ReLU(),torch.nn.Linear(32,3))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,has):
        return self.net(x)*has[:,None]


def main():
    torch.set_num_threads(2)
    start=time.perf_counter()
    data=np.load(HERE/'features.npz')
    t=pd.read_csv(HERE/'targets.csv',dtype={'mmsi':str})
    h,n,g,m=[data[k] for k in ('target','neighbor','geometry','mask')]
    if set(t.class_id)!=set(range(3)):
        raise ValueError('Frozen pilot vocabulary is bulk/fishing/general cargo')
    y=t.class_id.to_numpy(); identity=t.mmsi.to_numpy(); groups=t.pass_group.to_numpy()
    has=m.any(1).astype(np.float32)
    count=m.sum(1).clip(1)
    gm=(g*m[:,:,None]).sum(1)/count[:,None]
    gv=(((g-gm[:,None])**2)*m[:,:,None]).sum(1)/count[:,None]
    geom=np.c_[gm,gv,m.sum(1)/16]
    group_list=sorted(set(groups))
    rows,predictions,splits,diagnostics=[],[],[],[]
    all_ids=np.arange(len(t))
    for fold,test_group in enumerate(group_list):
        test=all_ids[groups==test_group]
        available=all_ids[(groups!=test_group)&~np.isin(identity,identity[test])]
        source_groups=sorted(set(groups[available]))
        # Fixed cyclic source acquisition holdout; never choose from target performance.
        val_group=source_groups[fold%len(source_groups)]
        val=available[groups[available]==val_group]
        train=available[(groups[available]!=val_group)&~np.isin(identity[available],identity[val])]
        if set(y[train])!=set(range(3)) or min(len(train),len(val),len(test))==0:
            raise ValueError(f'Insufficient closed-set source training in fold {fold}')
        assert not(set(identity[train])&set(identity[val]))
        assert not(set(identity[train])&set(identity[test]))
        assert not(set(identity[val])&set(identity[test]))
        assert not(set(groups[train])&set(groups[test]))
        for role,ii in [('train',train),('validation',val),('test',test)]:
            splits.extend(dict(fold=fold,role=role,instance_id=t.iloc[i].instance_id) for i in ii)
        choices=[]
        for alpha in (1.,10.,100.):
            candidate=Baseline(h[train],y[train],identity[train],alpha)
            choices.append((risk(y[val],candidate.predict(h[val])),alpha,candidate))
        _,alpha,base=min(choices,key=lambda z:z[0])
        base_score=base.predict(h)
        # The residual sees out-of-acquisition base predictions, not perfect in-sample labels.
        oof=np.zeros((len(t),3),np.float32)
        legal_train=[]
        for sg in sorted(set(groups[train])):
            held=train[groups[train]==sg]
            fit=train[(groups[train]!=sg)&~np.isin(identity[train],identity[held])]
            if set(y[fit])!=set(range(3)) or len(fit)<5:
                continue
            local=Baseline(h[fit],y[fit],identity[fit],alpha)
            oof[held]=local.predict(h[held])
            legal_train.extend(held.tolist())
        rt=np.asarray(sorted(legal_train),dtype=int)
        if not len(rt):
            raise ValueError('No legal source out-of-fold residual training')
        projected=base.project(n.reshape(-1,n.shape[-1])).reshape(len(t),16,-1)
        bag=(projected*m[:,:,None]).sum(1)/count[:,None]
        bases=dict(A=dict(score=base_score,epoch=0,params=0))
        for seed in (17,29,43):
            run=dict(bases)
            for arm,features in [('B',geom),('C',np.c_[geom,bag])]:
                torch.manual_seed(seed)
                scaler=StandardScaler().fit(features[train])
                x=torch.tensor(scaler.transform(features),dtype=torch.float32)
                mask=torch.tensor(has)
                model=Residual(x.shape[1])
                optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                target=torch.tensor(y,dtype=torch.long)
                samplew=torch.tensor(weights(y[rt],identity[rt]),dtype=torch.float32)
                fixed=torch.tensor(oof)
                best=copy.deepcopy(model.state_dict())
                best_loss=risk(y[val],base_score[val]); best_epoch=0
                for epoch in range(1,201):
                    model.train(); optimizer.zero_grad()
                    score=fixed[rt]+model(x[rt],mask[rt])
                    loss=(torch.nn.functional.cross_entropy(score,target[rt],reduction='none')*samplew).mean()
                    loss.backward(); optimizer.step()
                    if epoch%5==0:
                        model.eval()
                        with torch.no_grad():
                            score=base_score[val]+model(x[val],mask[val]).numpy()
                        objective=risk(y[val],score)
                        if objective<best_loss-1e-6:
                            best_loss=objective; best_epoch=epoch; best=copy.deepcopy(model.state_dict())
                model.load_state_dict(best); model.eval()
                with torch.no_grad():
                    delta=model(x,mask).numpy()
                assert np.array_equal(delta[has==0],np.zeros_like(delta[has==0]))
                final=base_score+delta
                run[arm]=dict(score=final,epoch=best_epoch,params=sum(p.numel() for p in model.parameters()))
                # Small model state supports exact replay; do not archive pretrained backbone.
                torch.save(dict(state=best,mean=scaler.mean_,scale=scaler.scale_,base_alpha=alpha),HERE/f'fold{fold}_{arm}_seed{seed}.pt')
            for arm,result in run.items():
                score=result['score']
                for scope,ids in [('all',test),('context',test[has[test]>0])]:
                    if not len(ids):
                        continue
                    bp=base_score[ids].argmax(1); p=score[ids].argmax(1)
                    rows.append(dict(fold=fold,test_group=test_group,port=t.iloc[test[0]].port,seed=seed,arm=arm,scope=scope,n=len(ids),
                        **metrics(y[ids],score[ids]),rescued=int(((bp!=y[ids])&(p==y[ids])).sum()),
                        harmed=int(((bp==y[ids])&(p!=y[ids])).sum()),epoch=result['epoch'],parameters=result['params']))
                for i in test:
                    predictions.append(dict(fold=fold,seed=seed,arm=arm,instance_id=t.iloc[i].instance_id,
                        class_id=int(y[i]),prediction=int(score[i].argmax()),base_prediction=int(base_score[i].argmax()),
                        neighbors=int(m[i].sum()),**{f's{c}':float(score[i,c]) for c in range(3)},
                        **{f'b{c}':float(base_score[i,c]) for c in range(3)}))
            diagnostics.append(dict(fold=fold,seed=seed,ntrain=len(train),nval=len(val),ntest=len(test),
                n_oof=len(rt),context_train=int(has[rt].sum()),context_val=int(has[val].sum()),context_test=int(has[test].sum()),
                base_alpha=alpha,B_epoch=run['B']['epoch'],C_epoch=run['C']['epoch']))
        print('fold',fold,'test',len(test),'train',len(train),'val',len(val),'context',has[test].sum(),flush=True)
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(HERE/'predictions.csv',index=False)
    pd.DataFrame(splits).to_csv(HERE/'splits.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(HERE/'training_diagnostics.csv',index=False)
    p=pd.DataFrame(predictions)
    pooled=[]
    for (seed,arm),q in p.groupby(['seed','arm']):
        for scope,q in [('all',q),('context',q[q.neighbors.gt(0)])]:
            pooled.append(dict(seed=seed,arm=arm,scope=scope,n=len(q),**metrics(q.class_id.to_numpy(),q[['s0','s1','s2']].to_numpy()),
                rescued=int(((q.base_prediction!=q.class_id)&(q.prediction==q.class_id)).sum()),
                harmed=int(((q.base_prediction==q.class_id)&(q.prediction!=q.class_id)).sum())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    (HERE/'run_summary.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,folds=len(group_list),seeds=[17,29,43],
        protocol='LO acquisition + MMSI exclusion, source acquisition validation, source OOF residual; no target labels for fitting',
        note='A is deterministic; repeating its predictions across neural seeds is not independent baseline evidence'),indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
