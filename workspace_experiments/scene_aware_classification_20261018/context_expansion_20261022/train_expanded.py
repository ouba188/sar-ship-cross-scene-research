"""Frozen cross-acquisition expansion and matched role-projection comparisons."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'context_region_relation_20261022'))
from run_relations import Relation,unit,summary_geometry
from run_ports24 import Baseline,weights,risk,metrics

ARMS=['whole_unit','split_unit','conditional','split_role','conditional_role','attention_role']
SEEDS=[17,29,43]
MODEL_ARM={'split_role':'split_unit','conditional_role':'conditional','attention_role':'attention'}


class Projection:
    def __init__(self,x):
        self.scale=StandardScaler().fit(x)
        self.pca=PCA(n_components=32,svd_solver='full').fit(self.scale.transform(x))
        self.post=StandardScaler().fit(self.pca.transform(self.scale.transform(x)))
    def project(self,x):return self.post.transform(self.pca.transform(self.scale.transform(x)))


def reconstruct(p,x):
    return p.scale.inverse_transform(p.pca.inverse_transform(p.post.inverse_transform(p.project(x))))


def split_data(t):
    groups=np.asarray(sorted(t.pass_group.unique()));np.random.default_rng(20261022).shuffle(groups)
    allocation={g:i%6 for i,g in enumerate(groups)}
    bins=t.pass_group.map(allocation).to_numpy();identity=t.mmsi.astype(str).to_numpy()
    folds=[];rows=[]
    for fold in range(6):
        test=np.flatnonzero(bins==fold)
        val=np.flatnonzero((bins==(fold+1)%6)&~np.isin(identity,identity[test]))
        train=np.flatnonzero((bins!=fold)&(bins!=(fold+1)%6)&~np.isin(identity,identity[np.r_[test,val]]))
        assert not set(identity[train])&set(identity[np.r_[test,val]])
        assert not set(identity[val])&set(identity[test])
        for role,ix in [('train',train),('validation',val),('test',test)]:
            rows.extend(dict(fold=fold,role=role,instance_id=t.iloc[i].instance_id,mmsi=identity[i],pass_group=t.iloc[i].pass_group) for i in ix)
        folds.append((train,val,test))
    pd.DataFrame(rows).to_csv(HERE/'splits.csv',index=False)
    pd.DataFrame([dict(pass_group=g,group_bin=b) for g,b in allocation.items()]).to_csv(HERE/'pass_assignment.csv',index=False)
    return folds,bins


def prepare(fold,indices,bins,t,f,edges,projection_rows):
    train,val,test=indices;y=t.class_id.to_numpy();identity=t.mmsi.astype(str).to_numpy();h=f['target'];m=f['mask']
    assert set(y[train])==set(range(3))
    choices=[]
    for alpha in [1.,10.,100.]:
        base=Baseline(h[train],y[train],identity[train],alpha)
        choices.append((risk(y[val],base.predict(h[val])),alpha,base))
    _,alpha,base=min(choices,key=lambda a:a[0]);score=base.predict(h)
    oof=np.zeros_like(score);rt=[]
    for b in sorted(set(bins[train])):
        held=train[bins[train]==b]
        fit=train[(bins[train]!=b)&~np.isin(identity[train],identity[held])]
        assert set(y[fit])==set(range(3))
        oof[held]=Baseline(h[fit],y[fit],identity[fit],alpha).predict(h[held]);rt.extend(held.tolist())
    assert set(rt)==set(train)
    st=summary_geometry(f['geometry'],m);ss=StandardScaler().fit(st[train]);st=ss.transform(st).astype(np.float32)
    gs=StandardScaler().fit(f['geometry'][train][m[train]])
    g=gs.transform(f['geometry'].reshape(-1,7)).reshape(len(t),16,7).astype(np.float32)
    whole=unit(base.project(f['neighbor'].reshape(-1,256)).reshape(len(t),16,32))
    whole=np.concatenate([whole,whole],-1)/np.sqrt(2)
    common=[];dedicated=[]
    source_ids=set(t.iloc[train].instance_id)
    unique=edges[edges.instance_id.isin(source_ids)].drop_duplicates(['product','detection_id'])
    loc=t.reset_index().set_index('instance_id')['index']
    ii=unique.instance_id.map(loc).to_numpy();jj=unique.slot.to_numpy(dtype=int)
    for region in ['fg','bg']:
        raw=f['neighbor_'+region];valid=f[region+'_valid']
        z=unit(base.project(raw.reshape(-1,256)).reshape(len(t),16,32));z[~valid]=0;common.append(z)
        selected=valid[ii,jj];source=raw[ii[selected],jj[selected]]
        proj=Projection(source)
        rz=unit(proj.project(raw.reshape(-1,256)).reshape(len(t),16,32));rz[~valid]=0;dedicated.append(rz)
        center=source.mean(0)
        for role,ix in [('train',train),('validation',val),('test',test)]:
            values=raw[ix][valid[ix]&m[ix]]
            denominator=float(((values-center)**2).sum())
            for kind,pp in [('target_projection',base),('role_projection',proj)]:
                err=float(((values-reconstruct(pp,values))**2).sum())/max(denominator,1e-12)
                projection_rows.append(dict(fold=fold,region=region,role=role,projection=kind,normalized_reconstruction_error=err,
                    edges=len(values),unique_fit_objects=len(source)))
    q=unit(base.project(h)).astype(np.float32)
    return dict(train=train,validation=val,test=test,base=score,oof=oof,q=q,whole=whole.astype(np.float32),
        split=(np.concatenate(common,-1)/np.sqrt(2)).astype(np.float32),
        role=(np.concatenate(dedicated,-1)/np.sqrt(2)).astype(np.float32),g=g,stats=st,m=m,alpha=alpha)


def inputs_for(arm,d,device='cpu'):
    n=d['whole'] if arm=='whole_unit' else d['role'] if arm.endswith('_role') else d['split']
    return [torch.as_tensor(a,dtype=torch.bool if k=='m' else torch.float32,device=device) for k,a in
        [('q',d['q']),('n',n),('g',d['g']),('stats',d['stats']),('m',d['m'])]]


def main():
    torch.set_num_threads(2);started=time.perf_counter();device='cuda' if torch.cuda.is_available() else 'cpu'
    t=pd.read_csv(HERE/'readout/targets.csv',dtype={'mmsi':str});f=np.load(HERE/'readout/features.npz')
    edges=pd.read_csv(HERE/'readout/neighbors.csv')
    assert len(t)==len(f['target']) and len(edges)==f['mask'].sum()
    if 'slot' not in edges:edges['slot']=edges.groupby('instance_id').cumcount()
    assert t.groupby('mmsi').class_id.nunique().max()==1
    y=t.class_id.to_numpy();identity=t.mmsi.to_numpy();folds,bins=split_data(t)
    (HERE/'models').mkdir(exist_ok=True);predictions=[];records=[];projection_rows=[];coverage=[];curves=[]
    for fold,indices in enumerate(folds):
        d=prepare(fold,indices,bins,t,f,edges,projection_rows);train,val,test=indices
        np.savez_compressed(HERE/f'fold{fold}_inputs.npz',**d)
        for role,ix in [('train',train),('validation',val),('test',test)]:
            for c in range(3):
                sel=ix[y[ix]==c]
                coverage.append(dict(fold=fold,role=role,class_id=c,n=len(sel),ships=t.iloc[sel].mmsi.nunique(),
                    context=int(f['mask'][sel].any(1).sum()),edges=int(f['mask'][sel].sum())))
        target=torch.tensor(y,dtype=torch.long,device=device);oof=torch.tensor(d['oof'],dtype=torch.float32,device=device)
        weight=torch.tensor(weights(y[train],identity[train]),dtype=torch.float32,device=device)
        for seed in SEEDS:
            for arm in ARMS:
                torch.manual_seed(seed);model=Relation(MODEL_ARM.get(arm,arm)).to(device);inputs=inputs_for(arm,d,device)
                opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=0;best_risk=risk(y[val],d['base'][val])
                start=time.perf_counter()
                for epoch in range(1,201):
                    opt.zero_grad();logits=oof[train]+model(*[x[train] for x in inputs])
                    loss=(torch.nn.functional.cross_entropy(logits,target[train],reduction='none')*weight).mean()
                    loss.backward();opt.step()
                    if epoch%5==0:
                        with torch.no_grad():sv=d['base'][val]+model(*[x[val] for x in inputs]).cpu().numpy()
                        vr=risk(y[val],sv);curves.append(dict(fold=fold,seed=seed,arm=arm,epoch=epoch,loss=float(loss.detach()),val_nll=vr))
                        if vr<best_risk-1e-6:
                            best_risk=vr;best_epoch=epoch;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(best)
                with torch.no_grad():delta=model(*inputs).cpu().numpy()
                assert not delta[~d['m'].any(1)].any()
                scores=d['base']+delta;bp=d['base'][test].argmax(1);pred=scores[test].argmax(1)
                torch.save(best,HERE/'models'/f'fold{fold}_{arm}_{seed}.pt')
                records.append(dict(fold=fold,seed=seed,arm=arm,epoch=best_epoch,val_nll=best_risk,base_alpha=d['alpha'],
                    seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),n=len(test),
                    **metrics(y[test],scores[test]),rescued=int(((bp!=y[test])&(pred==y[test])).sum()),harmed=int(((bp==y[test])&(pred!=y[test])).sum())))
                for i in test:
                    predictions.append(dict(fold=fold,seed=seed,arm=arm,instance_id=t.iloc[i].instance_id,class_id=int(y[i]),
                        prediction=int(scores[i].argmax()),base_prediction=int(d['base'][i].argmax()),neighbors=int(d['m'][i].sum()),
                        **{f's{c}':float(scores[i,c]) for c in range(3)},**{f'b{c}':float(d['base'][i,c]) for c in range(3)}))
                print('fold',fold,'seed',seed,arm,'epoch',best_epoch,'net',records[-1]['rescued']-records[-1]['harmed'],flush=True)
        pd.DataFrame(records).to_csv(HERE/'metrics.csv',index=False)
        pd.DataFrame(predictions).to_csv(HERE/'predictions.csv',index=False)
        pd.DataFrame(coverage).to_csv(HERE/'coverage.csv',index=False)
        pd.DataFrame(projection_rows).to_csv(HERE/'projection_diagnostics.csv',index=False)
        pd.DataFrame(curves).to_csv(HERE/'curves.csv',index=False)
    p=pd.DataFrame(predictions);pooled=[]
    for (arm,seed),q in p.groupby(['arm','seed']):
        for scope,part in [('all',q),('context',q[q.neighbors.gt(0)])]:
            pooled.append(dict(arm=arm,seed=seed,scope=scope,n=len(part),**metrics(part.class_id.to_numpy(),part[['s0','s1','s2']].to_numpy()),
                rescued=int(((part.base_prediction!=part.class_id)&(part.prediction==part.class_id)).sum()),
                harmed=int(((part.base_prediction==part.class_id)&(part.prediction!=part.class_id)).sum())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    (HERE/'run_summary.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,heads=len(records),targets=len(t),
        ships=t.mmsi.nunique(),passes=t.pass_group.nunique(),context=int(f['mask'].any(1).sum()),edges=int(f['mask'].sum()),device=device),indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).query('scope == "all"').groupby('arm')[['OA','BA','rescued','harmed']].mean().to_string(),flush=True)

if __name__=='__main__':main()
