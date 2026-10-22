"""Target-only context and independent class-bias controls for anchored modulation."""
import json
import numpy as np
import pandas as pd
import torch
from run_anchored import HERE, EXP, Anchored, SEEDS, weights, risk, metrics
from train_expanded import Baseline, unit

ARMS = ['own_regions_matched','own_regions_full','anchored_plus_bias']


class WithBias(Anchored):
    def __init__(self):
        super().__init__()
        self.bias_head = torch.nn.Linear(32,3)
        torch.nn.init.zeros_(self.bias_head.weight)
        torch.nn.init.zeros_(self.bias_head.bias)

    def forward(self,env,contributions,valid):
        return super().forward(env,contributions,valid)+self.bias_head(self.net[1](self.net[0](env)))*valid[:,None]


def own_input(t,f,fold):
    d = np.load(EXP/f'fold{fold}_inputs.npz')
    train = d['train']
    base = Baseline(f['target'][train],t.class_id.to_numpy()[train],t.mmsi.to_numpy()[train],float(d['alpha']))
    assert np.allclose(base.predict(f['target']),d['base'],atol=1e-5)
    regions=[]
    for role in ['fg','bg']:
        values=unit(base.project(f['target_'+role]))
        values[~f['target_'+role+'_valid']]=0
        regions.append(values)
    source = train[d['m'][train].any(1)]
    stats = np.broadcast_to(d['stats'][source].mean(0),(len(t),15))
    return np.c_[np.concatenate(regions,1)/np.sqrt(2),stats].astype(np.float32)


def main():
    torch.set_num_threads(2)
    t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    f=np.load(EXP/'readout/features.npz')
    y,ids=t.class_id.to_numpy(),t.mmsi.to_numpy()
    rows,records,checks=[],[],[]
    for fold in range(6):
        d=np.load(HERE/f'fold{fold}_inputs.npz')
        train,val,test=[d[k] for k in ['train','validation','test']]
        own=own_input(t,f,fold)
        np.savez_compressed(HERE/f'fold{fold}_own_input.npz',env=own)
        q=torch.tensor(d['q'],dtype=torch.float32,device='cuda')
        oq=torch.tensor(d['oq'],dtype=torch.float32,device='cuda')
        oo=torch.tensor(d['oof'],dtype=torch.float32,device='cuda')
        yy=torch.tensor(y,device='cuda')
        w=torch.tensor(weights(y[train],ids[train]),dtype=torch.float32,device='cuda')
        for seed in SEEDS:
            for arm in ARMS:
                env=torch.tensor(d['env'] if arm=='anchored_plus_bias' else own,device='cuda',dtype=torch.float32)
                active=np.ones(len(t),bool) if arm=='own_regions_full' else d['has']
                valid=torch.tensor(active,dtype=torch.float32,device='cuda')
                torch.manual_seed(seed)
                model=(WithBias() if arm=='anchored_plus_bias' else Anchored()).cuda()
                opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                br,be=risk(y[val],d['base'][val]),0
                for epoch in range(1,201):
                    opt.zero_grad()
                    s=oo[train]+model(env[train],oq[train],valid[train])
                    loss=(torch.nn.functional.cross_entropy(s,yy[train],reduction='none')*w).mean()
                    loss.backward();opt.step()
                    if epoch%5==0:
                        with torch.no_grad(): ss=d['base'][val]+model(env[val],q[val],valid[val]).cpu().numpy()
                        rr=risk(y[val],ss)
                        if rr<br-1e-6:
                            br,be=rr,epoch
                            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(best)
                with torch.no_grad():delta=model(env,q,valid).cpu().numpy()
                assert not delta[~active].any()
                score=d['base']+delta
                path=HERE/'models'/f'fold{fold}_{arm}_{seed}.pt'
                torch.save(best,path)
                replay=(WithBias() if arm=='anchored_plus_bias' else Anchored()).cuda().eval()
                replay.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
                with torch.no_grad():back=d['base'][test]+replay(env[test],q[test],valid[test]).cpu().numpy()
                err=float(np.abs(back-score[test]).max())
                assert err<1e-5 and np.array_equal(back.argmax(1),score[test].argmax(1))
                checks.append(dict(fold=fold,arm=arm,seed=seed,error=err,activated_test=int(active[test].sum())))
                bp,pp=d['base'][test].argmax(1),score[test].argmax(1)
                records.append(dict(fold=fold,arm=arm,seed=seed,epoch=be,val_nll=br,parameters=sum(p.numel() for p in model.parameters()),
                    **metrics(y[test],score[test]),rescued=int(((bp!=y[test])&(pp==y[test])).sum()),harmed=int(((bp==y[test])&(pp!=y[test])).sum())))
                for i in test:rows.append(dict(fold=fold,arm=arm,seed=seed,instance_id=t.iloc[i].instance_id,class_id=int(y[i]),
                    prediction=int(score[i].argmax()),base_prediction=int(d['base'][i].argmax()),active=bool(active[i]),has_external=bool(d['has'][i]),
                    **{f's{c}':float(score[i,c]) for c in range(3)}))
                print(fold,arm,seed,'step',be,'net',records[-1]['rescued']-records[-1]['harmed'],flush=True)
        pd.DataFrame(rows).to_csv(HERE/'control_predictions.csv',index=False)
        pd.DataFrame(records).to_csv(HERE/'control_metrics.csv',index=False)
        pd.DataFrame(checks).to_csv(HERE/'control_replay.csv',index=False)
    p=pd.DataFrame(rows);pooled=[]
    for (arm,seed),q0 in p.groupby(['arm','seed']):
        for scope,part in [('all',q0),('external',q0[q0.has_external]),('no_external',q0[~q0.has_external])]:
            pooled.append(dict(arm=arm,seed=seed,scope=scope,n=len(part),**metrics(part.class_id.to_numpy(),part[['s0','s1','s2']].to_numpy()),
                net=int((part.prediction==part.class_id).sum()-(part.base_prediction==part.class_id).sum())))
    pd.DataFrame(pooled).to_csv(HERE/'control_pooled.csv',index=False)
    print(pd.DataFrame(pooled).query('scope == "all"').to_string(index=False),flush=True)


if __name__=='__main__':main()
