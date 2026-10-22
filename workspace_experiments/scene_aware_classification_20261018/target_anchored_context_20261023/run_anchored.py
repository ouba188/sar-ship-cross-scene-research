"""Target-anchored contextual modulation with exact linear-reader grouping."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
STRONG = HERE.parent/'context_strong_reader_20261022'
EXP = HERE.parent/'context_expansion_20261022'
sys.path.insert(0, str(STRONG))
from run_strong import View, KINDS, combine, fit_head, weights, risk, metrics, SEEDS

ARMS = ['anchored', 'constant_env', 'scene_replacement']


def grouped_reader(raw, names, fit, y, identities, alpha):
    models = {k: View(raw[k], fit, k) for k in names}
    views = {k: models[k].transform(raw[k]) for k in names}
    z = combine(views, names)
    head = fit_head(z, y, identities, fit, alpha)
    # Fit identically to the saved float32 reader, then evaluate its linear map
    # in float64 so the channel-group decomposition is algebraically checkable.
    views64 = {k: models[k].transform(raw[k].astype(np.float64)) for k in names}
    score = head.predict(combine(views64,names))*3
    q = np.zeros((len(y), 32, 3), np.float64)
    offset = 0
    reference_z = []
    for name in names:
        v, h = models[name], raw[name]
        dim = views[name].shape[1]
        normalization = np.sqrt(dim) if len(names) > 1 else 1.
        w = ((head.coef_[:,offset:offset+dim]/v.post.scale_[None,:]) @ v.pca.components_)/v.scale.scale_[None,:]*3/normalization
        center = h[fit].mean(0, dtype=np.float64)
        reference_z.append(v.transform(center[None]))
        if name == 'stem':
            groups = [np.r_[p*128+np.arange(g*8,(g+1)*8),p*128+64+np.arange(g*8,(g+1)*8)] for p in range(2) for g in range(8)]
            start = 0
        else:
            groups = np.array_split(np.arange(h.shape[1]),16)
            start = 16
        for g, cols in enumerate(groups):
            q[:,start+g] = (h[:,cols]-center[cols]) @ w[:,cols].T
        offset += dim
    if len(names) > 1:
        zz = np.concatenate([a/np.sqrt(a.shape[1]) for a in reference_z],1)
    else:
        zz = reference_z[0]
    bias = head.predict(zz)[0]*3
    err = float(np.max(np.abs(q.sum(1)+bias-score)))
    assert err < 1e-7,err
    return score,q,bias,err


class Anchored(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(79,32),torch.nn.ReLU(),torch.nn.Linear(32,32))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, env, contributions, valid):
        a = .5*torch.tanh(self.net(env))
        return (a[:,:,None]*contributions).sum(1)*valid[:,None]


def prepare(fold,raw,t,bins):
    old = dict(np.load(EXP/f'fold{fold}_inputs.npz'))
    strong = np.load(STRONG/f'fold{fold}_base.npz')
    train,val,test = [old[k] for k in ['train','validation','test']]
    names = KINDS[str(strong['selected_kind'])]
    alpha = float(strong['selected_alpha'])
    y,ids = t.class_id.to_numpy(),t.mmsi.to_numpy()
    bs,q,bias,err = grouped_reader(raw,names,train,y,ids,alpha)
    assert np.allclose(bs,strong['base'],atol=3e-4)
    assert np.array_equal(bs[test].argmax(1),strong['base'][test].argmax(1))
    oo,oq = np.zeros_like(bs),np.zeros_like(q)
    errors = [err]
    for b in sorted(set(bins[train])):
        held = train[bins[train] == b]
        fit = train[(bins[train] != b)&~np.isin(ids[train],ids[held])]
        assert not set(ids[fit])&set(ids[held])
        ss,qq,_,ee = grouped_reader(raw,names,fit,y,ids,alpha)
        oo[held],oq[held] = ss[held],qq[held]
        errors.append(ee)
    assert np.allclose(oo[train],strong['oof'][train],atol=3e-4)
    mask = old['m']
    nodes = (old['split']*mask[:,:,None]).sum(1)/np.maximum(mask.sum(1,keepdims=True),1)
    env = np.c_[nodes,old['stats']].astype(np.float32)
    has = mask.any(1)
    mapping = pd.read_csv(HERE.parent/'context_reference_20261022/replacement_map.csv').set_index('instance_id')
    loc = dict(zip(t.instance_id,range(len(t))))
    donors = np.arange(len(t))
    for iid,row in mapping.iterrows():
        donors[loc[iid]] = loc[row.donor]
    assert (t.iloc[donors]['product'].to_numpy() == t['product'].to_numpy()).all()
    constant = np.broadcast_to(env[train[has[train]]].mean(0),env.shape).copy()
    return dict(base=bs,oof=oo,q=q,oq=oq,bias=bias,env=env,constant_env=constant,
        scene_replacement=env[donors],has=has,train=train,validation=val,test=test,
        max_group_replay_error=max(errors),selected_kind=str(strong['selected_kind']))


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    t = pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    f = np.load(EXP/'readout/features.npz')
    deep = np.load(STRONG/'deep_features.npz')
    assert np.array_equal(f['instance_ids'],deep['instance_ids'])
    raw = {'stem':f['target'],'layer3':deep['layer3'],'layer4':deep['layer4']}
    y,ids = t.class_id.to_numpy(),t.mmsi.to_numpy()
    alloc = pd.read_csv(EXP/'pass_assignment.csv').set_index('pass_group').group_bin
    bins = t.pass_group.map(alloc).to_numpy()
    records,rows,curves,checks = [],[],[],[]
    (HERE/'models').mkdir(exist_ok=True)
    for fold in range(6):
        d = prepare(fold,raw,t,bins)
        np.savez_compressed(HERE/f'fold{fold}_inputs.npz',**d)
        train,val,test = [d[k] for k in ['train','validation','test']]
        oo = torch.tensor(d['oof'],dtype=torch.float32,device='cuda')
        oq = torch.tensor(d['oq'],dtype=torch.float32,device='cuda')
        q = torch.tensor(d['q'],dtype=torch.float32,device='cuda')
        has = torch.tensor(d['has'],dtype=torch.float32,device='cuda')
        yy = torch.tensor(y,dtype=torch.long,device='cuda')
        w = torch.tensor(weights(y[train],ids[train]),dtype=torch.float32,device='cuda')
        for seed in SEEDS:
            for arm in ARMS:
                env = torch.tensor(d['env' if arm == 'anchored' else arm],device='cuda',dtype=torch.float32)
                torch.manual_seed(seed)
                model = Anchored().cuda()
                opt = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                br,be = risk(y[val],d['base'][val]),0
                for epoch in range(1,201):
                    opt.zero_grad()
                    s = oo[train]+model(env[train],oq[train],has[train])
                    loss = (torch.nn.functional.cross_entropy(s,yy[train],reduction='none')*w).mean()
                    loss.backward()
                    opt.step()
                    if epoch%5 == 0:
                        with torch.no_grad():
                            ss = d['base'][val]+model(env[val],q[val],has[val]).cpu().numpy()
                        rr = risk(y[val],ss)
                        curves.append(dict(fold=fold,arm=arm,seed=seed,epoch=epoch,loss=float(loss.detach()),val_nll=rr))
                        if rr < br-1e-6:
                            br,be = rr,epoch
                            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(best)
                with torch.no_grad():
                    delta = model(env,q,has).cpu().numpy()
                    null = model(env[test],torch.zeros_like(q[test]),has[test]).cpu().numpy()
                assert not delta[~d['has']].any() and not null.any()
                score = d['base']+delta
                torch.save(best,HERE/'models'/f'fold{fold}_{arm}_{seed}.pt')
                bp,pp = d['base'][test].argmax(1),score[test].argmax(1)
                records.append(dict(fold=fold,arm=arm,seed=seed,epoch=be,val_nll=br,parameters=sum(p.numel() for p in model.parameters()),
                    **metrics(y[test],score[test]),rescued=int(((bp!=y[test])&(pp==y[test])).sum()),harmed=int(((bp==y[test])&(pp!=y[test])).sum())))
                checks.append(dict(fold=fold,arm=arm,seed=seed,group_replay_error=d['max_group_replay_error'],zero_evidence_exact=True,no_context_exact=True))
                for i in test:
                    rows.append(dict(fold=fold,arm=arm,seed=seed,instance_id=t.iloc[i].instance_id,class_id=int(y[i]),
                        prediction=int(score[i].argmax()),base_prediction=int(d['base'][i].argmax()),neighbors=int(f['mask'][i].sum()),
                        **{f's{c}':float(score[i,c]) for c in range(3)},**{f'b{c}':float(d['base'][i,c]) for c in range(3)}))
                print('fold',fold,arm,seed,'epoch',be,'net',records[-1]['rescued']-records[-1]['harmed'],flush=True)
        for filename,values in [('metrics',records),('predictions',rows),('curves',curves),('verification',checks)]:
            pd.DataFrame(values).to_csv(HERE/(filename+'.csv'),index=False)
    p = pd.DataFrame(rows)
    pooled=[]
    for (arm,seed),q0 in p.groupby(['arm','seed']):
        pooled.append(dict(arm=arm,seed=seed,**metrics(q0.class_id.to_numpy(),q0[['s0','s1','s2']].to_numpy()),
            net=int((q0.class_id==q0.prediction).sum()-(q0.class_id==q0.base_prediction).sum())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    (HERE/'summary.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,heads=54,targets=len(t),
        source_only=True,grouping='fixed pretrained channels, not PCA axes',zero_evidence_no_environment_delta=True),indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
