"""One fixed-layer3, new-acquisition-partition replication; no feature extraction."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run_anchored import EXP, STRONG, Anchored, View, fit_head, weights, risk, metrics, SEEDS
from controls import WithBias
from train_expanded import Baseline, unit, summary_geometry

PARTITION_SEED = 20261023
ARMS = ['centered_own_full', 'uncentered_own_full', 'centered_plus_bias']


def partition(t):
    groups = np.asarray(sorted(t.pass_group.unique()))
    assert len(groups) == 163
    np.random.default_rng(PARTITION_SEED).shuffle(groups)
    allocation = {g:i % 6 for i,g in enumerate(groups)}
    pd.DataFrame([dict(pass_group=g, group_bin=b) for g,b in allocation.items()]).to_csv(HERE/'pass_assignment.csv', index=False)
    bins = t.pass_group.map(allocation).to_numpy()
    ids = t.mmsi.to_numpy()
    folds, rows, summaries = [], [], []
    for fold in range(6):
        test = np.flatnonzero(bins == fold)
        val = np.flatnonzero((bins == (fold+1) % 6) & ~np.isin(ids, ids[test]))
        train = np.flatnonzero((bins != fold) & (bins != (fold+1) % 6) & ~np.isin(ids, ids[np.r_[test,val]]))
        for a,b in [(train,val), (train,test), (val,test)]:
            assert not set(ids[a]) & set(ids[b])
            assert not set(bins[a]) & set(bins[b])
        assert len(set(bins[train])) == 4
        for role, ix in [('train',train), ('validation',val), ('test',test)]:
            assert set(t.iloc[ix].class_id) == {0,1,2}
            rows.extend(dict(fold=fold, role=role, instance_id=t.iloc[i].instance_id,
                             mmsi=ids[i], pass_group=t.iloc[i].pass_group) for i in ix)
            summaries.append(dict(fold=fold, role=role, n=len(ix), mmsi=t.iloc[ix].mmsi.nunique(),
                passes=t.iloc[ix].pass_group.nunique(), ports=t.iloc[ix].port.nunique(),
                **{f'class_{c}':int(t.iloc[ix].class_id.eq(c).sum()) for c in range(3)}))
        folds.append((train,val,test))
    pd.DataFrame(rows).to_csv(HERE/'splits.csv', index=False)
    pd.DataFrame(summaries).to_csv(HERE/'split_summary.csv', index=False)
    return folds, bins


def decompose(h, fit, view, head, z):
    # Keep the direct, original-precision A scores for every arm. The float64
    # map is only for checking the algebra and constructing channel groups.
    base = head.predict(z)*3
    direct64 = head.predict(view.transform(h.astype(np.float64)))*3
    w = ((head.coef_.astype(np.float64)/view.post.scale_[None,:]) @ view.pca.components_.astype(np.float64))/view.scale.scale_[None,:]*3
    center = h[fit].mean(0, dtype=np.float64)
    q = np.zeros((len(h),32,3), np.float64)
    raw = np.zeros_like(q)
    origin = np.zeros((32,3), np.float64)
    for j, cols in enumerate(np.array_split(np.arange(h.shape[1]),16)):
        q[:,16+j] = (h[:,cols]-center[cols]) @ w[:,cols].T
        raw[:,16+j] = h[:,cols].astype(np.float64) @ w[:,cols].T
        origin[16+j] = w[:,cols] @ center[cols]
    bias = head.predict(view.transform(center[None]))[0]*3
    intercept = bias-origin.sum(0)
    checks = dict(centered_error=float(abs(q.sum(1)+bias-direct64).max()),
                  raw_error=float(abs(raw.sum(1)+intercept-direct64).max()),
                  origin_error=float(abs(raw-q-origin).max()),
                  precision_score_error=float(abs(base-direct64).max()),
                  precision_top1_changes=int((base.argmax(1)!=direct64.argmax(1)).sum()))
    assert max(checks[k] for k in ['centered_error','raw_error','origin_error']) < 1e-7
    assert not q[:,:16].any() and not raw[:,:16].any()
    return base, q, raw, dict(W=w, h0=center, origin=origin, bias=bias, raw_intercept=intercept), checks


def own_environment(t, f, train, alpha):
    # Same projection and constant-layout construction as parent controls,
    # refitted to this new outer source split. Ridge does not affect project().
    stem = Baseline(f['target'][train], t.class_id.to_numpy()[train], t.mmsi.to_numpy()[train], alpha)
    regions = []
    for role in ['fg','bg']:
        z = unit(stem.project(f['target_'+role]))
        z[~f['target_'+role+'_valid']] = 0
        regions.append(z)
    stats = summary_geometry(f['geometry'],f['mask'])
    stats = StandardScaler().fit(stats[train]).transform(stats).astype(np.float32)
    source = train[f['mask'][train].any(1)]
    constant = stats[source].mean(0)
    env = np.c_[np.concatenate(regions,1)/np.sqrt(2), np.broadcast_to(constant,(len(t),15))].astype(np.float32)
    assert np.isfinite(env).all() and not np.ptp(env[:,64:],axis=0).any()
    return env


def prepare(fold, indices, bins, t, f, h):
    train,val,test = indices
    y,ids = t.class_id.to_numpy(), t.mmsi.to_numpy()
    v = View(h,train,'layer3')
    z = v.transform(h)
    candidates = []
    heads = {}
    for alpha in [1.,10.,100.]:
        head = fit_head(z,y,ids,train,alpha)
        vr = risk(y[val],head.predict(z[val])*3)
        heads[alpha] = head
        candidates.append(dict(fold=fold, kind='layer3', alpha=alpha, val_nll=vr))
    selected = min(candidates,key=lambda row:row['val_nll'])
    alpha = selected['alpha']
    base,q,qraw,details,check = decompose(h,train,v,heads[alpha],z)
    np.savez_compressed(HERE/'origins'/f'fold{fold}_outer.npz',fit=train,**details)
    checks = [dict(fold=fold, fit_role='outer', fit_n=len(train), held_n=len(t), **check)]
    oo,oq,oraw = np.zeros_like(base),np.zeros_like(q),np.zeros_like(qraw)
    coverage = np.zeros(len(t),int)
    for b in sorted(set(bins[train])):
        held = train[bins[train]==b]
        fit = train[(bins[train]!=b) & ~np.isin(ids[train],ids[held])]
        assert not set(ids[fit]) & set(ids[held])
        assert set(y[fit]) == {0,1,2}
        inner_view = View(h,fit,'layer3')
        zi = inner_view.transform(h)
        inner_head = fit_head(zi,y,ids,fit,alpha)
        ss,qq,rr,details,check = decompose(h,fit,inner_view,inner_head,zi)
        oo[held],oq[held],oraw[held] = ss[held],qq[held],rr[held]
        coverage[held] += 1
        np.savez_compressed(HERE/'origins'/f'fold{fold}_inner{b}.npz',fit=fit,held=held,**details)
        checks.append(dict(fold=fold, fit_role=f'inner{b}', fit_n=len(fit),held_n=len(held),**check))
    assert (coverage[train]==1).all() and coverage.sum()==len(train)
    env = own_environment(t,f,train,alpha)
    d = dict(base=base,oof=oo,q=q,qraw=qraw,oq=oq,oqraw=oraw,env=env,
             train=train,validation=val,test=test,alpha=alpha,instance_ids=t.instance_id.to_numpy(str))
    np.savez_compressed(HERE/f'fold{fold}_inputs.npz',**d)
    return d,candidates,selected,checks


def prediction_rows(t, test, score, base, fold, arm, seed):
    return [dict(fold=fold,arm=arm,seed=seed,instance_id=t.iloc[i].instance_id,
                 class_id=int(t.iloc[i].class_id),prediction=int(score[i].argmax()),base_prediction=int(base[i].argmax()),
                 **{f's{c}':float(score[i,c]) for c in range(3)},**{f'b{c}':float(base[i,c]) for c in range(3)}) for i in test]


def main():
    torch.set_num_threads(2)
    start = time.perf_counter()
    for name in ['models','origins']:
        (HERE/name).mkdir(exist_ok=True)
    t = pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    with np.load(EXP/'readout/features.npz') as cache:
        f = {k:cache[k] for k in ['target','target_fg','target_bg','target_fg_valid','target_bg_valid','geometry','mask','instance_ids']}
    with np.load(STRONG/'deep_features.npz') as deep:
        h = deep['layer3']
        np.testing.assert_array_equal(deep['instance_ids'],t.instance_id)
    np.testing.assert_array_equal(f['instance_ids'],t.instance_id)
    assert h.shape == (5695,2048) and h.dtype == np.float32
    folds,bins = partition(t)
    y,ids = t.class_id.to_numpy(),t.mmsi.to_numpy()
    records,rows,candidates,selections,checks,curves,replays,times = [],[],[],[],[],[],[],[]
    for fold,indices in enumerate(folds):
        tick = time.perf_counter()
        d,cc,ss,vv = prepare(fold,indices,bins,t,f,h)
        candidates.extend(cc);selections.append(ss);checks.extend(vv)
        times.append(dict(fold=fold,arm='reader_preparation',seed=-1,seconds=time.perf_counter()-tick))
        train,val,test = indices
        rows.extend(prediction_rows(t,test,d['base'],d['base'],fold,'A',-1))
        records.append(dict(fold=fold,arm='A',seed=-1,epoch=0,val_nll=ss['val_nll'],parameters=0,
                            rescued=0,harmed=0,**metrics(y[test],d['base'][test])))
        env = torch.tensor(d['env'],device='cuda')
        oo = torch.tensor(d['oof'],dtype=torch.float32,device='cuda')
        yy = torch.tensor(y,device='cuda',dtype=torch.long)
        valid = torch.ones(len(t),device='cuda')
        weight = torch.tensor(weights(y[train],ids[train]),dtype=torch.float32,device='cuda')
        for seed in SEEDS:
            for arm in ARMS:
                tick = time.perf_counter()
                raw = arm == 'uncentered_own_full'
                q = torch.tensor(d['qraw' if raw else 'q'],dtype=torch.float32,device='cuda')
                oq = torch.tensor(d['oqraw' if raw else 'oq'],dtype=torch.float32,device='cuda')
                torch.manual_seed(seed)
                model = (WithBias() if arm=='centered_plus_bias' else Anchored()).cuda()
                opt = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                br,be = risk(y[val],d['base'][val]),0
                with torch.no_grad():
                    assert not model(env,q,valid).any()
                for epoch in range(1,201):
                    opt.zero_grad()
                    score = oo[train]+model(env[train],oq[train],valid[train])
                    loss = (torch.nn.functional.cross_entropy(score,yy[train],reduction='none')*weight).mean()
                    loss.backward()
                    opt.step()
                    if epoch % 5 == 0:
                        with torch.no_grad():
                            ss = d['base'][val]+model(env[val],q[val],valid[val]).cpu().numpy()
                        rr = risk(y[val],ss)
                        curves.append(dict(fold=fold,arm=arm,seed=seed,epoch=epoch,loss=float(loss.detach()),val_nll=rr))
                        if rr < br-1e-6:
                            br,be = rr,epoch
                            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(best)
                model.eval()
                with torch.no_grad():
                    delta = model(env,q,valid).cpu().numpy()
                assert np.isfinite(delta).all()
                if be == 0:
                    assert not delta.any()
                score = d['base']+delta
                path = HERE/'models'/f'fold{fold}_{arm}_{seed}.pt'
                torch.save(best,path)
                replay = (WithBias() if arm=='centered_plus_bias' else Anchored()).cuda().eval()
                replay.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
                with torch.no_grad():
                    back = d['base'][test]+replay(env[test],q[test],valid[test]).cpu().numpy()
                error = float(abs(back-score[test]).max())
                assert error < 1e-5 and np.array_equal(back.argmax(1),score[test].argmax(1))
                replays.append(dict(fold=fold,arm=arm,seed=seed,error=error,top1_changes=0))
                bp,pp = d['base'][test].argmax(1),score[test].argmax(1)
                records.append(dict(fold=fold,arm=arm,seed=seed,epoch=be,val_nll=br,parameters=sum(p.numel() for p in model.parameters()),
                    rescued=int(((bp!=y[test])&(pp==y[test])).sum()),harmed=int(((bp==y[test])&(pp!=y[test])).sum()),**metrics(y[test],score[test])))
                rows.extend(prediction_rows(t,test,score,d['base'],fold,arm,seed))
                times.append(dict(fold=fold,arm=arm,seed=seed,seconds=time.perf_counter()-tick))
                print(fold,arm,seed,'epoch',be,'net',records[-1]['rescued']-records[-1]['harmed'],flush=True)
        for name,table in [('metrics',records),('predictions',rows),('baseline_candidates',candidates),('selections',selections),
                           ('decomposition_verification',checks),('curves',curves),('replay',replays),('timing',times)]:
            pd.DataFrame(table).to_csv(HERE/(name+'.csv'),index=False)
    p = pd.DataFrame(rows)
    assert len(replays)==54 and len(checks)==30 and not p.duplicated(['arm','seed','instance_id']).any()
    assert p.groupby(['arm','seed']).size().eq(len(t)).all()
    pooled = []
    for (arm,seed),part in p.groupby(['arm','seed']):
        net = part.prediction.eq(part.class_id).astype(int)-part.base_prediction.eq(part.class_id).astype(int)
        pooled.append(dict(arm=arm,seed=seed,n=len(part),rescued=int(net.eq(1).sum()),harmed=int(net.eq(-1).sum()),
                           net=int(net.sum()),**metrics(part.class_id.to_numpy(),part[['s0','s1','s2']].to_numpy())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    summary = dict(status='complete',seconds=time.perf_counter()-start,heads=54,baseline_folds=6,reader_fits=30,
        targets=len(t),mmsi=t.mmsi.nunique(),passes=t.pass_group.nunique(),ports=t.port.nunique(),
        partition_seed=PARTITION_SEED,seeds=SEEDS,steps=200,reader='fixed layer3; source-validation ridge alpha only',
        env_scope='outer-source stem projection and constant layout; not whole-pipeline OOF',
        no_independent_data=True,max_replay_error=max(r['error'] for r in replays))
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
