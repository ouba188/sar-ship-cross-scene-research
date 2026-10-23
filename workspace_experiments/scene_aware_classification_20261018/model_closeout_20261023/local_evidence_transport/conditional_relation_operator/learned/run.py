"""Shared learned conditional reconstruction, relative to frozen identity evidence."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import itertools
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('fixed_conditional_parent', HERE.parent/'run.py')
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
edge, dep = fixed.edge, fixed.dep
ARMS = fixed.ARMS
SEEDS = (17, 29, 43)
CHECKPOINTS = (0, 20, 50, 100, 200)
BATCH = 128
STATE_KEYS = ('Ub', 'Vb', 'Un', 'Vn', 'a', 'Mb', 'Mn')


class Metric(torch.nn.Module):
    def __init__(self, device='cuda', dtype=torch.float32):
        super().__init__()
        gen = torch.Generator().manual_seed(17)
        for label in ('b', 'n'):
            u = torch.randn(32, 4, generator=gen)*.01
            setattr(self, 'U'+label, torch.nn.Parameter(u.to(device=device, dtype=dtype)))
            setattr(self, 'V'+label, torch.nn.Parameter(torch.zeros(32, 4, device=device, dtype=dtype)))
        self.a = torch.nn.Parameter(torch.full((4,), .25, device=device, dtype=dtype))
        self.register_buffer('I', torch.eye(32, device=device, dtype=dtype))

    def matrices(self):
        return self.I+self.Ub@self.Vb.T, self.I+self.Un@self.Vn.T

    def regularizer(self):
        mb, mn = self.matrices()
        return .01*self.a.square().sum()+.01*((mb-self.I).square().sum()+(mn-self.I).square().sum())/32

    def state(self):
        mb, mn = self.matrices()
        return {k:v.detach().cpu().numpy().copy() for k,v in
                [(k,getattr(self,k)) for k in STATE_KEYS[:5]]+[('Mb',mb),('Mn',mn)]}

    def restore(self, state):
        with torch.no_grad():
            for k in STATE_KEYS[:5]:
                getattr(self,k).copy_(torch.as_tensor(state[k], device=self.a.device, dtype=self.a.dtype))


def cross(q, s):
    """g,b,i,d and g,r,j,d -> g,b,r,i,j via four dense matrix products."""
    g, batch, e, d = q.shape
    r = s.shape[1]
    value = torch.bmm(q.reshape(g,batch*e,d), s.reshape(g,r*e,d).transpose(1,2))
    return value.reshape(g,batch,e,r,e).permute(0,1,3,2,4)


def reference_operator(s, counts):
    gram = s@s.transpose(-1,-2)
    lam = (s.square().sum((-2,-1))/counts.clamp_min(1)).clamp_min(1e-8)
    system = gram+lam[...,None,None]*torch.eye(s.shape[-2],device=s.device,dtype=s.dtype)
    # solve_ex avoids a CUDA-to-CPU error check inside every gradient batch.
    solution = torch.linalg.solve_ex(system,s,check_errors=False)[0]
    return solution, gram, lam


def improvement(A, q, s, gram, mq):
    linear = (A*cross(q,s)).sum((-2,-1))
    energy = ((A@gram[:,None])*A).sum((-2,-1))
    return (2*linear-energy)/mq[:,:,None].clamp_min(1)


class Reader:
    def __init__(self, x, mass, bank, arm, device='cuda', dtype=torch.float32, legal_override=None):
        shuffle = int(arm.split('shuffle')[1]) if 'shuffle' in arm else None
        b,n,legal,mapping = edge.edges(x,mass,shuffle)
        if legal_override is not None:
            legal = legal_override
        b,n,counts = fixed.center(b,n,legal)
        self.arm, self.bank, self.mapping = arm, bank, mapping
        self.numpy_legal = legal
        self.b = torch.as_tensor(b.transpose(1,0,2,3).copy(),device=device,dtype=dtype)
        self.n = torch.as_tensor(n.transpose(1,0,2,3).copy(),device=device,dtype=dtype)
        self.counts = torch.as_tensor(counts.T.copy(),device=device)
        self.sb = self.b[:,bank.ravel()]
        self.sn = self.n[:,bank.ravel()]
        self.ms = self.counts[:,bank.ravel()]
        self.rv = self.ms>=2
        self.reference = torch.full((len(x),8,4),float('nan'),device=device,dtype=dtype)
        self.ready = np.zeros(len(x),bool)
        self.divisor = torch.ones(4,device=device,dtype=dtype)

    def scores(self, ids, mb, mn, trace=False):
        qb,qn = self.b[:,ids]@mb,self.n[:,ids]@mn
        sb,sn = self.sb@mb,self.sn@mn
        mq = self.counts[:,ids]
        solveb,gb,lam = reference_operator(sb,self.ms)
        A = cross(qb,solveb)
        gn = sn@sn.transpose(-1,-2)
        T = improvement(A,qn,sn,gn,mq)
        V = sn.square().sum((-2,-1))[:,None]/(self.ms-1).clamp_min(1)[:,None]
        V = V*(A.square().sum((-2,-1))-A.sum(-1).square().sum(-1)/self.ms.clamp_min(1)[:,None])
        V = V/mq[:,:,None].clamp_min(1)
        if self.arm == 'independent':
            solven,_,_ = reference_operator(sn,self.ms)
            An = cross(qn,solven)
            body = improvement(A,qb,sb,gb,mq)
            neighbor = improvement(An,qn,sn,gn,mq)
            value = .5*(body+neighbor)
        else:
            value = T if self.arm == 'transfer' else T+V
        ref_valid = self.rv[:,None] & (mq>=2)[:,:,None]
        value = torch.where(ref_valid,value,0.)
        grouped = value.reshape(4,len(ids),8,8)
        refcount = self.rv.reshape(4,8,8).sum(-1)
        raw = (grouped.sum(-1)/refcount[:,None].clamp_min(1)).permute(1,2,0)
        valid = ((mq>=2)[:,:,None] & (refcount>0)[:,None]).permute(1,2,0)
        center = (raw*valid).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)
        z = torch.where(valid,(raw-center)/self.divisor,0.)
        if trace:
            details = dict(A=A,T=T,V=V,reference_scores=grouped,
                reference_valid=ref_valid.reshape(4,len(ids),8,8),lambda_body=lam,
                qb=qb,qn=qn,sb=sb,sn=sn,mq=mq,ms=self.ms)
            if self.arm == 'independent':
                details.update(body=body,neighbor=neighbor,An=An)
            return z,raw,valid,details
        return z,raw,valid

    def initialize(self, utility):
        I = torch.eye(32,device=self.b.device,dtype=self.b.dtype)
        sums = torch.zeros(4,device=self.b.device,dtype=self.b.dtype)
        counts = torch.zeros_like(sums)
        with torch.no_grad():
            for start in range(0,len(utility),BATCH):
                z,_,valid = self.scores(utility[start:start+BATCH],I,I)
                sums += z.square().sum((0,1))
                counts += valid.sum((0,1))
            rms = (sums/counts.clamp_min(1)).sqrt()
            self.divisor = torch.where(rms>1e-8,rms,1.)
        return rms.cpu().numpy(),counts.cpu().numpy()

    def cache_reference(self, ids):
        missing = np.asarray(ids)[~self.ready[ids]]
        I = torch.eye(32,device=self.b.device,dtype=self.b.dtype)
        with torch.no_grad():
            for start in range(0,len(missing),BATCH):
                part = missing[start:start+BATCH]
                self.reference[part] = self.scores(part,I,I)[0]
        self.ready[missing] = True


def delta(z, a):
    d = (z*a).sum(-1)
    return d-d.mean(-1,keepdim=True)


def source_data():
    table,x,mass,bank = edge.support.load()
    idx = np.flatnonzero(table.role.isin(['utility_train','selection','evaluation']))
    role = table.iloc[idx].role.to_numpy()
    tr,va,te = [np.flatnonzero(role==r) for r in ('utility_train','selection','evaluation')]
    assert [len(tr),len(va),len(te)] == [644,612,485]
    return table,x,mass,bank,idx,tr,va,te


def smoke(data):
    table,x,mass,bank,idx,tr,va,te = data
    utility = idx[tr]
    b,n,legal,_ = edge.edges(x,mass)
    partial = utility[(legal[utility].sum(-1)==3).any(-1)]
    missing = utility[~legal[utility,2].any(-1)]
    ids = np.unique(np.r_[utility[:3],partial[:1],missing[:1]])
    checks = dict(utility_indices=ids.tolist(),actual_partial_indices=partial.tolist(),
                  missing_VH_indices=missing[:1].tolist(),no_evaluation=True)
    I = torch.eye(32,dtype=torch.float64)
    error = 0.
    for override in (None, 'partial'):
        l = legal.copy()
        if override:
            l[ids[0],0,0] = False
            l[bank[0,0],0,1] = False
        expected,valid,_ = fixed.raw_scores(b,n,l,bank,ids)
        for arm in ('transfer','excess'):
            reader = Reader(x,mass,bank,arm,device='cpu',dtype=torch.float64,legal_override=l)
            _,raw,mask = reader.scores(ids,I,I)
            error = max(error,float(np.max(np.abs(raw.detach().numpy()-expected[arm]))))
            np.testing.assert_array_equal(mask.numpy(),valid)
    assert error<1e-10,error
    checks['parent_identity_actual_partial_max_error'] = error
    checks['scalar_null'] = fixed.checks()
    full = np.load(ROOT/'reference_dependence/frozen_marginal/predictions_17.npz')['full']
    labels = table.class_id.to_numpy(int)
    classcounts = np.bincount(labels[utility],minlength=8)
    target = torch.tensor(labels[ids])
    base = torch.tensor(full[ids],dtype=torch.float64)
    weights = torch.tensor(.5/644+.5/(8*classcounts[labels[ids]]),dtype=torch.float64)
    # Independent NumPy comparisons and gradients detect inactive transform controls.
    grads = {}
    for arm in ARMS:
        reader = Reader(x,mass,bank,arm,device='cpu',dtype=torch.float64)
        reader.initialize(utility)
        reader.cache_reference(ids)
        model = Metric(device='cpu',dtype=torch.float64)
        def objective():
            z,_,_ = reader.scores(ids,*model.matrices())
            return (F.cross_entropy(base+delta(z-reader.reference[ids],model.a),target,reduction='none')*weights).sum()
        z,raw,valid = reader.scores(ids,*model.matrices())
        assert torch.count_nonzero(delta(z-reader.reference[ids],model.a)) == 0
        objective().backward()
        grad = {}
        for name in ('Vb','Vn'):
            p = getattr(model,name)
            pos = np.unravel_index(int(p.grad.abs().argmax()),p.shape)
            exact = float(p.grad[pos])
            assert abs(exact)>1e-12,(arm,name,exact)
            eps = 1e-5
            with torch.no_grad():
                p[pos] = eps
                plus = float(objective())
                p[pos] = -eps
                minus = float(objective())
                p[pos] = 0
            numeric = (plus-minus)/(2*eps)
            assert abs(numeric-exact)<1e-8,(arm,name,numeric,exact)
            grad[name] = dict(norm=float(p.grad.norm()),analytic=exact,numeric=numeric,error=abs(numeric-exact))
        grads[arm] = grad
        if arm == 'independent':
            body = fixed.raw_scores(b,b,legal,bank,ids)[0]['independent']
            neighbor = fixed.raw_scores(n,n,legal,bank,ids)[0]['independent']
            err = float(np.max(np.abs(raw.detach().numpy()-.5*(body+neighbor))))
            assert err<1e-10,err
            checks['independent_both_maps_score_error'] = err
    checks['initial_gradients'] = grads
    checks['initial_delta_max'] = 0.
    # The weighting test covers the actual imbalanced 644-label cohort and final short batch.
    yt = torch.tensor(labels[utility])
    logits = torch.tensor(full[utility],dtype=torch.float64)
    w = torch.tensor(.5/644+.5/(8*classcounts[labels[utility]]),dtype=torch.float64)
    accumulated = sum((F.cross_entropy(logits[k:k+BATCH],yt[k:k+BATCH],reduction='none')*w[k:k+BATCH]).sum()
                      for k in range(0,len(utility),BATCH))
    loss_error = abs(float(accumulated-dep.joint_loss(logits,yt)))
    assert loss_error<1e-12,loss_error
    checks['global_batch_loss_error'] = loss_error
    gpu_error = 0.
    expected,_,_ = fixed.raw_scores(b,n,legal,bank,ids)
    for arm in ('transfer','excess'):
        reader = Reader(x,mass,bank,arm)
        reader.cache_reference(ids)
        model = Metric()
        with torch.no_grad():
            z,raw,_ = reader.scores(ids,*model.matrices())
            assert torch.count_nonzero(z-reader.reference[ids]) == 0
        gpu_error = max(gpu_error,float(np.max(np.abs(raw.cpu().numpy()-expected[arm]))))
    assert gpu_error<1e-6,gpu_error
    checks['gpu_identity_parent_error'] = gpu_error
    (HERE/'smoke.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('SMOKE PASSED',json.dumps(checks),flush=True)


def infer(reader, model, ids, full):
    scores,zs,raws = [],[],[]
    with torch.no_grad():
        mb,mn = model.matrices()
        for start in range(0,len(ids),BATCH):
            part = ids[start:start+BATCH]
            z,raw,_ = reader.scores(part,mb,mn)
            relative = z-reader.reference[part]
            scores.append(full[part]+delta(relative,model.a).cpu().numpy())
            zs.append(relative.cpu().numpy())
            raws.append(raw.cpu().numpy())
    return np.concatenate(scores),np.concatenate(zs),np.concatenate(raws)


def fit(reader,full,table,idx,tr,va,seed,arm):
    model = Metric()
    opt = torch.optim.Adam(model.parameters(),lr=.01)
    ids = idx[tr]
    y = table.class_id.to_numpy(int)
    classcounts = np.bincount(y[ids],minlength=8)
    w = torch.tensor(.5/len(ids)+.5/(8*classcounts[y[ids]]),device='cuda',dtype=torch.float32)
    target = torch.tensor(y[ids],device='cuda')
    baseline = torch.tensor(full[ids],device='cuda')
    best = dep.exact_j(y[idx[va]],full[idx[va]].argmax(-1))
    chosen,selected = model.state(),0
    history,states = [],{}
    tick = time.perf_counter()
    for step in range(201):
        if step:
            opt.zero_grad(set_to_none=True)
            for start in range(0,len(ids),BATCH):
                part = ids[start:start+BATCH]
                z,_,_ = reader.scores(part,*model.matrices())
                logits = baseline[start:start+BATCH]+delta(z-reader.reference[part],model.a)
                loss = (F.cross_entropy(logits,target[start:start+BATCH],reduction='none')*w[start:start+BATCH]).sum()
                loss.backward()
            model.regularizer().backward()
            opt.step()
        if step not in CHECKPOINTS:
            continue
        train_score,train_z,_ = infer(reader,model,ids,full)
        val_score,val_z,_ = infer(reader,model,idx[va],full)
        if step == 0:
            assert np.count_nonzero(train_z)==0 and np.count_nonzero(val_z)==0
        assert np.isfinite(train_score).all() and np.isfinite(val_score).all(),(arm,seed,step)
        um,vm = dep.metrics(y[ids],train_score),dep.metrics(y[idx[va]],val_score)
        j = dep.exact_j(y[idx[va]],val_score.argmax(-1))
        state = model.state()
        states.update({f'{key}_{step}':v for key,v in state.items()})
        reg = float(model.regularizer().detach().cpu())
        history.append(dict(seed=seed,arm=arm,step=step,train_CE=.5*(um['NLL']+um['balanced_NLL']),
            regularizer=reg,train_objective=.5*(um['NLL']+um['balanced_NLL'])+reg,
            **{f'utility_{k}':v for k,v in um.items()},**{f'selection_{k}':v for k,v in vm.items()},
            Mb_change=float(np.linalg.norm(state['Mb']-np.eye(32))),
            Mn_change=float(np.linalg.norm(state['Mn']-np.eye(32))),a_norm=float(np.linalg.norm(state['a'])),
            exact_j=str(j),seconds=time.perf_counter()-tick))
        if j>best:
            best,selected,chosen = j,step,state
        print('CHECKPOINT',seed,arm,step,'utilityCE',history[-1]['train_CE'],'selectionJ',vm['J'],
              'seconds',round(time.perf_counter()-tick,1),flush=True)
    np.savez_compressed(HERE/f'checkpoints_{arm}_{seed}.npz',**states)
    model.restore(chosen)
    return model,chosen,selected,history


def trace_query(table_index,arm,Mb,Mn,x=None,mass=None,bank=None):
    """NumPy trace for a single original-table row; see METHOD_TRACE.md for axes."""
    if x is None:
        _,x,mass,bank = edge.support.load()
    reader = Reader(x,mass,bank,arm,device='cpu',dtype=torch.float64)
    with torch.no_grad():
        _,raw,valid,trace = reader.scores(np.array([table_index]),torch.tensor(Mb,dtype=torch.float64),
            torch.tensor(Mn,dtype=torch.float64),trace=True)
    out = {k:v.cpu().numpy() for k,v in trace.items()}
    out.update(raw_scores=raw[0].numpy(),valid=valid[0].numpy(),bank=bank,
               legal_edges=reader.numpy_legal[np.r_[table_index,bank.ravel()]],
               neighbor_map=reader.mapping[np.r_[table_index,bank.ravel()]])
    return out


def main(data):
    started = time.perf_counter()
    table,x,mass,bank,idx,tr,va,te = data
    y = table.class_id.to_numpy(int)
    readers,scales = {},[]
    for arm in ARMS:
        reader = Reader(x,mass,bank,arm)
        rms,count = reader.initialize(idx[tr])
        reader.cache_reference(idx[tr])
        reader.cache_reference(idx[va])
        readers[arm] = reader
        for g,group in enumerate(edge.GROUPS):
            scales.append(dict(arm=arm,group=group,feature=g,rms=rms[g],divisor=float(reader.divisor[g].cpu()),n=count[g]))
    pd.DataFrame(scales).to_csv(HERE/'scales.csv',index=False)
    results,classes,pairs,receipts,curves,coefficients = [],[],[],[],[],[]
    weights = {}
    max_replay = 0.
    for seed in SEEDS:
        old = np.load(ROOT/f'reference_dependence/frozen_marginal/predictions_{seed}.npz')
        np.testing.assert_array_equal(old['idx'],idx)
        full = old['full'].astype(np.float32)
        predictions = dict(idx=idx,bank=bank,full=full)
        features = dict(idx=idx,bank=bank)
        for arm in ARMS:
            tick = time.perf_counter()
            reader = readers[arm]
            model,state,step,history = fit(reader,full,table,idx,tr,va,seed,arm)
            # The selected state is frozen before evaluating any held-out row.
            reader.cache_reference(idx)
            score,z,raw = infer(reader,model,idx,full)
            predictions[arm] = full.copy()
            predictions[arm][idx] = score
            for key,value in [(arm,z),(arm+'_raw',raw),(arm+'_referencefeatures',reader.reference[idx].cpu().numpy())]:
                features[key] = np.full((len(table),8,4),np.nan,dtype=np.float32)
                features[key][idx] = value
            replay = np.sum(z*state['a'],axis=-1)
            replay -= replay.mean(-1,keepdims=True)
            err = float(np.max(np.abs((score-full[idx])-replay)))
            assert err<3e-5,err
            max_replay = max(max_replay,err)
            weights.update({f'{arm}_{seed}_{key}':v for key,v in state.items()})
            receipts.append(dict(seed=seed,arm=arm,base='full',selected_step=step,optimizer_steps=200,
                parameters=516,transform_parameters=512,seconds=time.perf_counter()-tick,replay_error=err,
                Mb_change=float(np.linalg.norm(state['Mb']-np.eye(32))),Mn_change=float(np.linalg.norm(state['Mn']-np.eye(32)))))
            curves.extend(history)
            coefficients.extend(dict(seed=seed,arm=arm,group=group,a=float(state['a'][g])) for g,group in enumerate(edge.GROUPS))
            for role,loc in [('utility_train',tr),('selection',va),('evaluation',te)]:
                ids = idx[loc]
                pred = predictions[arm][ids]
                results.append(dict(seed=seed,arm=arm,role=role,**dep.metrics(y[ids],pred)))
                before,after = full[ids].argmax(-1)==y[ids],pred.argmax(-1)==y[ids]
                pairs.append(dict(seed=seed,arm=arm,role=role,reference='full',rescued=int((~before&after).sum()),
                    harmed=int((before&~after).sum()),net=int(after.sum()-before.sum())))
                for c in range(8):
                    select = y[ids]==c
                    cm = dep.metrics(y[ids][select],pred[select])
                    classes.append(dict(seed=seed,arm=arm,role=role,class_id=c,
                        full_correct=int(before[select].sum()),correct=int(after[select].sum()),
                        rescued=int((~before[select]&after[select]).sum()),harmed=int((before[select]&~after[select]).sum()),
                        net=int(after[select].sum()-before[select].sum()),**cm))
            for name,values in [('metrics',results),('class_results',classes),('paired_results',pairs),
                                ('training_receipts',receipts),('training_curve',curves),('coefficients',coefficients)]:
                pd.DataFrame(values).to_csv(HERE/(name+'.csv'),index=False)
            np.savez_compressed(HERE/'weights.npz',**weights)
            np.savez_compressed(HERE/f'predictions_{seed}.npz',**predictions)
            np.savez_compressed(HERE/f'features_{seed}.npz',**features)
            print('SELECTED',seed,arm,step,dep.metrics(y[idx[te]],score[te]),flush=True)
        for role,loc in [('utility_train',tr),('selection',va),('evaluation',te)]:
            results.append(dict(seed=seed,arm='full',role=role,**dep.metrics(y[idx[loc]],full[idx[loc]])))
    pd.DataFrame(results).to_csv(HERE/'metrics.csv',index=False)
    summary = pd.DataFrame(results).query("role=='evaluation'").groupby('arm')[['OA','BA','J','NLL','balanced_NLL','Brier']].mean()
    summary.to_csv(HERE/'summary.csv')
    assert len(receipts)==15
    receipt = dict(status='complete',fits=len(receipts),optimizer_steps_per_fit=200,batch_size=BATCH,
        checkpoints=list(CHECKPOINTS),configs=list(SEEDS),arms=list(ARMS),base='clean full only',
        transform_parameters=512,coefficients=4,initial_U_seed=17,initial_U_std=.01,initial_V_zero=True,a_init=.25,
        regularizer_once_per_update=True,global_balanced_CE=True,utility=644,selection=612,evaluation=485,
        evaluation_after_selection=True,per_arm_RMS='utility identity frozen',max_feature_replay_error=max_replay,
        no_external=True,no_git=True,no_raw_analysis=True,seconds=time.perf_counter()-started)
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(summary.to_string(),flush=True)
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = source_data()
    if '--smoke' in sys.argv:
        smoke(data)
    else:
        assert (HERE/'smoke.json').exists(),'Run --smoke before training.'
        main(data)
