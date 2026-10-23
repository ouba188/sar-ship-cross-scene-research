"""Reference-relative edge metric update; source-only frozen-protocol experiment."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location('relative_frozen_learned_metric', ROOT/'learned_edge_metric/run.py')
learned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learned)
edge = learned.edge
trace_query = learned.trace_query
ARMS, SEEDS, CHECKPOINTS, BATCH = learned.ARMS, learned.SEEDS, learned.CHECKPOINTS, learned.BATCH


class Metric(learned.Metric):
    def __init__(self, device='cuda', dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        with torch.no_grad():
            self.a.fill_(.25)


class Reader(learned.Reader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference = torch.full((len(self.b), 8, 4), float('nan'), device=self.b.device, dtype=self.b.dtype)
        self.reference_ready = np.zeros(len(self.b), dtype=bool)

    def cache_reference(self, ids):
        missing = np.asarray(ids)[~self.reference_ready[ids]]
        identity = torch.eye(32, device=self.b.device, dtype=self.b.dtype)
        with torch.inference_mode():
            for start in range(0, len(missing), BATCH):
                part = missing[start:start+BATCH]
                self.reference[part] = super().scores(part, identity)[0]
        self.reference_ready[missing] = True

    def scores(self, ids, W):
        z, raw, valid = super().scores(ids, W)
        return z-self.reference[ids], raw, valid


def smoke(data):
    table, x, mass, bank, idx, tr, va, te, divisors = data
    utility = idx[tr]
    missing = utility[mass[utility, 1].sum(-1) == 0]
    ids = np.unique(np.r_[utility[:4], missing[:1]])
    checks = dict(utility_indices=ids.tolist(), includes_missing_VH=bool(len(missing)),
                  no_evaluation=True, tests='relative zero init, log ratio, initial data gradient only')
    initial_error = 0.
    for arm in ARMS:
        reader = Reader(x,mass,bank,arm,divisors[arm])
        reader.cache_reference(ids)
        model = Metric()
        with torch.inference_mode():
            z, _, _ = reader.scores(ids,model.matrix())
            initial_error = max(initial_error,float(learned.delta(z,model.a).abs().max()))
    assert initial_error == 0., initial_error
    checks['initial_delta_max'] = initial_error

    # Data-gradient test at the actual initialization, independent of regularization.
    reader = Reader(x,mass,bank,'mixed',divisors['mixed'],device='cpu',dtype=torch.float64)
    reader.cache_reference(ids)
    model = Metric(device='cpu',dtype=torch.float64)
    anchor = np.load(ROOT/'reference_dependence/frozen_marginal/predictions_17.npz')['anchor'][ids]
    labels = table.class_id.to_numpy(dtype=int)
    counts = np.bincount(labels[utility],minlength=8)
    weight = torch.tensor(.5/644+.5/(8*counts[labels[ids]]),dtype=torch.float64)
    target = torch.tensor(labels[ids])
    base = torch.tensor(anchor,dtype=torch.float64)
    def objective():
        z,_,_ = reader.scores(ids,model.matrix())
        return (F.cross_entropy(base+learned.delta(z,model.a),target,reduction='none')*weight).sum()
    objective().backward()
    position = np.unravel_index(int(model.V.grad.abs().argmax()),tuple(model.V.shape))
    analytic = float(model.V.grad[position])
    assert abs(analytic)>1e-12
    eps=1e-5
    with torch.no_grad():
        model.V[position]=eps
        plus=float(objective())
        model.V[position]=-eps
        minus=float(objective())
        model.V[position]=0
    numeric=(plus-minus)/(2*eps)
    assert abs(numeric-analytic)<1e-8
    checks['initial_V_gradient']=dict(position=[int(v) for v in position],analytic=analytic,
                                      finite_difference=numeric,error=abs(numeric-analytic),
                                      U_gradient_norm=float(model.U.grad.norm()),a_data_gradient_norm=float(model.a.grad.norm()))

    # Non-identity W makes the log-partition-ratio check nontrivial.
    with torch.no_grad():
        model.V.copy_(torch.arange(128,dtype=torch.float64).reshape(32,4)*.001)
    W=model.matrix().detach().numpy()
    maximum=0.
    for arm in ARMS:
        for i in ids:
            origin=trace_query(int(i),arm,np.eye(32),x,mass,bank)
            changed=trace_query(int(i),arm,W,x,mass,bank)
            for g,name in enumerate(edge.GROUPS):
                rho=origin[f'{name}_responsibility']
                logrho=np.full_like(rho,-np.inf)
                np.log(rho,out=logrho,where=rho>0)
                shift=(changed[f'{name}_similarity']-origin[f'{name}_similarity'])/edge.TAU
                valid=rho.sum((-1,-2))>0
                value=logsumexp(logrho+shift,axis=(-1,-2))
                value=np.where(valid,value,0.)
                count=max(int(origin['legal_edges'][0,g].sum()),1)
                ratio=value.sum(-1)/count
                difference=changed['raw_scores'][:,g]-origin['raw_scores'][:,g]
                maximum=max(maximum,float(np.max(np.abs(ratio-difference))))
    assert maximum<1e-10,maximum
    checks['raw_logratio_max_error']=maximum
    (HERE/'smoke.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks),flush=True)


def fit(reader,anchor,table,idx,tr,va,seed,arm):
    model=Metric()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    checkpoint_path=HERE/f'checkpoints_{arm}_{seed}.npz'
    saved=np.load(checkpoint_path) if checkpoint_path.exists() else None
    y=table.class_id.to_numpy(dtype=int)
    ids=idx[tr]
    counts=np.bincount(y[ids],minlength=8)
    global_weights=.5/len(ids)+.5/(8*counts[y[ids]])
    b=torch.tensor(anchor[ids],device='cuda')
    labels=torch.tensor(y[ids],device='cuda')
    lw=torch.tensor(global_weights,device='cuda',dtype=torch.float32)
    best=edge.dep.exact_j(y[idx[va]],anchor[idx[va]].argmax(-1))
    chosen=model.state()
    selected=0
    history,states=[],{}
    for step in (CHECKPOINTS if saved is not None else range(201)):
        if saved is not None:
            model.restore({key:saved[f'{key}_{step}'] for key in ('U','V','a','W')})
        if step>0 and saved is None:
            optimizer.zero_grad(set_to_none=True)
            for start in range(0,len(ids),BATCH):
                end=min(start+BATCH,len(ids))
                z,_,_=reader.scores(ids[start:end],model.matrix())
                loss=(F.cross_entropy(b[start:end]+learned.delta(z,model.a),labels[start:end],reduction='none')*lw[start:end]).sum()
                loss.backward()
            model.regularizer().backward()
            optimizer.step()
        if step not in CHECKPOINTS:
            continue
        utility,_,_=learned.infer(reader,model,ids,anchor[ids])
        selection,_,_=learned.infer(reader,model,idx[va],anchor[idx[va]])
        um=edge.dep.metrics(y[ids],utility)
        sm=edge.dep.metrics(y[idx[va]],selection)
        j=edge.dep.exact_j(y[idx[va]],selection.argmax(-1))
        state=model.state()
        states.update({f'{key}_{step}':value for key,value in state.items()})
        reg=float(model.regularizer().detach())
        history.append(dict(seed=seed,arm=arm,step=step,train_CE=.5*(um['NLL']+um['balanced_NLL']),
                            regularizer=reg,train_objective=.5*(um['NLL']+um['balanced_NLL'])+reg,
                            **{f'utility_{k}':v for k,v in um.items()},**{f'selection_{k}':v for k,v in sm.items()},
                            W_correction_norm=float(np.linalg.norm(state['W']-np.eye(32))),a_norm=float(np.linalg.norm(state['a'])),
                            exact_j=str(j)))
        if j>best:
            best,selected,chosen=j,step,state
        print('checkpoint',seed,arm,step,'utility_CE',history[-1]['train_CE'],'selection_J',sm['J'],flush=True)
    if saved is None:
        np.savez_compressed(checkpoint_path,**states)
    model.restore(chosen)
    return model,chosen,selected,history


def main(data):
    started=time.perf_counter()
    table,x,mass,bank,idx,tr,va,te,divisors=data
    y=table.class_id.to_numpy(dtype=int)
    allids=np.arange(len(table))
    pd.read_csv(ROOT/'learned_edge_metric/scales.csv').to_csv(HERE/'scales.csv',index=False)
    readers={arm:Reader(x,mass,bank,arm,divisors[arm]) for arm in ARMS}
    for reader in readers.values():
        reader.cache_reference(idx[np.r_[tr,va]])
    results,receipts,curves,classes,pairs,margins=[],[],[],[],[],[]
    weights={}
    for seed in SEEDS:
        old=np.load(ROOT/f'reference_dependence/frozen_marginal/predictions_{seed}.npz')
        anchor=old['anchor'].astype(np.float32)
        predictions=dict(anchor=anchor,full=old['full'])
        features={}
        for arm in ARMS:
            tick=time.perf_counter()
            reader=readers[arm]
            resumed=(HERE/f'checkpoints_{arm}_{seed}.npz').exists()
            model,state,step,history=fit(reader,anchor,table,idx,tr,va,seed,arm)
            # No evaluation prediction until the source checkpoint is frozen.
            reader.cache_reference(allids)
            score,z,raw=learned.infer(reader,model,allids,anchor)
            reference=reader.reference.cpu().numpy()
            predictions[arm]=score
            features[arm]=z
            features[f'{arm}_rawZ_W']=z+reference
            features[f'{arm}_referencefeatures']=reference
            weights.update({f'{arm}_{seed}_{key}':value for key,value in state.items()})
            receipts.append(dict(seed=seed,arm=arm,selected_step=step,optimizer_steps=200,parameters=260,
                                 seconds=time.perf_counter()-tick,resumed_completed_checkpoints=resumed,
                                 W_correction_norm=float(np.linalg.norm(state['W']-np.eye(32)))))
            curves.extend(history)
            absolute=np.load(ROOT/f'learned_edge_metric/predictions_{seed}.npz')[arm]
            for role,select in [('utility_train',tr),('selection',va),('evaluation',te)]:
                select=idx[select]
                m=edge.dep.metrics(y[select],score[select])
                results.append(dict(seed=seed,arm=arm,role=role,joint_CE=.5*(m['NLL']+m['balanced_NLL']),**m))
                for reference_name,prior_score in [('anchor',anchor),('absolute_same_arm',absolute)]:
                    new=score[select].argmax(-1)==y[select]
                    prior=prior_score[select].argmax(-1)==y[select]
                    pairs.append(dict(seed=seed,arm=arm,role=role,reference=reference_name,
                                      rescued=int((new&~prior).sum()),harmed=int((~new&prior).sum()),net=int(new.sum()-prior.sum())))
            rival_scores=anchor[idx].copy()
            rival_scores[np.arange(len(idx)),y[idx]]=-np.inf
            rival=rival_scores.argmax(-1)
            for j,i in enumerate(idx):
                for g,name in enumerate(edge.GROUPS):
                    gap=z[i,y[i],g]-z[i,rival[j],g]
                    reference_gap=reference[i,y[i],g]-reference[i,rival[j],g]
                    margins.append(dict(seed=seed,arm=arm,table_index=int(i),obs_id=int(table.iloc[i].obs_id),
                                        role=table.iloc[i].role,true_class=int(y[i]),rival=int(rival[j]),group=name,
                                        true_score=float(z[i,y[i],g]),rival_score=float(z[i,rival[j],g]),gap=float(gap),
                                        reference_gap=float(reference_gap),learned_gap=float(reference_gap+gap),
                                        coefficient=float(state['a'][g]),margin_contribution=float(gap*state['a'][g])))
            for c in range(8):
                select=idx[te][y[idx[te]]==c]
                m=edge.dep.metrics(y[select],score[select])
                new=score[select].argmax(-1)==y[select]
                prior=anchor[select].argmax(-1)==y[select]
                classes.append(dict(seed=seed,arm=arm,class_id=c,
                                    rescued=int((new&~prior).sum()),harmed=int((~new&prior).sum()),
                                    net=int(new.sum()-prior.sum()),**m))
            pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
            pd.DataFrame(curves).to_csv(HERE/'training_curve.csv',index=False)
            np.savez_compressed(HERE/'weights.npz',**weights)
            print('SELECTED',seed,arm,step,edge.dep.metrics(y[idx[te]],score[idx[te]]),flush=True)
        for arm in ('anchor','full'):
            for role,select in [('utility_train',tr),('selection',va),('evaluation',te)]:
                m=edge.dep.metrics(y[idx[select]],predictions[arm][idx[select]])
                results.append(dict(seed=seed,arm=arm,role=role,joint_CE=.5*(m['NLL']+m['balanced_NLL']),**m))
        for arm in ARMS:
            current=predictions[arm][idx[te]].argmax(-1)==y[idx[te]]
            for reference_name in ('body','additive','mixed'):
                prior=predictions[reference_name][idx[te]].argmax(-1)==y[idx[te]]
                pairs.append(dict(seed=seed,arm=arm,role='evaluation',reference=reference_name,
                                  rescued=int((current&~prior).sum()),harmed=int((~current&prior).sum()),net=int(current.sum()-prior.sum())))
        np.savez_compressed(HERE/f'predictions_{seed}.npz',idx=idx,bank=bank,**predictions)
        np.savez_compressed(HERE/f'features_{seed}.npz',idx=idx,bank=bank,**features)
    np.savez_compressed(HERE/'referencefeatures.npz',**{arm:reader.reference.cpu().numpy() for arm,reader in readers.items()})
    for name,rows in [('metrics',results),('class_results',classes),('paired_results',pairs),('query_group_margins',margins)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    summary=pd.DataFrame(results).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','balanced_NLL','joint_CE','Brier']].mean()
    summary.to_csv(HERE/'summary.csv')
    receipt=dict(status='complete',fits=len(receipts),optimizer_steps_per_fit=200,batch_size=BATCH,
                 global_utility_count=644,regularizer_once_per_optimizer_step=True,a_init=.25,
                 a_regularizer_center=0,initial_V_zero=True,initial_U_seed=17,initial_U_std=.01,
                 comparison_caveat='Absolute arm used a_init=0; relative uses .25 with unchanged zero-centered regularizer.',
                 feature_definition='arm = Z_W-Z_I; arm_rawZ_W = standardized Z_W; arm_referencefeatures = standardized Z_I',
                 reference_computation='Same numerical Reader at I with frozen source RMS; no reference gradients or refitting.',
                 completed_heads_resumed=sum(r['resumed_completed_checkpoints'] for r in receipts),
                 resume_note='First process exit1 at duplicate n field in class report; completed body17 checkpoints replayed without optimizer updates.',
                 evaluation_after_selection_only=True,checkpoint_steps=list(CHECKPOINTS),
                 no_external=True,no_git=True,no_images=True,seconds=time.perf_counter()-started)
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(summary.to_string(),flush=True)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    data=learned.source_data()
    if '--smoke' in sys.argv:
        smoke(data)
    else:
        assert (HERE/'smoke.json').exists(),'Run --smoke first.'
        main(data)
