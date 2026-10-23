"""Per-source-sample exact jointCE gradients, fixed heads, CPU only."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.func import functional_call, grad, vmap
from torch.nn.functional import cross_entropy

HERE = Path(__file__).resolve().parent
OSS = HERE.parent
sys.path.insert(0, str(OSS))
import train_heads as source

FOCUS = 'pub2_opensarship2_005904'


def gradients(model, arm, x, base, y, coefficient):
    params = {k: p.detach().clone().requires_grad_(True) for k, p in model.named_parameters()}
    def apply(p, data):
        args = (data,) if arm == 'additive' else (data, None)
        return functional_call(model, p, args)
    def one(p, xx, bb, yy, cc):
        score = bb[None] + apply(p, xx[None])
        ce = cross_entropy(score, yy[None])
        return ce*cc, (ce, score[0])
    calculate = vmap(grad(one, has_aux=True), in_dims=(None, 0, 0, 0, 0))
    vectors, losses, scores = [], [], []
    for first in range(0, len(x), 128):
        stop = first+128
        gs, (ce, score) = calculate(params, x[first:stop], base[first:stop],
                                   y[first:stop], coefficient[first:stop])
        vectors.append(torch.cat([gs[k].flatten(1) for k in params], 1).detach().double())
        losses.append(ce.detach())
        scores.append(score.detach())
    individual = torch.cat(vectors)
    ce, score = torch.cat(losses), torch.cat(scores)
    full_score = base + apply(params, x)
    loss = source.parent.joint_loss(full_score, y)
    full_grad = torch.autograd.grad(loss, tuple(params.values()))
    aggregate = torch.cat([p.flatten() for p in full_grad]).detach().double()
    summed = individual.sum(0)
    error = (summed-aggregate).norm()/aggregate.norm().clamp_min(1e-30)
    loss_error = abs(float((ce*coefficient).sum())-float(loss.detach()))
    assert float(error) < 5e-5
    assert loss_error < 2e-6
    assert torch.equal(score.argmax(1), full_score.detach().argmax(1))
    energy = individual.square().sum(1)
    assert torch.isfinite(energy).all() and energy.sum()>0
    projection = (individual @ aggregate)/aggregate.square().sum().clamp_min(1e-30)
    branch_energy = {}
    start = 0
    for k,p in params.items():
        group = k.split('.')[0] if arm == 'additive' else 'local'
        end = start+p.numel()
        branch_energy[group] = branch_energy.get(group, torch.zeros(len(x),dtype=torch.float64)) + individual[:,start:end].square().sum(1)
        start = end
    return ce.numpy(), score.numpy(), energy.numpy(), projection.numpy(), {
        k:v.numpy() for k,v in branch_energy.items()}, dict(
        jointCE=float(loss.detach()), weighted_loss_replay_error=loss_error,
        summed_gradient_relative_l2_error=float(error),
        summed_gradient_max_abs_error=float((summed-aggregate).abs().max()),
        sum_individual_gradient_sq=float(energy.sum()),
        full_batch_gradient_sq=float(aggregate.square().sum()),
        score_replay_max_abs_error=float((score-full_score.detach()).abs().max()),
        top1_replay_changes=0)


def main():
    started=time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table=pd.read_csv(OSS/'cohort.csv',dtype={'mmsi':str})
    with np.load(OSS/'moments.npz') as z:
        cache={k:z[k] for k in z.files}
    valid=cache['valid'].astype(bool)
    rows, summaries, classes, focus=[] ,[],[],[]
    for fold in range(6):
        d, _, _, om, od=source.load_fold(fold,table,cache)
        tr=d['train']
        f=table.iloc[tr].reset_index(drop=True)
        y=torch.tensor(f.class_id.to_numpy(),dtype=torch.long)
        counts=torch.bincount(y,minlength=8)
        assert (counts>0).all()
        coefficient=.5/len(tr)+.5/(8*counts[y].float())
        np.testing.assert_allclose(float(coefficient.sum()),1.,atol=2e-7)
        for seed in [17,29,43]:
            with np.load(OSS/'calibrated'/f'fold{fold}_seed{seed}.npz') as z:
                np.testing.assert_array_equal(z['sample_ids'],table.sample_id)
                b=z['train_base'][tr]
            base=torch.tensor(b,dtype=torch.float32)
            for arm in ['local','additive']:
                ck, selected=source.restore(OSS/'heads'/f'fold{fold}_{arm}_seed{seed}.pt')
                x,carrier,mu,sd=source.descriptor(arm,b,om[tr],od[tr],valid[tr],ck['mu'],ck['scale'])
                _,_,fresh_mu,fresh_sd=source.descriptor(arm,b,om[tr],od[tr],valid[tr])
                np.testing.assert_array_equal(mu,fresh_mu)
                np.testing.assert_array_equal(sd,fresh_sd)
                torch.manual_seed(seed)
                initial=source.make_model(arm).eval()
                with torch.no_grad():
                    assert not source.delta(initial,arm,x,carrier).any()
                if ck['epoch']==0:
                    for k,v in initial.state_dict().items():
                        assert torch.equal(v,ck['state'][k])
                for state,model in [('initial',initial),('selected',selected)]:
                    ce,score,energy,projection,branch,check=gradients(model,arm,x,base,y,coefficient)
                    share=energy/energy.sum()
                    order=np.argsort(-energy,kind='stable')
                    ranks=np.empty(len(f),int);ranks[order]=np.arange(1,len(f)+1)
                    g=f[['sample_id','class_id','class_name','mmsi','pass_group']].copy()
                    for key,value in dict(fold=fold,seed=seed,arm=arm,state=state,
                        selected_epoch=ck['epoch'],state_epoch=0 if state=='initial' else ck['epoch']).items():g[key]=value
                    g['class_source_n']=counts[y].numpy()
                    g['jointCE_coefficient']=coefficient.numpy()
                    g['CE']=ce;g['weighted_loss']=ce*coefficient.numpy()
                    g['B_prediction']=base.argmax(1).numpy()
                    g['B_correct']=g.B_prediction.eq(g.class_id)
                    g['prediction']=score.argmax(1)
                    g['true_probability']=torch.softmax(torch.tensor(score),1)[torch.arange(len(y)),y].numpy()
                    g['gradient_sq']=energy;g['gradient_sq_share']=share;g['gradient_rank']=ranks
                    g['signed_aggregate_projection']=projection
                    for key,value in branch.items():g[key+'_gradient_sq']=value
                    rows.append(g)
                    largest=g.iloc[order[0]]
                    item=dict(fold=fold,seed=seed,arm=arm,state=state,
                        selected_epoch=ck['epoch'],source_n=len(f),
                        max_id=largest.sample_id,max_class=int(largest.class_id),
                        max_share=float(largest.gradient_sq_share),max_CE=float(largest.CE),
                        max_B_prediction=int(largest.B_prediction),
                        top5_share=float(share[order[:5]].sum()),
                        top1pct_share=float(share[order[:int(np.ceil(.01*len(f)))]].sum()),**check)
                    summaries.append(item)
                    hit=g[g.sample_id.eq(FOCUS)]
                    if len(hit):
                        focus.append(dict(hit.iloc[0],in_source=True))
                    else:
                        focus.append(dict(fold=fold,seed=seed,arm=arm,state=state,
                            sample_id=FOCUS,in_source=False,
                            absent_reason='not in source train; no gradient evaluated'))
                    for c,sub in g.groupby('class_id'):
                        classes.append(dict(fold=fold,seed=seed,arm=arm,state=state,class_id=c,
                            class_name=sub.class_name.iloc[0],n=len(sub),CE=float(sub.CE.mean()),
                            weighted_loss=float(sub.weighted_loss.sum()),gradient_sq_share=float(sub.gradient_sq_share.sum()),
                            B_OA=float(sub.B_correct.mean()),max_id=sub.loc[sub.gradient_sq.idxmax(),'sample_id']))
                print(f'fold={fold} seed={seed} arm={arm} epoch={ck["epoch"]} done',flush=True)
    pd.concat(rows,ignore_index=True).to_csv(HERE/'gradient_samples.csv',index=False)
    pd.DataFrame(summaries).to_csv(HERE/'gradient_summary.csv',index=False)
    pd.DataFrame(classes).to_csv(HERE/'gradient_classes.csv',index=False)
    pd.DataFrame(focus).to_csv(HERE/'gradient_005904.csv',index=False)
    info=dict(heads=36,states=72,device='cpu',trained_steps=0,
        seconds=time.perf_counter()-started,
        max_gradient_relative_l2_error=max(s['summed_gradient_relative_l2_error'] for s in summaries),
        max_loss_error=max(s['weighted_loss_replay_error'] for s in summaries))
    (HERE/'gradient_receipt.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
