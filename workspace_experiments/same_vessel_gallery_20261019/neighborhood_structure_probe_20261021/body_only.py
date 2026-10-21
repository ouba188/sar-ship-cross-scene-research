"""Mechanism control: frozen pre-convolution body-only baseline plus R branch."""
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import run as exp

OUT=exp.ROOT/'body_only'


def fit():
    OUT.mkdir(exist_ok=True)
    start=time.perf_counter();torch.set_num_threads(2)
    f,splits=exp.old.relation.load_data();y=f.class_id.to_numpy();yy=torch.tensor(y,device='cuda')
    prior=exp.old.control.ROOT
    body_features=np.load(prior/'features.npz')['body_global']
    legacy=np.load(prior/'scores_body_global.npz')
    original_cache=np.load(exp.old.ROOT/'features.npz')
    cache=np.load(exp.ROOT/'features.npz')
    metrics=[];selections=[];replays=[]
    for fold,(tr,val,test) in enumerate(splits):
        fitted=np.load(prior/f'head_f{fold}_body_global.npz')
        x=(body_features-fitted['mean'])/fitted['std']/np.sqrt(int(fitted['dimension']))
        base_np=(x@fitted['coef'].T+fitted['intercept']).astype(np.float32)
        lookup=dict(zip(legacy['rows'],range(len(legacy['rows']))))
        err=float(np.abs(base_np[test]-legacy['scores'][[lookup[i] for i in test]]).max());assert err<2e-5
        replays.append(dict(fold=fold,max_abs=err))
        base=torch.tensor(base_np,device='cuda')
        weights=torch.tensor(exp.old.prior.weights(f,tr),device='cuda',dtype=torch.float32)
        for seed in range(3):
            metrics.append(dict(fold=fold,seed=seed,arm='baseline',mode='reference',partition='test',step=0,
                                **exp.old.control.metrics(y[test],base_np[test])))
            for arm in ('original',*exp.ARMS):
                if arm=='original':
                    material,stats=exp.old.standardize(original_cache,tr)
                    xx=material['separated']
                else:
                    key=exp.feature_key(arm,seed)
                    xx,stats=exp.matrices(cache,key,tr)
                    fixed_x,_=exp.matrices(cache,key,tr,reference_stats)
                    with torch.no_grad():fixed_scores=base_np[test]+reference_head(torch.tensor(fixed_x[test],device='cuda')).cpu().numpy()
                    np.savez_compressed(OUT/f'scores_f{fold}_{arm}_fixed_s{seed}.npz',rows=test,scores=fixed_scores,
                        base=base_np[test],reference=reference_scores)
                    metrics.append(dict(fold=fold,seed=seed,arm=arm,mode='fixed',partition='test',step=reference_step,
                                        **exp.old.control.metrics(y[test],fixed_scores)))
                np.savez_compressed(OUT/f'normalization_f{fold}_{arm}_s{seed}.npz',**stats)
                xx=torch.tensor(xx,device='cuda');torch.manual_seed(20261021+seed)
                head=exp.old.Residual().cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
                rng=np.random.default_rng(20261021+seed+31*fold);best=None;chosen=0;state=None
                for step in range(151):
                    if step in (0,50,100,150):
                        with torch.no_grad():score=(base[val]+head(xx[val])).cpu().numpy()
                        met=exp.old.control.metrics(y[val],score)
                        selections.append(dict(fold=fold,seed=seed,arm=arm,step=step,**met))
                        rank=(met['ba'],-met['brier'])
                        if best is None or rank>best:
                            best=rank;chosen=step;state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                        if step==0:np.testing.assert_array_equal(score,base_np[val])
                    if step==150:break
                    take=rng.integers(len(tr),size=128);ii=tr[take]
                    target=2*F.one_hot(yy[ii],8).float()-1
                    loss=((base[ii]+head(xx[ii])-target).square().mean(-1)*weights[take]).mean()
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                with torch.no_grad():
                    for part,ii in (('terminal_train',tr),('terminal_val',val)):
                        scores=(base[ii]+head(xx[ii])).cpu().numpy()
                        metrics.append(dict(fold=fold,seed=seed,arm=arm,mode='matched',partition=part,step=150,
                                            **exp.old.control.metrics(y[ii],scores)))
                head.load_state_dict(state)
                torch.save(dict(state=state,step=chosen),OUT/f'head_f{fold}_{arm}_s{seed}.pt')
                with torch.no_grad():residual=head(xx).cpu().numpy()
                for part,ii in (('train',tr),('val',val),('test',test)):
                    metrics.append(dict(fold=fold,seed=seed,arm=arm,mode='matched',partition=part,step=chosen,
                                        **exp.old.control.metrics(y[ii],base_np[ii]+residual[ii])))
                scores=base_np[test]+residual[test]
                if arm=='original':
                    reference_head=head;reference_stats=stats;reference_scores=scores;reference_step=chosen
                    np.savez_compressed(OUT/f'scores_f{fold}_reference_s{seed}.npz',rows=test,scores=scores,base=base_np[test])
                    metrics.append(dict(fold=fold,seed=seed,arm='reference',mode='reference',partition='test',step=chosen,
                                        **exp.old.control.metrics(y[test],scores)))
                np.savez_compressed(OUT/f'scores_f{fold}_{arm}_matched_s{seed}.npz',rows=test,scores=scores,
                    base=base_np[test],reference=reference_scores,source_constant=base_np[test]+residual[tr].mean(0))
                print('body fit',fold,seed,arm,chosen,metrics[-1]['oa'],metrics[-1]['ba'],flush=True)
            pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False,lineterminator='\n')
            pd.DataFrame(selections).to_csv(OUT/'validation_selection.csv',index=False,lineterminator='\n')
    pd.DataFrame(replays).to_csv(OUT/'baseline_replay.csv',index=False,lineterminator='\n')
    (OUT/'TRAINING.json').write_text(json.dumps(dict(status='complete',fits=45,folds=3,seeds=3,
        source_only=True,body_input='target mask before convolution, matched previously trained body_global head',
        seconds=time.perf_counter()-start),indent=2),encoding='utf-8')


if __name__=='__main__':fit()
