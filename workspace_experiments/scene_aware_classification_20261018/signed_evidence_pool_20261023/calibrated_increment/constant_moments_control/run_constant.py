"""Replace both signed descriptors and carrier with fold-source OOF means."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys
sys.dont_write_bytecode=True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
import run_increment as source
ref=source.ref


def main():
    assert (HERE/'PROTOCOL_ZH.md').exists()
    started=time.perf_counter()
    torch.set_num_threads(2)
    (HERE/'models').mkdir(exist_ok=True)
    table=pd.read_csv(ref.parent.EXP/'readout/targets.csv',dtype={'mmsi':str})
    y=table.class_id.to_numpy()
    yy=torch.tensor(y,dtype=torch.long)
    selections,curves,costs,checks,constants=[],[],[],[],[]
    for partition in source.CONFIGS:
        for fold in range(6):
            for seed in source.SEEDS:
                tick=time.perf_counter()
                d,_,oof,calibrator_epoch=source.load_fold(partition,fold,seed)
                tr,va=d['train'],d['validation']
                mean=oof[tr].mean(axis=0)
                assert mean.shape==(3,2)
                fixed=np.broadcast_to(mean,(len(table),3,2))
                raw=ref.raw_input('signed_only',d['oof'],fixed)
                rawv=ref.raw_input('signed_only',d['base'],fixed)
                mu,scale=raw[tr].mean((0,1)),raw[tr].std((0,1)).clip(1e-8)
                x,xv=[torch.tensor((r-mu)/scale,dtype=torch.float32) for r in [raw,rawv]]
                moments=torch.tensor(fixed.copy(),dtype=torch.float32)
                b,bv=[torch.tensor(d[k],dtype=torch.float32) for k in ['oof','base']]
                descriptor_ptp=float(np.ptp(raw[...,1:],axis=0).max())
                inference_ptp=float(np.ptp(rawv[...,1:],axis=0).max())
                carrier_ptp=float(np.ptp(fixed,axis=0).max())
                assert descriptor_ptp==inference_ptp==carrier_ptp==0
                torch.manual_seed(seed)
                model=ref.make_model('signed_only')
                assert sum(p.numel() for p in model.parameters())==113
                with torch.no_grad():
                    zero=float(model(xv[va],moments[va]).abs().max())
                    assert zero==0
                best_value=ref.exact_j(y[va],bv[va].numpy().argmax(1))
                best_epoch,best=0,ref.frozen_state(model)
                meta=dict(partition=partition,fold=fold,seed=seed,arm='constant_signed')
                curves.append(dict(**meta,epoch=0,train_loss=np.nan,**ref.assessment(y[va],bv[va].numpy().astype(float))))
                opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                for epoch in range(1,201):
                    opt.zero_grad()
                    score=b[tr]+model(x[tr],moments[tr])
                    loss=.5*ref.parent.balanced_loss(score,yy[tr])+.5*torch.nn.functional.cross_entropy(score,yy[tr])
                    loss.backward()
                    opt.step()
                    if epoch%5:
                        continue
                    with torch.no_grad():
                        score=(bv[va]+model(xv[va],moments[va])).numpy().astype(float)
                    value=ref.exact_j(y[va],score.argmax(1))
                    curves.append(dict(**meta,epoch=epoch,train_loss=float(loss.detach()),**ref.assessment(y[va],score)))
                    if value>best_value:
                        best_value,best_epoch,best=value,epoch,ref.frozen_state(model)
                path=HERE/'models'/f'{partition}_fold{fold}_constant_signed_seed{seed}.pt'
                ck=dict(state=best,mu=mu,scale=scale,mean_moments=mean,epoch=best_epoch,**meta)
                torch.save(ck,path)
                selections.append(dict(**meta,epoch=best_epoch,validation_J=float(best_value),checkpoint=str(path),
                    calibrator_epoch=calibrator_epoch,parameters=113))
                clone=ref.make_model('signed_only').eval()
                clone.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state'])
                with torch.inference_mode():
                    replay=(bv[va]+clone(xv[va],moments[va])).numpy().astype(float)
                assert ref.exact_j(y[va],replay.argmax(1))==best_value
                checks.append(dict(**meta,descriptor_sample_ptp=descriptor_ptp,inference_descriptor_sample_ptp=inference_ptp,
                    carrier_sample_ptp=carrier_ptp,zero_head_error=zero,validation_J_replay_error=abs(float(ref.exact_j(y[va],replay.argmax(1)))-float(best_value))))
                for c in range(3):
                    constants.append(dict(**meta,class_direction=c,source_n=len(tr),sigma=float(mean[c,0]),kappa=float(mean[c,1])))
                costs.append(dict(**meta,steps=200,seconds=time.perf_counter()-tick))
                print(f'{partition} fold{fold} seed{seed} constant_signed epoch={best_epoch}',flush=True)
    pd.DataFrame(selections).to_csv(HERE/'selections.csv',index=False)
    pd.DataFrame(curves).to_csv(HERE/'selection_curves.csv',index=False)
    pd.DataFrame(costs).to_csv(HERE/'trajectory_costs.csv',index=False)
    pd.DataFrame(constants).to_csv(HERE/'source_constants.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'constant_checks.csv',index=False)
    (HERE/'selection_frozen.json').write_text(json.dumps(dict(heads=len(selections),test_evaluated=False),indent=2),encoding='utf8')

    eval_started=time.perf_counter()
    original=pd.read_csv(PARENT/'predictions.csv',dtype={'mmsi':str})
    original=original[original.arm.eq('signed_only')]
    predictions,validation,baseline_checks=[],[],[]
    for row in selections:
        partition,fold,seed=row['partition'],row['fold'],row['seed']
        d,_,_,_=source.load_fold(partition,fold,seed)
        ck=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        fixed=np.broadcast_to(ck['mean_moments'],(len(table),3,2))
        raw=ref.raw_input('signed_only',d['base'],fixed)
        model=ref.make_model('signed_only').eval()
        model.load_state_dict(ck['state'])
        for role in ['validation','test']:
            ix=d[role]
            with torch.inference_mode():
                delta=model(torch.tensor((raw[ix]-ck['mu'])/ck['scale'],dtype=torch.float32),
                    torch.tensor(fixed[ix].copy(),dtype=torch.float32)).numpy().astype(float)
            base=d['base'][ix]
            score=base+delta
            g=table.iloc[ix][['instance_id','class_id','mmsi','port','pass_group']].copy()
            for key in ['partition','fold','seed','arm','epoch']:
                g[key]=row[key]
            g['prediction'],g['base_prediction']=score.argmax(1),base.argmax(1)
            g['net']=(g.prediction==g.class_id).astype(int)-(g.base_prediction==g.class_id).astype(int)
            g['outcome']=np.select([g.net.eq(1),g.net.eq(-1),g.prediction.eq(g.class_id)],['rescued','harmed','CC'],default='WW')
            for c in range(3):
                g[f's{c}'],g[f'b{c}'],g[f'delta{c}']=score[:,c],base[:,c],delta[:,c]
            (predictions if role=='test' else validation).append(g)
            if role=='test':
                expected=original[original.partition.eq(partition)&original.fold.eq(fold)&original.seed.eq(seed)].set_index('instance_id').loc[g.instance_id]
                error=float(abs(expected[['b0','b1','b2']].to_numpy()-base).max())
                assert error<1e-12 and np.array_equal(expected.base_prediction,g.base_prediction)
                baseline_checks.append(dict(partition=partition,fold=fold,seed=seed,B_replay_error=error))
    p=pd.concat(predictions,ignore_index=True)
    p.to_csv(HERE/'predictions.csv',index=False)
    pd.concat(validation,ignore_index=True).to_csv(HERE/'validation_predictions.csv',index=False)
    pooled=[]
    for (partition,seed),g in p.groupby(['partition','seed']):
        assert len(g)==len(table) and g.instance_id.is_unique
        before=ref.assessment(g.class_id.to_numpy(),g[['b0','b1','b2']].to_numpy())
        after=ref.assessment(g.class_id.to_numpy(),g[['s0','s1','s2']].to_numpy())
        pooled.append(dict(partition=partition,arm='constant_signed',seed=seed,net=int(g.net.sum()),
            rescued=int(g.net.eq(1).sum()),harmed=int(g.net.eq(-1).sum()),**after,
            **{f'delta_{k}':after[k]-before[k] for k in after}))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    p.groupby(['partition','seed','class_id']).agg(n=('net','size'),net=('net','sum')).to_csv(HERE/'class_effects.csv')
    p.groupby(['partition','seed','pass_group']).agg(n=('net','size'),net=('net','sum')).to_csv(HERE/'pass_effects.csv')
    pd.DataFrame(baseline_checks).to_csv(HERE/'baseline_replays.csv',index=False)
    receipt=dict(heads=len(selections),steps=200*len(selections),train_seconds=sum(r['seconds'] for r in costs),
        evaluation_seconds=time.perf_counter()-eval_started,total_seconds=time.perf_counter()-started,
        frozen_calibrator=True,calibrator_OOF=False,test_selection=False,constant_descriptors_and_carrier=True)
    (HERE/'summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
