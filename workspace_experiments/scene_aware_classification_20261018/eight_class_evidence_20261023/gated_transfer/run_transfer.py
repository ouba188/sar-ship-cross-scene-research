"""Three fixed heads on frozen eight-class OOF and inference caches."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp, softmax

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
BASELINE=PARENT/'baseline'
sys.path.insert(0,str(BASELINE))
from run_baseline import weights, isolated, indices

ARMS=['original_centered','score_scale_decoupled','additive']
SEEDS=[17,29,43]
CLASS_COUNT=8


class Head(torch.nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        self.net=torch.nn.Sequential(torch.nn.Linear(64,32),torch.nn.ReLU(),
            torch.nn.Linear(32,8 if arm=='additive' else 16))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def control(self,env):
        z=self.net(env)
        return z if self.arm=='additive' else .5*torch.tanh(z)

    def forward(self,env,q,base):
        a=self.control(env)
        if self.arm=='additive':
            return base+a
        if self.arm=='original_centered':
            return base+(a[:,:,None]*q).sum(1)
        m=a.mean(1,keepdim=True)
        return (1+m)*(base+((a-m)[:,:,None]*q).sum(1))


def explicit_score(arm,a,q,base):
    a=a.astype(np.float64)
    if arm=='additive':
        return base+a
    if arm=='original_centered':
        return base+(a[:,:,None]*q).sum(1)
    m=a.mean(1,keepdims=True)
    return (1+m)*(base+((a-m)[:,:,None]*q).sum(1))


def per_row(y,s):
    p=softmax(s,axis=1)
    nll=logsumexp(s,axis=1)-s[np.arange(len(y)),y]
    brier=np.sum((p-np.eye(8)[y])**2,axis=1)
    return (s.argmax(1)==y).astype(float),nll,brier


def metrics(y,s):
    correct,nll,brier=per_row(y,s)
    classes=np.unique(y)
    return dict(n=len(y),classes=len(classes),OA=float(correct.mean()*100),
        BA=float(np.mean([correct[y==c].mean() for c in classes])*100),
        NLL=float(nll.mean()),balanced_NLL=float(np.mean([nll[y==c].mean() for c in classes])),
        Brier=float(brier.mean()),balanced_Brier=float(np.mean([brier[y==c].mean() for c in classes])))


def load_frame():
    f=pd.read_csv(PARENT/'cohort.csv',dtype={'mmsi':str})
    f['pass_group']=f['product'].map(lambda s:'_'.join([s.split('_')[0],s.split('_')[6],s.split('_')[7]]))
    return f


def load_fold(f,fold):
    d=dict(np.load(BASELINE/f'fold{fold}_inputs.npz'))
    np.testing.assert_array_equal(d['instance_ids'],f.instance_id)
    assert len(f)==15251 and int(d['class_count'])==8
    assert d['q'].shape==d['oq'].shape==(len(f),16,8)
    assert d['whole64'].shape==d['oof_whole64'].shape==(len(f),64)
    for k in ('base','q','whole64','oof','oq','oof_whole64'):
        assert np.isfinite(d[k]).all(),k
    for key,ix in zip(('train','validation','test'),indices(f,fold)):
        np.testing.assert_array_equal(d[key],ix)
    train,val,test=[d[k] for k in ('train','validation','test')]
    for a,b in ((train,val),(train,test),(val,test)):
        isolated(f,a,b)
    np.testing.assert_array_equal(np.flatnonzero(d['oof_valid']),train)
    np.testing.assert_allclose(d['q'].sum(1)+d['b'],d['base'],atol=1e-7)
    np.testing.assert_allclose(d['oq'].sum(1)+d['ob'],d['oof'],atol=1e-7)
    for inner in range(4):
        with np.load(BASELINE/'models'/f'fold{fold}_inner{inner}.npz') as model:
            held=model['held'];fit=model['fit']
            isolated(f,fit,held)
            np.testing.assert_array_equal(held,np.flatnonzero(d['inner_bin']==inner))
    np.testing.assert_allclose(np.linalg.norm(d['oof_whole64'][train],axis=1),np.sqrt(32),atol=3e-6)
    return d


def frame_predictions(f,ix,score,base,fold,arm,seed,valid):
    p=f.iloc[ix][['instance_id','obs_id','class_id','class_name','mmsi','port','product','pass_group','time']].copy()
    y=p.class_id.to_numpy()
    p['row_index']=ix
    p['fold'],p['arm'],p['seed']=fold,arm,seed
    p['prediction'],p['base_prediction']=score.argmax(1),base.argmax(1)
    p['net']=p.prediction.eq(y).astype(int)-p.base_prediction.eq(y).astype(int)
    p['outcome']=np.select([p.net.eq(1),p.net.eq(-1),p.prediction.eq(y)],['rescued','harmed','CC'],default='WW')
    p['vh_valid']=valid[ix,1]
    for c in range(8):
        p[f's{c}'],p[f'b{c}']=score[:,c],base[:,c]
    return p


def main():
    started=time.perf_counter()
    torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    f=load_frame();y=f.class_id.to_numpy()
    (HERE/'models').mkdir(exist_ok=True)
    (HERE/'readouts').mkdir(exist_ok=True)
    records,curves,checks,sources,allpred,basepred=[],[],[],[],[],[]
    original_predictions=pd.read_csv(BASELINE/'predictions.csv')
    for fold in range(3):
        d=load_fold(f,fold)
        train,val,test=[d[k] for k in ('train','validation','test')]
        old=original_predictions[original_predictions.outer_fold.eq(fold)&original_predictions.role.eq('test')].set_index('instance_id').loc[f.iloc[test].instance_id]
        np.testing.assert_allclose(old[[f's{c}' for c in range(8)]],d['base'][test],atol=1e-12)
        np.testing.assert_array_equal(old.prediction,d['base'][test].argmax(1))
        basepred.append(frame_predictions(f,test,d['base'][test],d['base'][test],fold,'A',-1,d['valid']))
        # Materialize train tensors only from the three matching inner OOF arrays.
        te=torch.tensor(d['oof_whole64'][train],dtype=torch.float32,device=device)
        tq=torch.tensor(d['oq'][train],dtype=torch.float32,device=device)
        tb=torch.tensor(d['oof'][train],dtype=torch.float32,device=device)
        yy=torch.tensor(y[train],dtype=torch.long,device=device)
        w=torch.tensor(weights(f,train),dtype=torch.float32,device=device)
        env=torch.tensor(d['whole64'],dtype=torch.float32,device=device)
        sources.append(dict(fold=fold,train=len(train),validation=len(val),test=len(test),
            train_arrays='oof,oq,oof_whole64',inference_arrays='base,q,whole64',groups=16,
            train_outer_oof_env_max_difference=float(abs(d['whole64'][train]-d['oof_whole64'][train]).max()),
            missing_vh=int((~d['valid'][:,1]).sum()),
            missing_vh_max_abs_q=float(abs(d['q'][~d['valid'][:,1],8:]).max())))
        for seed in SEEDS:
            for arm in ARMS:
                head_start=time.perf_counter()
                torch.manual_seed(seed)
                model=Head(arm).to(device)
                opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                parameters=sum(p.numel() for p in model.parameters())
                assert parameters==(2344 if arm=='additive' else 2608)
                with torch.no_grad():
                    assert torch.equal(model(te,tq,tb),tb)
                    init=model.control(env).cpu().numpy()
                np.testing.assert_array_equal(explicit_score(arm,init,d['q'],d['base']),d['base'])
                best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_risk=metrics(y[val],d['base'][val])['balanced_NLL'];best_step=0
                init_loss=float((torch.nn.functional.cross_entropy(tb,yy,reduction='none')*w).mean())
                curves.append(dict(fold=fold,arm=arm,seed=seed,step=0,loss=init_loss,val_balanced_NLL=best_risk))
                for step in range(1,201):
                    opt.zero_grad()
                    score=model(te,tq,tb)
                    loss=(torch.nn.functional.cross_entropy(score,yy,reduction='none')*w).mean()
                    assert torch.isfinite(loss)
                    loss.backward();opt.step()
                    if step%5==0:
                        with torch.no_grad():
                            a=model.control(env[val]).cpu().numpy()
                        vs=explicit_score(arm,a,d['q'][val],d['base'][val])
                        vr=metrics(y[val],vs)['balanced_NLL']
                        curves.append(dict(fold=fold,arm=arm,seed=seed,step=step,loss=float(loss.detach()),val_balanced_NLL=vr))
                        if vr<best_risk-1e-6:
                            best_risk,best_step=vr,step
                            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(best);model.eval()
                with torch.inference_mode():
                    controls=model.control(env).cpu().numpy()
                    scores=explicit_score(arm,controls,d['q'],d['base'])
                    gpu=model(env[test],torch.tensor(d['q'][test],dtype=torch.float32,device=device),
                        torch.tensor(d['base'][test],dtype=torch.float32,device=device)).cpu().numpy()
                assert np.isfinite(scores).all()
                numeric_error=float(np.max(abs(scores[test]-gpu)))
                assert numeric_error<1e-5,numeric_error
                common_checks=0
                if arm!='additive':
                    a=controls.astype(np.float64)
                    assert abs(a).max()<=.5
                    m=a.mean(1,keepdims=True)
                    common=1+m
                    assert common.min()>0
                    np.testing.assert_array_equal((common*d['base']).argmax(1),d['base'].argmax(1))
                    if arm=='score_scale_decoupled':
                        contrast=d['base']+((a-m)[:,:,None]*d['q']).sum(1)
                        np.testing.assert_array_equal(scores.argmax(1),contrast.argmax(1))
                    common_checks=len(f)
                if best_step==0:
                    np.testing.assert_array_equal(scores,d['base'])
                name=f'fold{fold}_{arm}_{seed}'
                path=HERE/'models'/f'{name}.pt'
                torch.save(best,path)
                replay=Head(arm).to(device).eval()
                replay.load_state_dict(torch.load(path,map_location=device,weights_only=True))
                with torch.inference_mode():
                    replay_a=replay.control(env).cpu().numpy()
                back=explicit_score(arm,replay_a,d['q'],d['base'])
                replay_error=float(abs(back-scores).max())
                assert replay_error==0
                np.testing.assert_array_equal(back.argmax(1),scores.argmax(1))
                np.savez_compressed(HERE/'readouts'/f'{name}.npz',instance_ids=d['instance_ids'],control=controls,
                    score=scores,selected_step=best_step,train=train,validation=val,test=test)
                p=frame_predictions(f,test,scores[test],d['base'][test],fold,arm,seed,d['valid'])
                allpred.append(p)
                records.append(dict(fold=fold,arm=arm,seed=seed,parameters=parameters,step=best_step,
                    val_balanced_NLL=best_risk,rescued=int(p.net.eq(1).sum()),harmed=int(p.net.eq(-1).sum()),net=int(p.net.sum()),
                    seconds=time.perf_counter()-head_start,**metrics(y[test],scores[test])))
                checks.append(dict(fold=fold,arm=arm,seed=seed,zero_init_exact=True,step0_fallback_exact=True,
                    replay_error=replay_error,float32_formula_error=numeric_error,
                    float32_top1_differences=int((scores[test].argmax(1)!=gpu.argmax(1)).sum()),
                    positive_common_argmax_rows=common_checks,all_16_groups=True))
                print(f'fold={fold} arm={arm} seed={seed} step={best_step} net={p.net.sum()} OA={records[-1]["OA"]:.4f} BA={records[-1]["BA"]:.4f}',flush=True)
        for name,items in [('metrics',records),('curves',curves),('verification',checks),('sources',sources)]:
            pd.DataFrame(items).to_csv(HERE/(name+'.csv'),index=False)
        pd.concat(allpred,ignore_index=True).to_csv(HERE/'predictions.csv',index=False)
        pd.concat(basepred,ignore_index=True).to_csv(HERE/'baseline_predictions.csv',index=False)
    p=pd.concat(allpred,ignore_index=True)
    assert len(records)==27 and len(p)==4744*9
    assert not p.duplicated(['arm','seed','instance_id']).any()
    pooled=[]
    for (arm,seed),part in p.groupby(['arm','seed']):
        assert len(part)==4744
        pooled.append(dict(arm=arm,seed=seed,rescued=int(part.net.eq(1).sum()),harmed=int(part.net.eq(-1).sum()),
            net=int(part.net.sum()),**metrics(part.class_id.to_numpy(),part[[f's{c}' for c in range(8)]].to_numpy())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    bp=pd.concat(basepred,ignore_index=True)
    bm=metrics(bp.class_id.to_numpy(),bp[[f's{c}' for c in range(8)]].to_numpy())
    pd.DataFrame([bm]).to_csv(HERE/'baseline_pooled.csv',index=False)
    summary=dict(status='training_complete',heads=27,steps=200,seeds=SEEDS,arms=ARMS,device=device,
        seconds=time.perf_counter()-started,train_val_seconds=sum(r['seconds'] for r in records),
        rows=len(f),test_rows=4744,baseline=bm,zero_step_heads=sum(r['step']==0 for r in records),
        max_replay_error=max(c['replay_error'] for c in checks),
        max_float32_formula_error=max(c['float32_formula_error'] for c in checks),
        float32_top1_differences=sum(c['float32_top1_differences'] for c in checks),
        baseline_refits=0,new_feature_extractions=0,
        scope='Fixed eight-class development transfer, not old A+D or full-model victory. No test tuning.')
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
