"""Four fixed readouts, 36 source trajectories, then frozen target inference."""
from common import *
import argparse
import time
from fractions import Fraction
from scipy.special import logsumexp
from torch import nn

ARMS=('global','fixed','dynamic','merged')
SEEDS=(17,29,43)
SCORE=[f's{c}' for c in range(8)]
BASE=[f'b{c}' for c in range(8)]


def assessment(y,s):
    pred=s.argmax(1); lp=s-logsumexp(s,axis=1,keepdims=True)
    loss=-lp[np.arange(len(y)),y]
    return dict(n=len(y),OA=float((pred==y).mean()*100),
        BA=float(np.mean([(pred[y==c]==c).mean() for c in np.unique(y)])*100),
        NLL=float(loss.mean()),balanced_NLL=float(np.mean([loss[y==c].mean() for c in np.unique(y)])),
        Brier=float(((np.exp(lp)-np.eye(8)[y])**2).sum(1).mean()))


def exact_j(y,pred):
    assert len(np.unique(y))==8
    return Fraction(int((pred==y).sum()),2*len(y))+sum((Fraction(int(((pred==c)&(y==c)).sum()),16*int((y==c).sum())) for c in range(8)),Fraction())


def joint_loss(s,y):
    loss=nn.functional.cross_entropy(s,y,reduction='none')
    return .5*loss.mean()+.5*torch.stack([loss[y==c].mean() for c in range(8)]).mean()


class Head(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,12),nn.GELU(),nn.Linear(12,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        out=self.net(x).squeeze(-1)
        return out-out.mean(1,keepdim=True)


def frozen(model): return {k:v.detach().clone() for k,v in model.state_dict().items()}


def load_info(fold,role,table):
    ix=splits(fold)[role]
    directory=HERE/('summaries_test' if role=='test' else 'summaries_source')
    tags=[f'inner{k}' for k in range(4)] if role=='train' else ['outer']
    summary=np.zeros((len(ix),2,7,8,3)); quality=np.zeros((len(ix),2,3,4))
    geo=np.zeros((len(ix),2,4)); flags=np.zeros((len(ix),2,7),bool)
    count=np.zeros(len(ix),int); lookup={int(i):j for j,i in enumerate(ix)}
    for tag in tags:
        with np.load(directory/f'fold{fold}_{tag}.npz') as z:
            rows=z['indices']; np.testing.assert_array_equal(z['instance_ids'],table.iloc[rows].instance_id)
            loc=np.array([lookup[int(i)] for i in rows])
            summary[loc]=z['summary']; quality[loc]=z['quality']; geo[loc]=z['geometry']; flags[loc]=z['region_valid']; count[loc]+=1
    assert (count==1).all()
    return dict(indices=ix,summary=summary,quality=quality,geometry=geo,region_valid=flags)


def common_quality(info):
    g=info['geometry']; q=info['quality']
    fields=[]
    for p in range(2):
        fields.extend([np.log1p(g[:,p,0]),np.log1p(g[:,p,1]),g[:,p,2],g[:,p,3]])
        for r in range(3):
            fields.extend([np.log1p(q[:,p,r,0]),q[:,p,r,1],np.log1p(q[:,p,r,2]),q[:,p,r,3]])
    out=np.stack(fields,axis=-1)
    assert out.shape[1]==32
    return out


def raw_inputs(base,info,arm):
    value=info['summary']
    if arm=='global': summary=value[:,:,0:1]
    elif arm=='fixed': summary=value[:,:,4:7]
    elif arm=='dynamic': summary=value[:,:,1:4]
    else:
        flags=info['region_valid'][:,:,1:4]
        summary=(value[:,:,1:4]*flags[:,:,:,None,None]).sum(2,keepdims=True)/np.maximum(flags.sum(2,keepdims=True)[:,:,:,None,None],1)
    cols=[base-base.mean(1,keepdims=True)]
    for p in range(2):
        for r in range(summary.shape[2]):
            for stat in range(3):
                v=summary[:,p,r,:,stat]
                cols.extend([v,v-v.mean(1,keepdims=True)])
    quality=common_quality(info)
    raw=np.concatenate((np.stack(cols,-1),np.broadcast_to(quality[:,None,:],(len(base),8,32))),axis=-1)
    assert raw.shape[-1]==(69 if arm in ('fixed','dynamic') else 45)
    assert np.isfinite(raw).all()
    return raw


def standardize(raw,mu=None,scale=None):
    if mu is None:
        mu=raw.mean((0,1),keepdims=True); scale=np.maximum(raw.std((0,1),keepdims=True),1e-8)
    return torch.tensor((raw-mu)/scale,dtype=torch.float32),mu,scale


def reference(stage):
    f=pd.read_csv(ATT/f'{stage}_predictions.csv',dtype={'mmsi':str},float_precision='round_trip')
    f=f[f.arm.eq('two_sided_attention')].copy()
    assert not f.duplicated(['fold','seed','role','instance_id']).any()
    return f.set_index(['fold','seed','role','instance_id'])


def baseline(ref,table,ix,fold,seed,role):
    keys=pd.MultiIndex.from_arrays([[fold]*len(ix),[seed]*len(ix),[role]*len(ix),table.iloc[ix].instance_id])
    rows=ref.loc[keys]
    for key in ['obs_id','mmsi','class_id']:
        np.testing.assert_array_equal(rows[key].to_numpy(),table.iloc[ix][key].to_numpy())
    return rows[SCORE].to_numpy(float)


def predict_frame(table,ix,b,s,fold,seed,role,arm,epoch):
    out=table.iloc[ix][['instance_id','obs_id','mmsi','port','product','pass_group','class_id','class_name']].copy()
    out['row_index']=ix
    for k,v in dict(fold=fold,seed=seed,role=role,arm=arm,epoch=epoch).items():out[k]=v
    for c in range(8):out[f'b{c}']=b[:,c];out[f's{c}']=s[:,c]
    out['prediction']=s.argmax(1);out['base_prediction']=b.argmax(1)
    out['net']=(out.prediction==out.class_id).astype(int)-(out.base_prediction==out.class_id).astype(int)
    return out


def pooled(frame):
    rows=[]
    for (role,arm,seed),g in frame.groupby(['role','arm','seed']):
        y=g.class_id.to_numpy(); m=assessment(y,g[SCORE].to_numpy()); b=assessment(y,g[BASE].to_numpy())
        rows.append(dict(role=role,arm=arm,seed=seed,**m,net=int(g.net.sum()),rescued=int(g.net.eq(1).sum()),harmed=int(g.net.eq(-1).sum()),
            **{f'delta_{k}_B':m[k]-b[k] for k in ['OA','BA','NLL','Brier']}))
    return pd.DataFrame(rows)


def summary_checks(info,fold,role):
    records=[]; x=info['summary']; valid=info['region_valid']
    for p in range(2):
        for r in range(3):
            both=valid[:,p,1+r]&valid[:,p,4+r]
            for k,stat in enumerate(['mu','h','b']):
                diff=abs(x[both,p,1+r,:,k]-x[both,p,4+r,:,k])
                records.append(dict(fold=fold,role=role,pol=POL[p],region=REGIONS[r],stat=stat,n=int(both.sum()),
                    mean_abs=float(diff.mean()),max_abs=float(diff.max()),fraction_sample_different=float((diff.max(1)>1e-8).mean())))
    return records


def source_train():
    started=time.perf_counter(); torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert not (HERE/'source_selection_frozen.json').exists()
    table,_,_=load_data(); ref=reference('source')
    (HERE/'models').mkdir(exist_ok=True)
    assert not list((HERE/'models').glob('*.pt'))
    curves=[]; selections=[]; checks=[]; costs=[]; frames=[]; summary_diff=[]
    for fold in range(3):
        load_tick=time.perf_counter(); info={role:load_info(fold,role,table) for role in ['train','validation']}
        load_seconds=time.perf_counter()-load_tick
        for role in info:summary_diff.extend(summary_checks(info[role],fold,role))
        tr=info['train']['indices'];va=info['validation']['indices']
        y=torch.tensor(table.iloc[tr].class_id.to_numpy(),dtype=torch.long);yv=table.iloc[va].class_id.to_numpy()
        for seed in SEEDS:
            b={role:baseline(ref,table,info[role]['indices'],fold,seed,role) for role in info}
            bt=torch.tensor(b['train'],dtype=torch.float32)
            previous_quality=None; previous_init=None; previous_quality_scaler=None
            for arm in ARMS:
                tick=time.perf_counter()
                raw=raw_inputs(b['train'],info['train'],arm); rv=raw_inputs(b['validation'],info['validation'],arm)
                xt,mu,scale=standardize(raw);xv,_,_=standardize(rv,mu,scale)
                current_quality=(raw[...,-32:],rv[...,-32:])
                if previous_quality is not None:
                    for a,z in zip(previous_quality,current_quality):np.testing.assert_array_equal(a,z)
                    np.testing.assert_array_equal(mu[...,-32:],previous_quality_scaler[0])
                    np.testing.assert_array_equal(scale[...,-32:],previous_quality_scaler[1])
                previous_quality=current_quality;previous_quality_scaler=(mu[...,-32:],scale[...,-32:])
                torch.manual_seed(seed); model=Head(raw.shape[-1]); init=frozen(model)
                if arm=='fixed':previous_init=init
                if arm=='dynamic':assert all(torch.equal(init[k],previous_init[k]) for k in init)
                params=sum(p.numel() for p in model.parameters());assert params==(853 if arm in ['fixed','dynamic'] else 565)
                with torch.no_grad():assert not model(xt).any() and not model(xv).any()
                best=init;epoch_best=0;best_j=exact_j(yv,b['validation'].argmax(1))
                meta=dict(fold=fold,seed=seed,arm=arm)
                curves.append(dict(**meta,epoch=0,train_loss=np.nan,**assessment(yv,b['validation'])))
                opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                optimizer_tick=time.perf_counter()
                for epoch in range(1,201):
                    opt.zero_grad(); loss=joint_loss(bt+model(xt),y);assert torch.isfinite(loss)
                    loss.backward();opt.step()
                    if epoch%5:continue
                    with torch.no_grad():s=b['validation']+model(xv).numpy().astype(float)
                    j=exact_j(yv,s.argmax(1));curves.append(dict(**meta,epoch=epoch,train_loss=float(loss.detach()),**assessment(yv,s)))
                    if j>best_j:best=frozen(model);epoch_best=epoch;best_j=j
                optimization_validation_seconds=time.perf_counter()-optimizer_tick
                terminal=frozen(model)
                path=HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt'
                torch.save(dict(**meta,state=best,initial_state=init,terminal_state=terminal,epoch=epoch_best,
                    mu=mu,scale=scale,input_dim=raw.shape[-1],parameters=params,calibration_and_keep_fully_oof=False),path)
                ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['state']);model.eval()
                for role,xx in [('train',xt),('validation',xv)]:
                    with torch.inference_mode():s=b[role]+model(xx).numpy().astype(float)
                    if role=='validation':assert exact_j(yv,s.argmax(1))==best_j
                    frames.append(predict_frame(table,info[role]['indices'],b[role],s,fold,seed,role,arm,epoch_best))
                selections.append(dict(**meta,epoch=epoch_best,validation_J=float(best_j),parameters=params,selected_utc=now()))
                costs.append(dict(**meta,cache_load_seconds_shared_fold=load_seconds,optimization_and_validation_seconds=optimization_validation_seconds,cached_adaptation_wall_seconds=time.perf_counter()-tick))
                checks.append(dict(**meta,common_quality_tensor_exact=True,common_quality_scaler_exact=True,fixed_dynamic_init_exact=True,zero_step_exact=True,selected_replay_exact_J=True))
                for name,rows in [('selections',selections),('selection_curves',curves),('training_checks',checks),('training_costs',costs)]:pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
                print(fold,seed,arm,'epoch',epoch_best,'J',float(best_j),flush=True)
    assert len(selections)==36
    pred=pd.concat(frames,ignore_index=True);pred.to_csv(HERE/'source_predictions.csv',index=False)
    pooled(pred).to_csv(HERE/'source_pooled.csv',index=False)
    pd.DataFrame(summary_diff).to_csv(HERE/'source_fixed_dynamic_summary_difference.csv',index=False)
    save_json(HERE/'source_selection_frozen.json',dict(completed_utc=now(),heads=36,readers=15,steps=7200,zero_epochs=sum(r['epoch']==0 for r in selections),seconds=time.perf_counter()-started,test_evaluated=False,old_stage_fits=0))
    print(pooled(pred).query("role=='validation'").to_string(index=False),flush=True)


def test_evaluate():
    started=time.perf_counter();torch.set_num_threads(2)
    assert (HERE/'source_selection_frozen.json').exists()
    assert not (HERE/'test_predictions.csv').exists()
    table,_,_=load_data(); ref=reference('test');frames=[];checks=[];differences=[]
    for fold in range(3):
        info=load_info(fold,'test',table);ix=info['indices']
        differences.extend(summary_checks(info,fold,'test'))
        for seed in SEEDS:
            b=baseline(ref,table,ix,fold,seed,'test')
            for arm in ARMS:
                ck=torch.load(HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt',map_location='cpu',weights_only=False)
                raw=raw_inputs(b,info,arm);x,_,_=standardize(raw,ck['mu'],ck['scale'])
                model=Head(ck['input_dim']);model.load_state_dict(ck['state']);model.eval()
                with torch.inference_mode():s=b+model(x).numpy().astype(float)
                if ck['epoch']==0:np.testing.assert_array_equal(s,b)
                frames.append(predict_frame(table,ix,b,s,fold,seed,'test',arm,ck['epoch']))
                checks.append(dict(fold=fold,seed=seed,arm=arm,epoch=ck['epoch'],n=len(ix),baseline_cache_exact=True,new_fits=0))
    pred=pd.concat(frames,ignore_index=True);pred.to_csv(HERE/'test_predictions.csv',index=False)
    pooled(pred).to_csv(HERE/'test_pooled.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'test_checks.csv',index=False)
    pd.DataFrame(differences).to_csv(HERE/'test_fixed_dynamic_summary_difference.csv',index=False)
    save_json(HERE/'test_complete.json',dict(completed_utc=now(),cached_inference_seconds=time.perf_counter()-started,new_fits=0,unique_objects=pred.instance_id.nunique(),heads=36))
    print(pooled(pred).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['source','test']);args=parser.parse_args()
    source_train() if args.stage=='source' else test_evaluate()
