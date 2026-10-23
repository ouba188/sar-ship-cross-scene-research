"""Nine frozen-B source heads: PNG summary, own display anchor, matched anchor."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT))
import train_heads as original
P=original.parent
ARMS=['png_summary','own_scale','matched_scale']
SEEDS=[17,29,43]


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(14,16),nn.GELU(),nn.Linear(16,8))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        out=self.net(x)
        return out-out.mean(1,keepdim=True)


def measures(y,s):
    y=np.asarray(y,int);shift=s-s.max(1,keepdims=True)
    prob=np.exp(shift);prob/=prob.sum(1,keepdims=True)
    losses=np.log(np.exp(shift).sum(1))-shift[np.arange(len(y)),y]
    correct=s.argmax(1)==y;classes=np.unique(y)
    oa=float(correct.mean()*100);ba=float(np.mean([correct[y==c].mean() for c in classes])*100)
    return dict(n=len(y),n_classes=len(classes),present_classes=';'.join(map(str,classes)),
        OA=oa,BA=ba,J=(oa+ba)/200,NLL=float(losses.mean()),
        balanced_NLL=float(np.mean([losses[y==c].mean() for c in classes])),
        Brier=float(((prob-np.eye(8)[y])**2).sum(1).mean()))


def load():
    with np.load(ROOT/'pair_local_evidence/source_moments.npz') as z:
        ids,tr,va,reader=[z[k] for k in ['sample_ids','train','validation','reader']]
    cohort=pd.read_csv(ROOT/'cohort.csv',dtype={'mmsi':str})
    mapping=pd.Series(np.arange(len(cohort)),index=cohort.sample_id).loc[ids].to_numpy()
    table=cohort.iloc[mapping].reset_index(drop=True)
    assert len(tr)==1542 and len(va)==279 and len(ids)==1821
    with np.load(ROOT/'readers/fold0_inputs.npz') as z:
        np.testing.assert_array_equal(mapping[tr],z['train']);np.testing.assert_array_equal(mapping[va],z['validation'])
    for k in range(4):
        with np.load(ROOT/'readers/models'/f'fold0_inner{k}.npz') as z:fit,held=z['fit'],z['held']
        assert (reader[np.isin(mapping,held)]==k+1).all()
        for name in ['mmsi','pass_group']:assert not set(cohort.iloc[fit][name])&set(cohort.iloc[held][name])
    assert (reader[tr]>0).all() and (reader[va]==0).all()
    for name in ['mmsi','pass_group']:assert not set(table.iloc[tr][name])&set(table.iloc[va][name])
    table['source_row']=np.arange(len(table));table['global_row']=mapping;table['reader']=reader
    table['role']=np.where(reader==0,'validation','train')
    correspondence=pd.read_csv(HERE.parent/'source_correspondence.csv',dtype={'mmsi':str}).set_index(['sample_id','pol'])
    anchors=np.empty((len(ids),2));png=np.empty_like(anchors);sizes=np.empty_like(anchors)
    for p,pol in enumerate(['VV','VH']):
        rows=correspondence.loc[pd.MultiIndex.from_arrays([ids,np.repeat(pol,len(ids))])].reset_index()
        np.testing.assert_array_equal(rows.source_row,np.arange(len(ids)))
        np.testing.assert_array_equal(rows.role,table.role)
        anchor=rows.amplitude_anchor_255.to_numpy()
        assert np.isfinite(anchor).all() and (anchor>0).all()
        np.testing.assert_allclose(anchor,255/rows.amplitude_to_display_slope.to_numpy(),rtol=1e-12)
        assert rows.finite.all() and rows.negative.eq(0).all()
        anchors[:,p]=np.log(anchor)
        for col in ['height','width','band','calibrated_path','amplitude_anchor_255','amplitude_to_display_slope','fitted_display_mae','fitted_round_equal','float_q1','png_saturated_fraction']:
            table[f'{pol.lower()}_{col}']=rows[col].to_numpy()
        for i,path in enumerate(rows.png_path):
            with Image.open(path) as im:raw=np.array(im)
            assert raw.shape==(rows.iloc[i].height,rows.iloc[i].width)
            rms=float(np.sqrt(np.mean((raw.astype(float)/255)**2)));assert rms>0
            png[i,p]=np.log(rms);sizes[i,p]=np.log(raw.size)
    bases={}
    for seed in SEEDS:
        with np.load(ROOT/'calibrated'/f'fold0_seed{seed}.npz') as z:
            np.testing.assert_array_equal(z['sample_ids'][mapping],ids)
            b=np.zeros((len(ids),8));b[tr]=z['train_base'][mapping[tr]];b[va]=z['base'][mapping[va]]
            bases[seed]=b
    return table,tr,va,png,sizes,anchors,bases


def match(table,train,png,b17):
    coords=np.column_stack([b17-b17.mean(1,keepdims=True),png])
    mu=coords[train].mean(0);scale=coords[train].std(0).clip(1e-8)
    coords=(coords-mu)/scale
    keys=['role','reader','pass_group','vv_height','vv_width','vh_height','vh_width']
    donor=np.arange(len(table));distance=np.full(len(table),np.nan);eligible=np.zeros(len(table),int)
    block_size=np.zeros(len(table),int)
    for _,g in table.groupby(keys,sort=True):
        indices=g.index.to_numpy();block_size[indices]=len(g);edges=[]
        for pos,i in enumerate(indices):
            for j in indices[pos+1:]:
                if table.at[i,'mmsi']==table.at[j,'mmsi']:continue
                eligible[i]+=1;eligible[j]+=1
                dist=float(np.linalg.norm(coords[i]-coords[j]))
                first,second=sorted([str(table.at[i,'sample_id']),str(table.at[j,'sample_id'])])
                edges.append((dist,first,second,int(i),int(j)))
        occupied=set()
        for dist,_,__,i,j in sorted(edges):
            if i in occupied or j in occupied:continue
            donor[i]=j;donor[j]=i;distance[i]=distance[j]=dist;occupied.update([i,j])
    ix=np.arange(len(table));matched=donor!=ix
    np.testing.assert_array_equal(donor[donor],ix)
    assert np.all(table.mmsi.to_numpy()[matched]!=table.mmsi.to_numpy()[donor[matched]])
    for key in keys:np.testing.assert_array_equal(table[key].to_numpy(),table[key].to_numpy()[donor])
    table=table.copy();table['donor_row']=donor;table['donor_id']=table.sample_id.to_numpy()[donor]
    table['matched']=matched;table['match_distance']=distance;table['candidate_count']=eligible
    table['block_size']=block_size
    table['unmatched_reason']=np.select([matched,block_size==1,eligible==0],['matched','singleton_exact_block','no_other_MMSI'],default='greedy_remaining')
    table['donor_class']=table.class_id.to_numpy()[donor]
    table['same_true_class']=table.class_id.to_numpy()==table.donor_class.to_numpy()
    table['B17_same_prediction']=b17.argmax(1)==b17.argmax(1)[donor]
    np.savez_compressed(HERE/'matching_scaler.npz',mu=mu,scale=scale)
    return table,donor


def restore(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=Head().eval().requires_grad_(False);model.load_state_dict(ck['state'])
    return ck,model


def main():
    started=time.perf_counter();assert (HERE/'PROTOCOL_ZH.md').exists()
    (HERE/'heads').mkdir(exist_ok=True);torch.set_num_threads(2)
    table,tr,va,png,sizes,anchors,bases=load()
    table,donor=match(table,tr,png,bases[17]);table.to_csv(HERE/'source_inputs.csv',index=False)
    matchrows=[]
    for role,g in table.groupby('role'):
        mg=g[g.matched]
        matchrows.append(dict(role=role,n=len(g),matched=len(mg),matched_fraction=len(mg)/len(g),pairs=len(mg)//2,
            same_true_class=float(mg.same_true_class.mean()),B17_same_prediction=float(mg.B17_same_prediction.mean()),
            distance_min=float(mg.match_distance.min()),distance_median=float(mg.match_distance.median()),distance_max=float(mg.match_distance.max()),
            singleton_exact_block=int(g.unmatched_reason.eq('singleton_exact_block').sum()),
            no_other_MMSI=int(g.unmatched_reason.eq('no_other_MMSI').sum()),greedy_remaining=int(g.unmatched_reason.eq('greedy_remaining').sum())))
    pd.DataFrame(matchrows).to_csv(HERE/'matching_coverage.csv',index=False)
    coverage=[]
    for kind in ['class_id','pass_group','vv_height']:
        for (role,value),g in table.groupby(['role',kind]):
            coverage.append(dict(role=role,grouping=kind,group=value,n=len(g),matched=int(g.matched.sum()),matched_fraction=float(g.matched.mean())))
    pd.DataFrame(coverage).to_csv(HERE/'matching_strata.csv',index=False)
    for ix in [tr,va]:
        for p in range(2):np.testing.assert_array_equal(np.sort(anchors[ix,p]),np.sort(anchors[donor[ix],p]))
    np.savez_compressed(HERE/'source_inputs.npz',sample_ids=table.sample_id.to_numpy(str),train=tr,validation=va,
        reader=table.reader.to_numpy(),global_rows=table.global_row.to_numpy(),png_log_rms=png,log_pixels=sizes,
        log_anchor=anchors,donor=donor,**{f'B{seed}':base for seed,base in bases.items()})
    print('matching',json.dumps(matchrows),flush=True)
    curves=[];selections=[];checks=[];saved={};y=table.class_id.to_numpy()
    for seed in SEEDS:
        base=bases[seed];common=np.column_stack([base-base.mean(1,keepdims=True),png,sizes]);initial=None
        ownscale=None
        for arm in ARMS:
            extra=np.zeros_like(anchors) if arm=='png_summary' else anchors if arm=='own_scale' else anchors[donor]
            raw=np.column_stack([common,extra]);mu=raw[tr].mean(0);scale=raw[tr].std(0).clip(1e-8)
            if arm=='own_scale':ownscale=(mu.copy(),scale.copy())
            if arm=='matched_scale':
                np.testing.assert_allclose(mu,ownscale[0],atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(scale,ownscale[1],atol=1e-12,rtol=1e-12)
            x=torch.tensor((raw-mu)/scale,dtype=torch.float32)
            torch.manual_seed(seed);model=Head();assert sum(p.numel() for p in model.parameters())==376
            if initial is None:initial=P.frozen_state(model)
            else:
                for k,v in model.state_dict().items():assert torch.equal(v,initial[k])
            with torch.no_grad():assert not model(x[va]).any()
            best,epoch_best=P.frozen_state(model),0;best_j=P.exact_j(y[va],base[va].argmax(1))
            meta=dict(fold=0,seed=seed,arm=arm)
            initial_train=measures(y[tr],base[tr]);initial_loss=.5*(initial_train['NLL']+initial_train['balanced_NLL'])
            curves.append(dict(**meta,epoch=0,train_loss=initial_loss,train_loss_stage='initial_float64',**measures(y[va],base[va])))
            opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
            bb=torch.tensor(base[tr],dtype=torch.float32);yy=torch.tensor(y[tr],dtype=torch.long)
            for epoch in range(1,201):
                opt.zero_grad();loss=P.joint_loss(bb+model(x[tr]),yy);assert torch.isfinite(loss)
                loss.backward();opt.step()
                if epoch%5:continue
                with torch.no_grad():score=base[va]+model(x[va]).numpy().astype(float)
                j=P.exact_j(y[va],score.argmax(1))
                curves.append(dict(**meta,epoch=epoch,train_loss=float(loss.detach()),train_loss_stage='pre_update_float32',**measures(y[va],score)))
                if j>best_j:best_j,epoch_best,best=j,epoch,P.frozen_state(model)
            path=HERE/'heads'/f'{arm}_{seed}.pt'
            torch.save(dict(**meta,state=best,mu=mu,scale=scale,epoch=epoch_best,parameters=376),path)
            ck,replay=restore(path)
            with torch.no_grad():pred=base+replay(x).numpy().astype(float)
            assert P.exact_j(y[va],pred[va].argmax(1))==best_j
            selections.append(dict(**meta,epoch=epoch_best,validation_J=float(best_j),parameters=376))
            checks.append(dict(**meta,initial_exact_B=True,selected_J_replay=True,source_only=True,
                residual_sum_error=float(abs((pred-base).sum(1)).max())))
            saved[(seed,arm)]=pred
            print('selected',seed,arm,epoch_best,float(best_j),flush=True)
    pd.DataFrame(selections).to_csv(HERE/'selections.csv',index=False)
    pd.DataFrame(curves).to_csv(HERE/'selection_curves.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'checks.csv',index=False)
    (HERE/'selection_frozen.json').write_text(json.dumps(dict(heads=9,test_used=False)),encoding='utf-8')
    frames=[];metrics=[];classes=[];strata=[]
    for (seed,arm),score in saved.items():
        base=bases[seed]
        for role,ix in [('train',tr),('validation',va)]:
            g=table.iloc[ix][['sample_id','source_row','global_row','reader','class_id','class_name','mmsi','pass_group','matched','donor_row','donor_id','donor_class','match_distance','vv_height','vv_width','vh_height','vh_width']].copy().rename(columns={'sample_id':'instance_id'})
            g['seed']=seed;g['arm']=arm;g['role']=role
            g['epoch']=next(v['epoch'] for v in selections if v['seed']==seed and v['arm']==arm)
            g['base_prediction']=base[ix].argmax(1);g['prediction']=score[ix].argmax(1)
            g['net']=g.prediction.eq(g.class_id).astype(int)-g.base_prediction.eq(g.class_id).astype(int)
            g['outcome']=np.select([g.net.eq(1),g.net.eq(-1),g.prediction.eq(g.class_id)],['rescued','harmed','CC'],default='WW')
            for c in range(8):g[f'b{c}']=base[ix,c];g[f's{c}']=score[ix,c]
            frames.append(g)
            for subset,mask in [('all',np.ones(len(ix),bool)),('matched',table.iloc[ix].matched.to_numpy()),('unmatched',~table.iloc[ix].matched.to_numpy())]:
                sub=ix[mask]
                if not len(sub):continue
                before,after=measures(y[sub],base[sub]),measures(y[sub],score[sub])
                net=int(((score[sub].argmax(1)==y[sub]).astype(int)-(base[sub].argmax(1)==y[sub]).astype(int)).sum())
                metrics.append(dict(seed=seed,arm=arm,role=role,subset=subset,net=net,**after,
                    **{f'base_{k}':v for k,v in before.items() if k not in ['n','n_classes','present_classes']},
                    **{f'delta_{k}':after[k]-before[k] for k in ['OA','BA','NLL','balanced_NLL','Brier']}))
            for kind in ['class_id','pass_group','vv_height']:
                for value,gg in g.groupby(kind):
                    before=measures(gg.class_id.to_numpy(),gg[[f'b{c}' for c in range(8)]].to_numpy())
                    after=measures(gg.class_id.to_numpy(),gg[[f's{c}' for c in range(8)]].to_numpy())
                    strata.append(dict(seed=seed,arm=arm,role=role,grouping=kind,group=value,net=int(gg.net.sum()),
                        matched=int(gg.matched.sum()),**after,base_OA=before['OA'],base_BA=before['BA']))
    pred=pd.concat(frames,ignore_index=True);met=pd.DataFrame(metrics)
    pred.to_csv(HERE/'predictions.csv',index=False);met.to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(strata).to_csv(HERE/'strata.csv',index=False)
    paired=[]
    for (seed,role,subset),g in met.groupby(['seed','role','subset']):
        part=g.set_index('arm')
        for control in ['png_summary','matched_scale']:
            paired.append(dict(seed=seed,role=role,subset=subset,comparison=f'own_minus_{control}',
                **{f'delta_{k}':float(part.loc['own_scale',k]-part.loc[control,k]) for k in ['OA','BA','NLL','balanced_NLL','Brier','net']}))
    pd.DataFrame(paired).to_csv(HERE/'paired_metrics.csv',index=False)
    info=dict(heads=9,parameters=376,steps=1800,train=1542,validation=279,test_used=False,GPU_used=False,
        seconds=time.perf_counter()-started,zero_epoch_heads=sum(v['epoch']==0 for v in selections),
        scale_feature='log(amplitude_anchor_255=255/fitted_pixel_slope)',max_feature_used=False,
        matching=matchrows,source_scaler_only=True)
    (HERE/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(met[met.role.eq('validation')&met.subset.eq('all')][['seed','arm','net','OA','BA','NLL','delta_OA','delta_BA']].to_string(index=False))
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
