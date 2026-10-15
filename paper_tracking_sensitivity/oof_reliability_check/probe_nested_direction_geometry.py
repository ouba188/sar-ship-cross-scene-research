"""Diagnose whether shared source geometry misranks original versus C4 branches."""

import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='2'
import argparse
import json
import time
import numpy as np
import pandas as pd

from run_c4_base_trial import OUT as C4, ROOT, SOURCE, GEOMETRY, write_json
from protected_drift_activation import protected_drift_basis, fit_source_heads
from residual_expert import fit_correction
from direction_score_fusion import fit_fusions, predict_fusion

TRIAL=ROOT/'directional_fusion_trial_20261009'
EPISODES=ROOT/'module_completion_20261008/ports_projection'
OUT=TRIAL/'nested_geometry_probe'


def archive(path):
    with np.load(path) as f: return {k:f[k] for k in f.files}


def inputs():
    frame=pd.read_csv(SOURCE/'manifest.csv',usecols=['instance_id','product','port','class_id'])
    side=pd.read_csv(ROOT.parent/'pixel_correction_review/chip_incidence_v1/incidence_by_instance.csv',
        usecols=['instance_id','product','status','incidence_deg'])
    frame=frame.merge(side,on=['instance_id','product'],how='left',sort=False,validate='one_to_one')
    frame.loc[frame.status.ne('ok'),'incidence_deg']=np.nan
    rel=pd.read_csv(GEOMETRY/'relative_conditions.csv')
    np.testing.assert_array_equal(frame.instance_id,rel.instance_id)
    valid=rel.relative_status.eq('ok').to_numpy(); angle=np.deg2rad(rel.relative_signed_deg.fillna(0).to_numpy()*2)
    t=(frame.incidence_deg.to_numpy()-38)/5
    u=np.nan_to_num(np.c_[t,t*t],nan=0.)
    return frame,np.c_[u,np.isfinite(t),valid,np.where(valid,np.cos(angle),0),np.where(valid,np.sin(angle),0)]


def cell_means(raw,rows,labels,domains,center,scale):
    means,classes=[],[]
    for cl in np.unique(labels[rows]):
        groups=[]
        for port in np.unique(domains[rows]):
            selected=rows[(labels[rows]==cl)&(domains[rows]==port)]
            if len(selected)>=5:
                total=np.zeros(raw.shape[1],dtype=float)
                for lo in range(0,len(selected),512):
                    total+=raw[selected[lo:lo+512]].sum(0,dtype=float)
                groups.append((total/len(selected)-center)/scale)
        if len(groups)>=3:
            means.extend(groups); classes.extend([cl]*len(groups))
    return np.asarray(means),np.asarray(classes)


def metrics(labels,predictions,classes):
    correct=labels==predictions
    return dict(acc=float(correct.mean()),ba=float(np.mean([correct[labels==c].mean() for c in np.unique(labels)])))


def source_port(port,frame,context,raw,averaged,projector=None):
    folder=OUT/'ports'/port; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists(): return
    start=time.perf_counter()
    all_ports=sorted(frame.port.unique())
    items=json.loads((EPISODES/port/'EPISODES_COMPLETE.json').read_text())['episodes']
    labels=frame.class_id.to_numpy(); domains=frame.port.to_numpy()
    data={k:[] for k in ['original','c4','labels','rows','episode','weights','context']}
    metric_records=[]
    for i,item in enumerate(items):
        ep=archive(item['path']); train,rows,support,query=ep['train_rows'],ep['rows'],ep['support'],ep['query']
        key='_'.join(f'{j:02d}' for j in sorted([all_ports.index(port),all_ports.index(item['port'])]))
        geometry=archive(EPISODES/'episode_geometry_cache'/f'{key}.npz')
        pair=archive(ROOT/'evidence_aware_handoff_20261008/source_pair_cache'/f'{key}.npz')
        np.testing.assert_array_equal(train,geometry['train_rows']); np.testing.assert_array_equal(train,pair['train_rows'])
        if np.isin(domains[train],[port,item['port']]).any() or np.intersect1d(train,rows).size:
            raise ValueError('Held port enters nested geometry')
        center,scale,qs=pair['raw_center'],pair['raw_scale'],pair['q']
        cache_path=OUT/'pair_fit'/f'{key}.npz'
        cache_path.parent.mkdir(exist_ok=True)
        cached=archive(cache_path) if cache_path.exists() else None
        if cached is None:
            means,classes=cell_means(raw,train,labels,domains,center,scale)
            qd=protected_drift_basis(means,classes,qs,rank=16)['q_drift']
        else:
            np.testing.assert_array_equal(cached['train_rows'],train)
            qd=cached['qd']
        r,d=qs.shape[1],qd.shape[1]
        train_n=len(train) if cached is None else 0
        ids=np.r_[train,rows] if cached is None else rows
        pc=geometry['base_components']
        def project(values):
            xs=((values-geometry['base_mean'])@pc.T-geometry['base_center'])/geometry['base_scale']
            z=(values-center)/scale
            return xs,z@qs,z@qd
        xs,zs,zd=project(np.asarray(raw[ids],dtype=float)) if projector is None else projector(ids,geometry,center,scale,qs,qd,False)
        stable_center=zs[:train_n].mean(0) if cached is None else cached['stable_center']
        stable_scale=zs[:train_n].std(0)+1e-6 if cached is None else cached['stable_scale']
        dxs=(qd.T*scale[None,:])@pc.T/geometry['base_scale']
        transform=np.zeros((128+r+d+2,128+r+2))
        transform[:128,:128]=np.eye(128)
        transform[128+r:128+r+d,:128]=-.5*dxs
        transform[128:128+r,128:128+r]=np.eye(r)*np.sqrt(128/r)
        transform[-2:,-2:]=np.eye(2)
        predictions={}
        fitted={}
        for name,values in [('original',None),('c4',averaged)]:
            if values is not None:
                xs,zs,zd=project(np.asarray(values[ids],dtype=float)) if projector is None else projector(ids,geometry,center,scale,qs,qd,True)
            stable=(zs-stable_center)/stable_scale
            u=context[ids,:2]
            bank=np.c_[xs,stable,zd,u]
            head=fit_source_heads(bank[:train_n],labels[train],{'base':transform})['base'] if cached is None else {
                'coef':cached[name+'_coef'],'intercept':cached[name+'_intercept']}
            fitted[name+'_coef']=head['coef']; fitted[name+'_intercept']=head['intercept']
            core=xs[train_n:]-.5*zd[train_n:]@dxs
            phi=np.c_[(core-pair['base_residual_center'])/pair['base_residual_scale']/np.sqrt(128),
                stable[train_n:]/np.sqrt(r),context[rows,:2]/np.sqrt(2),np.ones(len(rows))]
            prior=bank[train_n:]@head['coef']+head['intercept']
            score=prior+phi@fit_correction(phi[support],labels[rows[support]],prior[support])
            predictions[name]=score[query]
            m=metrics(labels[rows[query]],score[query].argmax(1),8)
            metric_records.append(dict(outer_port=port,held_port=item['port'],mode=name,oa=m['acc'],ba=m['ba']))
        if cached is None:
            np.savez_compressed(cache_path,train_rows=train,qd=qd,stable_center=stable_center,stable_scale=stable_scale,**fitted)
        _,inverse,counts=np.unique(labels[rows[query]],return_inverse=True,return_counts=True)
        values=dict(**predictions,labels=labels[rows[query]],rows=rows[query],episode=np.full(len(query),i),
            weights=1./(len(items)*len(counts)*counts[inverse]),context=context[rows[query]])
        for k,v in values.items(): data[k].append(v)
        if (i+1)%6==0: print(f'NESTED {port} {i+1}/{len(items)} {time.perf_counter()-start:.1f}s',flush=True)
    data={k:np.concatenate(v) for k,v in data.items()}
    np.savez_compressed(folder/'source_predictions.npz',**data)
    pd.DataFrame(metric_records).to_csv(folder/'branch_metrics.csv',index=False)
    models=fit_fusions(data['original'],data['c4'],data['labels'],data['weights'],data['context'])
    write_json(folder/'fusion_models.json',models)
    evaluation=[]
    for name,model in models.items():
        scores,_=predict_fusion(model,data['original'],data['c4'],data['context'])
        for episode in np.unique(data['episode']):
            mask=data['episode']==episode
            m=metrics(data['labels'][mask],scores[mask].argmax(1),8)
            evaluation.append(dict(port=port,episode=int(episode),mode=name,oa=m['acc'],ba=m['ba']))
    pd.DataFrame(evaluation).to_csv(folder/'fusion_fit_metrics.csv',index=False)
    write_json(folder/'COMPLETE.json',dict(port=port,episodes=len(items),geometry_inner_held=True,
        qs='Existing pair-source cache, recomputed Qd/stable normalization; no held labels in geometry',
        seconds=time.perf_counter()-start,target_predictions_not_evaluated=True))
    print(pd.DataFrame(metric_records).groupby('mode')[['oa','ba']].mean().to_string(),flush=True)
    print('GLOBAL_ALPHA',models['global_mix']['alpha'],flush=True)


def gpu_projector(raw,averaged):
    import torch
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    tensors=[torch.as_tensor(np.array(x),device='cuda') for x in [raw,averaged]]
    def project(ids,geometry,center,scale,qs,qd,use_average):
        pc=geometry['base_components']
        operator=np.c_[pc.T/geometry['base_scale'],qs/scale[:,None],qd/scale[:,None]]
        offset=np.r_[(-geometry['base_mean']@pc.T-geometry['base_center'])/geometry['base_scale'],
            -(center/scale)@qs,-(center/scale)@qd]
        with torch.inference_mode():
            selected=tensors[int(use_average)][torch.as_tensor(ids,device='cuda')]
            projected=selected@torch.as_tensor(operator,dtype=torch.float32,device='cuda')
            result=projected.cpu().numpy().astype(float)+offset
        return result[:,:128],result[:,128:128+qs.shape[1]],result[:,128+qs.shape[1]:]
    return project


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--ports',nargs='+',required=True)
    parser.add_argument('--gpu',action='store_true'); args=parser.parse_args()
    frame,context=inputs(); raw=np.load(SOURCE/'pooled_features.npy',mmap_mode='r'); averaged=np.load(C4/'pooled_c4.npy',mmap_mode='r')
    projector=None
    if args.gpu:
        OUT=TRIAL/'nested_geometry_gpu'
        projector=gpu_projector(raw,averaged)
    for port in args.ports: source_port(port,frame,context,raw,averaged,projector)
