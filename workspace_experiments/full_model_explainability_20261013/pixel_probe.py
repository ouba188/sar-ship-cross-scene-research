"""Matched trained R18 query interventions inside the frozen full OSS2 model."""

from run_paths import OUT,ROOT,read,write,arrays,mapped,Geometry
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from PIL import Image
import torch
from torchvision.models import resnet18
from torchvision import transforms
from projection_residual_expert import discarded_coordinates

sys.path.insert(0,str(ROOT/'region_cue_probe_20261013'))
from transforms_regions import read_pair,transform

CONDITIONS=('original','target_shuffle','outer_shuffle','target_flat','sham_flat','target_display','outer_display')


def main():
    dest=OUT/'pixel_probe';dest.mkdir(exist_ok=True)
    task=next(t for t in read(ROOT/'paper_core_freeze_20261013/FROZEN_REFERENCE.json')['tasks'] if t['task']=='oss2/oss2')
    folder=OUT/'tasks/oss2/oss2'
    data={k:np.load(folder/'input'/(k+'.npy'),mmap_mode='r') for k in
          ('rows','ids','support','support_labels','eval_masks','phi','mean','conditions','prior')}
    frame=pd.read_csv(ROOT/'region_cue_probe_20261013/cohort.csv',keep_default_na=False)
    frame=frame.set_index('sample_id').loc[data['ids']].reset_index()
    regions=pd.read_csv(ROOT/'region_cue_probe_20261013/region_metadata.csv').set_index('sample_id').loc[data['ids']]
    eligible=data['eval_masks'].all(0)&regions.region_valid.to_numpy(bool)&regions.sham_valid.to_numpy(bool)
    rng=np.random.default_rng(20261013)
    chosen=[]
    for cl in range(8):
        rows=np.flatnonzero(eligible&(frame.class_id.to_numpy()==cl))
        chosen.extend(rng.choice(rows,min(16,len(rows)),replace=False).tolist())
    chosen=np.array(sorted(chosen),int)
    frame.iloc[chosen].assign(query_index=chosen).to_csv(dest/'selected.csv',index=False)
    write(dest/'SELECTION_FROZEN.json',dict(seed=20261013,query_indices=chosen.tolist(),
        classes=frame.iloc[chosen].class_id.value_counts().sort_index().to_dict(),
        conditions=CONDITIONS,selection_uses_predictions=False,main_all_query_task=False))
    torch.set_num_threads(1)
    net=resnet18(weights=None)
    ck=torch.load(ROOT/'visual_reliable_baseline/e206_out/ssl_resnet18.pt',map_location='cpu',weights_only=True)
    missing,unexpected=net.load_state_dict(ck['backbone'],strict=False)
    assert set(missing)=={'fc.weight','fc.bias'} and not unexpected
    tailpath='E:/Hermes/paper_tracking/sensitivity/module_completion_20261008/oss2/joint/layer4.pt'
    net.layer4.load_state_dict(torch.load(tailpath,map_location='cpu',weights_only=True))
    encoder=torch.nn.Sequential(*list(net.children())[:-1]).eval()
    tf=transforms.Compose([transforms.Resize(128),transforms.ToTensor()])
    features={};controls=[];started=time.perf_counter()
    # Same batch size and arithmetic path for original and every intervention.
    with torch.inference_mode():
        for condition in CONDITIONS:
            values=[]
            for lo in range(0,len(chosen),8):
                batch=[]
                for q in chosen[lo:lo+8]:
                    image,info=transform(read_pair(frame.iloc[q]),frame.iloc[q],condition,int(q))
                    assert info['region_valid'] and info['sham_valid']
                    batch.append(tf(Image.fromarray(image[1]).convert('RGB')))
                    controls.append(dict(condition=condition,query_index=int(q),**info))
                values.append(encoder(torch.stack(batch)).flatten(1).numpy())
            features[condition]=np.concatenate(values)
            np.save(dest/(condition+'_extra.npy'),features[condition])
            print('PIXELS',condition,len(chosen),round(time.perf_counter()-started,1),flush=True)
    pd.DataFrame(controls).to_csv(dest/'pixel_controls.csv',index=False)
    coeff=arrays(task['coefficients'])
    geo=Geometry(task,data,coeff,folder)
    prior_reference=geo.encode(chosen)
    support_union=np.unique(data['support']);refs=geo.encode(support_union)
    kr,_=geo.kernels(prior_reference,refs)
    raw=geo.raw[chosen]
    m,stats=geo.ctx.model,geo.ctx.stats
    oldextra=np.asarray(geo.extra[chosen])
    error=np.linalg.norm(features['original']-oldextra,axis=1)/np.maximum(np.linalg.norm(oldextra,axis=1),1e-12)
    if error.max()>1e-4:
        write(dest/'REPLAY_FAILED.json',dict(feature_max_relative_l2=float(error.max()),reason='encoder/input mismatch'))
        raise AssertionError('Original features mismatch; do not interpret pixel intervention')
    frozen=np.load(folder/'predictions.npy',mmap_mode='r')
    parts=np.load(folder/'parts.npy',mmap_mode='r')
    result=np.empty((len(CONDITIONS),5,len(chosen),8),float)
    maxprob=0.
    for ci,condition in enumerate(CONDITIONS):
        extra=features[condition]
        from compact_fusion_source import _expert_inputs
        x=_expert_inputs(raw,extra,m)
        br=(x/np.sqrt(192)).astype(float)[:,128:]
        residual=discarded_coordinates(extra,m['pca_ssl'])
        bd=(residual-stats['discarded_center'][2048:])/stats['discarded_scale'][2048:]/np.sqrt(2560)
        kb=.5*(br@refs['br'].T+bd@refs['bd'].T)
        pt=(x@coeff['dual_coef']+coeff['dual_intercept']).astype(np.float32).astype(float)
        for draw,s in enumerate(data['support']):
            col=np.searchsorted(support_union,s)
            expert=pt+(kr[:,col]+kb[:,col])@coeff[f'{draw}_dual_refit_expert']
            base=np.asarray(parts[draw,0,chosen])
            prob=.5*softmax(base,axis=1)+.5*softmax(expert,axis=1)
            result[ci,draw]=prob
            if condition=='original':
                expected=frozen[draw,7,chosen]
                oldexpert=np.asarray(parts[draw,1,chosen])+np.asarray(parts[draw,2,chosen])+np.asarray(parts[draw,3,chosen])+np.asarray(parts[draw,4,chosen])
                oldprob=.5*softmax(base,axis=1)+.5*softmax(oldexpert,axis=1)
                maxprob=max(maxprob,float(abs(prob-oldprob).max()))
                if not np.array_equal(prob.argmax(1),expected):
                    write(dest/'REPLAY_FAILED.json',dict(feature_max_relative_l2=float(error.max()),
                        probability_max_error=maxprob,draw=draw,flips=int((prob.argmax(1)!=expected).sum())))
                    raise AssertionError('Pixel original decisions mismatch; no tolerance relaxation')
    np.savez_compressed(dest/'predictions.npz',probability=result,query_indices=chosen,conditions=np.array(CONDITIONS))
    write(dest/'PREDICTIONS_FROZEN.json',dict(n=len(chosen),draws=5,conditions=CONDITIONS,
        checkpoint=tailpath,feature_max_relative_l2=float(error.max()),probability_max_error=maxprob,
        original_decision_replays=5,changed_path='query R18 prior and support relation; R50/base/supports fixed',
        new_training=False,seconds=time.perf_counter()-started))
    # Diagnostic labels were used for balanced sampling, never method selection or fitting.
    y=frame.iloc[chosen].class_id.to_numpy(int)
    _,ix,counts=np.unique(y,return_inverse=True,return_counts=True)
    w=100./(len(counts)*counts[ix]);records=[];classes=[]
    for draw in range(5):
        r=frozen[draw,0,chosen];t=frozen[draw,7,chosen]
        rescue=(r!=y)&(t==y);harm=(r==y)&(t!=y)
        for ci,condition in enumerate(CONDITIONS):
            pred=result[ci,draw].argmax(1)
            records.append(dict(draw=draw,condition=condition,n=len(y),oa=100*np.mean(pred==y),
                ba=float(w@(pred==y)),delta_ba=float(w@((pred==y).astype(int)-(t==y))),
                old_rescue_mass=float(w@rescue),old_rescue_lost=float(w@(rescue&(pred!=y))),
                old_harm_removed=float(w@(harm&(pred==y)))))
            for cl in np.unique(y):
                use=y==cl
                classes.append(dict(draw=draw,condition=condition,class_id=int(cl),n=int(use.sum()),
                    recall=100*np.mean(pred[use]==cl),delta_recall=100*np.mean((pred[use]==cl).astype(int)-(t[use]==cl))))
    records=pd.DataFrame(records);records.to_csv(dest/'metrics.csv',index=False)
    records.groupby('condition').mean(numeric_only=True).reset_index().to_csv(dest/'summary.csv',index=False)
    pd.DataFrame(classes).to_csv(dest/'classes.csv',index=False)
    print(records.groupby('condition').mean(numeric_only=True).to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
