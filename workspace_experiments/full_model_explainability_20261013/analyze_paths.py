"""Full-system path necessity, original rescue tracking, and concrete cases."""

from run_paths import OUT,ROOT,MODES,read,write,arrays
import numpy as np
import pandas as pd
from scipy.special import softmax


def weights(y):
    _,inverse,counts=np.unique(y,return_inverse=True,return_counts=True)
    return 100./(len(counts)*counts[inverse])


def manifest(task):
    dataset=task['task'].split('/')[0]
    if dataset=='self24ports':
        return Path('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv'),'instance_id'
    suffix={'oss2':'opensarship2_transfer_20261007','f0921':'f0921_detector_recrop_20261007'}[dataset]
    return Path('E:/Hermes/paper_tracking/sensitivity')/suffix/'target_manifest.csv','sample_id'


from pathlib import Path


def main():
    tasks=read(ROOT/'paper_core_freeze_20261013/FROZEN_REFERENCE.json')['tasks']
    if not all((OUT/'tasks'/t['task']/'FROZEN.json').exists() for t in tasks):
        raise RuntimeError('Finish all frozen tasks first')
    write(OUT/'ALL_PATHS_FROZEN.json',dict(tasks=25,draws=125,modes=MODES,new_fits=0))
    records,necessity,bridges,class_rows,cases=[],[],[],[],[]
    tables={}
    reference=pd.read_csv('E:/Hermes/paper_tracking/sensitivity/backbone_ablation_20261010/metrics.csv')
    for task in tasks:
        suffix=task['task'];dataset,name=suffix.split('/')
        dest=OUT/'tasks'/suffix
        ids=np.load(dest/'input/ids.npy',mmap_mode='r')
        rows=np.load(dest/'input/rows.npy',mmap_mode='r')
        ev=np.load(dest/'input/eval_masks.npy',mmap_mode='r')
        pred=np.load(dest/'predictions.npy',mmap_mode='r')
        ep=np.load(dest/'expert_predictions.npy',mmap_mode='r')
        parts=np.load(dest/'parts.npy',mmap_mode='r')
        path,idcol=manifest(task)
        if str(path) not in tables:
            cols=pd.read_csv(path,nrows=0).columns
            use=[k for k in (idcol,'class_id','class_name','mmsi','mmsi_or_ship_id') if k in cols]
            tables[str(path)]=pd.read_csv(path,usecols=use,keep_default_na=False,
                dtype={k:str for k in use if k!='class_id'})
        frame=tables[str(path)].iloc[rows].reset_index(drop=True)
        np.testing.assert_array_equal(frame[idcol].to_numpy(str),ids)
        yall=frame.class_id.to_numpy(int)
        for draw in range(5):
            mask=ev[draw];y=yall[mask];w=weights(y)
            meta=dict(dataset=dataset,task=name,draw=draw,n=len(y),present_classes=len(np.unique(y)))
            full0,full1=pred[draw,0,mask],pred[draw,7,mask]
            r,t=ep[draw,0,mask],ep[draw,7,mask]
            rescue=(full0!=y)&(full1==y)
            harm=(full0==y)&(full1!=y)
            basepred=np.asarray(parts[draw,0]).argmax(1)[mask]
            for layer,pp in [('full',pred),('expert',ep)]:
                start,end=pp[draw,0,mask],pp[draw,7,mask]
                rr=(start!=y)&(end==y);hh=(start==y)&(end!=y)
                for m,mode in enumerate(MODES):
                    p=pp[draw,m,mask]
                    records.append(dict(**meta,layer=layer,mode=mode,oa=100*np.mean(p==y),ba=float(w@(p==y)),
                        versus_single_ba=float(w@((p==y).astype(int)-(start==y))),
                        versus_full_ba=float(w@((p==y).astype(int)-(end==y))),
                        original_rescue_ba=float(w@rr),original_harm_ba=float(w@hh),
                        rescued_kept_ba=float(w@(rr&(p==y))),rescued_lost_ba=float(w@(rr&(p!=y))),
                        original_harm_removed_ba=float(w@(hh&(p==y))),
                        new_rescue_ba=float(w@((start!=y)&(end!=y)&(p==y))),
                        new_harm_ba=float(w@((start==y)&(end==y)&(p!=y)))))
                    if layer=='full':
                        for cl in np.unique(y):
                            cm=y==cl
                            class_rows.append(dict(**meta,mode=mode,class_id=int(cl),
                                class_name=frame.loc[frame.class_id==cl,'class_name'].iloc[0],class_n=int(cm.sum()),
                                recall=100*np.mean(p[cm]==cl),
                                delta_recall=100*np.mean((p[cm]==cl).astype(int)-(start[cm]==cl)),
                                ba_contribution=float(w[cm]@((p[cm]==cl).astype(int)-(start[cm]==cl)))))
                for mi,oldmode in [(0,'r50'),(7,'dual_refit')]:
                    key=oldmode+('_late' if layer=='full' else '_expert')
                    old=reference[(reference.dataset==dataset)&(reference.task==name)&
                                  (reference.draw==draw)&(reference['mode']==key)].iloc[0]
                    np.testing.assert_allclose(float(w@(pp[draw,mi,mask]==y)),old.ba,atol=1e-9)
            for bit,label in [(0,'prior'),(1,'reweight'),(2,'extra_relation')]:
                mode=list('111');mode[bit]='0';mode=''.join(mode)
                p=pred[draw,MODES.index(mode),mask]
                lost=rescue&(p!=y)
                rr=np.flatnonzero(mask)
                rival=full0
                delta=np.asarray(parts[draw,bit+2,rr])
                margin=delta[np.arange(len(y)),y]-delta[np.arange(len(y)),rival]
                necessity.append(dict(**meta,path=label,full_minus_removed_ba=float(w@((full1==y).astype(int)-(p==y))),
                    original_rescue_mass=float(w@rescue),lost_original_rescue=float(w@lost),
                    lost_rescue_fraction=float(w@lost/(w@rescue)) if (w@rescue)>0 else np.nan,
                    positive_rescue_margin_fraction=float(w@(rescue&(margin>0))/(w@rescue)) if (w@rescue)>0 else np.nan,
                    corrected_original_harm=float(w@(harm&(p==y)))))
            categories={'expert_rescue':(r!=y)&(t==y),'expert_harm':(r==y)&(t!=y),
                        'experts_both_correct':(r==y)&(t==y),'experts_both_wrong':(r!=y)&(t!=y)}
            masses=[]
            for label,group in categories.items():
                masses.append(float(w@(rescue&group)))
                bridges.append(dict(**meta,expert_state=label,full_rescue_ba=masses[-1],
                    full_harm_ba=float(w@(harm&group)),
                    base_correct_rescue_ba=float(w@(rescue&group&(basepred==y))),
                    base_wrong_rescue_ba=float(w@(rescue&group&(basepred!=y)))))
            np.testing.assert_allclose(sum(masses),w@rescue,atol=1e-10)
            if draw==0:
                # One typical rescue and harm per task, selected by median full-model winning margin.
                indices=np.flatnonzero(mask)
                pp=np.asarray(parts[draw])[:,indices]
                z=pp[1]+pp[2]+pp[3]+pp[4]
                prob=.5*softmax(pp[0],axis=1)+.5*softmax(z,axis=1)
                sortedp=np.sort(prob,axis=1)
                winmargin=sortedp[:,-1]-sortedp[:,-2]
                for kind,group in [('rescue',rescue),('harm',harm)]:
                    eligible=np.flatnonzero(group)
                    if not len(eligible):continue
                    k=eligible[np.argmin(abs(winmargin[eligible]-np.median(winmargin[eligible])))]
                    q=int(indices[k]);yy=int(y[k]);other=int(full0[k] if kind=='rescue' else full1[k])
                    dmargin=parts[draw,:,q,yy]-parts[draw,:,q,other]
                    cases.append(dict(dataset=dataset,task=name,draw=draw,kind=kind,query_index=q,
                        manifest_row=int(rows[q]),sample_id=str(ids[q]),true_class=yy,
                        class_name=frame.iloc[q].class_name,comparison_class=other,
                        base_class=int(basepred[k]),single_full_class=int(full0[k]),dual_full_class=int(full1[k]),
                        single_expert_class=int(r[k]),dual_expert_class=int(t[k]),
                        base_logit_margin=float(dmargin[0]),single_expert_margin=float(dmargin[1]),
                        prior_margin_delta=float(dmargin[2]),reweight_margin_delta=float(dmargin[3]),
                        extra_relation_margin_delta=float(dmargin[4]),
                        single_true_probability=float((.5*softmax(pp[0][k])+.5*softmax(pp[1][k]))[yy]),
                        dual_true_probability=float(prob[k,yy]),
                        without_prior_class=int(pred[draw,MODES.index('011'),q]),
                        without_reweight_class=int(pred[draw,MODES.index('101'),q]),
                        without_extra_relation_class=int(pred[draw,MODES.index('110'),q])))
        print('ANALYZED',suffix,flush=True)
    frames={'metrics':pd.DataFrame(records),'path_necessity':pd.DataFrame(necessity),
            'expert_fusion_bridge':pd.DataFrame(bridges),'classes':pd.DataFrame(class_rows),'cases':pd.DataFrame(cases)}
    for name,df in frames.items():df.to_csv(OUT/(name+'.csv'),index=False)
    for name,group in [('metrics',['layer','mode']),('path_necessity',['path']),
                       ('expert_fusion_bridge',['expert_state']),('classes',['mode','class_id','class_name'])]:
        df=frames[name]
        cols=df.select_dtypes(include=np.number).columns.difference(['draw','class_id'])
        bytask=df.groupby(['dataset','task']+group)[cols].mean().reset_index()
        bytask.to_csv(OUT/(name+'_by_task.csv'),index=False)
        bytask.groupby(['dataset']+group)[cols].mean().reset_index().to_csv(OUT/(name+'_summary.csv'),index=False)
    write(OUT/'PATH_ANALYSIS_COMPLETE.json',dict(tasks=25,draws=125,modes=8,
        reference_metric_replays=500,bridge_partition_checks=125,case_selection='draw0 median winning margin within rescue/harm',
        new_fits=0,causal_semantics_claimed=False))


if __name__=='__main__':main()
