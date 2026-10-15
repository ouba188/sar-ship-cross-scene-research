"""Choose four frozen integration recipes using only the original 64 labels."""

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
from pathlib import Path
import json
import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from scipy.special import softmax
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
PREVIOUS=OUT.parent/'residual_integration'
sys.path.insert(0,str(PREVIOUS))
import experiment as old

ARMS=old.ARMS
SEEDS=(20261012,20261013,20261014)
PRIMARY='cv_oa'


def folds(y, groups, seed):
    splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed)
    return list(splitter.split(np.zeros((len(y),1)),y,groups))


def fold_predict(stats,h,pe,ke,pb,kb,train,labels):
    delta=old.increment(stats,h[train],labels,h)
    expert=old.integrate(pe,ke[:,train],train,labels,delta)
    base=pb+kb[:,train]@old.coefficients(kb[np.ix_(train,train)],pb[train],labels,8.)
    return .5*(softmax(base,axis=1)[None]+softmax(expert,axis=-1))


def pick(metrics,criterion='oa'):
    lookup={r['arm']:r for r in metrics}
    if criterion=='nll':
        return min(ARMS,key=lambda a:(round(lookup[a]['nll'],12),ARMS.index(a)))
    eligible=[a for a in ARMS if lookup[a]['ba']>=lookup['original']['ba']-1e-10]
    return min(eligible,key=lambda a:(-round(lookup[a]['oa'],10),-round(lookup[a]['ba'],10),
                                      round(lookup[a]['macro_nll'],12),ARMS.index(a)))


def support_groups(folder,data):
    info=json.loads((folder/'PREPARED.json').read_text())
    if info['dataset']=='self24ports':
        frame=pd.read_csv(old.SOURCE/'manifest.csv',usecols=['instance_id','mmsi'],
                          dtype=str,keep_default_na=False).iloc[data['rows']]
        idcol,shipcol='instance_id','mmsi'
    else:
        frame=pd.read_csv(old.ROOT/old.DATASETS['oss2'][0]/'target_manifest.csv',
            usecols=['sample_id','mmsi_or_ship_id'],dtype=str,keep_default_na=False).iloc[data['rows']]
        idcol,shipcol='sample_id','mmsi_or_ship_id'
    np.testing.assert_array_equal(frame[idcol].to_numpy(str),data['ids'])
    ships=frame[shipcol].str.strip()
    known=ships.str.fullmatch(r'[0-9]{9}')&ships.ne('000000000')
    groups=np.where(known,'ship:'+ships,'sample:'+frame[idcol]).astype(str)
    return groups,known.to_numpy()


def select():
    selected,metrics,per_repeat,inventory,fold_info=[],[],[],[],[]
    for folder in old.FOLDERS:
        info=json.loads((folder/'PREPARED.json').read_text())
        task=info['task']
        old.previous.MATCHED=info['dataset']=='oss2'
        ctx=old.ExpertContext(folder)
        data=old.load_task(folder)
        hs,ht=old.head_inputs(ctx,data)
        stats=old.source_statistics(hs,ctx.source_labels)
        groups,known=support_groups(folder,data)
        matched=None
        if info['dataset']=='oss2':
            matched=old.load_npz(old.ROOT/'backbone_ablation_20261010/targets/oss2/oss2/coefficients.npz')
        for draw,support in enumerate(data['support']):
            y=data['support_labels'][draw]
            g=groups[support]
            pe,ke=ctx.target_components(data,support)
            if matched is not None:
                pe=(ht[:,:-1]@matched['dual_coef']+matched['dual_intercept']).astype(np.float32).astype(float)
            h,pe,ke=ht[support],pe[support],ke[support]
            pb=data['prior'][support]
            kb=old.base_cross(data,support,support,info['eta'])
            oof=np.full((len(SEEDS),len(ARMS),64,8),np.nan)
            fold_ids=np.full((len(SEEDS),64),-1,int)
            seen=np.zeros((len(SEEDS),64),int)
            for repeat,seed in enumerate(SEEDS):
                with warnings.catch_warnings(record=True) as notes:
                    parts=folds(y,g,seed)
                for fold,(train,val) in enumerate(parts):
                    if set(g[train]) & set(g[val]):
                        raise ValueError('Same vessel straddles training and validation')
                    predictions=fold_predict(stats,h,pe,ke,pb,kb,train,y[train])
                    oof[repeat][:,val,:]=predictions[:,val,:]
                    fold_ids[repeat,val]=fold
                    seen[repeat,val]+=1
                    missing=np.setdiff1d(np.unique(y[val]),np.unique(y[train]))
                    fold_info.append(dict(task=task,draw=draw,repeat=repeat,seed=seed,fold=fold,
                        train_n=len(train),val_n=len(val),train_classes=len(np.unique(y[train])),
                        val_classes=len(np.unique(y[val])),missing_train_classes=missing.tolist(),
                        val_without_class_support=int(np.isin(y[val],missing).sum()),
                        split_warnings='; '.join(str(n.message) for n in notes)))
                np.testing.assert_array_equal(seen[repeat],np.ones(64,int))
                row_metrics=[dict(arm=a,**old.metric(y,oof[repeat,i])) for i,a in enumerate(ARMS)]
                per_repeat.append(dict(task=task,draw=draw,repeat=repeat,seed=seed,
                                       cv_oa=pick(row_metrics),cv_nll=pick(row_metrics,'nll')))
            if not np.isfinite(oof).all():
                raise ValueError('Incomplete internal validation predictions')
            yy=np.tile(y,len(SEEDS))
            pooled=oof.transpose(1,0,2,3).reshape(len(ARMS),-1,8)
            rows=[dict(arm=a,**old.metric(yy,pooled[i])) for i,a in enumerate(ARMS)]
            metrics.extend(dict(task=task,draw=draw,**r) for r in rows)
            selected.append(dict(task=task,draw=draw,cv_oa=pick(rows),cv_nll=pick(rows,'nll')))
            unique_predictions=np.unique(pooled.argmax(-1),axis=0).shape[0]
            inventory.append(dict(task=task,draw=draw,support_n=64,groups=len(np.unique(g)),
                known_ships=len(np.unique(g[known[support]])),unknown_rows=int((~known[support]).sum()),
                support_classes=len(np.unique(y)),counts=np.bincount(y,minlength=8).tolist(),
                distinct_candidate_prediction_vectors=unique_predictions,
                candidate_disagreement_fraction=float(np.any(pooled.argmax(-1)!=pooled[0].argmax(-1)[None],axis=0).mean())))
            dest=OUT/'oof'/task
            dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/f'draw_{draw}.npz',oof=oof,y=y,groups=g,fold_ids=fold_ids,
                support=support,support_ids=data['ids'][support],arms=np.array(ARMS),seeds=np.array(SEEDS))
            print('CV SELECTED',task,draw,selected[-1]['cv_oa'],selected[-1]['cv_nll'],flush=True)
        del ctx,data,hs,ht,stats
        gc.collect()
    pd.DataFrame(metrics).to_csv(OUT/'cv_metrics.csv',index=False)
    pd.DataFrame(per_repeat).to_csv(OUT/'repeat_selections.csv',index=False)
    pd.DataFrame(inventory).to_csv(OUT/'support_inventory.csv',index=False)
    pd.DataFrame(fold_info).to_csv(OUT/'fold_inventory.csv',index=False)
    old.write(OUT/'SELECTION_FROZEN.json',dict(selected=selected,seeds=SEEDS,folds=4,
        primary=PRIMARY,target_query_labels_used=False,new_labels=0,neural_training=False,
        primary_criterion='OOF OA with nondecreasing OOF BA; BA/macro NLL/small-change tie break',
        diagnostic_criterion='OOF micro NLL minimum, not a post-hoc replacement for primary'))


def evaluate():
    selections=json.loads((OUT/'SELECTION_FROZEN.json').read_text())['selected']
    source=json.loads((PREVIOUS/'SOURCE_SELECTION.json').read_text())['selected']
    chosen={(r['task'],r['draw']):r for r in selections}
    repeats=pd.read_csv(OUT/'repeat_selections.csv')
    rows,classes,regrets,ranking,repeat_evaluation=[],[],[],[],[]
    cv=pd.read_csv(OUT/'cv_metrics.csv')
    for folder in old.FOLDERS:
        info=json.loads((folder/'PREPARED.json').read_text())
        task=info['task']
        z=old.load_npz(PREVIOUS/'target'/task/'predictions.npz')
        if info['dataset']=='self24ports':
            frame=pd.read_csv(old.SOURCE/'manifest.csv',usecols=['instance_id','class_id']).iloc[z['rows']]
            idcol='instance_id'
        else:
            frame=pd.read_csv(old.ROOT/old.DATASETS['oss2'][0]/'target_manifest.csv',
                usecols=['sample_id','class_id']).iloc[z['rows']]
            idcol='sample_id'
        np.testing.assert_array_equal(frame[idcol].to_numpy(str),z['ids'])
        y=frame.class_id.to_numpy(int)
        for draw in range(5):
            saved=old.load_npz(OUT/'oof'/task/f'draw_{draw}.npz')
            np.testing.assert_array_equal(saved['support'],z['support'][draw])
            np.testing.assert_array_equal(saved['support_ids'],z['ids'][z['support'][draw]])
            np.testing.assert_array_equal(saved['y'],y[z['support'][draw]])
            if z['eval_masks'][draw,z['support'][draw]].any():
                raise ValueError('Support in query')
            policies={a:a for a in ARMS}
            policies.update(source_selected=source[task]['arm'],
                            cv_oa=chosen[task,draw]['cv_oa'],cv_nll=chosen[task,draw]['cv_nll'])
            for scope,mask in [('original',z['eval_masks'][draw]),('common',z['common_eval'])]:
                yy=y[mask]
                probs=z['full'][draw][:,mask]
                baseline=old.metric(yy,probs[0])
                all_metrics=[dict(arm=a,**old.metric(yy,probs[i])) for i,a in enumerate(ARMS)]
                lookup={m['arm']:m for m in all_metrics}
                oracle=max(lookup[a]['oa'] for a in ARMS)
                for policy,arm in policies.items():
                    p=probs[ARMS.index(arm)]
                    pred,orig=p.argmax(1),probs[0].argmax(1)
                    mm=lookup[arm]
                    meta=dict(task=task,draw=draw,scope=scope,policy=policy,arm=arm)
                    rows.append(dict(**meta,n=len(yy),**{k:mm[k] for k in baseline},
                        **{'delta_'+k:mm[k]-baseline[k] for k in baseline},
                        rescues=int(((orig!=yy)&(pred==yy)).sum()),harms=int(((orig==yy)&(pred!=yy)).sum())))
                    for c in np.unique(yy):
                        ix=yy==c
                        classes.append(dict(**meta,class_id=int(c),n=int(ix.sum()),
                            recall=float(100*(pred[ix]==c).mean()),
                            recall_delta=float(100*((pred[ix]==c).mean()-(orig[ix]==c).mean()))))
                    if policy in ('cv_oa','cv_nll','source_selected'):
                        regrets.append(dict(**meta,oracle_oa=oracle,
                            oracle_gain=oracle-baseline['oa'],regret_oa=oracle-mm['oa']))
                if scope=='original':
                    internal=cv[(cv.task==task)&(cv.draw==draw)].set_index('arm')
                    for arm in ARMS:
                        ranking.append(dict(task=task,draw=draw,arm=arm,
                            cv_delta_oa=internal.loc[arm,'oa']-internal.loc['original','oa'],
                            cv_delta_ba=internal.loc[arm,'ba']-internal.loc['original','ba'],
                            cv_delta_nll=internal.loc[arm,'nll']-internal.loc['original','nll'],
                            target_delta_oa=lookup[arm]['oa']-baseline['oa'],
                            target_delta_ba=lookup[arm]['ba']-baseline['ba'],
                            target_delta_nll=lookup[arm]['nll']-baseline['nll']))
                    for _,r in repeats[(repeats.task==task)&(repeats.draw==draw)].iterrows():
                        for policy in ('cv_oa','cv_nll'):
                            arm=r[policy]
                            repeat_evaluation.append(dict(task=task,draw=draw,repeat=int(r['repeat']),policy=policy,
                                arm=arm,delta_oa=lookup[arm]['oa']-baseline['oa'],delta_ba=lookup[arm]['ba']-baseline['ba']))
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_recalls.csv',index=False)
    pd.DataFrame(regrets).to_csv(OUT/'selection_regret.csv',index=False)
    pd.DataFrame(ranking).to_csv(OUT/'cv_target_alignment.csv',index=False)
    pd.DataFrame(repeat_evaluation).to_csv(OUT/'repeat_evaluation.csv',index=False)
    summaries=[]
    for (task,scope,policy),f in frame.groupby(['task','scope','policy'],sort=False):
        row=dict(task=task,scope=scope,policy=policy)
        for k in ('oa','ba','nll','brier','macro_nll','delta_oa','delta_ba','delta_nll','delta_brier','rescues','harms'):
            row[k]=float(f[k].mean())
        for k in ('oa','ba'):
            d=f['delta_'+k]
            row[k+'_delta_sd']=float(d.std(ddof=1))
            row[k+'_wins']=int((d>1e-10).sum())
            row[k+'_losses']=int((d<-1e-10).sum())
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'summary.csv',index=False)
    print(summary[(summary.scope=='original')&summary.policy.isin(['original','source_selected','cv_oa','cv_nll'])]
        [['task','policy','oa','ba','delta_oa','delta_ba','delta_nll']].to_string(index=False),flush=True)
    old.write(OUT/'COMPLETE.json',dict(target_tasks=4,support_draws=20,fold_fits=240,
        annotations_per_draw=64,new_annotations=0,neural_training=False,
        full64_predictions_reused_from_previous=True,main_thread_unchanged=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['select','evaluate'])
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        {'select':select,'evaluate':evaluate}[args.phase]()
