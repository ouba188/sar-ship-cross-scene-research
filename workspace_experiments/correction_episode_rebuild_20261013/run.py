"""Matched source-episode experiment with unchanged target predictions."""

import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
PREV=ROOT/'correction_choice_probe_20261013'
STUDY=ROOT/'same_vessel_condition_study_20261010'
sys.path.insert(0,str(PREV))
import run_probe as old
from choice_core import features, gain_labels, class_weights, predict_probe, selected_probabilities
from episode_core import make_episodes, fit_fixed_mass

CONTEXTS=old.CONTEXTS
RECIPES={'expanded':('expanded',), 'expanded_matched_mixed':('expanded','matched_mixed'),
         'expanded_local':('expanded','local')}
MODES={'score_only':8,'contextual':16}
load=old.load


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(name,value):
    path=OUT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def source_meta():
    meta=pd.read_csv(STUDY/'observations.csv',dtype={'mmsi':str})
    folds=load(STUDY/'crossfit_evidence_20261011/source_folds.npz')
    return meta,folds,meta.iloc[folds['fit_obs_ids']].reset_index(drop=True)


def plan():
    meta,folds,frame=source_meta()
    assert not set(frame.port)&set(CONTEXTS)
    eps,availability,skipped=make_episodes(frame,folds['assignment'])
    pd.DataFrame(availability).to_csv(OUT/'availability.csv',index=False)
    pd.DataFrame(skipped,columns=['fold','port','draw','reason']).to_csv(OUT/'skipped_pairs.csv',index=False)
    write('EPISODES_FROZEN.json',dict(episodes=eps,fit_obs_ids=folds['fit_obs_ids'].tolist(),
        recipes=RECIPES,source_only=True,target_labels_used=False,skipped_pairs=skipped))
    print('PLAN',pd.Series([e['kind'] for e in eps]).value_counts().to_dict(),
          'SKIPPED',len(skipped),flush=True)


def read_context(context):
    meta,folds,frame=source_meta()
    maps=load(STUDY/'case_relation_headroom_20261012'/context/'maps.npz',['obs_ids','single'])
    bd=load(STUDY/'support_budget_fusion_select_20261012/base_cache'/context/'base.npz')
    np.testing.assert_array_equal(maps['obs_ids'],bd['obs_ids'])
    np.testing.assert_array_equal(maps['obs_ids'][:len(frame)],folds['fit_obs_ids'])
    prior=load(STUDY/'case_relation_student_20261012'/context/'source_readers.npz',['fit_prior'])['fit_prior']
    models={f:load(STUDY/'case_completion_class_control_20261012'/context/f'fold_{f}_model.npz') for f in range(3)}
    return frame,folds,maps,bd,prior,models


def compute_episode(ep,frame,maps,bd,prior,models):
    s,q=np.asarray(ep['support']),np.asarray(ep['query'])
    ix=np.r_[s,q]
    kb=old.c.base_kernel({k:bd[k][ix] for k in ('phi','mean','conditions')},
        bd['embedding'][ix].astype(float),np.arange(16),float(bd['eta']))
    sy=frame.iloc[s].class_id.to_numpy(int)
    xs,xq=maps['single'][s],maps['single'][q]
    changed=old.j.corrected_support(xs,models[ep['fold']])
    base=bd['prior_oof'][q]+kb[16:]@old.c.coefficients(kb[:16],bd['prior_oof'][s],sy,2.)
    bp=softmax(base,axis=1)
    raw=.5*(bp+softmax(prior[q]+xq@xs.T@old.c.coefficients(xs@xs.T,prior[s],sy,2.),axis=1))
    initial=.5*(bp+softmax(prior[q]+xq@changed.T@old.c.coefficients(changed@changed.T,prior[s],sy,2.),axis=1))
    groups=frame.iloc[s].mmsi.to_numpy(str)
    parts=None
    if np.bincount(sy,minlength=8).max()<4:
        assert len(np.unique(groups))==len(groups)
        parts=[]
        for repeat,seed in enumerate(old.j.utility.SEEDS):
            for fold,(train,val) in enumerate(KFold(4,shuffle=True,random_state=seed).split(sy)):
                parts.append(dict(repeat=repeat,fold=fold,train=train.tolist(),validation=val.tolist()))
    profile,detail=old.make_profile(xs,prior[s],kb[:16],bd['prior_oof'][s],sy,groups,models[ep['fold']],parts)
    detail['splitter']='unique_vessel_random4' if parts is not None else 'stratified_group4'
    return dict(raw=raw,initial=initial,x=features(raw,initial,bp,profile),
        **{'profile_'+k:v for k,v in profile.items()}),detail


def build():
    started=time.perf_counter()
    frozen=read(OUT/'EPISODES_FROZEN.json')
    units,replays=[],[]
    oldunits=[u for u in read(PREV/'SOURCE_INPUTS.json')['units'] if u['family']=='source_fit']
    for context in CONTEXTS:
        frame,folds,maps,bd,prior,models=read_context(context)
        index=pd.Index(folds['fit_obs_ids'])
        # One legacy episode per fold detects mapping/base-kernel/refit errors.
        for fold in range(3):
            u=next(u for u in oldunits if u['task']==context and u['unit']==f'{fold}_0_16')
            z=load(u['path'])
            ep=dict(fold=fold,support=index.get_indexer(z['support_ids']),query=index.get_indexer(z['query_ids']))
            actual,_=compute_episode(ep,frame,maps,bd,prior,models)
            for key in ('raw','initial','x'):
                np.testing.assert_allclose(actual[key],z[key],atol=1e-10,rtol=0)
                replays.append(dict(context=context,fold=fold,field=key,max_error=float(abs(actual[key]-z[key]).max())))
        for ep in frozen['episodes']:
            values,detail=compute_episode(ep,frame,maps,bd,prior,models)
            uid=context.replace(' ','_')+'__'+ep['kind']+'__'+ep['key']
            path=OUT/'inputs'/(uid+'.npz')
            path.parent.mkdir(parents=True,exist_ok=True)
            s,q=np.array(ep['support']),np.array(ep['query'])
            np.savez_compressed(path,**values,support_ids=folds['fit_obs_ids'][s],
                query_ids=folds['fit_obs_ids'][q],labels=frame.iloc[q].class_id.to_numpy(int))
            units.append(dict(context=context,uid=uid,path=str(path),kind=ep['kind'],key=ep['key'],
                fold=ep['fold'],port=ep['port'],draw=ep['draw'],query_n=len(q),
                support_ports=frame.iloc[s].port.nunique(),query_ports=frame.iloc[q].port.nunique(),
                support_classes=frame.iloc[s].class_id.nunique(),query_classes=frame.iloc[q].class_id.nunique(),
                oof_splitter=detail['splitter']))
        print('BUILT',context,len(frozen['episodes']),'episodes',flush=True)
    pd.DataFrame(replays).to_csv(OUT/'readout_replays.csv',index=False)
    pd.DataFrame(units).to_csv(OUT/'training_units.csv',index=False)
    write('SOURCE_READY.json',dict(units=units,seconds=time.perf_counter()-started))


def fit():
    started=time.perf_counter()
    units=read(OUT/'SOURCE_READY.json')['units']
    meta,_,_=source_meta()
    orig=read(PREV/'FROZEN_MODELS.json')
    mass={m['context']:m['events'] for m in orig['models'] if m['name']=='score_only'}
    rows,coverage,events,fit_replays=[],[],[],[]
    # Preserve effective regularization and reproduce the original fitting recipe.
    ous=[u for u in read(PREV/'SOURCE_INPUTS.json')['units'] if u['family']=='source_fit']
    for context in CONTEXTS:
        xx,gg,ww=[],[],[]
        for u in ous:
            if u['task']!=context: continue
            z=load(u['path']); y=meta.iloc[z['query_ids']].class_id.to_numpy(int)
            xx.append(z['x']);gg.append(gain_labels(z['raw'],z['initial'],y));ww.append(class_weights(y))
        for mode,width in MODES.items():
            x,g,w=map(np.concatenate,(xx,gg,ww))
            m=fit_fixed_mass(x,g,w,width,mass[context])
            previous=load(PREV/(context+'__'+mode+'.npz'))
            error=float(abs(predict_probe(m,x)-predict_probe(previous,x)).max())
            np.testing.assert_allclose(predict_probe(m,x),predict_probe(previous,x),atol=1e-9,rtol=0)
            fit_replays.append(dict(context=context,mode=mode,max_probability_error=error))
    for recipe,kinds in RECIPES.items():
        for context in ('pooled',)+CONTEXTS:
            selected=[u for u in units if u['kind'] in kinds and (context=='pooled' or u['context']==context)]
            xx,gg,ww,ids=[],[],[],[]
            for u in selected:
                z=load(u['path']);y=z['labels'];g=gain_labels(z['raw'],z['initial'],y)
                xx.append(z['x']);gg.append(g);ww.append(class_weights(y)/len(selected));ids.append(z['query_ids'])
                for pos in np.flatnonzero(g):
                    oid=int(z['query_ids'][pos]); row=meta.iloc[oid]
                    events.append(dict(recipe=recipe,context=context,uid=u['uid'],obs_id=oid,
                        ship=str(row.mmsi),class_id=int(y[pos]),gain=int(g[pos]),weight=float(ww[-1][pos])))
            x,g,w,oid=map(np.concatenate,(xx,gg,ww,ids))
            coverage.append(dict(recipe=recipe,context=context,units=len(selected),query_exposures=len(g),
                query_observations=len(np.unique(oid)),query_ships=meta.iloc[np.unique(oid)].mmsi.nunique(),
                event_exposures=int((g!=0).sum()),event_observations=len(np.unique(oid[g!=0])),
                event_ships=meta.iloc[np.unique(oid[g!=0])].mmsi.nunique()))
            for mode,width in MODES.items():
                m=fit_fixed_mass(x,g,w,width,mass[context])
                dest=OUT/'models'/f'{recipe}__{context}__{mode}.npz'
                dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez(dest,**m)
                rows.append(dict(recipe=recipe,context=context,mode=mode,path=str(dest),
                    events=m['events'],event_weight_mass=m['event_weight_mass'],iterations=m['iterations']))
            print('FIT',recipe,context,'events',int((g!=0).sum()),flush=True)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'training_events.csv',index=False)
    pd.DataFrame(fit_replays).to_csv(OUT/'fit_replays.csv',index=False)
    write('FROZEN_MODELS.json',dict(models=rows,recipes=RECIPES,C=.1,threshold=.5,
        event_weight_mass=mass,held_target_labels_used=False,seconds=time.perf_counter()-started))


def predict():
    units=[u for u in read(PREV/'PREDICTIONS_FROZEN.json')['units'] if u['family']!='source_fit']
    models=read(OUT/'FROZEN_MODELS.json')['models']
    fitted={(m['recipe'],m['context'],m['mode']):load(m['path']) for m in models}
    for u in units:
        z=load(u['path']);context='pooled' if u['family']=='external' else u['task']
        answer={recipe+'__'+mode:predict_probe(fitted[(recipe,context,mode)],z['x'])
                for recipe in RECIPES for mode in MODES}
        dest=OUT/'decisions'/(u['uid']+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,query_ids=z['query_ids'],**answer)
    write('PREDICTIONS_FROZEN.json',dict(units=units,held_target_labels_used=False,
        reused_development_data=True,new_selector_fits=len(models),new_neural_fits=0))
    print('FROZEN',len(units),'evaluation units',flush=True)


def evaluate():
    units=read(OUT/'PREDICTIONS_FROZEN.json')['units']
    labels={};rows=[];classrows=[]
    for u in units:
        if u['label_path'] not in labels:
            labels[u['label_path']]=pd.read_csv(u['label_path'],usecols=['class_id'])
        z=load(u['path']);y=labels[u['label_path']].iloc[z['query_ids']].class_id.to_numpy(int)
        d=load(OUT/'decisions'/(u['uid']+'.npz'))
        np.testing.assert_array_equal(d['query_ids'],z['query_ids'])
        previous=load(PREV/'decisions'/(u['uid']+'.npz'))
        g=gain_labels(z['raw'],z['initial'],y);w=class_weights(y);event=g!=0
        key={k:u[k] for k in ('family','task','cohort','uid')}
        scores={**{k:v for k,v in d.items() if k!='query_ids'},
                **{'old__'+mode:previous[mode] for mode in MODES},'confidence':previous['confidence']}
        for name,value in scores.items():
            choice=value>(0 if name=='confidence' else .5)
            p=selected_probabilities(z['raw'],z['initial'],choice)
            result=old.metrics(y,p,z['raw'],z['initial'])
            np.testing.assert_allclose(result['versus_initial_ba'],result['harm_avoided_ba']-result['rescue_lost_ba'],atol=1e-10)
            av=float(roc_auc_score(g[event]>0,value[event],sample_weight=w[event])) if len(np.unique(g[event]))==2 else np.nan
            rows.append(dict(**key,arm=name,auc=av,choose_initial=float(choice.mean()),**result))
            for cls in np.unique(y):
                keep=y==cls
                classrows.append(dict(**key,arm=name,class_id=int(cls),query_n=int(keep.sum()),
                    recall=float(100*np.mean(p.argmax(1)[keep]==cls)),
                    ba_contribution_vs_initial=float(100*w[keep]@((p.argmax(1)[keep]==cls).astype(int)-(z['initial'].argmax(1)[keep]==cls)))))
        for arm in ('raw','initial'):
            rows.append(dict(**key,arm=arm,auc=np.nan,choose_initial=np.nan,**old.metrics(y,z[arm],z['raw'],z['initial'])))
    data=pd.DataFrame(rows);data.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(classrows).to_csv(OUT/'classes.csv',index=False)
    measures=['oa','ba','delta_ba','versus_initial_oa','versus_initial_ba','harm_avoided_ba','rescue_lost_ba','auc','choose_initial']
    per_task=data.groupby(['family','task','cohort','arm'])[measures].mean().reset_index()
    per_task.to_csv(OUT/'task_summary.csv',index=False)
    data.groupby(['family','task','cohort','arm']).agg(sd=('versus_initial_ba','std'),
        positive=('versus_initial_ba',lambda x:int((x>1e-10).sum())),units=('uid','size')).reset_index().to_csv(OUT/'draw_variation.csv',index=False)
    # Source contexts share query ships; summarize them as contexts, not independent ports.
    per_task['scope']=np.where(per_task.family=='external',per_task.task,per_task.family)
    summary=per_task.groupby(['scope','cohort','arm'])[measures].mean().reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    write('COMPLETE.json',dict(status='complete',new_neural_fits=0,new_selector_fits=24,
        evaluation_units=len(units),main_model_replaced=False,scope='B16 reused development tasks'))
    print(summary[['scope','cohort','arm','versus_initial_ba','auc']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','build','fit','predict','evaluate'])
    stage=parser.parse_args().stage
    with old.c.threadpool_limits(limits=2):
        globals()[stage]()
