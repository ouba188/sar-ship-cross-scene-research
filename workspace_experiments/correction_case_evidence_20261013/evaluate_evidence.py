"""Fixed source-only evidence probes and matched full-task readout."""

import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
PREV=ROOT/'correction_episode_rebuild_20261013'
sys.path.insert(0,str(ROOT/'correction_choice_probe_20261013'))
from choice_core import class_weights, gain_labels, predict_probe, selected_probabilities
from evidence_core import NAMES

CONTEXTS=('Los Angeles','Port Klang','Santos')
ARMS={'cases':np.arange(8),'sensitivity':np.arange(8,14),'both':np.arange(14)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def source_units():
    return [u for u in read(PREV/'SOURCE_READY.json')['units'] if u['kind']=='expanded']


def get(u):
    z=load(u['path']);e=load(OUT/'features'/(u['uid']+'.npz'))
    np.testing.assert_array_equal(z['query_ids'],e['query_ids'])
    return z,e['evidence']


def reference(ctx):
    return load(PREV/'models'/f'expanded__{ctx}__contextual.npz')


def auc(g,s,w):
    keep=g!=0
    return float(roc_auc_score(g[keep]>0,s[keep],sample_weight=w[keep])) if len(np.unique(g[keep]))==2 else np.nan


def fit():
    started=time.perf_counter()
    infos,replays,training,coefficients=[],[],[],[]
    (OUT/'models').mkdir(exist_ok=True)
    for ctx in ('pooled',)+CONTEXTS:
        units=[u for u in source_units() if ctx=='pooled' or u['context']==ctx]
        data=[get(u) for u in units];ref=reference(ctx)
        xx=np.concatenate([z['x'] for z,e in data])
        ev=np.concatenate([e for z,e in data])
        g=np.concatenate([gain_labels(z['raw'],z['initial'],z['labels']) for z,e in data])
        w=np.concatenate([class_weights(z['labels'])/len(data) for z,e in data])
        x=(xx-ref['mean'])/ref['scale']
        mean,scale=ev.mean(0),ev.std(0)
        scale=np.where(scale<1e-8,1.,scale)
        ne=(ev-mean)/scale
        keep=g!=0
        weights=w[keep]*(float(ref['event_weight_mass'])/w[keep].sum())
        for arm,cols in {'reference':None,**ARMS}.items():
            design=x if cols is None else np.c_[x,ne[:,cols]]
            model=LogisticRegression(C=.1,solver='lbfgs',max_iter=1000,tol=1e-9)
            model.fit(design[keep],g[keep]>0,sample_weight=weights)
            score=model.predict_proba(design)[:,1]
            if arm=='reference':
                expected=predict_probe(ref,xx)
                np.testing.assert_allclose(score,expected,atol=1e-9,rtol=0)
                replays.append(dict(context=ctx,error=float(abs(score-expected).max())))
            else:
                path=OUT/'models'/f'{ctx}__{arm}.npz'
                np.savez(path,coef=model.coef_[0],intercept=model.intercept_[0],mean=mean,scale=scale,cols=cols)
                infos.append(dict(context=ctx,arm=arm,path=str(path),events=int(keep.sum()),iterations=int(model.n_iter_[0])))
                coefficients.extend(dict(context=ctx,arm=arm,feature=NAMES[k],coefficient=float(v))
                                    for k,v in zip(cols,model.coef_[0,16:]))
            training.append(dict(context=ctx,arm=arm,auc=auc(g,score,w),micro_auc=auc(g,score,np.ones(len(w))),
                                 more_count_auc=auc(g[xx[:,13]>xx[:,12]],score[xx[:,13]>xx[:,12]],w[xx[:,13]>xx[:,12]])))
        print('FIT',ctx,flush=True)
    pd.DataFrame(replays).to_csv(OUT/'fit_replays.csv',index=False)
    pd.DataFrame(training).to_csv(OUT/'training_auc.csv',index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'new_coefficients.csv',index=False)
    write('FROZEN_MODELS.json',dict(models=infos,new_fits=12,reference_refits=4,target_labels_used=False,
                                   C=.1,threshold=.5,seconds=time.perf_counter()-started))


def predict():
    units=read(PREV/'PREDICTIONS_FROZEN.json')['units']
    models={(m['context'],m['arm']):load(m['path']) for m in read(OUT/'FROZEN_MODELS.json')['models']}
    (OUT/'decisions').mkdir(exist_ok=True)
    for u in units:
        ctx='pooled' if u['family']=='external' else u['task']
        z,e=get(u);ref=reference(ctx);x=(z['x']-ref['mean'])/ref['scale']
        values={}
        for arm in ARMS:
            m=models[ctx,arm];ne=(e-m['mean'])/m['scale']
            values[arm]=expit(np.c_[x,ne[:,m['cols']]]@m['coef']+m['intercept'])
        np.savez_compressed(OUT/'decisions'/(u['uid']+'.npz'),query_ids=z['query_ids'],**values)
    write('PREDICTIONS_FROZEN.json',dict(units=units,target_labels_used=False))
    print('PREDICTIONS FROZEN',len(units),flush=True)


def summarize(frame,fields,name):
    task=frame.groupby(['family','task','cohort','arm'])[fields].mean().reset_index()
    task.to_csv(OUT/(name+'_task.csv'),index=False)
    task['scope']=np.where(task.family=='external',task.task,task.family)
    summary=task.groupby(['scope','cohort','arm'])[fields].mean().reset_index()
    summary.to_csv(OUT/(name+'.csv'),index=False)
    return summary


def metrics(y,p,initial,w):
    delta=(p.argmax(1)==y).astype(int)-(initial.argmax(1)==y)
    return dict(oa=float(100*np.mean(p.argmax(1)==y)),ba=float(100*w@(p.argmax(1)==y)),
                versus_initial_oa=float(100*delta.mean()),versus_initial_ba=float(100*w@delta))


def evaluate():
    units=read(OUT/'PREDICTIONS_FROZEN.json')['units']
    tables={};rows=[];classes=[];pairs=[];checks=[];strata=[];diagnostics=[]
    anchors=pd.read_csv(ROOT/'correction_objective_balance_20261013/metrics.csv').set_index(['uid','arm'])
    for u in units:
        z,e=get(u);d=load(OUT/'decisions'/(u['uid']+'.npz'))
        np.testing.assert_array_equal(z['query_ids'],d['query_ids'])
        if u['label_path'] not in tables:
            tables[u['label_path']]=pd.read_csv(u['label_path'],usecols=['class_id']).class_id.to_numpy(int)
        y=tables[u['label_path']][z['query_ids']];w=class_weights(y)
        g=gain_labels(z['raw'],z['initial'],y)
        old=load(PREV/'decisions'/(u['uid']+'.npz'))['expanded__contextual']
        scores={a:d[a] for a in ARMS}
        scores.update(reference=old,confidence=z['initial'].max(1)-z['raw'].max(1))
        probs={a:selected_probabilities(z['raw'],z['initial'],s>(0. if a=='confidence' else .5)) for a,s in scores.items()}
        probs.update({a:z[a] for a in ('raw','initial','combined') if a in z})
        key={k:u[k] for k in ('uid','family','task','cohort')};actuals={}
        more=z['x'][:,13]>z['x'][:,12]
        for arm,p in probs.items():
            m=metrics(y,p,z['initial'],w);pred=p.argmax(1)
            avoided=float(100*w@((g==-1)&(pred==y)))
            lost=float(100*w@((g==1)&(pred!=y)))
            if arm in scores:
                np.testing.assert_allclose(m['versus_initial_ba'],avoided-lost,atol=1e-10)
            row=dict(**key,arm=arm,**m,harm_avoided_ba=avoided,rescue_lost_ba=lost,
                     auc=auc(g,scores[arm],w) if arm in scores else np.nan,
                     micro_auc=auc(g,scores[arm],np.ones(len(y))) if arm in scores else np.nan)
            rows.append(row);actuals[arm]=m
            if arm in ('reference','confidence','raw','initial'):
                anchor=anchors.loc[(u['uid'],'ba' if arm=='reference' else arm)]
                for field in m:
                    err=abs(m[field]-anchor[field]);assert err<1e-9
                    checks.append(dict(uid=u['uid'],arm=arm,field=field,error=err))
            parts=[]
            delta=(pred==y).astype(int)-(z['initial'].argmax(1)==y)
            for cls in np.unique(y):
                keep=y==cls
                v=dict(**key,arm=arm,class_id=int(cls),n=int(keep.sum()),recall=float(100*np.mean(pred[keep]==cls)),
                       oa_contribution=float(100*delta[keep].sum()/len(y)),ba_contribution=float(100*w[keep]@delta[keep]),
                       rescue_lost=int(((g==1)&(pred!=y)&keep).sum()),harm_avoided=int(((g==-1)&(pred==y)&keep).sum()))
                parts.append(v);classes.append(v)
            np.testing.assert_allclose([sum(v['oa_contribution'] for v in parts),sum(v['ba_contribution'] for v in parts)],
                                       [m['versus_initial_oa'],m['versus_initial_ba']],atol=1e-10)
            for group,keep in [('initial_more_support',more),('other',~more)]:
                strata.append(dict(**key,arm=arm,group=group,n=int(keep.sum()),
                    ba_contribution=float(100*w[keep]@delta[keep]),oa_contribution=float(100*delta[keep].sum()/len(y)),
                    auc=auc(g[keep],scores[arm][keep],w[keep]) if arm in scores else np.nan))
        for arm in ARMS:
            for control in ('reference','confidence','initial','raw','combined'):
                if control in actuals:
                    pairs.append(dict(**key,arm=arm+'__minus__'+control,
                                      oa=actuals[arm]['oa']-actuals[control]['oa'],ba=actuals[arm]['ba']-actuals[control]['ba']))
        for group,mask in [('all',np.ones(len(y),bool)),('initial_more_support',more)]:
            for sign,name in [(1,'rescue'),(-1,'harm')]:
                keep=mask&(g==sign)
                if not keep.any():continue
                ew=w[keep]/w[keep].sum()
                for k,nm in enumerate(NAMES):
                    diagnostics.append(dict(**key,group=group,event=name,feature=nm,n=int(keep.sum()),
                                            mean=float(e[keep,k].mean()),ba_weighted_mean=float(ew@e[keep,k])))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'classes.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'metric_replays.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'event_evidence.csv',index=False)
    summary=summarize(frame,['oa','ba','versus_initial_oa','versus_initial_ba','auc','micro_auc','harm_avoided_ba','rescue_lost_ba'],'summary')
    pair=pd.DataFrame(pairs);pair.to_csv(OUT/'comparisons.csv',index=False)
    summarize(pair,['oa','ba'],'comparison_summary')
    strata=pd.DataFrame(strata);strata.to_csv(OUT/'strata.csv',index=False)
    strata.groupby(['family','task','cohort','arm','group'])[['ba_contribution','oa_contribution','auc']].mean().reset_index().to_csv(OUT/'strata_summary.csv',index=False)
    frame.groupby(['family','task','cohort','arm']).agg(oa_sd=('versus_initial_oa','std'),ba_sd=('versus_initial_ba','std'),
        oa_positive=('versus_initial_oa',lambda x:int((x>1e-10).sum())),ba_positive=('versus_initial_ba',lambda x:int((x>1e-10).sum())),
        units=('uid','size')).reset_index().to_csv(OUT/'variation.csv',index=False)
    write('EVALUATION_COMPLETE.json',dict(units=len(units),new_fits=12,reference_refits=4,neural_fits=0,
            metric_replays=len(checks),class_sum_checks=2*len(frame),main_replaced=False))
    print(summary.query("arm in ['reference','cases','sensitivity','both']").to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fit','predict','evaluate'])
    globals()[parser.parse_args().stage]()
