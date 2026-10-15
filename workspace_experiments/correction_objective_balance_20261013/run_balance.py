"""Fixed objective replacement on identical source episodes and normalization."""

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
PREV=ROOT/'correction_episode_rebuild_20261013'
CHOICE=ROOT/'correction_choice_probe_20261013'
sys.path.insert(0,str(CHOICE))
from choice_core import class_weights,gain_labels,predict_probe,selected_probabilities

CONTEXTS=('Los Angeles','Port Klang','Santos')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def write(name,value):
    dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def fit_fixed(x,gain,weight,reference):
    keep=gain!=0
    w=weight[keep]
    mass=float(reference['event_weight_mass'])
    model=LogisticRegression(C=.1,solver='lbfgs',max_iter=1000,tol=1e-9)
    model.fit((x[keep]-reference['mean'])/reference['scale'],gain[keep]>0,
              sample_weight=w*(mass/w.sum()))
    return dict(mean=reference['mean'],scale=reference['scale'],coef=model.coef_[0],
        intercept=model.intercept_[0],width=16,event_weight_mass=mass,
        events=int(keep.sum()),iterations=int(model.n_iter_[0]))


def fit():
    start=time.perf_counter()
    units=[u for u in read(PREV/'SOURCE_READY.json')['units'] if u['kind']=='expanded']
    info=[];replays=[];weights=[]
    for context in ('pooled',)+CONTEXTS:
        selected=[u for u in units if context=='pooled' or u['context']==context]
        xx,gg,ba,oa=[],[],[],[]
        for u in selected:
            z=load(u['path']);y=z['labels'];g=gain_labels(z['raw'],z['initial'],y)
            xx.append(z['x']);gg.append(g)
            ba.append(class_weights(y)/len(selected));oa.append(np.full(len(y),1/(len(y)*len(selected))))
            for cls in np.unique(y):
                keep=y==cls
                weights.append(dict(context=context,uid=u['uid'],class_id=int(cls),query_n=int(keep.sum()),
                    oa_mass=float(oa[-1][keep].sum()),ba_mass=float(ba[-1][keep].sum()),
                    oa_event_mass=float(oa[-1][keep&(g!=0)].sum()),ba_event_mass=float(ba[-1][keep&(g!=0)].sum())))
        x,g,wb,wo=map(np.concatenate,(xx,gg,ba,oa))
        ref=load(PREV/'models'/f'expanded__{context}__contextual.npz')
        replay=fit_fixed(x,g,wb,ref)
        err=float(abs(predict_probe(replay,x)-predict_probe(ref,x)).max())
        np.testing.assert_allclose(predict_probe(replay,x),predict_probe(ref,x),atol=1e-9,rtol=0)
        replays.append(dict(context=context,max_probability_error=err))
        for name,w in [('micro',wo),('joint',.5*(wo+wb))]:
            fitted=fit_fixed(x,g,w,ref)
            dest=OUT/'models'/f'{context}__{name}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez(dest,**fitted)
            info.append(dict(context=context,objective=name,path=str(dest),events=int(fitted['events']),
                event_weight_mass=float(fitted['event_weight_mass'])))
        print('FIT',context,flush=True)
    pd.DataFrame(replays).to_csv(OUT/'fit_replays.csv',index=False)
    pd.DataFrame(weights).to_csv(OUT/'training_class_weights.csv',index=False)
    pd.DataFrame(weights).groupby(['context','class_id'])[
        ['oa_mass','ba_mass','oa_event_mass','ba_event_mass']].sum().reset_index().to_csv(OUT/'training_weight_summary.csv',index=False)
    write('FROZEN_MODELS.json',dict(models=info,seconds=time.perf_counter()-start,new_selector_fits=8,
        reference_refits=4,held_target_labels_used=False,normalization='frozen expanded BA reference',
        joint_macro_fraction=.5,C=.1,threshold=.5))


def predict():
    frozen=read(OUT/'FROZEN_MODELS.json')
    models={(m['context'],m['objective']):load(m['path']) for m in frozen['models']}
    units=read(PREV/'PREDICTIONS_FROZEN.json')['units']
    for u in units:
        context='pooled' if u['family']=='external' else u['task']
        z=load(u['path'])
        dest=OUT/'decisions'/(u['uid']+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,query_ids=z['query_ids'],**{
            obj:predict_probe(models[(context,obj)],z['x']) for obj in ('micro','joint')})
    write('PREDICTIONS_FROZEN.json',dict(units=units,held_target_labels_used=False,new_neural_fits=0))
    print('FROZEN',len(units),flush=True)


def evaluate():
    # Import only the existing metric helper; no preparation/training functions are called.
    from run_probe import metrics
    units=read(OUT/'PREDICTIONS_FROZEN.json')['units']
    tables={};rows=[];classrows=[];baselines=pd.read_csv(PREV/'metrics.csv').set_index(['uid','arm'])
    for u in units:
        if u['label_path'] not in tables:
            tables[u['label_path']]=pd.read_csv(u['label_path'],usecols=['class_id'])
        z=load(u['path']);y=tables[u['label_path']].iloc[z['query_ids']].class_id.to_numpy(int)
        d=load(OUT/'decisions'/(u['uid']+'.npz'))
        np.testing.assert_array_equal(d['query_ids'],z['query_ids'])
        prev=load(PREV/'decisions'/(u['uid']+'.npz'))
        old=load(CHOICE/'decisions'/(u['uid']+'.npz'))
        scores={**{k:d[k] for k in ('micro','joint')},'ba':prev['expanded__contextual'],
            'confidence':old['confidence'],'old_score_only':old['score_only']}
        g=gain_labels(z['raw'],z['initial'],y);event=g!=0;w=class_weights(y)
        key={k:u[k] for k in ('family','task','cohort','uid')}
        arms={k:selected_probabilities(z['raw'],z['initial'],s>(0 if k=='confidence' else .5)) for k,s in scores.items()}
        arms.update({k:z[k] for k in ('raw','initial')})
        for arm,p in arms.items():
            result=metrics(y,p,z['raw'],z['initial'])
            if arm in scores:
                np.testing.assert_allclose(result['versus_initial_ba'],result['harm_avoided_ba']-result['rescue_lost_ba'],atol=1e-10)
            if arm=='ba':
                for col in ('oa','ba','versus_initial_oa','versus_initial_ba'):
                    np.testing.assert_allclose(result[col],baselines.loc[(u['uid'],'expanded__contextual'),col],atol=1e-10)
            ba_auc=micro_auc=np.nan
            if arm in scores and len(np.unique(g[event]))==2:
                ba_auc=float(roc_auc_score(g[event]>0,scores[arm][event],sample_weight=w[event]))
                micro_auc=float(roc_auc_score(g[event]>0,scores[arm][event]))
            rows.append(dict(**key,arm=arm,ba_event_auc=ba_auc,micro_event_auc=micro_auc,**result))
            delta=(p.argmax(1)==y).astype(int)-(z['initial'].argmax(1)==y)
            for cls in np.unique(y):
                keep=y==cls
                classrows.append(dict(**key,arm=arm,class_id=int(cls),n=int(keep.sum()),
                    recall=float(100*np.mean(p.argmax(1)[keep]==cls)),
                    oa_contribution=float(100*delta[keep].sum()/len(y)),
                    ba_contribution=float(100*w[keep]@delta[keep])))
    data=pd.DataFrame(rows);data.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(classrows).to_csv(OUT/'classes.csv',index=False)
    cols=['oa','ba','versus_initial_oa','versus_initial_ba','harm_avoided_ba','rescue_lost_ba','ba_event_auc','micro_event_auc']
    task=data.groupby(['family','task','cohort','arm'])[cols].mean().reset_index()
    task.to_csv(OUT/'task_summary.csv',index=False)
    task['scope']=np.where(task.family=='external',task.task,task.family)
    summary=task.groupby(['scope','cohort','arm'])[cols].mean().reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    data.groupby(['family','task','cohort','arm']).agg(oa_sd=('versus_initial_oa','std'),ba_sd=('versus_initial_ba','std'),
        oa_positive=('versus_initial_oa',lambda x:int((x>1e-10).sum())),
        ba_positive=('versus_initial_ba',lambda x:int((x>1e-10).sum())),units=('uid','size')).reset_index().to_csv(OUT/'draw_variation.csv',index=False)
    write('EXPERIMENT_COMPLETE.json',dict(status='complete',new_selector_fits=8,reference_refits=4,
        new_neural_fits=0,evaluation_units=len(units),new_source_episodes=0,main_model_replaced=False))
    print(summary[['scope','cohort','arm','versus_initial_oa','versus_initial_ba']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fit','predict','evaluate'])
    globals()[parser.parse_args().stage]()
