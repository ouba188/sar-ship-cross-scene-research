"""Trace rollback decisions and test fixed predicted-class identity probes."""

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PREV = ROOT / 'correction_episode_rebuild_20261013'
CHOICE = ROOT / 'correction_choice_probe_20261013'
BALANCE = ROOT / 'correction_objective_balance_20261013'
sys.path.insert(0, str(CHOICE))
from choice_core import class_weights, gain_labels, predict_probe, selected_probabilities, NAMES

CONTEXTS = ('Los Angeles', 'Port Klang', 'Santos')
GROUPS = {'confidence': slice(0, 6), 'base_agreement': slice(6, 8),
          'support_oof': slice(8, 12), 'support_count': slice(12, 14),
          'support_coverage': slice(14, 16)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def source_units():
    return [u for u in read(PREV / 'SOURCE_READY.json')['units'] if u['kind'] == 'expanded']


def base_model(context):
    return load(PREV / 'models' / f'expanded__{context}__contextual.npz')


def design(z, reference, arm):
    x = (z['x'] - reference['mean']) / reference['scale']
    r, i = z['raw'].argmax(1), z['initial'].argmax(1)
    direction = np.eye(8)[i] - np.eye(8)[r]
    return x if arm == 'reference' else direction if arm == 'class_only' else np.c_[x, direction]


def fit():
    meta = pd.read_csv(ROOT / 'same_vessel_condition_study_20261010/observations.csv', dtype={'mmsi': str})
    units = source_units()
    infos, replays, coefs = [], [], []
    for context in ('pooled',) + CONTEXTS:
        subset = [u for u in units if context == 'pooled' or u['context'] == context]
        ref = base_model(context)
        data = [load(u['path']) for u in subset]
        for z in data:
            np.testing.assert_array_equal(z['labels'], meta.iloc[z['query_ids']].class_id)
        g = np.concatenate([gain_labels(z['raw'], z['initial'], z['labels']) for z in data])
        w = np.concatenate([class_weights(z['labels']) / len(data) for z in data])
        event = g != 0
        w = w[event] * (float(ref['event_weight_mass']) / w[event].sum())
        for arm in ('reference', 'class_only', 'role_class'):
            x = np.concatenate([design(z, ref, arm) for z in data])
            model = LogisticRegression(C=.1, solver='lbfgs', max_iter=1000, tol=1e-9)
            model.fit(x[event], g[event] > 0, sample_weight=w)
            if arm == 'reference':
                expected = np.concatenate([predict_probe(ref, z['x']) for z in data])
                actual = model.predict_proba(x)[:, 1]
                np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
                replays.append(dict(context=context, max_error=float(abs(actual-expected).max())))
                continue
            dest = OUT / 'models' / f'{context}__{arm}.npz'
            dest.parent.mkdir(exist_ok=True)
            np.savez(dest, coef=model.coef_[0], intercept=model.intercept_[0])
            infos.append(dict(context=context, arm=arm, path=str(dest), events=int(event.sum()),
                              iterations=int(model.n_iter_[0]), mass=float(w.sum())))
            names = (list(NAMES) if arm == 'role_class' else []) + [f'class_direction_{c}' for c in range(8)]
            coefs.extend(dict(context=context, arm=arm, feature=n, coefficient=float(v))
                         for n, v in zip(names, model.coef_[0]))
        print('FIT', context, flush=True)
    pd.DataFrame(replays).to_csv(OUT/'fit_replays.csv', index=False)
    pd.DataFrame(coefs).to_csv(OUT/'probe_coefficients.csv', index=False)
    write('FROZEN_MODELS.json', dict(models=infos, new_fits=8, reference_refits=4,
                                    target_labels_used=False, C=.1, threshold=.5))


def predict():
    frozen = read(OUT/'FROZEN_MODELS.json')
    fitted = {(m['context'], m['arm']): load(m['path']) for m in frozen['models']}
    units = read(PREV/'PREDICTIONS_FROZEN.json')['units']
    (OUT/'decisions').mkdir(exist_ok=True)
    for u in units:
        ctx = 'pooled' if u['family'] == 'external' else u['task']
        z, ref = load(u['path']), base_model(ctx)
        probs = {}
        for arm in ('class_only', 'role_class'):
            m = fitted[ctx, arm]
            probs[arm] = expit(design(z, ref, arm) @ m['coef'] + m['intercept'])
        np.savez_compressed(OUT/'decisions'/(u['uid']+'.npz'), query_ids=z['query_ids'], **probs)
    write('PREDICTIONS_FROZEN.json', dict(units=units, target_labels_used=False))
    print('FROZEN', len(units), flush=True)


def summarize(df, fields, filename):
    task = df.groupby(['family', 'task', 'cohort', 'arm'])[fields].mean().reset_index()
    task.to_csv(OUT/(filename+'_task.csv'), index=False)
    task['scope'] = np.where(task.family == 'external', task.task, task.family)
    task.groupby(['scope', 'cohort', 'arm'])[fields].mean().reset_index().to_csv(OUT/(filename+'.csv'), index=False)


def evaluate():
    units = read(OUT/'PREDICTIONS_FROZEN.json')['units']
    rows, clsrows, comparisons, maps, replays = [], [], [], [], []
    refs = pd.read_csv(BALANCE/'metrics.csv').set_index(['uid', 'arm'])
    tables = {}
    for u in units:
        if u['label_path'] not in tables:
            tables[u['label_path']] = pd.read_csv(u['label_path'], usecols=['class_id', 'class_name'])
            maps.extend(dict(dataset=u['dataset'], class_id=int(r.class_id), class_name=r.class_name)
                        for r in tables[u['label_path']].drop_duplicates().itertuples())
        z = load(u['path'])
        y = tables[u['label_path']].iloc[z['query_ids']].class_id.to_numpy(int)
        w, g = class_weights(y), gain_labels(z['raw'], z['initial'], y)
        d = load(OUT/'decisions'/(u['uid']+'.npz'))
        np.testing.assert_array_equal(d['query_ids'], z['query_ids'])
        old = load(PREV/'decisions'/(u['uid']+'.npz'))['expanded__contextual']
        bal = load(BALANCE/'decisions'/(u['uid']+'.npz'))['joint']
        conf = z['initial'].max(1) > z['raw'].max(1)
        scores = {a:d[a] for a in ('class_only','role_class')}
        choices = {a:s > .5 for a,s in scores.items()}
        choices.update(ba=old>.5, joint=bal>.5, confidence=conf)
        arms = {a:selected_probabilities(z['raw'], z['initial'], c) for a,c in choices.items()}
        arms.update({a:z[a] for a in ('raw', 'initial', 'combined') if a in z})
        actuals = {}
        for arm,p in arms.items():
            pred = p.argmax(1)
            delta = (pred==y).astype(int) - (z['initial'].argmax(1)==y)
            oa,ba = float(100*np.mean(pred==y)),float(100*w@(pred==y))
            avoided,lost = float(100*w@((g==-1)&(pred==y))),float(100*w@((g==1)&(pred!=y)))
            if arm in choices:
                np.testing.assert_allclose(100*w@delta, avoided-lost, atol=1e-10)
            auc = np.nan
            if arm in scores and len(np.unique(g[g!=0]))==2:
                auc = roc_auc_score(g[g!=0]>0, scores[arm][g!=0], sample_weight=w[g!=0])
            row = dict(**{k:u[k] for k in ('family','task','cohort','uid')}, arm=arm,
                       oa=oa,ba=ba,versus_initial_oa=float(100*delta.mean()),versus_initial_ba=float(100*w@delta),
                       harm_avoided_ba=avoided,rescue_lost_ba=lost,auc=auc)
            if arm in ('raw','initial','ba','joint','confidence'):
                for field in ('oa','ba','versus_initial_oa','versus_initial_ba'):
                    err = abs(row[field]-refs.loc[(u['uid'],arm),field])
                    assert err < 1e-9
                    replays.append(dict(uid=u['uid'],arm=arm,field=field,error=err))
            rows.append(row);actuals[arm]=row
            parts=[]
            for cl in np.unique(y):
                keep=y==cl
                part=dict(uid=u['uid'],family=u['family'],task=u['task'],cohort=u['cohort'],arm=arm,
                          class_id=int(cl),n=int(keep.sum()),recall=float(100*np.mean(pred[keep]==cl)),
                          oa_contribution=float(100*delta[keep].sum()/len(y)),ba_contribution=float(100*w[keep]@delta[keep]))
                parts.append(part);clsrows.append(part)
            np.testing.assert_allclose([sum(p['oa_contribution'] for p in parts),sum(p['ba_contribution'] for p in parts)],
                                       [row['versus_initial_oa'],row['versus_initial_ba']],atol=1e-10)
        for arm in ('class_only','role_class'):
            for control in ('ba','joint','confidence','raw','initial','combined'):
                if control in actuals:
                    comparisons.append(dict(**{k:u[k] for k in ('family','task','cohort','uid')},arm=arm+'__minus__'+control,
                        oa=actuals[arm]['oa']-actuals[control]['oa'],ba=actuals[arm]['ba']-actuals[control]['ba']))
    df=pd.DataFrame(rows);df.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(clsrows).to_csv(OUT/'classes.csv',index=False)
    pd.DataFrame(maps).drop_duplicates().to_csv(OUT/'class_mapping.csv',index=False)
    pd.DataFrame(replays).to_csv(OUT/'metric_replays.csv',index=False)
    fields=['oa','ba','versus_initial_oa','versus_initial_ba','harm_avoided_ba','rescue_lost_ba','auc']
    summarize(df,fields,'summary')
    cf=pd.DataFrame(comparisons);cf.to_csv(OUT/'comparisons.csv',index=False)
    summarize(cf,['oa','ba'],'comparison_summary')
    df['both_positive']=(df.versus_initial_oa>1e-10)&(df.versus_initial_ba>1e-10)
    df.groupby(['family','task','cohort','arm']).agg(oa_sd=('versus_initial_oa','std'),ba_sd=('versus_initial_ba','std'),
        oa_positive=('versus_initial_oa',lambda x:int((x>1e-10).sum())),ba_positive=('versus_initial_ba',lambda x:int((x>1e-10).sum())),
        both_positive=('both_positive','sum'),units=('uid','size')).reset_index().to_csv(OUT/'variation.csv',index=False)
    write('EVALUATION_COMPLETE.json',dict(units=len(units),new_fits=8,reference_refits=4,neural_fits=0,
        metric_replays=len(replays),class_sum_checks=2*len(df),main_replaced=False))
    print(pd.read_csv(OUT/'summary.csv').query("arm in ['role_class','class_only']").to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fit','predict','evaluate'])
    globals()[parser.parse_args().stage]()
