"""Small sufficient-statistic controls shared by two already-frozen protocols."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
Y8 = 2. * np.eye(8) - 1.
DIAG = ('SS', 'TS', 'ST')


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def solve(g, c):
    penalty = np.eye(g.shape[0])
    penalty[-1, -1] = 0.
    return np.linalg.solve(g + penalty, c)


def source_statistics(h, y):
    n = np.bincount(y, minlength=8)
    if (n == 0).any():
        raise ValueError('Expected all eight source classes')
    sums = np.stack([h[y == c].sum(0) for c in range(8)])
    grams = np.stack([h[y == c].T @ h[y == c] for c in range(8)])
    w0 = len(y)/(8.*n)
    g0 = np.einsum('c,cij->ij', w0, grams)
    c0 = (sums*w0[:,None]).T @ Y8
    return dict(n=n, sums=sums, grams=grams, g0=g0, c0=c0, theta0=solve(g0,c0))


def fit_variants(stats, ht, yt):
    n, sums, grams = (stats[k] for k in ('n','sums','grams'))
    k = np.bincount(yt,minlength=8)
    w1 = n.sum()/(8.*(n+k))
    wt = w1[yt]
    gs = np.einsum('c,cij->ij',w1,grams)
    cs = (sums*w1[:,None]).T @ Y8
    gt = ht.T @ (wt[:,None]*ht)
    ct = ht.T @ (wt[:,None]*Y8[yt])
    means_s = sums/n[:,None]
    means_t = means_s.copy()
    for c in np.flatnonzero(k):
        means_t[c] = ht[yt == c].mean(0)
    mass = wt.sum()
    ps, pt = np.full(8,1/8), k*w1/mass
    cross = {
        a+b: mass*(means*pi[:,None]).T @ Y8
        for a,pi in [('S',ps),('T',pt)]
        for b,means in [('S',means_s),('T',means_t)]
    }
    error = float(np.max(np.abs(cross['TT']-ct)))
    np.testing.assert_allclose(cross['TT'],ct,atol=1e-9,rtol=1e-12)
    heads = {'original':stats['theta0'],
             'joint':solve(gs+gt,cs+ct),
             'source_fixed':solve(stats['g0']+gt,stats['c0']+ct)}
    heads.update({name:solve(gs+gt,cs+cross[name]) for name in DIAG})
    details = dict(source_n=int(n.sum()),target_n=int(k.sum()),target_mass=float(mass),
        target_mass_fraction=float(mass/n.sum()),missing_classes=np.flatnonzero(k==0).tolist(),
        cross_replay_error=error,target_counts=k.tolist(),effective_target_proportions=pt.tolist())
    return heads,details


def measures(y, p):
    pred = p.argmax(1)
    correct = pred == y
    nll = -np.log(np.clip(p[np.arange(len(y)),y],1e-15,1.))
    brier = np.sum((p-np.eye(8)[y])**2,axis=1)
    recall = np.array([correct[y==c].mean() for c in range(8)])
    return dict(oa=100.*correct.mean(),ba=100.*recall.mean(),nll=nll.mean(),brier=brier.mean(),
        macro_nll=np.mean([nll[y==c].mean() for c in range(8)]),
        macro_brier=np.mean([brier[y==c].mean() for c in range(8)])),recall


def summarize(frame, keys, values):
    rows = []
    for group,part in frame.groupby(keys,sort=False):
        row = dict(zip(keys,group if isinstance(group,tuple) else (group,)))
        row['draws'] = len(part)
        for col in values:
            row[col] = float(part[col].mean())
            row[col+'_sd'] = float(np.sqrt(part.groupby('split_seed')[col].var(ddof=1).mean()))
            row[col+'_positive'] = int((part[col]>1e-10).sum())
            row[col+'_negative'] = int((part[col]<-1e-10).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_arrays(meta, y, masks, outputs, stage_names, arms):
    metrics, classes, contrasts, factors, events = [],[],[],[],[]
    pairs = [('joint_vs_original','original','joint'),
             ('fixed_vs_original','original','source_fixed'),
             ('fixed_vs_joint','joint','source_fixed'),
             ('composition_with_source_means','SS','TS'),
             ('means_with_source_composition','SS','ST'),
             ('means_with_target_composition','TS','joint'),
             ('composition_with_target_means','ST','joint')]
    for scope,mask in masks.items():
        yy = y[mask]
        if len(np.unique(yy)) != 8:
            raise ValueError('Query cohort no longer has all eight classes')
        for stage in stage_names:
            scores = {a:outputs[stage][i,mask] for i,a in enumerate(arms)}
            measured = {}
            for arm,p in scores.items():
                mm,rr = measures(yy,p)
                measured[arm] = mm
                metrics.append(dict(**meta,scope=scope,stage=stage,arm=arm,n=len(yy),**mm))
                for c in range(8):
                    classes.append(dict(**meta,scope=scope,stage=stage,arm=arm,class_id=c,
                        n=int(np.sum(yy==c)),recall=100.*rr[c]))
            for name,a,b in pairs:
                if a not in scores or b not in scores:
                    continue
                pa,pb = scores[a].argmax(1),scores[b].argmax(1)
                rescues,harms = (pa!=yy)&(pb==yy),(pa==yy)&(pb!=yy)
                contrasts.append(dict(**meta,scope=scope,stage=stage,comparison=name,
                    **{'delta_'+k:measured[b][k]-measured[a][k] for k in measured[a]},
                    rescues=int(rescues.sum()),harms=int(harms.sum())))
                if stage == 'full' and name in ('joint_vs_original','fixed_vs_original'):
                    positions = np.flatnonzero(mask)
                    for j in np.flatnonzero(pa!=pb):
                        events.append(dict(**meta,scope=scope,stage=stage,comparison=name,
                            position=int(positions[j]),true_class=int(yy[j]),before=int(pa[j]),after=int(pb[j]),
                            event='rescue' if rescues[j] else ('harm' if harms[j] else 'wrong_to_wrong')))
            for metric in ('oa','ba','nll','brier'):
                ss,ts,st,tt = (measured[a][metric] for a in ('SS','TS','ST','joint'))
                composition=.5*((ts-ss)+(tt-st))
                conditional=.5*((st-ss)+(tt-ts))
                np.testing.assert_allclose(composition+conditional,tt-ss,atol=1e-10)
                factors.append(dict(**meta,scope=scope,stage=stage,metric=metric,
                    composition=composition,conditional_mean=conditional,total=tt-ss,
                    interaction=tt-ts-st+ss))
    return metrics,classes,contrasts,factors,events


def save_reports(destination, results):
    names = ('metrics','class_recalls','contrasts','factors','events')
    frames = {}
    for name, rows in zip(names,results):
        frames[name] = pd.DataFrame(rows)
        frames[name].to_csv(destination/(name+'.csv'),index=False)
    for name,keys,values in [
        ('metrics',['task','scope','stage','arm'],['oa','ba','nll','brier','macro_nll','macro_brier']),
        ('contrasts',['task','scope','stage','comparison'],
         ['delta_oa','delta_ba','delta_nll','delta_brier','delta_macro_nll','delta_macro_brier']),
        ('factors',['task','scope','stage','metric'],['composition','conditional_mean','total','interaction']),
    ]:
        summary=summarize(frames[name],keys,values)
        summary.to_csv(destination/(name+'_summary.csv'),index=False)
    return frames
