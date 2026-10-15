"""Paired effects and fixed-query class/score/calibration decomposition."""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from analyze import readout, weights, oscr
from study import METHODS, write

ARMS = ('random', 'cover1', 'sham1', 'cover3')
METRICS = ('known_ba_before', 'known_ba_after', 'known_false_reject', 'unknown_tpr',
           'utility_05', 'utility_20', 'utility_50', 'total_05', 'total_20', 'total_50', 'auroc')


def measure(score, threshold, y, pred, unknown):
    r = readout(score, threshold, y, pred, unknown)
    for pi in (.05, .2, .5):
        r[f'total_{int(pi*100):02d}'] = (1-pi)*r['known_ba_after']+pi*r['unknown_tpr']
    return r


def main():
    plan = pd.DataFrame(json.loads((OUT/'SUPPORT_PLAN.json').read_text(encoding='utf-8')))
    lookup = {(r.task, r.unknown, r.draw, r.arm): r for r in plan.itertuples()}
    metrics, class_rows, paths = [], [], []
    checks = 0
    for base_path in sorted((OUT/'tasks').rglob('unknown_*_random.npz')):
        task = base_path.parent.relative_to(OUT/'tasks').as_posix()
        unknown = int(base_path.stem.split('_')[1])
        base = np.load(base_path)
        y = base['labels']; known, w, uw = weights(y, unknown)
        for arm in ARMS:
            z = np.load(base_path.with_name(f'unknown_{unknown}_{arm}.npz'))
            np.testing.assert_array_equal(z['query'], base['query'])
            np.testing.assert_array_equal(z['labels'], y)
            for draw in range(5):
                rec = lookup[(task, unknown, draw, arm)]
                oldcounts = np.array(rec.old_counts); counts = np.array(rec.counts)
                pred = z['classes'][z['probability'][draw].argmax(1)]
                pred0 = base['classes'][base['probability'][draw].argmax(1)]
                if rec.changed == 0:
                    np.testing.assert_array_equal(z['probability'][draw], base['probability'][draw])
                    np.testing.assert_array_equal(z['score'][draw], base['score'][draw])
                for mi, method in enumerate(METHODS):
                    score = z['score'][draw, :, mi]; threshold = z['thresholds'][draw, mi]
                    np.testing.assert_allclose(threshold, np.quantile(z['calibration_score'][draw, :, mi], .95), atol=1e-12)
                    rejection = score > threshold
                    score0 = base['score'][draw, :, mi]; threshold0 = base['thresholds'][draw, mi]
                    rej0 = score0 > threshold0
                    result = measure(score, threshold, y, pred, unknown)
                    auc = 100*roc_auc_score(y == unknown, score, sample_weight=w+uw)
                    metrics.append(dict(task=task, unknown=unknown, draw=draw, arm=arm, method=method,
                        changed=rec.changed, original_missing=rec.original_missing, missing_after=rec.missing_after,
                        threshold=threshold, delta_threshold=threshold-threshold0,
                        auroc=auc, oscr=oscr(score, y, pred, unknown), **result))
                    for c in range(8):
                        use = y == c
                        if not use.any():
                            continue
                        group = ('unknown' if c == unknown else 'original_missing' if oldcounts[c] == 0
                                 else 'donor' if counts[c] < oldcounts[c] else 'original_supported')
                        class_rows.append(dict(task=task, unknown=unknown, draw=draw, arm=arm, method=method,
                            class_id=c, group=group, n=int(use.sum()), old_support=int(oldcounts[c]),
                            new_support=int(counts[c]), original_missing=rec.original_missing,
                            reject_before=100*rej0[use].mean(), reject_after=100*rejection[use].mean(),
                            delta_reject=100*(rejection[use].astype(int)-rej0[use]).mean(),
                            delta_score=float((score[use]-score0[use]).mean()),
                            ba_before=100*(pred0[use] == c).mean(), ba_after=100*(pred[use] == c).mean(),
                            accepted_correct_before=100*((pred0[use] == c)&~rej0[use]).mean(),
                            accepted_correct_after=100*((pred[use] == c)&~rejection[use]).mean(),
                            rescued_rejects=100*(rej0[use]&~rejection[use]).mean(),
                            new_rejects=100*(~rej0[use]&rejection[use]).mean()))
                    # Independent class means and decision-transition identity.
                    cls = np.unique(y[known])
                    frr = 100*np.mean([rejection[y == c].mean() for c in cls])
                    after = 100*np.mean([((pred[y == c] == c)&~rejection[y == c]).mean() for c in cls])
                    tpr = 100*rejection[~known].mean()
                    np.testing.assert_allclose([frr, after, tpr],
                        [result['known_false_reject'], result['known_ba_after'], result['unknown_tpr']], atol=1e-9)
                    before_decision = ((pred0 == y)&~rej0).astype(int)
                    after_decision = ((pred == y)&~rejection).astype(int)
                    direct = 100*(.8*w@(after_decision-before_decision)+.2*uw@(rejection.astype(int)-rej0))
                    baseline = measure(score0, threshold0, y, pred0, unknown)
                    np.testing.assert_allclose(result['total_20']-baseline['total_20'], direct, atol=1e-9)
                    checks += 4
                    if method == 'support_knn':
                        for stage, sc, th, pp in (
                            ('original', score0, threshold0, pred0),
                            ('distance_only', score, threshold0, pred0),
                            ('distance_calibration', score, threshold, pred0),
                            ('actual', score, threshold, pred)):
                            paths.append(dict(task=task, unknown=unknown, draw=draw, arm=arm,
                                original_missing=rec.original_missing, stage=stage,
                                **measure(sc, th, y, pp, unknown)))
    m = pd.DataFrame(metrics); c = pd.DataFrame(class_rows); p = pd.DataFrame(paths)
    m.to_csv(OUT/'metrics.csv', index=False)
    c.to_csv(OUT/'class_effects.csv', index=False)
    p.to_csv(OUT/'path_decomposition.csv', index=False)
    keys = ['task', 'unknown', 'draw', 'method']
    ref = m[m.arm == 'random'][keys+list(METRICS)].rename(columns={x:'random_'+x for x in METRICS})
    paired = m.merge(ref, on=keys, validate='many_to_one')
    for metric in METRICS:
        paired['delta_'+metric] = paired[metric]-paired['random_'+metric]
    paired.to_csv(OUT/'paired.csv', index=False)
    sham = m[m.arm == 'sham1'][keys+list(METRICS)].rename(columns={x:'sham_'+x for x in METRICS})
    versus_sham = m[m.arm == 'cover1'].merge(sham, on=keys, validate='one_to_one')
    for metric in METRICS:
        versus_sham['delta_'+metric] = versus_sham[metric]-versus_sham['sham_'+metric]
    versus_sham.to_csv(OUT/'coverage_vs_sham.csv', index=False)
    summaries = []
    for scope, use in [('all', paired), ('original_missing', paired[paired.original_missing > 0]),
                       ('fishing_unknown', paired[paired.unknown == 1])]:
        agg = use.groupby(['task', 'arm', 'method']).mean(numeric_only=True).reset_index()
        agg['scope'] = scope
        summaries.append(agg)
        seed = use.groupby(['task', 'arm', 'method', 'draw'])[['delta_total_20', 'delta_known_false_reject', 'delta_unknown_tpr']].mean().reset_index()
        seed.to_csv(OUT/f'draws_{scope}.csv', index=False)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT/'summary.csv', index=False)
    c.groupby(['task', 'arm', 'method', 'group'])[
        ['reject_before', 'reject_after', 'delta_reject', 'ba_before', 'ba_after',
         'accepted_correct_before', 'accepted_correct_after', 'delta_score']
    ].mean().reset_index().to_csv(OUT/'class_summary.csv', index=False)
    p.groupby(['task', 'arm', 'stage'])[['known_false_reject', 'unknown_tpr', 'total_20']].mean().reset_index().to_csv(OUT/'path_summary.csv', index=False)
    plan.drop(columns=['supports', 'counts', 'old_counts']).to_csv(OUT/'support_changes.csv', index=False)
    print('COVERAGE COUNTS', plan[plan.arm == 'cover1'].groupby('task')[['original_missing','changed','missing_after']].agg(['mean','max']).to_string())
    cols = ['task','arm','delta_known_ba_before','delta_known_false_reject','delta_unknown_tpr','delta_total_20','delta_utility_20']
    for scope in ('all', 'original_missing', 'fishing_unknown'):
        print('SUPPORT DISTANCE', scope)
        print(summary[(summary.method == 'support_knn')&(summary.scope == scope)][cols].to_string(index=False))
    focus = summary[(summary.scope == 'all')&(summary.method == 'support_knn')]
    names = list(focus.task.unique())
    fig, axes = plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax, metric, title in zip(axes, ['delta_known_false_reject','delta_unknown_tpr','delta_total_20'],
        ['Known false rejection (lower is better)','Unknown detection (higher is better)','Total utility at 20% unknown']):
        for j,(arm,color) in enumerate([('cover1','#318a9f'),('sham1','#888888'),('cover3','#b46a40')]):
            table = focus[focus.arm == arm].set_index('task')
            ax.bar(np.arange(len(names))+(j-1)*.24,table.loc[names,metric],.24,label=arm,color=color)
        ax.set_xticks(np.arange(len(names)),[n.removeprefix('ports/') for n in names],rotation=30,ha='right')
        ax.axhline(0,color='#555555',lw=.8);ax.set_title(title,fontsize=10);ax.set_ylabel('Change vs random (percentage points)')
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Fixed 64 supports: class coverage intervention, compact frozen-feature pilot',fontsize=11)
    fig.savefig(OUT/'COVERAGE_EFFECTS.png',dpi=170);fig.savefig(OUT/'COVERAGE_EFFECTS.pdf');plt.close(fig)
    write(OUT/'ANALYSIS_COMPLETE.json',dict(metric_rows=len(m),independent_metric_checks=checks,
        class_rows=len(c),new_neural_module=False,primary_quantile=.95))


if __name__ == '__main__':
    main()
